# Compatibility and performance notes

Documentation and pinned source checked **2026-11-21**. Upstream revisions and exact artifact
sizes/SHA256 values are recorded in `Dockerfile` and `models.json`.

## Validation performed

The complete Docker image built successfully, including both CUDA `sm_70`
engines (about 20 minutes with two build jobs per engine). Five CPU-only
harness contract tests pass, including the 128K launch profile and the capacity
check's full-output/early-EOS handling. OpenCode 2.2.6 also completed a real CLI request
against a mock local API with non-strict tools and a streamed response. GPU
model loading, generation quality, memory use and throughput still require a
V100 test run; no NVIDIA GPU/driver was available on the build host.

## V100 requirements

- Qwen3.8-27B is a real published checkpoint. BF16 weights alone exceed 32 GB.
  llama.cpp uses Unsloth UD-Q4_K_M; Turbo uses the mixed NVFP4/FP8 NInfer artifact.
- V100 is Volta, compute capability 7.0. Both engines compile explicitly for
  `sm_70` with CUDA 12.8.1. CUDA 13 cannot compile this architecture. Turbo's
  generic Dockerfile targets Blackwell, so its V100 build instructions matter.
- Turbo executes NVFP4 **in software using FP16/SIMT kernels**; V100 does not
  have native FP4, FP8, or BF16 tensor cores. INT8 KV is supported by this port;
  NVFP4/K8V4 KV is not. llama.cpp uses its own q8_0 KV representation.
- Turbo needs CMake >=3.28, a C++20 compiler, Python for CUTLASS configuration,
  FFmpeg development libraries and libcurl. Ubuntu 24.04 supplies these.
- Context length, graph/workspace buffers, recurrent state, draft weights and
  cache slots all consume VRAM. The default text-only profile is 32K with one
  request lane; `CONTEXT_SIZE=131072` enables the lower-memory 128K profile.
- Vast SSH mode replaces the image entrypoint. Template environment variables
  may not appear in interactive SSH shells; the README explicitly sets PATH.

## 128K memory budget and OOM controls

**128K (131,072 total tokens) is a credible target on one V100 32 GB.** The
pinned V100 report includes a 150,000-token MTP run at 43.63 decode tok/s and
79.2% acceptance. That is real long-context execution evidence, although it is
one maintainer-reported measurement, not an OOM guarantee for every rental.

The model's [configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
has 64 layers: **16 full-attention** and **48 recurrent/linear-attention** layers,
4 KV heads and head dimension 256. The main attention KV payload is therefore:

`tokens × 16 layers × 4 KV heads × 256 dimensions × 2 (K and V) × storage bytes`

| Main KV storage | At 32K | At 128K | Increase |
|---|---:|---:|---:|
| FP16/BF16 | 2.00 GiB | 8.00 GiB | 6.00 GiB |
| llama.cpp q8_0 (34 bytes per 32 values) | 1.06 GiB | 4.25 GiB | 3.19 GiB |
| NInfer INT8 group-64 (64 codes + 2-byte scale) | 1.03 GiB | 4.13 GiB | 3.09 GiB |

These are **calculated main-KV payloads**, not measured total VRAM. Page tables,
alignment, recurrent states, MTP/DFlash state, graphs, workspace and the driver
also need memory. A basic FP32 recurrent matrix state alone is about 144 MiB
across the 48 recurrent layers, before convolution/history and extra copies.
The artifact card reports 19.729 GiB resident MTP3 weights on RTX 5090; that is
useful sizing context, not a V100 memory measurement. Its 22.09 GiB disk file
includes components which are only loaded when selected, including vision.

### Changes applied by the launcher

| Control | Default 32K profile | 128K profile | Effect |
|---|---|---|---|
| Active requests | 1 | 1 | Avoid duplicated live histories and workspaces |
| Vision | off | off | Avoid vision weights and workspace |
| Main KV | q8_0 / INT8 | q8_0 / INT8 | Keep the existing 8-bit precision |
| Turbo KV sizing | auto | auto | Require full configured capacity plus 1 GiB sizing headroom |
| Turbo prefill chunk | 2048 | 1024 | Reduce prefill-dependent temporary buffers |
| Turbo extra GPU state slots | 1 | 0 | Keep one active state; retain checkpoints through host tiers |
| llama.cpp physical microbatch | 512 | 256 | Reduce prompt-processing temporary buffers |
| llama.cpp context checkpoints | 32 | 4 | Bound **host RAM** used by saved recurrent snapshots |
| llama.cpp prompt RAM cache | 8192 MiB | 2048 MiB | Bound **host RAM**, separate from live GPU KV |

CUDA graphs, prefix reuse and the selected speculative backend remain enabled.
Smaller batches can lower prefill throughput; host checkpoint restores can add
latency. These choices aim to buy memory without changing the weight/KV quant
or silently shrinking context. Benchmark the tradeoff on your card.

**Pinned-version detail:** NInfer's planner requires `kv_capacity >= max_context`.
At concurrency one its minimum and maximum pool capacities both cover exactly
the configured context (rounded to pages). `--kv-capacity auto` checks the
complete planned runtime allocation against available memory after weights,
including workspace and graph allowance, while retaining **1 GiB headroom**.
If that cannot fit, startup fails. It does **not** quietly substitute a smaller
KV pool. This corrects the earlier suggestion that auto could provide less
than the requested logical context on this pinned version. Explicit capacity
does not add this automatic headroom check.

### Qualification and fallback order

1. Rent an otherwise idle 32 GB V100 and **64 GB host RAM**; keep one server.
2. Run `CONTEXT_SIZE=131072 serve.sh turbo`, then `context-check.py` from another
   terminal with the same environment. Repeat for each intended backend/profile.
   The check measures almost-full prompt plus full 4,096-token generation on two
   requests, including repeat/cache pressure. A short prompt after startup is
   insufficient qualification. A successful result is evidence for the tested
   workload; 0.5-second sampling can miss short-lived peaks.
3. Keep at least **1 GiB observed free VRAM**; 1.5–2 GiB is a preferable margin
   where achievable (`--min-free-mib 1536` sets a stricter check). The sampler
   measures headroom, it does not reserve memory or replace CUDA's allocator.
4. If tight, use `PREFILL_CHUNK=512` for Turbo or `UBATCH_SIZE=128` for llama.cpp.
5. Next test `turbo-nospec` / `llama` to release speculative weights/state.
   Re-qualify this changed profile; decode speed can decrease substantially.
6. Turbo's `--no-cuda-graph` is another last-resort saving with a speed cost.
   llama.cpp supports explicit partial layer offload (`--n-gpu-layers N`) if
   needed, with a potentially large PCIe/CPU penalty. Reduce context if the
   desired all-GPU profile cannot meet the margin.

`HOST_KV_MIB` and `HOST_STATE_SLOTS` control inactive-cache RAM, not extra VRAM
for a live 128K request. Increasing them cannot make insufficient active GPU
capacity fit. Keep OpenCode's context at 131,072 too: this total includes the
prompt, tool schemas/history, reasoning and output. The wrapper reserves its
normal 4,096-token model output limit; OpenCode compacts as the window fills.

Four-bit KV is not needed for this first 128K target. NInfer's NVFP4/K8V4 KV
paths are unsupported on Volta; llama.cpp q4_0 KV would change precision and
needs a different FlashAttention build matrix than this image. PyTorch allocator
environment variables do not tune these C++ engines.

Relevant pinned implementation references:

- [NInfer KV sizing/headroom](https://github.com/b-iurea/ninfer-v100-turbo/blob/56f7b091992fa9af85fecad646de954f6b459b77/src/runtime/engine/kv_capacity.cpp),
  [default 1 GiB margin](https://github.com/b-iurea/ninfer-v100-turbo/blob/56f7b091992fa9af85fecad646de954f6b459b77/include/ninfer/types.h#L48),
  [full-context minimum and graph/workspace planning](https://github.com/b-iurea/ninfer-v100-turbo/blob/56f7b091992fa9af85fecad646de954f6b459b77/src/targets/qwen3_6/impl/runtime/layouts_impl.h)
- [INT8 KV storage geometry](https://github.com/b-iurea/ninfer-v100-turbo/blob/56f7b091992fa9af85fecad646de954f6b459b77/src/core/paged_kv_storage.h),
  [device/host state semantics](https://github.com/b-iurea/ninfer-v100-turbo/blob/56f7b091992fa9af85fecad646de954f6b459b77/docs/maintainer/resource-scheduling-and-context-cache.md)
- [llama.cpp checkpoint storage](https://github.com/ggml-org/llama.cpp/blob/e95426f16982054c379ca7bdc8186996dd22f0fe/common/common.cpp#L2420),
  [server controls](https://github.com/ggml-org/llama.cpp/blob/e95426f16982054c379ca7bdc8186996dd22f0fe/tools/server/README.md)

## Is ninfer-v100-turbo valuable?

**Worth testing on this exact GPU.** Its Volta-specific FP16 activation staging
has a documented same-card, same-artifact before/after A/B:

| Mode | Before | Turbo | Change |
|---|---:|---:|---:|
| No speculation | 29.43 tok/s | 34.31 tok/s | +16.6% |
| MTP K=1, synthetic continuation | 220.12 tok/s | 259.99 tok/s | +18.1% |

The 260 tok/s headline uses approximately 99% draft acceptance on a synthetic
corpus. It is not an OpenCode throughput prediction. Published varied-context
MTP results are roughly 44–93 tok/s, depending on depth and acceptance, from a
different sweep. These are **maintainer-reported**, not independently measured
here. MTP K=3 is the repository's practical general-generation starting point.

The local harness compares complete stacks with **different quantizations**;
it cannot establish the fork's claimed gain over upstream NInfer in isolation.
`turbo-nospec` versus `turbo` measures speculation within this fork. `llama-mtp`
also enables llama.cpp's learned MTP path. DFlash2 is available as another test.

## OpenCode

The image pins OpenCode **2.2.6**, matching the current V2 provider/schema
documentation. It uses the OpenAI-compatible **Chat Completions** package,
explicit tool capability, and `reasoning_content` replay. Thinking effort is
sent as a request-body field so the exact Qwen levels (`low`, `medium`, `xhigh`,
and `none`) reach both engines. Each invocation uses `--standalone`, so changes
to endpoint/context environment cannot be hidden by an old background service.

Turbo parses calls and streams reasoning/results, but rejects strict JSON-schema
tools, required/named tool choice, JSON-constrained responses, and some other
OpenAI features. The configuration uses non-strict, automatically selected tools.
The smoke test verifies parsed calls, argument JSON, replay and a streamed answer;
the separate OpenCode coding task checks actual agent file/shell work.

## Sources

- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
  [GGUF files](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main),
  [NInfer manifest](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/blob/main/artifact-manifest.json)
- [llama.cpp CUDA build](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cuda)
  and [server options](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [Turbo V100 build, qualification and A/B](https://github.com/b-iurea/ninfer-v100-turbo/blob/master/docs/v100.md),
  [V100 Dockerfile](https://github.com/b-iurea/ninfer-v100-turbo/blob/master/Dockerfile.v100),
  [serving contract](https://github.com/b-iurea/ninfer-v100-turbo/blob/master/docs/serving.md)
- [NVIDIA CUDA 12.8.1 driver compatibility](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html)
- [Vast Docker/SSH environment](https://docs.vast.ai/documentation/instances/docker-environment)
- [OpenCode installation](https://opencode.ai/docs/),
  [custom providers](https://opencode.ai/docs/providers/#custom),
  [reasoning compatibility](https://opencode.ai/docs/models/#compatibility),
  [configuration schema](https://opencode.ai/config.json)
