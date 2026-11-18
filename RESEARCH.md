# Compatibility and performance notes

Documentation checked **2026-11-07**. Upstream revisions and exact artifact
sizes/SHA256 values are recorded in `Dockerfile` and `models.json`.

## Validation performed

The complete Docker image built successfully, including both CUDA `sm_70`
engines (about 20 minutes with two build jobs per engine). Three CPU-only
harness contract tests passed. OpenCode 2.2.6 also completed a real CLI request
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
  cache slots all consume VRAM. The initial text-only profile is deliberately
  32K context with one request lane, rather than the model's theoretical 262K.
- Vast SSH mode replaces the image entrypoint. Template environment variables
  may not appear in interactive SSH shells; the README explicitly sets PATH.

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
