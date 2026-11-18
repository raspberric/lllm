# Qwen3.8-27B · V100 32 GB · Vast.ai

One image with **llama.cpp**, **ninfer-v100-turbo**, and **OpenCode 2.2.6**.
Pinned engines/models, verified downloads, text-only, 32K context, one request.
Turbo is worth testing, but its 260 tok/s headline is a synthetic ceiling.
See [research, compatibility details and sources](RESEARCH.md).

## 1. Build and rent

On an **x86-64/AVX2 Docker host** (no GPU needed):

```bash
docker build --build-arg BUILD_JOBS=2 -t YOUR_DOCKERHUB_USER/qwen-v100:1 .
docker push YOUR_DOCKERHUB_USER/qwen-v100:1
```

Allow **25 GB free build disk** and **20–40+ minutes** for CUDA compilation.
Use `BUILD_JOBS=1` if memory is tight; BuildKit caches interrupted compilation.

Create a Vast custom template with your image, **SSH launch mode**, and:

- **1× V100 32 GB** (SXM2 preferred), **48 GB+ host RAM** (64 GB preferred).
- **100 GB disk**, NVIDIA driver **570.124.06+ with V100 support**.
- **On-start Script:** `/opt/qwen/scripts/entrypoint.sh true`.

Vast supplies the driver and replaces the Docker entrypoint in SSH mode.
CUDA 12.8.1 is included; CUDA 13 cannot compile Volta. Connect with Vast's SSH
command. In **each SSH terminal**, export:

```bash
export DATA_DIR=/workspace
export PATH=/opt/hf/bin:/opt/qwen/scripts:$PATH
```

## 2. Download and compare

```bash
nvidia-smi
download-models.py all
compare.py llama turbo
# Optional: learned MTP on both engines, plus Turbo controls:
compare.py llama-mtp turbo-nospec turbo turbo-dflash2
```

Downloads total **41.6 GB** and resume after interruption; no HF login required.
Use `tmux` for long runs. Engines run sequentially, with readiness checks,
a two-turn tool-call smoke test, and three coding prompts × one warmup + three
measurements. Results: `/workspace/results/TIMESTAMP/BACKEND/`:

- `summary.json`: median first-token latency, end-to-end tok/s, server decode tok/s.
- `requests.jsonl`: token counts, full answers/reasoning, timings and warmup labels.
- `tools.log`, `server.log`: compatibility result and engine diagnostics.

Both get the same prompts and sampling: low reasoning, 512 thinking tokens,
2,048 total output tokens. Leading markers change to avoid full-prefix cache
hits. Check answers and truncation flags: this is not an automatic quality score.
Tool failures are reported without blocking speed measurements.

`llama` uses **UD-Q4_K_M**, `turbo` uses **mixed NVFP4/FP8**: this compares
complete stacks, not identical quantization or isolated kernel speed.
Try `DRAFT_TOKENS=1 compare.py llama-mtp turbo` to compare another MTP window.

## 3. Test OpenCode

In one SSH terminal, start **one** backend and wait for readiness:

```bash
serve.sh turbo                  # or: serve.sh llama / serve.sh llama-mtp
```

In another terminal, with the exports above:

```bash
mkdir -p /workspace/agent-turbo
cd /workspace/agent-turbo
opencode-local.py
```

Paste `/opt/qwen/fixtures/coding-task.md`: it asks for a Python implementation,
tests and a real test run. Repeat in a **fresh directory/session** for each
engine. Or run the task directly:

```bash
opencode-local.py run "$(</opt/qwen/fixtures/coding-task.md)"
```

No `/connect` is needed. The wrapper configures local tools, reasoning replay,
low thinking effort and a 4,096-token output limit. Config lives at
`/opt/qwen/opencode.json`; quit/restart OpenCode after editing it.

## Useful adjustments

- Stop foreground servers with **Ctrl-C** before comparisons; an exclusive lock
  prevents two launchers sharing this card.
- OOM: try `CONTEXT_SIZE=16384 serve.sh turbo`; use the same `CONTEXT_SIZE` for
  OpenCode/comparisons. Turbo can also use `--device-state-slots 0`.
- `serve.sh` forwards extra engine flags. `DRAFT_TOKENS`, `THINKING_BUDGET`,
  `PORT` and `DATA_DIR` are configurable.
- Local GPU Docker: `docker run --rm -it --gpus all -v qwen-data:/workspace YOUR_DOCKERHUB_USER/qwen-v100:1 bash`.
- API is loopback-only by default. External access needs `HOST=0.0.0.0`,
  `API_KEY=your-secret`, and published port 8080; clients use `/v1`. Vast's
  **IP Port Info** shows its randomly mapped external port.
- Copy results off before **destroying** the Vast instance.

CPU-only harness checks: `python3 -m unittest discover -s tests -v`.
