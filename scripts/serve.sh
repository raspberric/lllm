#!/usr/bin/env bash
set -euo pipefail
backend="${1:-turbo}"
if (( $# )); then shift; fi
data="${DATA_DIR:-/workspace}"
host="${HOST:-127.0.0.1}"
port="${PORT:-8080}"
context="${CONTEXT_SIZE:-32768}"
model_id="${MODEL_ID:-qwen3.8-27b}"
mkdir -p "$data"/{logs,results}
# The same lock protects foreground launches and automated comparisons.
exec 9>"$data/inference.lock"
flock -n 9 || { printf 'Another inference server is running. Stop it first.\n' >&2; exit 1; }
auth=()
if [[ -n "${API_KEY:-}" ]]; then auth=(--api-key "$API_KEY"); fi
case "$backend" in
  llama|llama-mtp)
    model="$data/models/llama/Qwen3.8-27B-UD-Q4_K_M.gguf"
    spec=(--spec-type none)
    if [[ "$backend" == llama-mtp ]]; then
      draft="$data/models/llama-mtp/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
      [[ -f "$draft" ]] || { printf 'Run download-models.py all first.\n' >&2; exit 1; }
      spec=(--spec-type draft-mtp --spec-draft-model "$draft" --spec-draft-n-max "${DRAFT_TOKENS:-3}")
    fi
    [[ -f "$model" ]] || { printf 'Run download-models.py llama first.\n' >&2; exit 1; }
    exec llama-server --model "$model" --alias "$model_id" \
      --host "$host" --port "$port" --ctx-size "$context" --parallel 1 \
      --n-gpu-layers all --fit off --flash-attn on \
      --cache-type-k q8_0 --cache-type-v q8_0 \
      --batch-size 2048 --ubatch-size 512 --jinja --metrics \
      --reasoning-format deepseek --reasoning-budget "${THINKING_BUDGET:-1024}" \
      --reasoning-preserve --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
      "${auth[@]}" "${spec[@]}" "$@"
    ;;
  turbo|turbo-nospec|turbo-dflash2)
    model="$data/models/turbo/qwen3_8_27b_nvfp4.ninfer"
    [[ -f "$model" ]] || { printf 'Run download-models.py turbo first.\n' >&2; exit 1; }
    spec=()
    if [[ "$backend" == turbo ]]; then
      spec=(--spec mtp --draft-tokens "${DRAFT_TOKENS:-3}" --lm-head-draft)
    elif [[ "$backend" == turbo-dflash2 ]]; then
      spec=(--spec dflash2 --draft-tokens "${DRAFT_TOKENS:-7}" --lm-head-draft)
    fi
    exec ninfer-serve "$model" --model-id "$model_id" --host "$host" --port "$port" \
      --max-context "$context" --kv-capacity "$context" --max-concurrency 1 \
      --kv-dtype int8 --prefill-chunk 2048 --device-state-slots 1 \
      --host-state-slots 2 --host-kv-mib 2048 --preserve-thinking \
      --default-max-tokens 4096 --default-thinking-budget "${THINKING_BUDGET:-1024}" \
      --request-log-jsonl "$data/logs/ninfer-requests.jsonl" \
      "${auth[@]}" "${spec[@]}" "$@"
    ;;
  *) printf 'Usage: serve.sh {llama|llama-mtp|turbo|turbo-nospec|turbo-dflash2} [server flags...]\n' >&2; exit 2 ;;
esac
