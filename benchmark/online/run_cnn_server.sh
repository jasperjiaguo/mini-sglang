#!/usr/bin/env bash
set -euo pipefail

: "${RUN_NAME:?set RUN_NAME}"
: "${GPU:?set GPU}"
: "${PORT:?set PORT}"

ROOT_DIR="${ROOT_DIR:-$HOME/mini-sglang}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/mini-sglang-ngram}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/experiments/ngram-cnn-20260710}"
MODEL_PATH="${MODEL_PATH:-/shared/public/elr-models/Qwen/Qwen3-8B/2069b3fae1114555f3c020c81410e51fa0f656f2}"
GRAPH_BS="${GRAPH_BS:-32}"
MAX_RUNNING_REQ="${MAX_RUNNING_REQ:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1152}"
OVERLAP="${OVERLAP:-0}"

source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"
mkdir -p "$OUTPUT_ROOT/$RUN_NAME"

if [[ "$OVERLAP" == "1" ]]; then
  unset MINISGL_DISABLE_OVERLAP_SCHEDULING
else
  export MINISGL_DISABLE_OVERLAP_SCHEDULING=1
fi

command=(
  python -m minisgl
  --model-path "$MODEL_PATH"
  --attention-backend fi
  --cache-type radix
  --page-size 1
  --cuda-graph-max-bs "$GRAPH_BS"
  --max-running-requests "$MAX_RUNNING_REQ"
  --max-seq-len-override "$MAX_SEQ_LEN"
  --max-prefill-length 8192
  --port "$PORT"
)
if [[ -n "${NGRAM_SIZE:-}" || -n "${NUM_DRAFT_TOKENS:-}" ]]; then
  : "${NGRAM_SIZE:?set NGRAM_SIZE for speculative decoding}"
  : "${NUM_DRAFT_TOKENS:?set NUM_DRAFT_TOKENS for speculative decoding}"
  command+=(
    --spec-decoding ngram
    --spec-decoding-config
    "{\"ngram_size\":${NGRAM_SIZE},\"num_draft_tokens\":${NUM_DRAFT_TOKENS}}"
  )
fi

log_file="$OUTPUT_ROOT/$RUN_NAME/server.log"
{
  echo "=== COMMAND ==="
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
  printf '%q ' "${command[@]}"
  echo
  echo "MINISGL_DISABLE_OVERLAP_SCHEDULING=${MINISGL_DISABLE_OVERLAP_SCHEDULING:-0}"
  echo "==============="
} >"$log_file"

export CUDA_VISIBLE_DEVICES="$GPU"
exec "${command[@]}" 2>&1 | tee -a "$log_file"
