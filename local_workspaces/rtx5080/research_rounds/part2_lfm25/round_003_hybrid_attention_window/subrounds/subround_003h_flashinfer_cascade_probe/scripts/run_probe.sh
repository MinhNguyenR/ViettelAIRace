#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
  echo "usage: $0 [PORTABLE_OUTPUT_PATH]" >&2
  exit 2
fi

SUBROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$(git -C "$SUBROUND_DIR" rev-parse --show-toplevel)"
OUTPUT_RELATIVE="${1:-analysis/cascade_probe.json}"
OUTPUT_DIR="$SUBROUND_DIR/$(dirname "$OUTPUT_RELATIVE")"
OUTPUT_NAME="$(basename "$OUTPUT_RELATIVE")"
mkdir -p "$OUTPUT_DIR"
IMAGE="localhost/part2-lfm25-r002k:fused-bf16-greedy"
CONTAINER_NAME="part2-lfm25-r003h-cascade-$$"
trap 'docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true' EXIT
PROBE_ARGS=(--output "/r003h/$OUTPUT_RELATIVE")
if [[ -n "${R003H_BATCH:-}" ]]; then
  PROBE_ARGS+=(--batch "$R003H_BATCH")
fi
if [[ -n "${R003H_SEQ_LEN:-}" ]]; then
  PROBE_ARGS+=(--seq-len "$R003H_SEQ_LEN")
fi
if [[ -n "${R003H_PREFIX_TOKENS:-}" ]]; then
  PROBE_ARGS+=(--prefix-tokens "$R003H_PREFIX_TOKENS")
fi
if [[ -n "${R003H_RECENT_TOKENS:-}" ]]; then
  PROBE_ARGS+=(--recent-tokens "$R003H_RECENT_TOKENS")
fi
if [[ -n "${R003H_DTYPE:-}" ]]; then
  PROBE_ARGS+=(--dtype "$R003H_DTYPE")
fi
if [[ -n "${R003H_WARMUP:-}" ]]; then
  PROBE_ARGS+=(--warmup "$R003H_WARMUP")
fi
if [[ -n "${R003H_REPEATS:-}" ]]; then
  PROBE_ARGS+=(--repeats "$R003H_REPEATS")
fi

docker run --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --cpus 3 \
  --memory 8g \
  --memory-swap 8g \
  --shm-size 2g \
  --env OMP_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  --env TOKENIZERS_PARALLELISM=false \
  --volume "$SUBROUND_DIR:/r003h:rw" \
  --entrypoint python3 \
  "$IMAGE" \
  /r003h/tests/probe_cascade.py \
  "${PROBE_ARGS[@]}"

echo "R003h probe complete: $SUBROUND_DIR/$OUTPUT_RELATIVE"
