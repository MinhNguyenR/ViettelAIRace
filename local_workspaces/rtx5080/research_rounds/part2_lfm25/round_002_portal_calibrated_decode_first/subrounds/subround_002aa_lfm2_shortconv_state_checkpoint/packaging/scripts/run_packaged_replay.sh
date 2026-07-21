#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: run_packaged_replay.sh MODE LABEL [PORT]" >&2
  exit 2
fi

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBROUND_DIR="$(cd "$PACKAGE_DIR/.." && pwd)"
ROUND_DIR="$(cd "$SUBROUND_DIR/../.." && pwd)"
ROOT_DIR="$(git -C "$SUBROUND_DIR" rev-parse --show-toplevel)"
MODEL_DIR="$ROOT_DIR/local_workspaces/rtx5080/models/lfm25_1.2b_instruct"
TRACE_DIR="$ROOT_DIR/input_part2"
IMAGE="${R002AA_IMAGE:-localhost/part2-lfm25-r002aa:shortconv-sparse-nopad}"
MODE="$1"
LABEL="$2"
PORT="${3:-8000}"
GPU_UTILIZATION="${R002AA_GPU_UTILIZATION:-0.78}"

case "$MODE" in
  candidate) SPARSE_DUAL=1 ;;
  control)
    IMAGE="${R002AA_CONTROL_IMAGE:-localhost/part2-lfm25-r002k:fused-bf16-greedy}"
    SPARSE_DUAL=0
    ;;
  *) echo "unsupported mode: $MODE" >&2; exit 2 ;;
esac

STAMP="$(date +%Y%m%d-%H%M%S)"
UTIL_TAG="${GPU_UTILIZATION//./p}"
RUN_DIR="$PACKAGE_DIR/outputs/full_replay/${LABEL}-${MODE}-u${UTIL_TAG}-${STAMP}"
CONTAINER_NAME="part2-lfm25-r002aa-packaged-${PORT}"
# shellcheck disable=SC1091
source "$SUBROUND_DIR/scripts/gpu_idle_gate.sh"
mkdir -p "$RUN_DIR"
r002aa_require_idle_gpu "$RUN_DIR"
printf '%s\n' "$IMAGE" >"$RUN_DIR/image_ref.txt"

cleanup() {
  docker logs "$CONTAINER_NAME" >"$RUN_DIR/server.log" 2>&1 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run --detach \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --cpus 3 \
  --memory 8g \
  --memory-swap 8g \
  --shm-size 2g \
  --network host \
  --env OMP_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  --env NUMEXPR_NUM_THREADS=1 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env TOKENIZERS_PARALLELISM=false \
  --env R002AA_SPARSE_DUAL="$SPARSE_DUAL" \
  --env R002AA_PREFILL_PAD_MULTIPLE=0 \
  --volume "$MODEL_DIR:/model:ro" \
  --entrypoint python3 \
  "$IMAGE" \
  -m vllm.entrypoints.openai.api_server \
  --model=/model \
  --served-model-name=LFM2.5-1.2B-Instruct \
  --host=0.0.0.0 \
  --port="$PORT" \
  --max-model-len=32768 \
  --gpu-memory-utilization="$GPU_UTILIZATION" \
  --tensor-parallel-size=1 \
  --model-impl=vllm \
  --performance-mode=interactivity \
  --max-num-seqs=8 \
  --max-num-batched-tokens=16384 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --mamba-cache-mode=align \
  --mamba-cache-dtype=bfloat16 \
  --quantization=fp8_per_tensor \
  --kv-cache-dtype=fp8_e4m3 \
  --async-scheduling \
  --disable-uvicorn-access-log \
  --disable-log-stats >"$RUN_DIR/container_id.txt"

started="$(date +%s)"
deadline=$((started + 600))
ready=0
while (( $(date +%s) < deadline )); do
  if curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null; then
    ready="$(date +%s)"
    printf '%s\n' "$((ready - started))" >"$RUN_DIR/startup_seconds.txt"
    break
  fi
  if ! docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -qx true; then
    docker logs "$CONTAINER_NAME" >&2 || true
    exit 1
  fi
  sleep 2
done
if (( ready == 0 )); then
  docker logs "$CONTAINER_NAME" >&2 || true
  exit 1
fi

docker logs "$CONTAINER_NAME" >"$RUN_DIR/startup.log" 2>&1
grep -q 'R002AA_ALL_SHORTCONV hook_installed=1' "$RUN_DIR/startup.log"
grep -q 'R002K_FUSED_GREEDY runner_patched=1' "$RUN_DIR/startup.log"
if grep -q 'R002Y_PROFILE runner_patched=1' "$RUN_DIR/startup.log"; then
  echo "packaged serving path unexpectedly enabled the R002y profiler" >&2
  exit 1
fi
grep -q 'CUDAGraphMode.FULL_AND_PIECEWISE' "$RUN_DIR/startup.log"
grep -q 'Graph capturing finished' "$RUN_DIR/startup.log"
if [[ "$MODE" == "candidate" ]]; then
  grep -q 'R002AA_RUNTIME sparse_enabled=1 mode_file=0' "$RUN_DIR/startup.log"
else
  grep -q 'R002AA_RUNTIME sparse_enabled=0 mode_file=0' "$RUN_DIR/startup.log"
fi

set +e
docker run --rm \
  --network host \
  --env TOKENIZERS_PARALLELISM=false \
  --volume "$MODEL_DIR:/model:ro" \
  --volume "$TRACE_DIR:/trace:ro" \
  --volume "$ROUND_DIR:/round:ro" \
  --volume "$RUN_DIR:/results" \
  --entrypoint python3 \
  "$IMAGE" \
  /round/subrounds/subround_002a_turn_semantics_bracket/scripts/run_turn_semantics.py \
  --trace /trace/trace_grading_public.jsonl \
  --base-url "http://127.0.0.1:$PORT" \
  --tokenizer /model \
  --prompt-mode append_only_static_assistant \
  --capture-output-token-ids response \
  --time-scale 1.0 \
  --label "r002aa_packaged_${LABEL}_${MODE}_${STAMP}" \
  --output-dir /results | tee "$RUN_DIR/client.log"
CLIENT_STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$CLIENT_STATUS" >"$RUN_DIR/client_exit_code.txt"

curl --silent --show-error "http://127.0.0.1:$PORT/metrics" >"$RUN_DIR/metrics.prom" || true
if [[ ! -s "$RUN_DIR/summary.json" ]]; then
  echo "packaged replay did not produce summary.json" >&2
  exit 1
fi
jq -e '.failed_graded == 0 and .successful_graded == .graded_count and .total_count == 420' "$RUN_DIR/summary.json" >/dev/null
cat "$RUN_DIR/summary.json"
echo "R002aa packaged replay complete: $RUN_DIR (client_status=$CLIENT_STATUS)"
