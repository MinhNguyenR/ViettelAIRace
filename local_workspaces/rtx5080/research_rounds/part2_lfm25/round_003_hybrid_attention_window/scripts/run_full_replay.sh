#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: run_full_replay.sh MODE LABEL [PORT]" >&2
  echo "MODE is candidate or control" >&2
  exit 2
fi

ROUND_DIR="$(cd "$(dirname "$BASH_SOURCE")/.." && pwd)"
ROOT_DIR="$(git -C "$ROUND_DIR" rev-parse --show-toplevel)"
MODEL_DIR="$ROOT_DIR/local_workspaces/rtx5080/models/lfm25_1.2b_instruct"
TRACE_DIR="$ROOT_DIR/input_part2"
R002A_DIR="$ROOT_DIR/local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first"
OVERLAY_DIR="${R003_OVERLAY_DIR:-$ROUND_DIR/overlay}"
ANALYSIS_DIR="${R003_ANALYSIS_DIR:-$ROUND_DIR/analysis}"
MODE="$1"
LABEL="$2"
PORT="${3:-8000}"
GPU_UTILIZATION="${R003_GPU_UTILIZATION:-0.80}"
ALLOC_CONF="${R003_ALLOC_CONF:-expandable_segments:True}"
WINDOW_SIZE="${R003_ATTENTION_WINDOW:-8192}"
GLOBAL_LAYERS="${R003B_GLOBAL_LAYERS:-}"
WINDOW_MAP="${R003D_WINDOW_MAP:-}"
PROMPT_MODE="${R003_PROMPT_MODE:-append_only_static_assistant}"
CONVERSATION_LIMIT="${R003_CONVERSATION_LIMIT:-}"
case "$MODE" in
  candidate)
    IMAGE="${R003_IMAGE:-localhost/part2-lfm25-r002k:fused-bf16-greedy}"
    WINDOW_ENABLED=1
    ;;
  control)
    IMAGE="${R003_CONTROL_IMAGE:-localhost/part2-lfm25-r002k:fused-bf16-greedy}"
    WINDOW_ENABLED=0
    ;;
  *) echo "unsupported mode: $MODE" >&2; exit 2 ;;
esac

STAMP="$(date +%Y%m%d-%H%M%S)"
UTIL_TAG="${GPU_UTILIZATION//./p}"
RUN_DIR="$ANALYSIS_DIR/full-replay-${LABEL}-${MODE}-u${UTIL_TAG}-${STAMP}"
CONTAINER_NAME="part2-lfm25-r003-full-$PORT"
mkdir -p "$RUN_DIR"
printf '%s\n' "$IMAGE" >"$RUN_DIR/image_ref.txt"
printf '%s\n' "$MODE" >"$RUN_DIR/mode.txt"
printf '%s\n' "$WINDOW_ENABLED" >"$RUN_DIR/window_enabled.txt"
printf '%s\n' "$WINDOW_SIZE" >"$RUN_DIR/window_size.txt"
printf '%s\n' "$GLOBAL_LAYERS" >"$RUN_DIR/global_layers.txt"
printf '%s\n' "$WINDOW_MAP" >"$RUN_DIR/window_map.txt"
printf '%s\n' "$PROMPT_MODE" >"$RUN_DIR/prompt_mode.txt"
printf '%s\n' "$CONVERSATION_LIMIT" >"$RUN_DIR/conversation_limit.txt"

cleanup() {
  docker logs "$CONTAINER_NAME" >"$RUN_DIR/server.log" 2>&1 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv,noheader,nounits >"$RUN_DIR/gpu_before.csv"

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
  --env PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF" \
  --env TOKENIZERS_PARALLELISM=false \
  --env PYTHONPATH=/r003-overlay:/opt/r002k \
  --env VLLM_R003_WINDOW="$WINDOW_ENABLED" \
  --env R003_ATTENTION_WINDOW="$WINDOW_SIZE" \
  --env R003B_GLOBAL_LAYERS="$GLOBAL_LAYERS" \
  --env R003D_WINDOW_MAP="$WINDOW_MAP" \
  --volume "$OVERLAY_DIR:/r003-overlay:ro" \
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
deadline=$(( started + 600 ))
ready=0
while (( $(date +%s) < deadline )); do
  if curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null; then
    ready="$(date +%s)"
    echo "$(( ready - started ))" >"$RUN_DIR/startup_seconds.txt"
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
  echo "server health timeout" >&2
  exit 1
fi
curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null
docker logs "$CONTAINER_NAME" >"$RUN_DIR/startup.log" 2>&1

grep -q "R003_WINDOW hook_installed=1 enabled=$WINDOW_ENABLED" "$RUN_DIR/startup.log"
grep -q 'R002K_FUSED_GREEDY runner_patched=1' "$RUN_DIR/startup.log"
grep -q 'CUDAGraphMode.FULL_AND_PIECEWISE' "$RUN_DIR/startup.log"
grep -q 'Graph capturing finished' "$RUN_DIR/startup.log"
if [[ "$MODE" == "candidate" ]]; then
  grep -q "R003_WINDOW verified=1 window=$WINDOW_SIZE" "$RUN_DIR/startup.log"
elif grep -q 'R003_WINDOW verified=1' "$RUN_DIR/startup.log"; then
  echo "control unexpectedly enabled the R003 attention window" >&2
  exit 1
fi
nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv,noheader,nounits >"$RUN_DIR/gpu_ready.csv"

set +e
DRIVER_ARGS=(
  --trace /trace/trace_grading_public.jsonl
  --base-url "http://127.0.0.1:$PORT"
  --tokenizer /model
  --prompt-mode "$PROMPT_MODE"
  --capture-output-token-ids response
  --time-scale 1.0
  --label "r003_full_${LABEL}_${MODE}_$STAMP"
  --output-dir /results
)
if [[ -n "$CONVERSATION_LIMIT" ]]; then
  DRIVER_ARGS+=(--conversation-limit "$CONVERSATION_LIMIT")
fi
docker run --rm \
  --network host \
  --env TOKENIZERS_PARALLELISM=false \
  --volume "$MODEL_DIR:/model:ro" \
  --volume "$TRACE_DIR:/trace:ro" \
  --volume "$R002A_DIR:/round2:ro" \
  --volume "$RUN_DIR:/results" \
  --entrypoint python3 \
  "$IMAGE" \
  /round2/subrounds/subround_002a_turn_semantics_bracket/scripts/run_turn_semantics.py \
  "${DRIVER_ARGS[@]}" | tee "$RUN_DIR/client.log"
CLIENT_STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$CLIENT_STATUS" >"$RUN_DIR/client_exit_code.txt"

curl --silent --show-error "http://127.0.0.1:$PORT/metrics" >"$RUN_DIR/metrics.prom" || true
nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv,noheader,nounits >"$RUN_DIR/gpu_after.csv"
if [[ ! -s "$RUN_DIR/summary.json" ]]; then
  echo "full replay did not produce summary.json (client_status=$CLIENT_STATUS)" >&2
  exit 1
fi
cat "$RUN_DIR/summary.json"
echo "R003 full replay complete: $RUN_DIR (client_status=$CLIENT_STATUS)"
