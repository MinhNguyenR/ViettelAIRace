#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: run_shared_prefix_replay.sh MODE LABEL [PORT]" >&2
  echo "MODE is candidate or control" >&2
  exit 2
fi

ROUND_DIR="$(cd "$(dirname "$BASH_SOURCE")/.." && pwd)"
ROOT_DIR="$(git -C "$ROUND_DIR" rev-parse --show-toplevel)"
MODEL_DIR="$ROOT_DIR/local_workspaces/rtx5080/models/lfm25_1.2b_instruct"
TRACE_DIR="$ROOT_DIR/input_part2"
R002A_DIR="$ROOT_DIR/local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first"
OVERLAY_DIR="$ROUND_DIR/overlay"
ANALYSIS_DIR="${R003I_ANALYSIS_DIR:-$ROUND_DIR/analysis}"
MODE="$1"
LABEL="$2"
PORT="${3:-18140}"
IMAGE="${R003I_IMAGE:-localhost/part2-lfm25-r002k:fused-bf16-greedy}"
GPU_UTILIZATION="${R003I_GPU_UTILIZATION:-0.80}"
KV_DTYPE="${R003I_KV_DTYPE:-bfloat16}"
PROMPT_MODE="${R003I_PROMPT_MODE:-shared_fixed_prefix}"
FIXED_PREFIX_TOKENS="${R003I_FIXED_PREFIX_TOKENS:-4096}"
CONVERSATION_LIMIT="${R003I_CONVERSATION_LIMIT:-8}"
MAX_NUM_SEQS="${R003I_MAX_NUM_SEQS:-8}"
FORCE="${R003I_FORCE:-1}"
ALLOW_FP8="${R003I_ALLOW_FP8:-0}"
FORCE_FLASHINFER="${R003I_FORCE_FLASHINFER:-1}"
GLOBAL_CASCADE="${R003I_GLOBAL_CASCADE:-1}"
MIN_PREFIX="${R003I_MIN_PREFIX:-256}"
PRIME_SHARED_PREFIX="${R003I_PRIME_SHARED_PREFIX:-1}"
TIME_SCALE="${R003I_TIME_SCALE:-1.0}"
TRACE_PATH="${R003I_TRACE_PATH:-/trace/trace_grading_public.jsonl}"
case "$MODE" in
  candidate) ENABLED=1 ;;
  control) ENABLED=0 ;;
  *) echo "unsupported mode: $MODE" >&2; exit 2 ;;
esac
case "$GLOBAL_CASCADE" in
  1) CASCADE_FLAG=--no-disable-cascade-attn ;;
  0) CASCADE_FLAG=--disable-cascade-attn ;;
  *) echo "R003I_GLOBAL_CASCADE must be 0 or 1" >&2; exit 2 ;;
esac

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$ANALYSIS_DIR/replay-${LABEL}-${MODE}-${STAMP}"
CONTAINER_NAME="part2-lfm25-r003i-${PORT}-${STAMP}"
mkdir -p "$RUN_DIR"
printf '%s\n' "$IMAGE" >"$RUN_DIR/image_ref.txt"
printf '%s\n' "$MODE" >"$RUN_DIR/mode.txt"
printf '%s\n' "$KV_DTYPE" >"$RUN_DIR/kv_dtype.txt"
printf '%s\n' "$PROMPT_MODE" >"$RUN_DIR/prompt_mode.txt"
printf '%s\n' "$FIXED_PREFIX_TOKENS" >"$RUN_DIR/fixed_prefix_tokens.txt"
printf '%s\n' "$CONVERSATION_LIMIT" >"$RUN_DIR/conversation_limit.txt"
printf '%s\n' "$TIME_SCALE" >"$RUN_DIR/time_scale.txt"
printf '%s\n' "$ENABLED" >"$RUN_DIR/enabled.txt"

cleanup() {
  docker logs "$CONTAINER_NAME" >"$RUN_DIR/server.log" 2>&1 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT
if ss -ltnH "sport = :$PORT" | grep -q .; then
  echo "refusing to reuse occupied benchmark port $PORT" >&2
  exit 1
fi
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
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env TOKENIZERS_PARALLELISM=false \
  --env PYTHONPATH=/r003i/overlay:/opt/r002k \
  --env R003I_ENABLED="$ENABLED" \
  --env R003I_FORCE="$FORCE" \
  --env R003I_ALLOW_FP8="$ALLOW_FP8" \
  --env R003I_FORCE_FLASHINFER="$FORCE_FLASHINFER" \
  --env R003I_MIN_PREFIX="$MIN_PREFIX" \
  --volume "$ROUND_DIR:/r003i:ro" \
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
  "$CASCADE_FLAG" \
  --performance-mode=interactivity \
  --max-num-seqs="$MAX_NUM_SEQS" \
  --max-num-batched-tokens=16384 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --mamba-cache-mode=align \
  --mamba-cache-dtype=bfloat16 \
  --quantization=fp8_per_tensor \
  --kv-cache-dtype="$KV_DTYPE" \
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
  echo "server health timeout" >&2
  exit 1
fi
docker logs "$CONTAINER_NAME" >"$RUN_DIR/startup.log" 2>&1
grep -q 'R003I hook_installed=1' "$RUN_DIR/startup.log"
if [[ "$FORCE_FLASHINFER" == "1" ]] \
  && grep -q 'R003I verified=1 attention_layers=6 shortconv_layers=10 backend=FLASHINFER' "$RUN_DIR/startup.log"; then
  SELECTED_BACKEND=FLASHINFER
elif grep -q 'Using FLASH_ATTN attention backend' "$RUN_DIR/startup.log"; then
  SELECTED_BACKEND=FLASH_ATTN
elif grep -q 'Using FLASHINFER attention backend' "$RUN_DIR/startup.log"; then
  SELECTED_BACKEND=FLASHINFER
else
  echo "R003i could not identify the selected attention backend" >&2
  exit 1
fi
printf '%s\n' "$SELECTED_BACKEND" >"$RUN_DIR/selected_backend.txt"
if [[ "$FORCE_FLASHINFER" == "1" && "$SELECTED_BACKEND" != "FLASHINFER" ]]; then
  echo "R003i forced FlashInfer but selected $SELECTED_BACKEND" >&2
  exit 1
fi
if [[ "$SELECTED_BACKEND" == "FLASHINFER" ]]; then
  grep -q 'R003I builder_patched=1 backend=FLASHINFER' "$RUN_DIR/startup.log"
  if [[ "$FORCE_FLASHINFER" == "1" ]]; then
    grep -q 'R003I verified=1 attention_layers=6 shortconv_layers=10' "$RUN_DIR/startup.log"
  fi
  DECISION_MARKER=R003I_DECISION
  BUILD_MARKER=R003I_BUILD
else
  grep -q 'R003I fa_builder_patched=1 backend=FLASH_ATTN' "$RUN_DIR/startup.log"
  DECISION_MARKER=R003I_FA_DECISION
  BUILD_MARKER=R003I_FA_BUILD
fi
grep -q 'R002K_FUSED_GREEDY runner_patched=1' "$RUN_DIR/startup.log"
grep -q 'Graph capturing finished' "$RUN_DIR/startup.log"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv,noheader,nounits >"$RUN_DIR/gpu_ready.csv"

set +e
DRIVER_ARGS=(
  --trace "$TRACE_PATH"
  --base-url "http://127.0.0.1:$PORT"
  --tokenizer /model
  --prompt-mode "$PROMPT_MODE"
  --fixed-prefix-tokens "$FIXED_PREFIX_TOKENS"
  --capture-output-token-ids response
  --time-scale "$TIME_SCALE"
  --conversation-limit "$CONVERSATION_LIMIT"
  --label "r003i_${LABEL}_${MODE}_$STAMP"
  --output-dir /results
)
if [[ "$PRIME_SHARED_PREFIX" == "1" ]]; then
  DRIVER_ARGS+=(--prime-shared-prefix --prime-prompt-tokens "$FIXED_PREFIX_TOKENS")
fi
docker run --rm \
  --network host \
  --env TOKENIZERS_PARALLELISM=false \
  --volume "$MODEL_DIR:/model:ro" \
  --volume "$TRACE_DIR:/trace:ro" \
  --volume "$R002A_DIR:/round2:ro" \
  --volume "$ROUND_DIR:/r003i:ro" \
  --volume "$ROUND_DIR/scripts/run_shared_prefix_driver.py:/driver.py:ro" \
  --volume "$RUN_DIR:/results" \
  --entrypoint python3 \
  "$IMAGE" \
  /driver.py \
  "${DRIVER_ARGS[@]}" | tee "$RUN_DIR/client.log"
CLIENT_STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$CLIENT_STATUS" >"$RUN_DIR/client_exit_code.txt"
curl --silent --show-error "http://127.0.0.1:$PORT/metrics" >"$RUN_DIR/metrics.prom" || true
nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv,noheader,nounits >"$RUN_DIR/gpu_after.csv"
if [[ ! -s "$RUN_DIR/summary.json" ]]; then
  echo "replay did not produce summary.json (client_status=$CLIENT_STATUS)" >&2
  exit 1
fi
docker logs "$CONTAINER_NAME" >"$RUN_DIR/server.log" 2>&1
grep -c "$DECISION_MARKER" "$RUN_DIR/server.log" >"$RUN_DIR/decision_log_count.txt" || printf '0\n' >"$RUN_DIR/decision_log_count.txt"
grep -c "$DECISION_MARKER.*decision=1" "$RUN_DIR/server.log" >"$RUN_DIR/positive_decision_log_count.txt" || printf '0\n' >"$RUN_DIR/positive_decision_log_count.txt"
grep -c "$BUILD_MARKER.*use_cascade=1" "$RUN_DIR/server.log" >"$RUN_DIR/cascade_build_log_count.txt" || printf '0\n' >"$RUN_DIR/cascade_build_log_count.txt"
cat "$RUN_DIR/summary.json"
echo "R003i replay complete: $RUN_DIR (client_status=$CLIENT_STATUS)"
