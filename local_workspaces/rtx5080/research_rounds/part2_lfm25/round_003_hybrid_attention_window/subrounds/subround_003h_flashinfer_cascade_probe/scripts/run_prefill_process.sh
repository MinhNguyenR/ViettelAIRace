#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 )); then
  echo "usage: $0 LABEL MODE REPEATS CONVERSATION_IDS PORT" >&2
  exit 2
fi

LABEL="$1"
MODE="$2"
REPEATS="$3"
CONVERSATION_IDS="$4"
PORT="$5"
if [[ "$MODE" != "timing" && "$MODE" != "kineto" ]]; then
  echo "mode must be timing or kineto" >&2
  exit 2
fi

SUBROUND_DIR="$(cd "$(dirname "$BASH_SOURCE")/.." && pwd)"
ROOT_DIR="$(git -C "$SUBROUND_DIR" rev-parse --show-toplevel)"
MODEL_DIR="$ROOT_DIR/local_workspaces/rtx5080/models/lfm25_1.2b_instruct"
TRACE_DIR="$ROOT_DIR/input_part2"
IMAGE="localhost/part2-lfm25-r002k:fused-bf16-greedy"
STAMP="$(date +%Y%m%d-%H%M%S)"
R002Y_DIR="$ROOT_DIR/local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002y_cached_turn_prefill_attribution"
OVERLAY_DIR="$SUBROUND_DIR/overlay"
ANALYSIS_DIR="$SUBROUND_DIR/analysis"
WINDOW_ENABLED=1
WINDOW_SIZE="${R003H_WINDOW:-4096}"
PREFIX_TOKENS="${R003H_PREFIX_TOKENS:-128}"
[[ "${PREFIX_TOKENS}" =~ ^[0-9]+$ ]] || {
  echo "R003H_PREFIX_TOKENS must be a non-negative integer" >&2
  exit 2
}
RUN_DIR="$ANALYSIS_DIR/$LABEL-$MODE-$STAMP"
CONTAINER_NAME="part2-lfm25-r003h-$LABEL-$PORT-$STAMP"
mkdir -p "$RUN_DIR"
printf '%s\n' "$IMAGE" >"$RUN_DIR/image_ref.txt"
printf '%s\n' "$WINDOW_SIZE" >"$RUN_DIR/window.txt"
printf '%s\n' "$PREFIX_TOKENS" >"$RUN_DIR/prefix_tokens.txt"

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
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env TOKENIZERS_PARALLELISM=false \
  --env PYTHONPATH=/r003-overlay:/r002y-overlay:/opt/r002k \
  --env VLLM_R003_WINDOW="$WINDOW_ENABLED" \
  --env R003_ATTENTION_WINDOW="$WINDOW_SIZE" \
  --env R003H_ENABLED=1 \
  --env R003H_WINDOW="$WINDOW_SIZE" \
  --env R003H_PREFIX_TOKENS="$PREFIX_TOKENS" \
  --env R003H_CASCADE="${R003H_CASCADE:-1}" \
  --env R002Y_PROFILE_ARM=/r002y-out/arm.json \
  --env R002Y_PROFILE_FINALIZE=/r002y-out/finalize \
  --env R002Y_PROFILE_OUTPUT_ROOT=/r002y-out \
  --volume "$OVERLAY_DIR:/r003-overlay:ro" \
  --volume "$R002Y_DIR/overlay:/r002y-overlay:ro" \
  --volume "$RUN_DIR:/r002y-out" \
  --volume "$MODEL_DIR:/model:ro" \
  --entrypoint python3 \
  "$IMAGE" \
  -m vllm.entrypoints.openai.api_server \
  --model=/model \
  --served-model-name=LFM2.5-1.2B-Instruct \
  --host=0.0.0.0 \
  --port="$PORT" \
  --max-model-len=32768 \
  --gpu-memory-utilization=0.80 \
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
while (( $(date +%s) < deadline )); do
  if curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null; then
    ready="$(date +%s)"
    echo "$(( ready - started ))" >"$RUN_DIR/startup_seconds.txt"
    break
  fi
  if ! docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null \
    | grep -qx true; then
    docker logs "$CONTAINER_NAME" >&2 || true
    exit 1
  fi
  sleep 2
done
curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null
docker logs "$CONTAINER_NAME" >"$RUN_DIR/startup.log" 2>&1
grep -q 'R003H hook_installed=1 enabled=1 backend=FLASHINFER' "$RUN_DIR/startup.log"
grep -q 'R003H builder_patched=1 backend=FLASHINFER decode_only=1' "$RUN_DIR/startup.log"
grep -q 'R003H verified=1 attention_layers=6 shortconv_layers=10' "$RUN_DIR/startup.log"
grep -q 'R002Y_PROFILE runner_patched=1' "$RUN_DIR/startup.log"
grep -q 'R002K_FUSED_GREEDY runner_patched=1' "$RUN_DIR/startup.log"
grep -q 'CUDAGraphMode.FULL_AND_PIECEWISE' "$RUN_DIR/startup.log"
grep -q 'Graph capturing finished' "$RUN_DIR/startup.log"

KINETO_ARG=()
if [[ "$MODE" == "kineto" ]]; then
  KINETO_ARG=(--kineto)
fi

set +e
docker run --rm \
  --network host \
  --env TOKENIZERS_PARALLELISM=false \
  --volume "$MODEL_DIR:/model:ro" \
  --volume "$TRACE_DIR:/trace:ro" \
  --volume "$R002Y_DIR:/r002y:ro" \
  --volume "$RUN_DIR:/r002y-out" \
  --entrypoint python3 \
  "$IMAGE" \
  /r002y/scripts/run_prefill_panels.py \
  --trace /trace/trace_grading_public.jsonl \
  --tokenizer /model \
  --base-url "http://127.0.0.1:$PORT" \
  --output-root /r002y-out \
  --arm /r002y-out/arm.json \
  --finalize /r002y-out/finalize \
  --repeats "$REPEATS" \
  --conversation-ids "$CONVERSATION_IDS" \
  "${KINETO_ARG[@]}" 2> >(tee "$RUN_DIR/client.stderr.log" >&2) \
  | tee "$RUN_DIR/client.log"
client_exit=${PIPESTATUS[0]}
set -e
printf '%s\n' "$client_exit" >"$RUN_DIR/client_exit_code.txt"
if (( client_exit != 0 )); then
  echo "R003h client failed; see $RUN_DIR/client.stderr.log" >&2
  exit "$client_exit"
fi

docker logs "$CONTAINER_NAME" >"$RUN_DIR/server.log" 2>&1
expected_panels=$(( REPEATS * 6 ))
observed_panels="$(find "$RUN_DIR/panels" -mindepth 1 -maxdepth 1 -type d | wc -l)"
[[ "$observed_panels" -eq "$expected_panels" ]]
grep -q 'R002Y_PROFILE active=0 done=1' "$RUN_DIR/server.log"
if grep -q 'R003H_CASCADE active=1' "$RUN_DIR/server.log"; then
  printf '1\n' >"$RUN_DIR/cascade_activated.txt"
else
  printf '0\n' >"$RUN_DIR/cascade_activated.txt"
fi
jq -e ".repeats == $REPEATS and (.panels | length) == $expected_panels" \
  "$RUN_DIR/manifest.json" >/dev/null
echo "R003h process complete: $RUN_DIR"
