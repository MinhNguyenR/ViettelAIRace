#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: run_profile_process.sh MODE LABEL [PORT]" >&2
  exit 2
fi

SUBROUND_DIR="$(cd "$(dirname "$BASH_SOURCE")/.." && pwd)"
ROOT_DIR="$(git -C "$SUBROUND_DIR" rev-parse --show-toplevel)"
MODEL_DIR="$ROOT_DIR/local_workspaces/rtx5080/models/lfm25_1.2b_instruct"
MODE="$1"
LABEL="$2"
PORT="${3:-8000}"
CONTROL_IMAGE="localhost/part2-lfm25-r002k:fused-bf16-greedy"
CANDIDATE_IMAGE="localhost/part2-lfm25-r002w:shortconv-decode-fusion"

case "$MODE" in
  control)
    IMAGE="$CONTROL_IMAGE"
    ENABLE_R002W=0
    ;;
  candidate)
    IMAGE="$CANDIDATE_IMAGE"
    ENABLE_R002W=1
    ;;
  *)
    echo "unsupported mode: $MODE" >&2
    exit 2
    ;;
esac

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$SUBROUND_DIR/outputs/profiles/${LABEL}-${MODE}-${STAMP}"
CONTAINER_NAME="part2-lfm25-r002x-$MODE-$PORT"
mkdir -p "$RUN_DIR"
printf '%s\n' "$IMAGE" >"$RUN_DIR/image_ref.txt"

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
  --env PYTHONPATH=/r002x-overlay:/opt/r002w:/opt/r002k \
  --env R002X_ENABLE_R002W="$ENABLE_R002W" \
  --env R002X_PROFILE_ARM=/r002x-out/arm.json \
  --env R002X_PROFILE_OUTPUT_ROOT=/r002x-out \
  --env R002W_MAX_DECODE_ROWS=8 \
  --volume "$SUBROUND_DIR/overlay:/r002x-overlay:ro" \
  --volume "$RUN_DIR:/r002x-out" \
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
  if ! docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -qx true; then
    docker logs "$CONTAINER_NAME" >&2 || true
    exit 1
  fi
  sleep 2
done
curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null
docker logs "$CONTAINER_NAME" >"$RUN_DIR/startup.log" 2>&1
grep -q 'R002X_PROFILE runner_patched=1' "$RUN_DIR/startup.log"
grep -q 'CUDAGraphMode.FULL_AND_PIECEWISE' "$RUN_DIR/startup.log"
grep -q 'Graph capturing finished' "$RUN_DIR/startup.log"

if [[ "$MODE" == "candidate" ]]; then
  layer_count="$(grep -c 'R002W_SHORT_CONV_FUSION layer_ready=1.*eligible=1' "$RUN_DIR/startup.log")"
  dispatch_count="$(grep -c 'R002W_SHORT_CONV_FUSION dispatch=decode_fused' "$RUN_DIR/startup.log")"
  if [[ "$layer_count" != 10 || "$dispatch_count" != 10 ]]; then
    echo "invalid candidate startup: layers=$layer_count dispatch=$dispatch_count" >&2
    exit 1
  fi
elif grep -q 'R002W_SHORT_CONV_FUSION class_patched=1' "$RUN_DIR/startup.log"; then
  echo "control unexpectedly enabled R002w" >&2
  exit 1
fi

nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv,noheader,nounits >"$RUN_DIR/gpu_ready.csv"

python3 "$SUBROUND_DIR/scripts/run_profile_panels.py" \
  --base-url "http://127.0.0.1:$PORT" \
  --output-root "$RUN_DIR" \
  --steps 64

for batch in 1 2 4 8; do
  python3 "$SUBROUND_DIR/scripts/analyze_panel.py" \
    --panel-dir "$RUN_DIR/panels/batch$batch" \
    --output "$RUN_DIR/panels/batch$batch/summary.json" \
    >"$RUN_DIR/panels/batch$batch/summary.stdout.json"
done

docker logs "$CONTAINER_NAME" >"$RUN_DIR/server.log" 2>&1
grep -q 'R002X_PROFILE active=0 done=1 panel=batch8 steps=64 valid=1' "$RUN_DIR/server.log"
echo "R002x profile process complete: $RUN_DIR"
