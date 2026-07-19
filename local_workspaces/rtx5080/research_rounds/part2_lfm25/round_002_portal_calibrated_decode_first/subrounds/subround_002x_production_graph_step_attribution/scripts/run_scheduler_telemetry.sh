#!/usr/bin/env bash
set -euo pipefail

SUBROUND_DIR="$(cd "$(dirname "$BASH_SOURCE")/.." && pwd)"
ROUND_DIR="$(cd "$SUBROUND_DIR/../.." && pwd)"
ROOT_DIR="$(git -C "$SUBROUND_DIR" rev-parse --show-toplevel)"
MODEL_DIR="$ROOT_DIR/local_workspaces/rtx5080/models/lfm25_1.2b_instruct"
TRACE_DIR="$ROOT_DIR/input_part2"
IMAGE="localhost/part2-lfm25-r002k:fused-bf16-greedy"
PORT="${1:-8000}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$SUBROUND_DIR/outputs/telemetry/control-u0p80-$STAMP"
CONTAINER_NAME="part2-lfm25-r002x-telemetry-$PORT"
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
  --env PYTHONPATH=/r002x-overlay:/opt/r002k \
  --env R002X_ENABLE_R002W=0 \
  --env R002X_PROFILE_ARM=/r002x-out/unused-arm.json \
  --env R002X_PROFILE_OUTPUT_ROOT=/r002x-out \
  --env R002X_TELEMETRY=1 \
  --env R002X_TELEMETRY_OUTPUT=/r002x-out/scheduler_telemetry.json \
  --env R002X_TELEMETRY_DUMP=/r002x-out/dump-telemetry \
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
  --label "r002x_scheduler_telemetry_$STAMP" \
  --output-dir /results

touch "$RUN_DIR/dump-telemetry"
curl --silent --fail \
  --header 'Content-Type: application/json' \
  --data '{"model":"LFM2.5-1.2B-Instruct","messages":[{"role":"user","content":"flush telemetry"}],"temperature":0,"max_tokens":1}' \
  "http://127.0.0.1:$PORT/v1/chat/completions" >"$RUN_DIR/flush_response.json"

deadline=$(( $(date +%s) + 30 ))
while (( $(date +%s) < deadline )); do
  if [[ -f "$RUN_DIR/scheduler_telemetry.json" ]] \
    && jq -e '.final == true' "$RUN_DIR/scheduler_telemetry.json" >/dev/null; then
    break
  fi
  sleep 1
done
jq -e '.final == true' "$RUN_DIR/scheduler_telemetry.json" >/dev/null
docker logs "$CONTAINER_NAME" >"$RUN_DIR/server.log" 2>&1
grep -q 'R002X_TELEMETRY final=1' "$RUN_DIR/server.log"
echo "R002x scheduler telemetry complete: $RUN_DIR"
