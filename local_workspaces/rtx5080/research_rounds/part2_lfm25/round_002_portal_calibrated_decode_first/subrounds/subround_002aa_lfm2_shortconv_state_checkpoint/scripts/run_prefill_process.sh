#!/usr/bin/env bash
set -euo pipefail

if (( $# < 6 )); then
  echo "usage: $0 LABEL CACHE_MODE REPEATS CONVERSATION_IDS PORT GPU_MEMORY [MAMBA_BLOCK_SIZE]" >&2
  exit 2
fi

LABEL="$1"
CACHE_MODE="$2"
REPEATS="$3"
CONVERSATION_IDS="$4"
PORT="$5"
GPU_MEMORY="$6"
MAMBA_BLOCK_SIZE="${7:-auto}"
DRIVER_MODE="${R002AA_DRIVER_MODE:-timing}"
KINETO="${R002AA_KINETO:-0}"
PREFILL_PAD_MULTIPLE="${R002AA_PREFILL_PAD_MULTIPLE:-0}"
REQUIRE_IDLE_GPU="${R002AA_REQUIRE_IDLE_GPU:-1}"
IDLE_MAX_GPU_UTIL="${R002AA_IDLE_MAX_GPU_UTIL:-10}"
IDLE_MAX_MEMORY_MIB="${R002AA_IDLE_MAX_MEMORY_MIB:-2048}"
IDLE_SAMPLE_COUNT="${R002AA_IDLE_SAMPLE_COUNT:-5}"
if [[ "$CACHE_MODE" != "align" && "$CACHE_MODE" != "all" && "$CACHE_MODE" != "sparse" ]]; then
  echo "cache mode must be align, all or sparse" >&2
  exit 2
fi
if [[ "$DRIVER_MODE" != "timing" && "$DRIVER_MODE" != "parity" && "$DRIVER_MODE" != "ab" ]]; then
  echo "R002AA_DRIVER_MODE must be timing, parity or ab" >&2
  exit 2
fi
if [[ "$KINETO" != "0" && "$KINETO" != "1" ]]; then
  echo "R002AA_KINETO must be 0 or 1" >&2
  exit 2
fi
if [[ "$REQUIRE_IDLE_GPU" != "0" && "$REQUIRE_IDLE_GPU" != "1" ]]; then
  echo "R002AA_REQUIRE_IDLE_GPU must be 0 or 1" >&2
  exit 2
fi
for value in "$IDLE_MAX_GPU_UTIL" "$IDLE_MAX_MEMORY_MIB" "$IDLE_SAMPLE_COUNT"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "R002aa idle-GPU thresholds must be non-negative integers" >&2
    exit 2
  fi
done
if (( IDLE_SAMPLE_COUNT < 1 )); then
  echo "R002AA_IDLE_SAMPLE_COUNT must be positive" >&2
  exit 2
fi
SERVER_CACHE_MODE="$CACHE_MODE"
SPARSE_DUAL=0
if [[ "$CACHE_MODE" == "sparse" ]]; then
  SERVER_CACHE_MODE="align"
  SPARSE_DUAL=1
fi

SUBROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_DIR="$(cd "$SUBROUND_DIR/.." && pwd)"
R002Y_DIR="$PARENT_DIR/subround_002y_cached_turn_prefill_attribution"
ROOT_DIR="$(git -C "$SUBROUND_DIR" rev-parse --show-toplevel)"
MODEL_DIR="$ROOT_DIR/local_workspaces/rtx5080/models/lfm25_1.2b_instruct"
TRACE_DIR="$ROOT_DIR/input_part2"
IMAGE="localhost/part2-lfm25-r002k:fused-bf16-greedy"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$SUBROUND_DIR/outputs/$LABEL-$CACHE_MODE-$STAMP"
CONTAINER_NAME="part2-lfm25-r002aa-$LABEL-$PORT"
mkdir -p "$RUN_DIR"
printf '%s\n' "$IMAGE" >"$RUN_DIR/image_ref.txt"

IDLE_SAMPLE_FILE="$RUN_DIR/gpu_idle_samples.csv"
printf 'timestamp,memory_total_mib,memory_used_mib,memory_free_mib,gpu_util_percent,power_w\n' \
  >"$IDLE_SAMPLE_FILE"
for (( sample = 0; sample < IDLE_SAMPLE_COUNT; sample++ )); do
  printf '%s,' "$(date --iso-8601=seconds)" >>"$IDLE_SAMPLE_FILE"
  nvidia-smi \
    --query-gpu=memory.total,memory.used,memory.free,utilization.gpu,power.draw \
    --format=csv,noheader,nounits >>"$IDLE_SAMPLE_FILE"
  if (( sample + 1 < IDLE_SAMPLE_COUNT )); then
    sleep 1
  fi
done
if [[ "$REQUIRE_IDLE_GPU" == "1" ]] && ! awk -F, \
  -v max_util="$IDLE_MAX_GPU_UTIL" \
  -v max_memory="$IDLE_MAX_MEMORY_MIB" '
    NR == 1 { next }
    {
      gsub(/[[:space:]]/, "", $3)
      gsub(/[[:space:]]/, "", $5)
      if (($3 + 0) > max_memory || ($5 + 0) > max_util) {
        rejected = 1
      }
    }
    END { exit rejected ? 1 : 0 }
  ' "$IDLE_SAMPLE_FILE"; then
  echo "R002aa idle-GPU gate rejected the run; close GPU workloads and inspect $IDLE_SAMPLE_FILE" >&2
  exit 3
fi

MODE_FILE_CONTAINER=""
if [[ "$DRIVER_MODE" == "ab" ]]; then
  printf '0\n' >"$RUN_DIR/mode.txt"
  MODE_FILE_CONTAINER="/r002aa-out/mode.txt"
fi

cleanup() {
  docker logs "$CONTAINER_NAME" >"$RUN_DIR/server.log" 2>&1 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv,noheader,nounits >"$RUN_DIR/gpu_before.csv"

MAMBA_BLOCK_ARGS=()
if [[ "$MAMBA_BLOCK_SIZE" != "auto" ]]; then
  MAMBA_BLOCK_ARGS=(--mamba-block-size="$MAMBA_BLOCK_SIZE")
fi

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
  --env R002AA_MODE_FILE="$MODE_FILE_CONTAINER" \
  --env R002AA_PREFILL_PAD_MULTIPLE="$PREFILL_PAD_MULTIPLE" \
  --env PYTHONPATH=/r002aa-overlay:/r002y-overlay:/opt/r002k \
  --env R002Y_PROFILE_ARM=/r002aa-out/arm.json \
  --env R002Y_PROFILE_FINALIZE=/r002aa-out/finalize \
  --env R002Y_PROFILE_OUTPUT_ROOT=/r002aa-out \
  --volume "$SUBROUND_DIR/overlay:/r002aa-overlay:ro" \
  --volume "$R002Y_DIR/overlay:/r002y-overlay:ro" \
  --volume "$RUN_DIR:/r002aa-out" \
  --volume "$MODEL_DIR:/model:ro" \
  --entrypoint python3 \
  "$IMAGE" \
  -m vllm.entrypoints.openai.api_server \
  --model=/model \
  --served-model-name=LFM2.5-1.2B-Instruct \
  --host=0.0.0.0 \
  --port="$PORT" \
  --max-model-len=32768 \
  --gpu-memory-utilization="$GPU_MEMORY" \
  --tensor-parallel-size=1 \
  --model-impl=vllm \
  --performance-mode=interactivity \
  --max-num-seqs=8 \
  --max-num-batched-tokens=16384 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --mamba-cache-mode="$SERVER_CACHE_MODE" \
  --mamba-cache-dtype=bfloat16 \
  "${MAMBA_BLOCK_ARGS[@]}" \
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
grep -q 'R002AA_ALL_SHORTCONV hook_installed=1' "$RUN_DIR/startup.log"
grep -q 'R002Y_PROFILE runner_patched=1' "$RUN_DIR/startup.log"
grep -q 'R002K_FUSED_GREEDY runner_patched=1' "$RUN_DIR/startup.log"
grep -q 'CUDAGraphMode.FULL_AND_PIECEWISE' "$RUN_DIR/startup.log"
grep -q 'Graph capturing finished' "$RUN_DIR/startup.log"
if [[ "$CACHE_MODE" == "all" ]]; then
  grep -q 'R002AA_ALL_SHORTCONV lfm2_all_enabled=1' "$RUN_DIR/startup.log"
fi
if [[ "$CACHE_MODE" == "sparse" ]]; then
  grep -q 'R002AA_SPARSE_DUAL scheduler_patched=1' "$RUN_DIR/startup.log"
  grep -q 'R002AA_SPARSE_DUAL manager_patched=1' "$RUN_DIR/startup.log"
  grep -q 'R002AA_SPARSE_DUAL metadata_patched=1' "$RUN_DIR/startup.log"
fi

if [[ "$DRIVER_MODE" == "timing" ]]; then
  KINETO_ARGS=()
  if [[ "$KINETO" == "1" ]]; then
    KINETO_ARGS=(--kineto)
  fi
  docker run --rm \
    --network host \
    --env TOKENIZERS_PARALLELISM=false \
    --volume "$MODEL_DIR:/model:ro" \
    --volume "$TRACE_DIR:/trace:ro" \
    --volume "$R002Y_DIR:/r002y:ro" \
    --volume "$RUN_DIR:/r002aa-out" \
    --entrypoint python3 \
    "$IMAGE" \
    /r002y/scripts/run_prefill_panels.py \
    --trace /trace/trace_grading_public.jsonl \
    --tokenizer /model \
    --base-url "http://127.0.0.1:$PORT" \
    --output-root /r002aa-out \
    --arm /r002aa-out/arm.json \
    --finalize /r002aa-out/finalize \
    --repeats "$REPEATS" \
    --conversation-ids "$CONVERSATION_IDS" \
    "${KINETO_ARGS[@]}" | tee "$RUN_DIR/client.log"
elif [[ "$DRIVER_MODE" == "parity" ]]; then
  docker run --rm \
    --network host \
    --env TOKENIZERS_PARALLELISM=false \
    --volume "$MODEL_DIR:/model:ro" \
    --volume "$TRACE_DIR:/trace:ro" \
    --volume "$SUBROUND_DIR:/r002aa:ro" \
    --volume "$RUN_DIR:/r002aa-out" \
    --entrypoint python3 \
    "$IMAGE" \
    /r002aa/scripts/run_output_parity.py \
    --trace /trace/trace_grading_public.jsonl \
    --tokenizer /model \
    --base-url "http://127.0.0.1:$PORT" \
    --output /r002aa-out/parity.json \
    --conversation-ids "$CONVERSATION_IDS" \
    --max-tokens "${R002AA_PARITY_MAX_TOKENS:-16}" | tee "$RUN_DIR/client.log"
else
  docker run --rm \
    --network host \
    --env TOKENIZERS_PARALLELISM=false \
    --volume "$MODEL_DIR:/model:ro" \
    --volume "$TRACE_DIR:/trace:ro" \
    --volume "$R002Y_DIR:/r002y:ro" \
    --volume "$SUBROUND_DIR:/r002aa:ro" \
    --volume "$RUN_DIR:/r002aa-out" \
    --entrypoint python3 \
    "$IMAGE" \
    /r002aa/scripts/run_inprocess_ab.py \
    --trace /trace/trace_grading_public.jsonl \
    --tokenizer /model \
    --r002y-scripts /r002y/scripts \
    --base-url "http://127.0.0.1:$PORT" \
    --output-root /r002aa-out \
    --arm /r002aa-out/arm.json \
    --finalize /r002aa-out/finalize \
    --mode-file /r002aa-out/mode.txt \
    --pairs "$REPEATS" \
    --conversation-ids "$CONVERSATION_IDS" | tee "$RUN_DIR/client.log"
fi

docker logs "$CONTAINER_NAME" >"$RUN_DIR/server.log" 2>&1
if [[ "$DRIVER_MODE" == "timing" ]]; then
  expected_panels=$(( REPEATS * 6 ))
  observed_panels="$(find "$RUN_DIR/panels" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  [[ "$observed_panels" -eq "$expected_panels" ]]
  grep -q 'R002Y_PROFILE active=0 done=1' "$RUN_DIR/server.log"
  jq -e ".repeats == $REPEATS and (.panels | length) == $expected_panels" \
    "$RUN_DIR/manifest.json" >/dev/null
elif [[ "$DRIVER_MODE" == "parity" ]]; then
  expected_rows=$(( REPEATS * 6 ))
  jq -e ".schema == \"r002aa.output_parity.v1\" and (.rows | length) == $expected_rows" \
    "$RUN_DIR/parity.json" >/dev/null
else
  expected_panels=$(( REPEATS * 12 ))
  observed_panels="$(find "$RUN_DIR/panels" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  [[ "$observed_panels" -eq "$expected_panels" ]]
  grep -q 'R002Y_PROFILE active=0 done=1' "$RUN_DIR/server.log"
  jq -e ".schema == \"r002aa.inprocess_ab.v1\" and .pairs == $REPEATS and (.panels | length) == $expected_panels" \
    "$RUN_DIR/manifest.json" >/dev/null
fi
echo "R002aa process complete: $RUN_DIR"
