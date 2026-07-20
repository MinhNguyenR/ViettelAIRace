#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "usage: $0 WINDOW LABEL [PORT]" >&2
  echo "WINDOW is control, 4096, 6144, 8192, 12288 or 16384" >&2
  exit 2
fi

WINDOW="$1"
LABEL="$2"
PORT="${3:-8000}"
case "$WINDOW" in
  control) WINDOW_ENABLED=0; WINDOW_SIZE=8192 ;;
  4096|6144|8192|12288|16384) WINDOW_ENABLED=1; WINDOW_SIZE="$WINDOW" ;;
  *) echo "unsupported window: $WINDOW" >&2; exit 2 ;;
esac

SUBROUND_DIR="$(cd "$(dirname "$BASH_SOURCE")/.." && pwd)"
PARENT_DIR="$(cd "$SUBROUND_DIR/../.." && pwd)"
ROOT_DIR="$(git -C "$SUBROUND_DIR" rev-parse --show-toplevel)"
MODEL_DIR="$ROOT_DIR/local_workspaces/rtx5080/models/lfm25_1.2b_instruct"
IMAGE="localhost/part2-lfm25-r002k:fused-bf16-greedy"
OVERLAY_DIR="${R003_OVERLAY_DIR:-$PARENT_DIR/overlay}"
ANALYSIS_DIR="${R003_ANALYSIS_DIR:-$SUBROUND_DIR/analysis}"
CLIENT_SCRIPT="${R003_RETRIEVAL_CLIENT:-$SUBROUND_DIR/scripts/run_long_context_retrieval.py}"
GLOBAL_LAYERS="${R003B_GLOBAL_LAYERS:-}"
WINDOW_MAP="${R003D_WINDOW_MAP:-}"
R003E_DOCKER_ARGS=()
if [[ -n "${R003E_ENABLED:-}" ]]; then
  : "${R003E_WINDOW_MAP:?R003E_WINDOW_MAP is required when R003E is enabled}"
  : "${R003E_PREFIX_TOKENS:?R003E_PREFIX_TOKENS is required when R003E is enabled}"
  [[ "${R003E_PREFIX_TOKENS}" =~ ^[0-9]+$ ]] || {
    echo "R003E_PREFIX_TOKENS must be a non-negative integer" >&2
    exit 2
  }
  R003E_DOCKER_ARGS=(
    --env "R003E_ENABLED=${R003E_ENABLED}"
    --env "R003E_WINDOW_MAP=${R003E_WINDOW_MAP}"
    --env "R003E_PREFIX_TOKENS=${R003E_PREFIX_TOKENS}"
  )
fi
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$ANALYSIS_DIR/retrieval-${LABEL}-${WINDOW}-${STAMP}"
CONTAINER_NAME="part2-lfm25-r003a-retrieval-$PORT"
mkdir -p "$RUN_DIR"
printf '%s\n' "$IMAGE" >"$RUN_DIR/image_ref.txt"
printf '%s\n' "$WINDOW" >"$RUN_DIR/window.txt"
printf '%s\n' "$GLOBAL_LAYERS" >"$RUN_DIR/global_layers.txt"
printf '%s\n' "$WINDOW_MAP" >"$RUN_DIR/window_map.txt"

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
  --env PYTHONPATH=/r003-overlay:/opt/r002k \
  --env VLLM_R003_WINDOW="$WINDOW_ENABLED" \
  --env R003_ATTENTION_WINDOW="$WINDOW_SIZE" \
  --env R003B_GLOBAL_LAYERS="$GLOBAL_LAYERS" \
  --env R003D_WINDOW_MAP="$WINDOW_MAP" \
  "${R003E_DOCKER_ARGS[@]}" \
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
ready=0
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
if (( ready == 0 )); then
  docker logs "$CONTAINER_NAME" >&2 || true
  echo "server health timeout" >&2
  exit 1
fi

docker logs "$CONTAINER_NAME" >"$RUN_DIR/startup.log" 2>&1
grep -q "R003_WINDOW hook_installed=1 enabled=$WINDOW_ENABLED" "$RUN_DIR/startup.log"
grep -q 'R002K_FUSED_GREEDY runner_patched=1' "$RUN_DIR/startup.log"
grep -q 'CUDAGraphMode.FULL_AND_PIECEWISE' "$RUN_DIR/startup.log"
grep -q 'Graph capturing finished' "$RUN_DIR/startup.log"
if [[ "$WINDOW_ENABLED" == "1" ]]; then
  grep -q "R003_WINDOW verified=1 window=$WINDOW_SIZE" "$RUN_DIR/startup.log"
fi
if [[ -n "${R003E_ENABLED:-}" ]]; then
  grep -q "R003E hook_installed=1 prefix_tokens=${R003E_PREFIX_TOKENS} map=${R003E_WINDOW_MAP}" "$RUN_DIR/startup.log"
  if (( R003E_PREFIX_TOKENS > 0 )); then
    grep -q 'R003E runner_patched=1 prefix_metadata=1' "$RUN_DIR/startup.log"
    grep -q "R003E_PREFIX_RANGE active=1 prefix_tokens=${R003E_PREFIX_TOKENS}" "$RUN_DIR/startup.log"
  else
    grep -q 'R003E runner_patched=1 prefix_metadata=0' "$RUN_DIR/startup.log"
    if grep -q 'R003E_PREFIX_RANGE active=1' "$RUN_DIR/startup.log"; then
      echo "R003e zero-prefix control unexpectedly installed a prefix range" >&2
      exit 1
    fi
  fi
fi

docker run --rm \
  --network host \
  --env TOKENIZERS_PARALLELISM=false \
  --volume "$MODEL_DIR:/model:ro" \
  --volume "$CLIENT_SCRIPT:/client.py:ro" \
  --volume "$RUN_DIR:/out" \
  --entrypoint python3 \
  "$IMAGE" \
  -S \
  /client.py \
  --tokenizer /model \
  --base-url "http://127.0.0.1:$PORT" \
  --model LFM2.5-1.2B-Instruct \
  --label "$LABEL" \
  --window "$WINDOW" \
  --output /out/results.json | tee "$RUN_DIR/client.log"

docker logs "$CONTAINER_NAME" >"$RUN_DIR/server.log" 2>&1
jq -e '.total_cases == 14 and .successful_cases == 14' "$RUN_DIR/results.json" >/dev/null
echo "R003a retrieval complete: $RUN_DIR"
