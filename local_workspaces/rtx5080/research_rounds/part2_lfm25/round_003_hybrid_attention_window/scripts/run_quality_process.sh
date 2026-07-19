#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "usage: $0 LABEL WINDOW_ENABLED [PORT]" >&2
  exit 2
fi
LABEL="$1"
WINDOW_ENABLED="$2"
WINDOW_SIZE="${WINDOW_SIZE:-8192}"
PACKAGING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$(git -C "${PACKAGING_DIR}" rev-parse --show-toplevel)"
MODEL_DIR="${ROOT_DIR}/local_workspaces/rtx5080/models/lfm25_1.2b_instruct"
DATASET="${ROOT_DIR}/input/quality/private/gpqa_diamond_100.jsonl"
EVALUATOR="${ROOT_DIR}/local_workspaces/rtx5080/research_rounds/warmup_rounds/round_007_kv_cache_dtype/subrounds/round_007b_gpqa_diamond_accuracy/scripts/evaluate_gpqa_mcq.py"
IMAGE_TAG="${R002K_IMAGE_TAG:-localhost/part2-lfm25-r002k:fused-bf16-greedy}"
PORT="${3:-18120}"
CONTAINER_NAME="part2-lfm25-r003-quality-${LABEL}-${PORT}"
OVERLAY_DIR="${R003_OVERLAY_DIR:-${PACKAGING_DIR}/overlay}"
ANALYSIS_DIR="${R003_ANALYSIS_DIR:-${PACKAGING_DIR}/analysis}"
GLOBAL_LAYERS="${R003B_GLOBAL_LAYERS:-}"
WINDOW_MAP="${R003D_WINDOW_MAP:-}"
RUN_DIR="${ANALYSIS_DIR}/quality-${LABEL}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${RUN_DIR}"
printf '%s\n' "${GLOBAL_LAYERS}" >"${RUN_DIR}/global_layers.txt"
printf '%s\n' "${WINDOW_MAP}" >"${RUN_DIR}/window_map.txt"

cleanup() {
  docker logs "${CONTAINER_NAME}" >"${RUN_DIR}/server.log" 2>&1 || true
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run --detach \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --cpus 3 \
  --memory 8g \
  --memory-swap 8g \
  --shm-size 2g \
  --network host \
  --env OMP_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  --env TOKENIZERS_PARALLELISM=false \
  --env PYTHONPATH=/r003-overlay:/opt/r002k \
  --env VLLM_R003_WINDOW="${WINDOW_ENABLED}" \
  --env R003_ATTENTION_WINDOW="${WINDOW_SIZE}" \
  --env R003B_GLOBAL_LAYERS="${GLOBAL_LAYERS}" \
  --env R003D_WINDOW_MAP="${WINDOW_MAP}" \
  --volume "${OVERLAY_DIR}:/r003-overlay:ro" \
  --volume "${MODEL_DIR}:/model:ro" \
  --entrypoint python3 \
  "${IMAGE_TAG}" \
  -m vllm.entrypoints.openai.api_server \
  --model=/model \
  --served-model-name=LFM2.5-1.2B-Instruct \
  --host=0.0.0.0 \
  --port="${PORT}" \
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
  --disable-log-stats \
  >"${RUN_DIR}/container_id.txt"

started="$(date +%s)"
deadline=$((started + 600))
while (( $(date +%s) < deadline )); do
  if curl --silent --fail "http://127.0.0.1:${PORT}/health" >/dev/null; then
    ready="$(date +%s)"
    printf '%s\n' "$((ready - started))" >"${RUN_DIR}/startup_seconds.txt"
    docker logs "${CONTAINER_NAME}" >"${RUN_DIR}/startup.log" 2>&1 || true
    grep -q "R003_WINDOW hook_installed=1 enabled=${WINDOW_ENABLED}" "${RUN_DIR}/startup.log"
    grep -q 'R002K_FUSED_GREEDY runner_patched=1' "${RUN_DIR}/startup.log"
    grep -q 'Graph capturing finished' "${RUN_DIR}/startup.log"
    if [[ "${WINDOW_ENABLED}" == "1" ]]; then
      grep -q "R003_WINDOW verified=1 window=${WINDOW_SIZE}" "${RUN_DIR}/startup.log"
    fi
    break
  fi
  if ! docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null | grep -qx true; then
    break
  fi
  sleep 2
done

if ! curl --silent --fail "http://127.0.0.1:${PORT}/health" >/dev/null; then
  tail -n 200 "${RUN_DIR}/startup.log" >&2 || true
  echo "R003 quality server failed to become ready" >&2
  exit 1
fi

docker run --rm \
  --network host \
  --env TOKENIZERS_PARALLELISM=false \
  --volume "${DATASET}:/dataset.jsonl:ro" \
  --volume "${EVALUATOR}:/evaluate_gpqa_mcq.py:ro" \
  --volume "${RUN_DIR}:/out" \
  --entrypoint python3 \
  "${IMAGE_TAG}" \
  /evaluate_gpqa_mcq.py \
  --dataset /dataset.jsonl \
  --output-jsonl /out/results.jsonl \
  --summary-json /out/summary.json \
  --summary-md /out/summary.md \
  --api-base "http://127.0.0.1:${PORT}/v1" \
  --model LFM2.5-1.2B-Instruct \
  --candidate "${LABEL}" \
  --max-cases 100

echo "R003 quality process complete: ${RUN_DIR}"
