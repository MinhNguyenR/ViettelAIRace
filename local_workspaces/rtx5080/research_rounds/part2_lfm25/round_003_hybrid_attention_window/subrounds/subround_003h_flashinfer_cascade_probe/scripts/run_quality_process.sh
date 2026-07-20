#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "usage: $0 LABEL [PORT]" >&2
  exit 2
fi
LABEL="$1"
WINDOW_SIZE="${R003H_WINDOW:-4096}"
PREFIX_TOKENS="${R003H_PREFIX_TOKENS:-128}"
PACKAGING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$(git -C "${PACKAGING_DIR}" rev-parse --show-toplevel)"
MODEL_DIR="${ROOT_DIR}/local_workspaces/rtx5080/models/lfm25_1.2b_instruct"
DATASET="${ROOT_DIR}/input/quality/private/gpqa_diamond_100.jsonl"
EVALUATOR="${ROOT_DIR}/local_workspaces/rtx5080/research_rounds/warmup_rounds/round_007_kv_cache_dtype/subrounds/round_007b_gpqa_diamond_accuracy/scripts/evaluate_gpqa_mcq.py"
IMAGE_TAG="${R002K_IMAGE_TAG:-localhost/part2-lfm25-r002k:fused-bf16-greedy}"
PORT="${2:-18120}"
OVERLAY_DIR="${R003H_OVERLAY_DIR:-${PACKAGING_DIR}/overlay}"
ANALYSIS_DIR="${R003H_ANALYSIS_DIR:-${PACKAGING_DIR}/analysis}"
STAMP="$(date +%Y%m%d-%H%M%S)"
CONTAINER_NAME="part2-lfm25-r003-quality-${LABEL}-${PORT}-${STAMP}"
RUN_DIR="${ANALYSIS_DIR}/quality-${LABEL}-${STAMP}"
mkdir -p "${RUN_DIR}"
printf '%s\n' "${WINDOW_SIZE}" >"${RUN_DIR}/window.txt"
printf '%s\n' "${PREFIX_TOKENS}" >"${RUN_DIR}/prefix_tokens.txt"

cleanup() {
  docker logs "${CONTAINER_NAME}" >"${RUN_DIR}/server.log" 2>&1 || true
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

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
  --env VLLM_R003_WINDOW=1 \
  --env R003_ATTENTION_WINDOW="${WINDOW_SIZE}" \
  --env R003H_ENABLED=1 \
  --env R003H_WINDOW="${WINDOW_SIZE}" \
  --env R003H_PREFIX_TOKENS="${PREFIX_TOKENS}" \
  --env R003H_CASCADE="${R003H_CASCADE:-1}" \
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
    grep -q 'R003H hook_installed=1 enabled=1 backend=FLASHINFER' "${RUN_DIR}/startup.log"
    grep -q 'R003H builder_patched=1 backend=FLASHINFER decode_only=1' "${RUN_DIR}/startup.log"
    grep -q 'R003H verified=1 attention_layers=6 shortconv_layers=10' "${RUN_DIR}/startup.log"
    grep -q 'R002K_FUSED_GREEDY runner_patched=1' "${RUN_DIR}/startup.log"
    grep -q 'Graph capturing finished' "${RUN_DIR}/startup.log"
    break
  fi
  if ! docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null | grep -qx true; then
    break
  fi
  sleep 2
done

if ! curl --silent --fail "http://127.0.0.1:${PORT}/health" >/dev/null; then
  tail -n 200 "${RUN_DIR}/startup.log" >&2 || true
  echo "R003h quality server failed to become ready" >&2
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

# The cleanup trap normally writes server.log only during shell exit. Capture
# it before checking activation so a successful run is recorded accurately.
docker logs "${CONTAINER_NAME}" >"${RUN_DIR}/server.log" 2>&1 || true
if grep -q 'R003H_CASCADE active=1' "${RUN_DIR}/server.log"; then
  printf '1\n' >"${RUN_DIR}/cascade_activated.txt"
else
  printf '0\n' >"${RUN_DIR}/cascade_activated.txt"
fi
echo "R003h quality process complete: ${RUN_DIR}"
