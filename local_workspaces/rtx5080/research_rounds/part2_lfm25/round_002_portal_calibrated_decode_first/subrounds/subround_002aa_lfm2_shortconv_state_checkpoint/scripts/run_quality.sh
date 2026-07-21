#!/usr/bin/env bash
set -euo pipefail

SUBROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROUND_DIR="$(cd "$SUBROUND_DIR/../.." && pwd)"
ROOT_DIR="$(git -C "$SUBROUND_DIR" rev-parse --show-toplevel)"
R002Y_DIR="$ROUND_DIR/subrounds/subround_002y_cached_turn_prefill_attribution"
MODEL_DIR="$ROOT_DIR/local_workspaces/rtx5080/models/lfm25_1.2b_instruct"
DATASET="$ROOT_DIR/input/quality/private/gpqa_diamond_100.jsonl"
EVALUATOR="$ROOT_DIR/local_workspaces/rtx5080/research_rounds/warmup_rounds/round_007_kv_cache_dtype/subrounds/round_007b_gpqa_diamond_accuracy/scripts/evaluate_gpqa_mcq.py"
ANALYZER="$SUBROUND_DIR/scripts/analyze_quality.py"
REFERENCE_DIR="$ROUND_DIR/subrounds/subround_002k_fused_bf16_greedy_head/packaging/outputs/quality/fused"
IMAGE="localhost/part2-lfm25-r002k:fused-bf16-greedy"
GPU_UTILIZATION="${R002AA_GPU_UTILIZATION:-0.78}"
PREFILL_PAD_MULTIPLE="${R002AA_PREFILL_PAD_MULTIPLE:-32}"
if [[ ! "$PREFILL_PAD_MULTIPLE" =~ ^[0-9]+$ ]]; then
  echo "R002AA_PREFILL_PAD_MULTIPLE must be a non-negative integer" >&2
  exit 2
fi
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="$SUBROUND_DIR/outputs/quality/sparse-pad${PREFILL_PAD_MULTIPLE}-$STAMP"
CONTAINER_NAME="part2-lfm25-r002aa-quality"
# shellcheck disable=SC1091
source "$SUBROUND_DIR/scripts/gpu_idle_gate.sh"
mkdir -p "$OUTPUT_DIR"
r002aa_require_idle_gpu "$OUTPUT_DIR"

cleanup() {
  docker logs "$CONTAINER_NAME" >"$OUTPUT_DIR/server.log" 2>&1 || true
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
  --network none \
  --env OMP_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  --env NUMEXPR_NUM_THREADS=1 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env TOKENIZERS_PARALLELISM=false \
  --env R002AA_SPARSE_DUAL=1 \
  --env R002AA_PREFILL_PAD_MULTIPLE="$PREFILL_PAD_MULTIPLE" \
  --env PYTHONPATH=/r002aa-overlay:/r002y-overlay:/opt/r002k \
  --env R002Y_PROFILE_ARM=/quality/output/arm.json \
  --env R002Y_PROFILE_FINALIZE=/quality/output/finalize \
  --env R002Y_PROFILE_OUTPUT_ROOT=/quality/output \
  --volume "$SUBROUND_DIR/overlay:/r002aa-overlay:ro" \
  --volume "$R002Y_DIR/overlay:/r002y-overlay:ro" \
  --volume "$MODEL_DIR:/model:ro" \
  --volume "$DATASET:/quality/gpqa.jsonl:ro" \
  --volume "$EVALUATOR:/quality/evaluate_gpqa_mcq.py:ro" \
  --volume "$OUTPUT_DIR:/quality/output" \
  --entrypoint python3 \
  "$IMAGE" \
  -m vllm.entrypoints.openai.api_server \
  --model=/model \
  --served-model-name=LFM2.5-1.2B-Instruct \
  --host=0.0.0.0 \
  --port=8000 \
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
  --disable-log-stats >"$OUTPUT_DIR/container_id.txt"

ready=0
for _ in $(seq 1 300); do
  if docker exec "$CONTAINER_NAME" python3 -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2).read()" \
    >/dev/null 2>&1; then
    ready=1
    break
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" == "false" ]]; then
    docker logs "$CONTAINER_NAME" >&2 || true
    exit 1
  fi
  sleep 2
done
if (( ready == 0 )); then
  docker logs "$CONTAINER_NAME" >&2 || true
  echo "R002aa quality server health timeout" >&2
  exit 1
fi

docker logs "$CONTAINER_NAME" >"$OUTPUT_DIR/startup.log" 2>&1
grep -q 'R002AA_ALL_SHORTCONV hook_installed=1' "$OUTPUT_DIR/startup.log"
grep -q 'R002AA_RUNTIME sparse_enabled=1 mode_file=0' "$OUTPUT_DIR/startup.log"
grep -q 'R002AA_SPARSE_DUAL scheduler_patched=1' "$OUTPUT_DIR/startup.log"
grep -q 'R002K_FUSED_GREEDY runner_patched=1' "$OUTPUT_DIR/startup.log"
grep -q 'CUDAGraphMode.FULL_AND_PIECEWISE' "$OUTPUT_DIR/startup.log"
grep -q 'Graph capturing finished' "$OUTPUT_DIR/startup.log"

for repeat in a b; do
  docker exec "$CONTAINER_NAME" python3 /quality/evaluate_gpqa_mcq.py \
    --dataset /quality/gpqa.jsonl \
    --output-jsonl "/quality/output/sparse_pad${PREFILL_PAD_MULTIPLE}_${repeat}_results.jsonl" \
    --summary-json "/quality/output/sparse_pad${PREFILL_PAD_MULTIPLE}_${repeat}_summary.json" \
    --summary-md "/quality/output/sparse_pad${PREFILL_PAD_MULTIPLE}_${repeat}_summary.md" \
    --api-base http://localhost:8000/v1 \
    --model LFM2.5-1.2B-Instruct \
    --candidate "r002aa_sparse_pad32_${repeat}" \
    --max-cases 100 \
    --timeout 180
done

python3 "$ANALYZER" \
  --reference-summary "$REFERENCE_DIR/summary.json" \
  --reference-results "$REFERENCE_DIR/results.jsonl" \
  --candidate-summary "$OUTPUT_DIR/sparse_pad${PREFILL_PAD_MULTIPLE}_a_summary.json" \
  --candidate-summary "$OUTPUT_DIR/sparse_pad${PREFILL_PAD_MULTIPLE}_b_summary.json" \
  --candidate-results "$OUTPUT_DIR/sparse_pad${PREFILL_PAD_MULTIPLE}_a_results.jsonl" \
  --candidate-results "$OUTPUT_DIR/sparse_pad${PREFILL_PAD_MULTIPLE}_b_results.jsonl" \
  --output "$OUTPUT_DIR/quality_report.json"

jq -e '.passed_accuracy_gate and .passed_reference_correctness_guard' \
  "$OUTPUT_DIR/quality_report.json" >/dev/null
cat "$OUTPUT_DIR/quality_report.json"
echo "R002aa quality complete: $OUTPUT_DIR"
