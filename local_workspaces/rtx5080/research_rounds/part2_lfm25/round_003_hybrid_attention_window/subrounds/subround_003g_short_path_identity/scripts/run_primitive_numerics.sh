#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${R003G_IMAGE:-localhost/part2-lfm25-r002k:fused-bf16-greedy}"
OUTPUT="${R003G_NUMERICS_OUTPUT:-${ROOT}/analysis/native_identity.json}"

docker run --rm \
  --gpus all \
  --ipc=host \
  --shm-size=2g \
  -v "${ROOT}:/r003g:ro" \
  -v "$(dirname "${OUTPUT}"):/r003g-output" \
  --entrypoint python3 \
  "${IMAGE}" \
  /r003g/tests/test_split_range_kernel.py \
  --output "/r003g-output/$(basename "${OUTPUT}")"
