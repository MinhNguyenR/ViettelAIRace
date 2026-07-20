#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${R003F_IMAGE:-localhost/part2-lfm25-r002k:fused-bf16-greedy}"
OUTPUT="${R003F_NUMERICS_OUTPUT:-${ROOT}/analysis/primitive_numerics.json}"

docker run --rm \
  --gpus all \
  --ipc=host \
  --shm-size=2g \
  -v "${ROOT}:/r003f:ro" \
  -v "$(dirname "${OUTPUT}"):/r003f-output" \
  --entrypoint python3 \
  "${IMAGE}" \
  /r003f/tests/test_split_range_kernel.py \
  --output "/r003f-output/$(basename "${OUTPUT}")"
