#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3.11 -m llm_race_lab.cli run-experiment \
  --name "${EXPERIMENT_NAME:-mock-v1-dry}" \
  --requests-per-case "${REQUESTS_PER_CASE:-4}" \
  --concurrency "${CONCURRENCY_GRID:-1,4}" \
  --prompt-classes "${PROMPT_CLASSES:-short,medium}" \
  --decode-classes "${DECODE_CLASSES:-short,medium}" \
  --patterns "${PATTERNS:-continuous,burst}" \
  --dry-run
