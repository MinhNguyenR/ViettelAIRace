#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3.11 -m llm_race_lab.cli generate-workload \
  --output workloads/mock_v0.jsonl \
  --requests "${REQUESTS:-64}"

PYTHONPATH=src python3.11 -m llm_race_lab.cli run-benchmark \
  --workload workloads/mock_v0.jsonl \
  --output metrics/dry_run_results.jsonl \
  --dry-run \
  --concurrency "${CONCURRENCY:-8}"

PYTHONPATH=src python3.11 -m llm_race_lab.cli report \
  --input metrics/dry_run_results.jsonl \
  --output reports/dry_run_report.html
