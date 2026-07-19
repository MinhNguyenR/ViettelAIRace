#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "usage: $0 LABEL GLOBAL_LAYERS [PORT]" >&2
  exit 2
fi

SUBROUND_DIR="$(cd "$(dirname "$BASH_SOURCE")/.." && pwd)"
ROUND_DIR="$(cd "$SUBROUND_DIR/../.." && pwd)"
LABEL="$1"
GLOBAL_LAYERS="$2"
PORT="${3:-8000}"

export R003_OVERLAY_DIR="$SUBROUND_DIR/overlay"
export R003_ANALYSIS_DIR="$SUBROUND_DIR/analysis"
export R003B_GLOBAL_LAYERS="$GLOBAL_LAYERS"
export WINDOW_ENABLED=1
export WINDOW_SIZE=4096

exec bash "$ROUND_DIR/scripts/run_prefill_process.sh" \
  "$LABEL" timing 1 15 "$PORT"
