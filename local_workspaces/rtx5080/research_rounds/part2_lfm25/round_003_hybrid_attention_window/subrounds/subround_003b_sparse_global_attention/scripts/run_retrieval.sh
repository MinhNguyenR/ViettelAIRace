#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "usage: $0 LABEL GLOBAL_LAYERS [PORT]" >&2
  exit 2
fi

SUBROUND_DIR="$(cd "$(dirname "$BASH_SOURCE")/.." && pwd)"
R003A_DIR="$(cd "$SUBROUND_DIR/../subround_003a_window_sweetspot" && pwd)"
LABEL="$1"
GLOBAL_LAYERS="$2"
PORT="${3:-8000}"

export R003_OVERLAY_DIR="$SUBROUND_DIR/overlay"
export R003_ANALYSIS_DIR="$SUBROUND_DIR/analysis"
export R003_RETRIEVAL_CLIENT="$R003A_DIR/scripts/run_long_context_retrieval.py"
export R003B_GLOBAL_LAYERS="$GLOBAL_LAYERS"

exec bash "$R003A_DIR/scripts/run_long_context_retrieval.sh" \
  4096 "$LABEL" "$PORT"
