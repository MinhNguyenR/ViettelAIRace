#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "usage: $0 LABEL WINDOW_MAP [PORT]" >&2
  exit 2
fi

LABEL="$1"
WINDOW_MAP="$2"
PORT="${3:-8010}"
SUBROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R003_DIR="$(cd "$SUBROUND_DIR/../.." && pwd)"
R003A_DIR="$R003_DIR/subrounds/subround_003a_window_sweetspot"

export R003_OVERLAY_DIR="$SUBROUND_DIR/overlay"
export R003_ANALYSIS_DIR="$SUBROUND_DIR/analysis"
export R003_RETRIEVAL_CLIENT="$R003A_DIR/scripts/run_long_context_retrieval.py"
export R003B_GLOBAL_LAYERS=""
export R003D_WINDOW_MAP="$WINDOW_MAP"

exec bash "$R003A_DIR/scripts/run_long_context_retrieval.sh" 4096 "$LABEL" "$PORT"
