#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
  echo "usage: $0 LABEL WINDOW_MAP PREFIX_TOKENS [PORT]" >&2
  exit 2
fi

LABEL="$1"
WINDOW_MAP="$2"
PREFIX_TOKENS="$3"
PORT="${4:-8062}"
SUBROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R003_DIR="$(cd "$SUBROUND_DIR/../.." && pwd)"
R003A_DIR="$R003_DIR/subrounds/subround_003a_window_sweetspot"

export R003_OVERLAY_DIR="$SUBROUND_DIR/overlay"
export R003_ANALYSIS_DIR="$SUBROUND_DIR/analysis"
export R003_RETRIEVAL_CLIENT="$R003A_DIR/scripts/run_long_context_retrieval.py"
export R003E_ENABLED=1
export R003E_WINDOW_MAP="$WINDOW_MAP"
export R003E_PREFIX_TOKENS="$PREFIX_TOKENS"
export R003D_WINDOW_MAP="$WINDOW_MAP"
export R003B_GLOBAL_LAYERS=""

exec bash "$R003A_DIR/scripts/run_long_context_retrieval.sh" 4096 "$LABEL" "$PORT"
