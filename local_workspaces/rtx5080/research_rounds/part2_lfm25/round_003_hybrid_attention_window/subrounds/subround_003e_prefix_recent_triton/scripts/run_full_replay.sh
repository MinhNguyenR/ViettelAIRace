#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
  echo "usage: $0 LABEL WINDOW_MAP PREFIX_TOKENS [PORT]" >&2
  exit 2
fi

LABEL="$1"
WINDOW_MAP="$2"
PREFIX_TOKENS="$3"
PORT="${4:-8063}"
SUBROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R003_DIR="$(cd "$SUBROUND_DIR/../.." && pwd)"

export R003_OVERLAY_DIR="$SUBROUND_DIR/overlay"
export R003_ANALYSIS_DIR="$SUBROUND_DIR/analysis"
export R003E_ENABLED=1
export R003E_WINDOW_MAP="$WINDOW_MAP"
export R003E_PREFIX_TOKENS="$PREFIX_TOKENS"
export R003D_WINDOW_MAP="$WINDOW_MAP"
export R003B_GLOBAL_LAYERS=""
export R003_ATTENTION_WINDOW=4096

exec bash "$R003_DIR/scripts/run_full_replay.sh" candidate "$LABEL" "$PORT"
