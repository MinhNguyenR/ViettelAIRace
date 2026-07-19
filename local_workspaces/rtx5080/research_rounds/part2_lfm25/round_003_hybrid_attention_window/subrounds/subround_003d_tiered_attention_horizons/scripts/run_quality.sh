#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "usage: $0 LABEL WINDOW_MAP [PORT]" >&2
  exit 2
fi

LABEL="$1"
WINDOW_MAP="$2"
PORT="${3:-18120}"
SUBROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R003_DIR="$(cd "$SUBROUND_DIR/../.." && pwd)"

export R003_OVERLAY_DIR="$SUBROUND_DIR/overlay"
export R003_ANALYSIS_DIR="$SUBROUND_DIR/analysis"
export R003B_GLOBAL_LAYERS=""
export R003D_WINDOW_MAP="$WINDOW_MAP"
export WINDOW_SIZE=4096

exec bash "$R003_DIR/scripts/run_quality_process.sh" "$LABEL" 1 "$PORT"
