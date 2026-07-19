#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "usage: $0 LABEL GLOBAL_LAYERS [PORT]" >&2
  exit 2
fi

LABEL="$1"
GLOBAL_LAYERS="$2"
PORT="${3:-18130}"
SUBROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R003_DIR="$(cd "$SUBROUND_DIR/../.." && pwd)"
R003B_DIR="$R003_DIR/subrounds/subround_003b_sparse_global_attention"

export R003_OVERLAY_DIR="$R003B_DIR/overlay"
export R003_ANALYSIS_DIR="$SUBROUND_DIR/analysis"
export R003B_GLOBAL_LAYERS="$GLOBAL_LAYERS"
export WINDOW_SIZE=4096

exec bash "$R003_DIR/scripts/run_quality_process.sh" "$LABEL" 1 "$PORT"
