#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "usage: $0 LABEL [PORT]" >&2
  exit 2
fi

SUBROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export R003H_OVERLAY_DIR="$SUBROUND_DIR/overlay"
export R003H_ANALYSIS_DIR="$SUBROUND_DIR/analysis"
export R003H_WINDOW="${R003H_WINDOW:-4096}"
export R003H_PREFIX_TOKENS="${R003H_PREFIX_TOKENS:-128}"

exec bash "$SUBROUND_DIR/scripts/run_quality_process.sh" "$@"
