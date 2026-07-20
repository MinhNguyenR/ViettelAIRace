#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
  echo "usage: $0 LABEL WINDOW_MAP PREFIX_TOKENS [PORT]" >&2
  exit 2
fi

LABEL="$1"
export R003F_WINDOW_MAP="$2"
export R003F_PREFIX_TOKENS="$3"
PORT="${4:-8062}"
SUBROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec bash "$SUBROUND_DIR/scripts/run_prefill_process.sh" \
  "$LABEL" timing 1 15 "$PORT"
