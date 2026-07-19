#!/usr/bin/env bash
set -euo pipefail

if (( $# < 6 )); then
  echo "usage: $0 CONTROL OUTPUT RETRIEVAL QUALITY FULL_REPLAY CANDIDATE..." >&2
  echo "pass at least one NAME=MANIFEST argument after the five paths" >&2
  exit 2
fi

SUBROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTROL="$1"
OUTPUT="$2"
RETRIEVAL_SCREEN="$3"
QUALITY="$4"
FULL_REPLAY="$5"
shift 5

ARGS=(--control "$CONTROL")
for candidate in "$@"; do
  ARGS+=(--candidate "$candidate")
done
ARGS+=(
  --retrieval "$RETRIEVAL_SCREEN"
  --quality "$QUALITY"
  --full-replay "$FULL_REPLAY"
  --output "$OUTPUT"
)
exec python3 "$SUBROUND_DIR/scripts/analyze_sparse_screen.py" "${ARGS[@]}"
