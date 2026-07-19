#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "usage: $0 RUN_DIR [OUTPUT_DIR]" >&2
  exit 2
fi

SUBROUND_DIR="$(cd "$(dirname "$BASH_SOURCE")/.." && pwd)"
RUN_DIR="$1"
OUTPUT_DIR="${2:-$SUBROUND_DIR/analysis/$(basename "$RUN_DIR")}"
mkdir -p "$OUTPUT_DIR/panels"
printf '%s\n' "$RUN_DIR" >"$OUTPUT_DIR/source_run.txt"
for panel_dir in "$RUN_DIR"/panels/*; do
  [[ -d "$panel_dir" ]] || continue
  panel_id="$(basename "$panel_dir")"
  mkdir -p "$OUTPUT_DIR/panels/$panel_id"
  python3 "$SUBROUND_DIR/scripts/analyze_prefill_panel.py" \
    --panel-dir "$panel_dir" \
    --output "$OUTPUT_DIR/panels/$panel_id/summary.json"
done
echo "$OUTPUT_DIR"
