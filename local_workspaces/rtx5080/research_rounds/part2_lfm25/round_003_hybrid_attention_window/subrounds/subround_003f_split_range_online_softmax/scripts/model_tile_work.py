#!/usr/bin/env python3
"""Model Triton K/V tile visits for recent, native-prefix and split masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BLOCK_Q = 4
QUERY_ROWS_PER_BLOCK = 4
TILE_SIZE = 32
WINDOW = 4096


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def tile_sets(
    context_len: int,
    query_len: int,
    q_block: int,
    prefix_tokens: int,
) -> tuple[set[int], set[int], set[int]]:
    seq_len = context_len + query_len
    q_lo = q_block * BLOCK_Q
    q_hi = min(q_lo + QUERY_ROWS_PER_BLOCK - 1, query_len - 1)
    causal_end = min(context_len + q_hi + 1, seq_len)
    num_causal_tiles = ceil_div(causal_end, TILE_SIZE)

    first_recent_key = context_len + q_lo - WINDOW + 1
    recent_lo = max(0, first_recent_key // TILE_SIZE)
    recent_hi = min((context_len + q_hi) // TILE_SIZE + 1, num_causal_tiles)
    recent = set(range(recent_lo, recent_hi))

    native_mm = set(range(ceil_div(seq_len, TILE_SIZE)))

    prefix_end = min(prefix_tokens, causal_end)
    prefix = set(range(ceil_div(prefix_end, TILE_SIZE)))
    split = prefix | recent
    return recent, native_mm, split


def model_step(context_len: int, query_len: int, prefix_tokens: int) -> dict[str, int]:
    totals = {"recent": 0, "native_mm": 0, "ideal_split": 0}
    for q_block in range(ceil_div(query_len, BLOCK_Q)):
        recent, native_mm, split = tile_sets(
            context_len, query_len, q_block, prefix_tokens
        )
        totals["recent"] += len(recent)
        totals["native_mm"] += len(native_mm)
        totals["ideal_split"] += len(split)
    return totals


def load_panels(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    panel_root = manifest_path.parent / "panels"
    panels: list[dict[str, Any]] = []
    for item in manifest["panels"]:
        metadata = json.loads(
            (panel_root / item["panel_id"] / "profile_metadata.json").read_text()
        )
        panels.append(
            {
                "turn": int(item["turn_idx"]),
                "prompt_tokens": int(item["prompt_tokens"]),
                "gpu_ms": float(item["boundary_gpu_ms"]),
                "steps": metadata["step_rows"],
            }
        )
    return panels


def model_panels(
    recent_manifest: Path,
    native_manifest: Path,
    prefix_tokens: int,
    merge_overhead_ms: float,
) -> dict[str, Any]:
    recent_panels = load_panels(recent_manifest)
    native_panels = load_panels(native_manifest)
    if [p["turn"] for p in recent_panels] != [p["turn"] for p in native_panels]:
        raise ValueError("panel turn mismatch")

    rows: list[dict[str, Any]] = []
    for recent_panel, native_panel in zip(recent_panels, native_panels, strict=True):
        if recent_panel["steps"] != native_panel["steps"]:
            raise ValueError(f"step mismatch at turn {recent_panel['turn']}")
        totals = {"recent": 0, "native_mm": 0, "ideal_split": 0}
        for step in recent_panel["steps"]:
            modeled = model_step(
                int(step["computed_before"]),
                int(step["scheduled_tokens"]),
                prefix_tokens,
            )
            for key, value in modeled.items():
                totals[key] += value

        recent_ms = float(recent_panel["gpu_ms"])
        native_ms = float(native_panel["gpu_ms"])
        excess_tiles = totals["native_mm"] - totals["recent"]
        ideal_extra_tiles = totals["ideal_split"] - totals["recent"]
        if excess_tiles <= 0:
            projected_ms = recent_ms
        else:
            projected_ms = recent_ms + (native_ms - recent_ms) * (
                ideal_extra_tiles / excess_tiles
            )
        applied_merge_overhead_ms = (
            merge_overhead_ms if ideal_extra_tiles > 0 else 0.0
        )
        rows.append(
            {
                "turn": recent_panel["turn"],
                "prompt_tokens": recent_panel["prompt_tokens"],
                "tile_visits_per_kv_head_layer": totals,
                "native_to_recent_ratio": totals["native_mm"] / totals["recent"],
                "ideal_to_recent_ratio": totals["ideal_split"] / totals["recent"],
                "measured_recent_gpu_ms": recent_ms,
                "measured_native_mm_gpu_ms": native_ms,
                "projected_ideal_gpu_ms": projected_ms,
                "applied_merge_overhead_ms": applied_merge_overhead_ms,
                "projected_with_merge_overhead_ms": projected_ms
                + applied_merge_overhead_ms,
            }
        )

    return {
        "schema": "r003f.opportunity_model.v1",
        "constants": {
            "block_q": BLOCK_Q,
            "tile_size": TILE_SIZE,
            "window_tokens": WINDOW,
            "prefix_tokens": prefix_tokens,
            "merge_overhead_stress_ms": merge_overhead_ms,
        },
        "panels": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recent-manifest", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--merge-overhead-ms", type=float, default=10.0)
    args = parser.parse_args()
    if args.prefix_tokens <= 0:
        parser.error("--prefix-tokens must be positive")
    result = model_panels(
        args.recent_manifest,
        args.native_manifest,
        args.prefix_tokens,
        args.merge_overhead_ms,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
