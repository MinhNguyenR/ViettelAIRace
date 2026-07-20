#!/usr/bin/env python3
"""Summarize a matched R002aa timing control/candidate pair."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def load(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for row in payload["panels"]:
        key = (int(row["repeat"]), int(row["turn_idx"]))
        if key in rows:
            raise ValueError(f"duplicate panel key {key} in {path}")
        rows[key] = row
    return rows


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control = load(args.control)
    candidate = load(args.candidate)
    if control.keys() != candidate.keys():
        raise RuntimeError("control and candidate timing panel keys differ")

    rows: list[dict[str, Any]] = []
    by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for key in sorted(control):
        control_row = control[key]
        candidate_row = candidate[key]
        row = {
            "repeat": key[0],
            "turn_idx": key[1],
            "control_gpu_ms": float(control_row["boundary_gpu_ms"]),
            "candidate_gpu_ms": float(candidate_row["boundary_gpu_ms"]),
            "gpu_delta_ms": float(candidate_row["boundary_gpu_ms"])
            - float(control_row["boundary_gpu_ms"]),
            "control_ttft_ms": float(control_row["ttft_ms"]),
            "candidate_ttft_ms": float(candidate_row["ttft_ms"]),
            "ttft_delta_ms": float(candidate_row["ttft_ms"])
            - float(control_row["ttft_ms"]),
            "control_steps": int(control_row["step_count"]),
            "candidate_steps": int(candidate_row["step_count"]),
            "initial_computed_tokens": int(
                candidate_row["initial_computed_tokens"]
            ),
            "scheduled_prefill_tokens": int(
                candidate_row["scheduled_prefill_tokens"]
            ),
        }
        rows.append(row)
        by_turn[key[1]].append(row)

    turn_rows = []
    for turn_idx in sorted(by_turn):
        values = by_turn[turn_idx]
        turn_rows.append(
            {
                "turn_idx": turn_idx,
                "samples": len(values),
                "gpu_delta_ms": aggregate([row["gpu_delta_ms"] for row in values]),
                "ttft_delta_ms": aggregate(
                    [row["ttft_delta_ms"] for row in values]
                ),
                "control_steps": sorted(
                    {int(row["control_steps"]) for row in values}
                ),
                "candidate_steps": sorted(
                    {int(row["candidate_steps"]) for row in values}
                ),
            }
        )

    cached_rows = [row for row in rows if row["turn_idx"] > 0]
    output = {
        "schema": "r002aa.timing_pair.v1",
        "sample_count": len(rows),
        "all_turns": {
            "gpu_delta_ms": aggregate([row["gpu_delta_ms"] for row in rows]),
            "ttft_delta_ms": aggregate([row["ttft_delta_ms"] for row in rows]),
        },
        "cached_turns": {
            "sample_count": len(cached_rows),
            "gpu_delta_ms": aggregate(
                [row["gpu_delta_ms"] for row in cached_rows]
            ),
            "ttft_delta_ms": aggregate(
                [row["ttft_delta_ms"] for row in cached_rows]
            ),
        },
        "by_turn": turn_rows,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        "all_gpu_delta_ms="
        f"{output['all_turns']['gpu_delta_ms']['mean']:.6f} "
        "all_ttft_delta_ms="
        f"{output['all_turns']['ttft_delta_ms']['mean']:.6f} "
        "cached_gpu_delta_ms="
        f"{output['cached_turns']['gpu_delta_ms']['mean']:.6f} "
        "cached_ttft_delta_ms="
        f"{output['cached_turns']['ttft_delta_ms']['mean']:.6f}"
    )


if __name__ == "__main__":
    main()
