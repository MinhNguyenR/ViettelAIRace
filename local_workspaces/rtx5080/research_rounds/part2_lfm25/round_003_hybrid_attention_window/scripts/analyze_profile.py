#!/usr/bin/env python3
"""Summarize the registered R003 control/candidate panel bracket."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sorted(payload["panels"], key=lambda row: int(row["turn_idx"]))


def means(rows: list[dict[str, Any]]) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["turn_idx"]), []).append(
            float(row["boundary_gpu_ms"])
        )
    return {turn: statistics.fmean(values) for turn, values in grouped.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control_rows = load(args.control)
    candidate_rows = [load(path) for path in args.candidate]
    control = means(control_rows)
    candidate_means = [means(rows) for rows in candidate_rows]
    candidate = {
        str(turn): statistics.fmean(panel[turn] for panel in candidate_means)
        for turn in control
    }
    deltas = {
        str(turn): {
            "control_ms": control[turn],
            "candidate_ms": candidate[str(turn)],
            "saving_ms": control[turn] - candidate[str(turn)],
            "relative_saving": (control[turn] - candidate[str(turn)]) / control[turn],
        }
        for turn in control
    }
    control_progression = [
        (int(row["initial_computed_tokens"]), int(row["scheduled_prefill_tokens"]))
        for row in control_rows
    ]
    candidate_progressions = [
        [
            (int(row["initial_computed_tokens"]), int(row["scheduled_prefill_tokens"]))
            for row in rows
        ]
        for rows in candidate_rows
    ]
    payload = {
        "schema": "r003.profile_comparison.v1",
        "control": str(args.control),
        "candidates": [str(path) for path in args.candidate],
        "control_turn_gpu_ms": {str(k): v for k, v in control.items()},
        "candidate_turn_gpu_ms": candidate,
        "deltas": deltas,
        "turn0_regression": (candidate["0"] - control[0]) / control[0],
        "turn5_saving_ms": control[5] - candidate["5"],
        "control_cache_progression": control_progression,
        "candidate_cache_progressions": candidate_progressions,
        "cache_progression_matches_control": all(
            progression == control_progression
            for progression in candidate_progressions
        ),
        "opening_gate": {
            "turn0_regression_at_most_5pct": (candidate["0"] - control[0]) / control[0] <= 0.05,
            "turn5_saving_at_least_15ms": control[5] - candidate["5"] >= 15.0,
            "cache_progression_matches": all(
                progression == control_progression
                for progression in candidate_progressions
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
