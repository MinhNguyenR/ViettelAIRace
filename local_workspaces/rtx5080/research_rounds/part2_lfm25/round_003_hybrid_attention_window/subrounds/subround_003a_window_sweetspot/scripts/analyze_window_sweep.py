#!/usr/bin/env python3
"""Aggregate the R003a isolated window-size sweep."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


KV_PATTERN = re.compile(r"GPU KV cache size: ([0-9,]+) tokens")
CONCURRENCY_PATTERN = re.compile(
    r"Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)x"
)


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot describe an empty sample")
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "pstdev": statistics.pstdev(values),
    }


def parse_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["panels"]
    if not rows or not all(row.get("success") for row in rows):
        raise ValueError(f"incomplete or failed panel in {path}")

    by_turn: dict[int, list[float]] = defaultdict(list)
    progressions: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        turn = int(row["turn_idx"])
        by_turn[turn].append(float(row["boundary_gpu_ms"]))
        key = (int(row["repeat"]), int(row["conv_id"]))
        progressions[key].append(
            (
                int(row["initial_computed_tokens"]),
                int(row["scheduled_prefill_tokens"]),
            )
        )

    startup = (path.parent / "startup.log").read_text(encoding="utf-8")
    kv_match = KV_PATTERN.search(startup)
    concurrency_match = CONCURRENCY_PATTERN.search(startup)
    return {
        "path": str(path),
        "turn_gpu_ms": {
            str(turn): describe(values) for turn, values in sorted(by_turn.items())
        },
        "cache_progressions": [
            [[computed, scheduled] for computed, scheduled in progression]
            for _, progression in sorted(progressions.items())
        ],
        "kv_cache_tokens": int(kv_match.group(1).replace(",", ""))
        if kv_match
        else None,
        "max_concurrency_32768": float(concurrency_match.group(1))
        if concurrency_match
        else None,
    }


def parse_window(value: str) -> tuple[str, Path]:
    try:
        name, raw_path = value.split("=", 1)
        int(name)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("window must be WINDOW=MANIFEST") from exc
    return name, Path(raw_path)


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    turns = sorted(runs[0]["turn_gpu_ms"], key=int)
    reference_progression = runs[0]["cache_progressions"][0]
    all_progressions = [
        progression for run in runs for progression in run["cache_progressions"]
    ]
    return {
        "run_count": len(runs),
        "runs": runs,
        "turn_gpu_ms": {
            turn: describe(
                [
                    value
                    for run in runs
                    for value in [run["turn_gpu_ms"][turn]["mean"]]
                ]
            )
            for turn in turns
        },
        "cache_progression": reference_progression,
        "all_cache_progressions_match": all(
            progression == reference_progression for progression in all_progressions
        ),
        "kv_cache_tokens": describe(
            [float(run["kv_cache_tokens"]) for run in runs]
        ),
        "max_concurrency_32768": describe(
            [float(run["max_concurrency_32768"]) for run in runs]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--window", type=parse_window, action="append", required=True,
        metavar="WINDOW=MANIFEST",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control = aggregate_runs([parse_manifest(path) for path in args.control])
    grouped: dict[str, list[Path]] = defaultdict(list)
    for window, path in args.window:
        grouped[window].append(path)
    windows = {
        window: aggregate_runs([parse_manifest(path) for path in paths])
        for window, paths in sorted(grouped.items(), key=lambda item: int(item[0]))
    }
    if "8192" not in windows:
        raise ValueError("the registered 8192 window is required as the reference")

    control_t0 = control["turn_gpu_ms"]["0"]["mean"]
    control_t5 = control["turn_gpu_ms"]["5"]["mean"]
    registered_saving = control_t5 - windows["8192"]["turn_gpu_ms"]["5"]["mean"]
    comparisons: dict[str, Any] = {}
    for window, aggregate in windows.items():
        turn0 = aggregate["turn_gpu_ms"]["0"]["mean"]
        turn5 = aggregate["turn_gpu_ms"]["5"]["mean"]
        saving = control_t5 - turn5
        cache_matches_control = (
            aggregate["cache_progression"] == control["cache_progression"]
            and aggregate["all_cache_progressions_match"]
        )
        comparisons[window] = {
            "turn0_regression": (turn0 - control_t0) / control_t0,
            "turn5_saving_ms": saving,
            "turn5_saving_retained_vs_8192": saving / registered_saving,
            "cache_progression_matches_control": cache_matches_control,
            "opening_gate": {
                "turn0_regression_at_most_5pct": (
                    (turn0 - control_t0) / control_t0 <= 0.05
                ),
                "turn5_saving_at_least_15ms": saving >= 15.0,
                "retains_at_least_70pct_of_8192": (
                    saving / registered_saving >= 0.70
                ),
                "cache_progression_matches": cache_matches_control,
            },
        }

    payload = {
        "schema": "r003a.window_profile_sweep.v1",
        "control": control,
        "windows": windows,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
