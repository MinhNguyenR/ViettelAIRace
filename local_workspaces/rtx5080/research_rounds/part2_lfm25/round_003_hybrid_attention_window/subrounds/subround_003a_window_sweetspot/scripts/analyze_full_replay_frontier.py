#!/usr/bin/env python3
"""Build a compact score frontier from R003a full-replay summaries."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SCALAR_PATHS = {
    "score_points": ("score_points",),
    "ttft_mean_ms": ("ttft_ms", "mean"),
    "ttft_p50_ms": ("ttft_ms", "p50"),
    "ttft_p95_ms": ("ttft_ms", "p95"),
    "tpot_mean_ms": ("tpot_ms", "mean"),
    "tpot_p50_ms": ("tpot_ms", "p50"),
    "tpot_p95_ms": ("tpot_ms", "p95"),
    "ttft_le_400_ms": ("ttft_le_400_ms",),
    "failed_graded": ("failed_graded",),
    "peak_active_requests": ("peak_active_requests",),
}


def nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        value = value[key]
    return value


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "pstdev": statistics.pstdev(values),
    }


def load_summary(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json" if path.is_dir() else path
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(payload["graded_count"]) != 330:
        raise ValueError(f"unexpected graded count in {summary_path}")
    return {
        "path": str(summary_path.parent),
        **{
            name: float(nested_value(payload, field_path))
            for name, field_path in SCALAR_PATHS.items()
        },
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_count": len(runs),
        "runs": runs,
        "metrics": {
            field: describe([float(run[field]) for run in runs])
            for field in SCALAR_PATHS
        },
    }


def parse_variant(value: str) -> tuple[str, Path]:
    try:
        name, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected NAME=RUN_PATH") from exc
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=RUN_PATH")
    return name, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        action="append",
        type=parse_variant,
        required=True,
        help="Repeatable NAME=RUN_PATH entry; use NAME=unbounded for controls.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name, path in args.variant:
        grouped[name].append(load_summary(path))
    if "unbounded" not in grouped:
        parser.error("at least one unbounded control is required")

    variants = {
        name: aggregate(runs)
        for name, runs in sorted(
            grouped.items(),
            key=lambda item: -1 if item[0] == "unbounded" else int(item[0]),
        )
    }
    control = variants["unbounded"]["metrics"]
    comparisons = {
        name: {
            field: {
                "candidate_mean": data["metrics"][field]["mean"],
                "control_mean": control[field]["mean"],
                "absolute": (
                    data["metrics"][field]["mean"] - control[field]["mean"]
                ),
            }
            for field in SCALAR_PATHS
        }
        for name, data in variants.items()
        if name != "unbounded"
    }
    payload = {
        "schema": "r003a.full_replay_frontier.v1",
        "variants": variants,
        "delta_vs_unbounded": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
