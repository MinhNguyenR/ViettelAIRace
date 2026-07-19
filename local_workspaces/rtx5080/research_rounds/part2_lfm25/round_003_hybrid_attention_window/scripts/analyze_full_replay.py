#!/usr/bin/env python3
"""Aggregate the R003 full-replay control/candidate bracket."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any


TTFT_FLOOR_MS = 10.0
TTFT_CEILING_MS = 400.0
TPOT_FLOOR_MS = 1.0
TPOT_CEILING_MS = 10.0


def component(value: float, floor: float, ceiling: float) -> float:
    normalized = max(0.0, min(1.0, (ceiling - value) / (ceiling - floor)))
    return normalized * normalized


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


def load_run(path: Path) -> dict[str, Any]:
    run_dir = path if path.is_dir() else path.parent
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    requests = [
        json.loads(line)
        for line in (run_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    graded = [row for row in requests if not row["in_warmup"]]
    ttft_components: list[float] = []
    tpot_components: list[float] = []
    by_turn: dict[int, dict[str, list[float]]] = {}
    for row in graded:
        successful = bool(row["success"])
        ttft = float(row["ttft_ms"]) if successful else math.inf
        tpot = float(row["tpot_ms"]) if successful else math.inf
        ttft_score = component(ttft, TTFT_FLOOR_MS, TTFT_CEILING_MS)
        tpot_score = component(tpot, TPOT_FLOOR_MS, TPOT_CEILING_MS)
        ttft_components.append(ttft_score)
        tpot_components.append(tpot_score)
        turn = int(row["turn_idx"])
        bucket = by_turn.setdefault(turn, {"ttft": [], "tpot": []})
        bucket["ttft"].append(ttft_score)
        bucket["tpot"].append(tpot_score)

    recomputed_ers = 0.5 * (
        statistics.fmean(ttft_components) + statistics.fmean(tpot_components)
    )
    if not math.isclose(recomputed_ers, float(summary["ers"]), abs_tol=1e-12):
        raise ValueError(
            f"ERS mismatch in {run_dir}: {recomputed_ers} != {summary['ers']}"
        )

    return {
        "path": str(run_dir),
        "score_points": float(summary["score_points"]),
        "ttft_points": 50.0 * statistics.fmean(ttft_components),
        "tpot_points": 50.0 * statistics.fmean(tpot_components),
        "ttft_p50_ms": float(summary["ttft_ms"]["p50"]),
        "ttft_p95_ms": float(summary["ttft_ms"]["p95"]),
        "tpot_p50_ms": float(summary["tpot_ms"]["p50"]),
        "tpot_p95_ms": float(summary["tpot_ms"]["p95"]),
        "ttft_le_400_ms": int(summary["ttft_le_400_ms"]),
        "failed_graded": int(summary["failed_graded"]),
        "peak_active_requests": int(summary["peak_active_requests"]),
        "wall_duration_s": float(summary["wall_duration_s"]),
        "output_sequence_sha256": summary["output_sequence_sha256"],
        "by_turn": {
            str(turn): {
                "ttft_points": 50.0 * statistics.fmean(scores["ttft"]),
                "tpot_points": 50.0 * statistics.fmean(scores["tpot"]),
                "score_points": 50.0
                * (
                    statistics.fmean(scores["ttft"])
                    + statistics.fmean(scores["tpot"])
                ),
                "ttft_mean_ms": float(summary["by_turn"][str(turn)]["ttft_ms"]["mean"]),
                "ttft_p50_ms": float(summary["by_turn"][str(turn)]["ttft_ms"]["p50"]),
                "ttft_p95_ms": float(summary["by_turn"][str(turn)]["ttft_ms"]["p95"]),
                "tpot_mean_ms": float(summary["by_turn"][str(turn)]["tpot_ms"]["mean"]),
                "tpot_p50_ms": float(summary["by_turn"][str(turn)]["tpot_ms"]["p50"]),
                "tpot_p95_ms": float(summary["by_turn"][str(turn)]["tpot_ms"]["p95"]),
            }
            for turn, scores in sorted(by_turn.items())
        },
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_fields = (
        "score_points",
        "ttft_points",
        "tpot_points",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "tpot_p50_ms",
        "tpot_p95_ms",
        "ttft_le_400_ms",
        "failed_graded",
        "peak_active_requests",
        "wall_duration_s",
    )
    turns = sorted(runs[0]["by_turn"], key=int)
    turn_fields = tuple(runs[0]["by_turn"][turns[0]])
    return {
        "run_count": len(runs),
        "runs": runs,
        "metrics": {
            field: describe([float(run[field]) for run in runs])
            for field in scalar_fields
        },
        "by_turn": {
            turn: {
                field: describe(
                    [float(run["by_turn"][turn][field]) for run in runs]
                )
                for field in turn_fields
            }
            for turn in turns
        },
    }


def delta(candidate: float, control: float) -> dict[str, float]:
    return {
        "candidate_mean": candidate,
        "control_mean": control,
        "absolute": candidate - control,
        "relative": (candidate - control) / control if control else math.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control = aggregate([load_run(path) for path in args.control])
    candidate = aggregate([load_run(path) for path in args.candidate])
    compare_fields = (
        "score_points",
        "ttft_points",
        "tpot_points",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "tpot_p50_ms",
        "tpot_p95_ms",
        "ttft_le_400_ms",
    )
    turns = sorted(control["by_turn"], key=int)
    payload = {
        "schema": "r003.full_replay_comparison.v1",
        "control": control,
        "candidate": candidate,
        "delta_candidate_minus_control": {
            field: delta(
                candidate["metrics"][field]["mean"],
                control["metrics"][field]["mean"],
            )
            for field in compare_fields
        },
        "by_turn_delta_candidate_minus_control": {
            turn: {
                field: delta(
                    candidate["by_turn"][turn][field]["mean"],
                    control["by_turn"][turn][field]["mean"],
                )
                for field in (
                    "score_points",
                    "ttft_points",
                    "tpot_points",
                    "ttft_mean_ms",
                    "ttft_p50_ms",
                    "ttft_p95_ms",
                    "tpot_mean_ms",
                )
            }
            for turn in turns
        },
        "qualification": {
            "candidate_min_score_above_control_max": (
                candidate["metrics"]["score_points"]["min"]
                > control["metrics"]["score_points"]["max"]
            ),
            "all_runs_zero_failures": all(
                run["failed_graded"] == 0
                for run in control["runs"] + candidate["runs"]
            ),
            "candidate_mean_ttft_p95_lower": (
                candidate["metrics"]["ttft_p95_ms"]["mean"]
                < control["metrics"]["ttft_p95_ms"]["mean"]
            ),
            "candidate_mean_tpot_p95_not_worse": (
                candidate["metrics"]["tpot_p95_ms"]["mean"]
                <= control["metrics"]["tpot_p95_ms"]["mean"]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
