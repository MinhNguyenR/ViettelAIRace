#!/usr/bin/env python3
"""Compare R002y timing controls and aggregate prefill kernel attribution."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any


TURNS = tuple(range(6))
MUTABLE_FAMILIES = {
    "attention",
    "short_conv",
    "fp8_gemm",
    "bf16_gemm",
    "normalization",
    "r002k_output_head",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-a", type=Path, required=True)
    parser.add_argument("--kineto", type=Path, required=True)
    parser.add_argument("--control-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_summaries(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "panels").glob("*/summary.json")):
        payload = json.loads(path.read_text())
        if not payload["valid"]:
            raise ValueError(f"invalid R002y summary: {path}")
        rows.append(payload)
    if not rows:
        raise ValueError(f"no R002y summaries under {root}")
    return rows


def group_by_turn(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["turn_idx"])].append(row)
    if set(grouped) != set(TURNS):
        raise ValueError(f"incomplete turn set: {sorted(grouped)}")
    return dict(grouped)


def family_value(row: dict[str, Any], name: str) -> float:
    for family in row["families"]:
        if family["name"] == name:
            return float(family["duration_ms"])
    return 0.0


def main() -> None:
    args = parse_args()
    control_a = group_by_turn(load_summaries(args.control_a))
    kineto = group_by_turn(load_summaries(args.kineto))
    control_b = group_by_turn(load_summaries(args.control_b))
    if any(len(kineto[turn]) != 1 for turn in TURNS):
        raise ValueError("Kineto process must contain one panel per turn")

    turn_rows = []
    for turn in TURNS:
        a_gpu = statistics.median(
            float(row["boundary_gpu_ms"]) for row in control_a[turn]
        )
        b_gpu = statistics.median(
            float(row["boundary_gpu_ms"]) for row in control_b[turn]
        )
        a_ttft = statistics.median(
            float(row["client_ttft_ms"]) for row in control_a[turn]
        )
        b_ttft = statistics.median(
            float(row["client_ttft_ms"]) for row in control_b[turn]
        )
        profile = kineto[turn][0]
        control_mean = statistics.fmean((a_gpu, b_gpu))
        turn_rows.append(
            {
                "turn_idx": turn,
                "prompt_tokens": int(profile["prompt_tokens"]),
                "initial_computed_tokens": int(
                    profile["initial_computed_tokens"]
                ),
                "scheduled_prefill_tokens": int(
                    profile["scheduled_prefill_tokens"]
                ),
                "cache_fraction": float(profile["cache_fraction"]),
                "step_count": int(profile["step_count"]),
                "control_a_gpu_median_ms": a_gpu,
                "control_b_gpu_median_ms": b_gpu,
                "control_gpu_mean_ms": control_mean,
                "control_drift_fraction": abs(a_gpu - b_gpu) / control_mean,
                "control_a_client_ttft_median_ms": a_ttft,
                "control_b_client_ttft_median_ms": b_ttft,
                "kineto_boundary_gpu_ms": float(profile["boundary_gpu_ms"]),
                "kineto_kernel_duration_ms": float(
                    profile["kernel_duration_ms"]
                ),
                "kineto_kernel_event_count": int(profile["kernel_event_count"]),
            }
        )

    weighted_control_a = statistics.fmean(
        row["control_a_gpu_median_ms"] for row in turn_rows
    )
    weighted_control_b = statistics.fmean(
        row["control_b_gpu_median_ms"] for row in turn_rows
    )
    weighted_control = statistics.fmean(
        row["control_gpu_mean_ms"] for row in turn_rows
    )
    weighted_drift = abs(weighted_control_a - weighted_control_b) / weighted_control

    family_names = {
        str(family["name"])
        for turn in TURNS
        for family in kineto[turn][0]["families"]
    }
    weighted_kernel = statistics.fmean(
        float(kineto[turn][0]["kernel_duration_ms"]) for turn in TURNS
    )
    family_rows = []
    for name in sorted(family_names):
        values = [family_value(kineto[turn][0], name) for turn in TURNS]
        weighted_duration = statistics.fmean(values)
        family_rows.append(
            {
                "name": name,
                "turn_duration_ms": values,
                "weighted_duration_ms": weighted_duration,
                "weighted_share": (
                    weighted_duration / weighted_kernel if weighted_kernel else 0.0
                ),
                "present_turns": sum(value > 0.0 for value in values),
                "mutable": name in MUTABLE_FAMILIES,
            }
        )
    family_rows.sort(key=lambda row: row["weighted_duration_ms"], reverse=True)
    qualified = [
        row
        for row in family_rows
        if row["mutable"]
        and row["weighted_duration_ms"] >= 2.0
        and row["weighted_share"] >= 0.15
        and row["present_turns"] >= 4
    ]
    cache_progression_valid = (
        turn_rows[0]["initial_computed_tokens"] == 0
        and all(row["initial_computed_tokens"] > 0 for row in turn_rows[1:])
        and all(3500 <= row["scheduled_prefill_tokens"] <= 5000 for row in turn_rows)
    )
    gate = {
        "all_six_turns_valid": True,
        "cache_progression_valid": cache_progression_valid,
        "weighted_control_drift_le_10pct": weighted_drift <= 0.10,
        "qualified_mutable_family_exists": bool(qualified),
    }
    gate["attribution_gate_passed"] = all(gate.values())
    output = {
        "schema": "r002y.prefill_comparison.v1",
        "control_a": str(args.control_a),
        "kineto": str(args.kineto),
        "control_b": str(args.control_b),
        "turn_weight": 1.0 / 6.0,
        "turn_rows": turn_rows,
        "weighted": {
            "control_a_gpu_ms": weighted_control_a,
            "control_b_gpu_ms": weighted_control_b,
            "control_gpu_ms": weighted_control,
            "control_drift_fraction": weighted_drift,
            "kineto_kernel_duration_ms": weighted_kernel,
        },
        "families": family_rows,
        "qualified_families": qualified,
        "opening_gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "weighted": output["weighted"],
                "qualified_families": qualified,
                "opening_gate": gate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
