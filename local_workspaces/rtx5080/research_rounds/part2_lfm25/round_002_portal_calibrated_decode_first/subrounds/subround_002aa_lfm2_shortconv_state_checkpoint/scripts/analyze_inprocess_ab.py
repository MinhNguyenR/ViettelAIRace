#!/usr/bin/env python3
"""Analyze native-vs-sparse panels captured inside one R002aa engine."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    if payload.get("schema") != "r002aa.inprocess_ab.v1":
        raise ValueError("unexpected in-process A/B schema")

    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for panel in payload["panels"]:
        key = (int(panel["pair"]), int(panel["turn_idx"]))
        mode = str(panel["mode"])
        if mode in grouped[key]:
            raise ValueError(f"duplicate mode {mode} for panel key {key}")
        grouped[key][mode] = panel

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        modes = grouped[key]
        if modes.keys() != {"control", "sparse"}:
            raise RuntimeError(f"incomplete mode pair for {key}: {modes.keys()}")
        control = modes["control"]
        sparse = modes["sparse"]
        if int(control["prompt_tokens"]) != int(sparse["prompt_tokens"]):
            raise RuntimeError(f"prompt length mismatch for {key}")
        rows.append(
            {
                "pair": key[0],
                "turn_idx": key[1],
                "prompt_tokens": int(control["prompt_tokens"]),
                "control_gpu_ms": float(control["boundary_gpu_ms"]),
                "sparse_gpu_ms": float(sparse["boundary_gpu_ms"]),
                "gpu_delta_ms": float(sparse["boundary_gpu_ms"])
                - float(control["boundary_gpu_ms"]),
                "control_ttft_ms": float(control["ttft_ms"]),
                "sparse_ttft_ms": float(sparse["ttft_ms"]),
                "ttft_delta_ms": float(sparse["ttft_ms"])
                - float(control["ttft_ms"]),
                "control_steps": int(control["step_count"]),
                "sparse_steps": int(sparse["step_count"]),
            }
        )

    cached_rows = [row for row in rows if row["turn_idx"] > 0]
    by_turn = []
    for turn_idx in range(6):
        turn_rows = [row for row in rows if row["turn_idx"] == turn_idx]
        by_turn.append(
            {
                "turn_idx": turn_idx,
                "samples": len(turn_rows),
                "gpu_delta_ms": aggregate(
                    [row["gpu_delta_ms"] for row in turn_rows]
                ),
                "ttft_delta_ms": aggregate(
                    [row["ttft_delta_ms"] for row in turn_rows]
                ),
            }
        )

    output = {
        "schema": "r002aa.inprocess_ab_analysis.v1",
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
        "by_turn": by_turn,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        f"all_gpu_delta_ms={output['all_turns']['gpu_delta_ms']['mean']:.6f} "
        f"all_ttft_delta_ms={output['all_turns']['ttft_delta_ms']['mean']:.6f} "
        "cached_gpu_delta_ms="
        f"{output['cached_turns']['gpu_delta_ms']['mean']:.6f} "
        "cached_ttft_delta_ms="
        f"{output['cached_turns']['ttft_delta_ms']['mean']:.6f}"
    )


if __name__ == "__main__":
    main()
