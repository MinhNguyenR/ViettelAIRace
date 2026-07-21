#!/usr/bin/env python3
"""Analyze matched full-replay pairs and exclude contaminated GPU runs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_requests(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("in_warmup"):
                continue
            key = (int(row["conv_id"]), int(row["turn_idx"]))
            if key in rows:
                raise ValueError(f"duplicate graded request key {key} in {path}")
            rows[key] = row
    return rows


def load_idle(path: Path) -> dict[str, float]:
    values: list[float] = []
    memory: list[float] = []
    with path.open() as handle:
        next(handle, None)
        for line in handle:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 5:
                continue
            memory.append(float(fields[2]))
            values.append(float(fields[4]))
    return {
        "sample_count": float(len(values)),
        "max_gpu_util_percent": max(values) if values else float("inf"),
        "max_memory_used_mib": max(memory) if memory else float("inf"),
        "clean_at_10_percent": float(bool(values) and max(values) <= 10.0),
    }


def clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def score(row: dict[str, Any]) -> float:
    if not row.get("success") or int(row.get("output_tokens", 0)) <= 0:
        return 0.0
    ttft = float(row["ttft_ms"])
    tpot = float(row["tpot_ms"])
    ttft_score = clamp((400.0 - ttft) / 390.0) ** 2
    tpot_score = clamp((10.0 - tpot) / 9.0) ** 2
    return 0.5 * ttft_score + 0.5 * tpot_score


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summarize_pair(control_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    control_summary = json.loads((control_dir / "summary.json").read_text())
    candidate_summary = json.loads((candidate_dir / "summary.json").read_text())
    control = load_requests(control_dir / "requests.jsonl")
    candidate = load_requests(candidate_dir / "requests.jsonl")
    if control.keys() != candidate.keys():
        raise ValueError("control and candidate graded request keys differ")

    keys = sorted(control)
    score_deltas = [score(candidate[key]) - score(control[key]) for key in keys]
    ttft_deltas = [
        float(candidate[key]["ttft_ms"]) - float(control[key]["ttft_ms"])
        for key in keys
    ]
    tpot_deltas = [
        float(candidate[key]["tpot_ms"]) - float(control[key]["tpot_ms"])
        for key in keys
    ]
    by_turn: dict[int, list[float]] = defaultdict(list)
    for key, delta in zip(keys, score_deltas):
        by_turn[key[1]].append(delta)

    control_idle = load_idle(control_dir / "gpu_idle_samples.csv")
    candidate_idle = load_idle(candidate_dir / "gpu_idle_samples.csv")
    clean = bool(
        control_idle["clean_at_10_percent"]
        and candidate_idle["clean_at_10_percent"]
        and control_idle["max_memory_used_mib"] <= 2048.0
        and candidate_idle["max_memory_used_mib"] <= 2048.0
    )
    return {
        "control": {
            "label": control_summary.get("label"),
            "score_points": float(control_summary["score_points"]),
            "failed_graded": int(control_summary["failed_graded"]),
        },
        "candidate": {
            "label": candidate_summary.get("label"),
            "score_points": float(candidate_summary["score_points"]),
            "failed_graded": int(candidate_summary["failed_graded"]),
        },
        "graded_count": len(keys),
        "score_delta_points": float(candidate_summary["score_points"])
        - float(control_summary["score_points"]),
        "request_score_delta_points": 100.0 * mean(score_deltas),
        "ttft_delta_ms": mean(ttft_deltas),
        "tpot_delta_ms": mean(tpot_deltas),
        "positive": bool(float(candidate_summary["score_points"]) > float(control_summary["score_points"])),
        "clean": clean,
        "control_idle": control_idle,
        "candidate_idle": candidate_idle,
        "score_delta_by_turn_points": {
            str(turn): 100.0 * mean(values)
            for turn, values in sorted(by_turn.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("CONTROL_DIR", "CANDIDATE_DIR"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs = [summarize_pair(Path(control), Path(candidate)) for control, candidate in args.pair]
    clean_pairs = [pair for pair in pairs if pair["clean"]]
    report = {
        "schema": "r002aa.full_replay_pairs.v1",
        "pair_count": len(pairs),
        "pairs": pairs,
        "clean_pair_count": len(clean_pairs),
        "clean_positive_pair_count": sum(pair["positive"] for pair in clean_pairs),
        "all_pair_score_delta_mean_points": mean(
            [pair["score_delta_points"] for pair in pairs]
        ),
        "clean_pair_score_delta_mean_points": mean(
            [pair["score_delta_points"] for pair in clean_pairs]
        ),
        "clean_pair_ttft_delta_mean_ms": mean(
            [pair["ttft_delta_ms"] for pair in clean_pairs]
        ),
        "clean_pair_tpot_delta_mean_ms": mean(
            [pair["tpot_delta_ms"] for pair in clean_pairs]
        ),
        "promotion_evidence": bool(
            len(clean_pairs) >= 2
            and all(pair["positive"] for pair in clean_pairs)
            and all(pair["candidate"]["failed_graded"] == 0 for pair in clean_pairs)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
