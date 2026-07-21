#!/usr/bin/env python3
"""Evaluate R002aa GPQA accuracy separately from output stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["id"])] = row
    return rows


def compare(
    reference: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, int]:
    keys = sorted(set(reference) & set(candidate))
    return {
        "common_cases": len(keys),
        "prediction_changes": sum(
            reference[key].get("parsed_answer", "")
            != candidate[key].get("parsed_answer", "")
            for key in keys
        ),
        "correctness_changes": sum(
            bool(reference[key].get("is_correct"))
            != bool(candidate[key].get("is_correct"))
            for key in keys
        ),
        "reference_correct_candidate_wrong": sum(
            bool(reference[key].get("is_correct"))
            and not bool(candidate[key].get("is_correct"))
            for key in keys
        ),
        "reference_wrong_candidate_correct": sum(
            not bool(reference[key].get("is_correct"))
            and bool(candidate[key].get("is_correct"))
            for key in keys
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, action="append", required=True)
    parser.add_argument("--candidate-results", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.candidate_summary) != len(args.candidate_results):
        raise ValueError("candidate summary/results counts differ")

    reference_summary = load_summary(args.reference_summary)
    reference_results = load_results(args.reference_results)
    reference_total = int(reference_summary["total"])
    reference_accuracy = float(reference_summary["accuracy"])
    candidates = []
    candidate_results = []
    for summary_path, results_path in zip(
        args.candidate_summary, args.candidate_results
    ):
        summary = load_summary(summary_path)
        results = load_results(results_path)
        comparison = compare(reference_results, results)
        accuracy = float(summary["accuracy"])
        candidates.append(
            {
                "summary_path": str(summary_path),
                "results_path": str(results_path),
                "total": int(summary["total"]),
                "ok": int(summary["ok"]),
                "errors": int(summary["errors"]),
                "parse_failures": int(summary["parse_failures"]),
                "correct": int(summary["correct"]),
                "accuracy": accuracy,
                "accuracy_drop_percentage_points": (
                    reference_accuracy - accuracy
                )
                * 100.0,
                "comparison_to_reference": comparison,
                "passed_accuracy_run": (
                    int(summary["total"]) == reference_total
                    and int(summary["ok"]) == reference_total
                    and int(summary["errors"]) == 0
                    and accuracy >= reference_accuracy
                ),
            }
        )
        candidate_results.append(results)

    pair_comparison = None
    if len(candidate_results) == 2:
        pair_comparison = compare(candidate_results[0], candidate_results[1])

    report = {
        "schema": "r002aa.quality_analysis.v1",
        "reference": {
            "candidate": reference_summary.get("candidate"),
            "total": reference_total,
            "errors": int(reference_summary["errors"]),
            "parse_failures": int(reference_summary["parse_failures"]),
            "correct": int(reference_summary["correct"]),
            "accuracy": reference_accuracy,
        },
        "candidates": candidates,
        "candidate_pair_comparison": pair_comparison,
        "passed_accuracy_gate": all(
            candidate["passed_accuracy_run"] for candidate in candidates
        ),
        "passed_reference_correctness_guard": all(
            candidate["comparison_to_reference"][
                "reference_correct_candidate_wrong"
            ]
            == 0
            for candidate in candidates
        ),
        "passed_exact_prediction_guard": all(
            candidate["comparison_to_reference"]["prediction_changes"] == 0
            for candidate in candidates
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
