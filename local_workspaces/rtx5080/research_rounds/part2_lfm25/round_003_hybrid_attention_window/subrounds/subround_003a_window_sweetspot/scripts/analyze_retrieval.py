#!/usr/bin/env python3
"""Compare R003a retrieval candidates case by case against control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("successful_cases") != payload.get("total_cases"):
        raise ValueError(f"incomplete retrieval run: {path}")
    return payload


def index(payload: dict[str, Any]) -> dict[tuple[int, int, str], dict[str, Any]]:
    return {
        (int(row["target_tokens"]), int(row["replicate"]), str(row["key"])): row
        for row in payload["rows"]
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument(
        "--candidate", action="append", required=True,
        metavar="WINDOW=RESULTS_JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control_payload = load(args.control)
    control = index(control_payload)
    candidates: dict[str, Any] = {}
    for raw in args.candidate:
        window, raw_path = raw.split("=", 1)
        payload = load(Path(raw_path))
        rows = index(payload)
        if rows.keys() != control.keys():
            raise ValueError(f"case mismatch for window {window}")

        lost: list[dict[str, Any]] = []
        gained: list[dict[str, Any]] = []
        output_hash_matches = 0
        by_target: dict[str, dict[str, int]] = {}
        for case, control_row in sorted(control.items()):
            target, replicate, key = case
            candidate_row = rows[case]
            control_pass = bool(control_row["contains_key"])
            candidate_pass = bool(candidate_row["contains_key"])
            if control_row["output_sha256"] == candidate_row["output_sha256"]:
                output_hash_matches += 1
            bucket = by_target.setdefault(
                str(target),
                {"cases": 0, "control_pass": 0, "candidate_pass": 0,
                 "lost": 0, "gained": 0},
            )
            bucket["cases"] += 1
            bucket["control_pass"] += int(control_pass)
            bucket["candidate_pass"] += int(candidate_pass)
            detail = {
                "target_tokens": target,
                "actual_prompt_tokens": int(control_row["actual_prompt_tokens"]),
                "replicate": replicate,
                "key": key,
                "control_output": control_row["output"],
                "candidate_output": candidate_row["output"],
            }
            if control_pass and not candidate_pass:
                bucket["lost"] += 1
                lost.append(detail)
            elif candidate_pass and not control_pass:
                bucket["gained"] += 1
                gained.append(detail)

        candidates[window] = {
            "path": raw_path,
            "control_contains_key": int(control_payload["contains_key"]),
            "candidate_contains_key": int(payload["contains_key"]),
            "control_pass_candidate_fail": len(lost),
            "control_fail_candidate_pass": len(gained),
            "output_hash_matches": output_hash_matches,
            "total_cases": len(control),
            "lost_cases": lost,
            "gained_cases": gained,
            "by_target": by_target,
            "quality_risk_gate": {
                "loses_at_most_one_control_pass": len(lost) <= 1,
                "all_requests_successful": (
                    payload["successful_cases"] == payload["total_cases"]
                ),
            },
        }

    output = {
        "schema": "r003a.long_context_retrieval_comparison.v1",
        "control_path": str(args.control),
        "control_contains_key": int(control_payload["contains_key"]),
        "total_cases": len(control),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
