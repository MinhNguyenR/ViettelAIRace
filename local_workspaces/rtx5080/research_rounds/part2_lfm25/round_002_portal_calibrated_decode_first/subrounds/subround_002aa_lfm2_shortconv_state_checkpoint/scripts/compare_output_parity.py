#!/usr/bin/env python3
"""Compare two deterministic R002aa output captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "r002aa.output_parity.v1":
        raise ValueError(f"unexpected parity schema: {path}")
    return {
        (int(row["conv_id"]), int(row["turn_idx"])): row
        for row in payload["rows"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control = load_rows(args.control)
    candidate = load_rows(args.candidate)
    if control.keys() != candidate.keys():
        raise RuntimeError("control and candidate panel keys differ")

    rows: list[dict[str, Any]] = []
    all_equal = True
    equal_first_token_count = 0
    equal_token_position_count = 0
    total_token_count = 0
    for key in sorted(control):
        control_ids = [int(value) for value in control[key]["token_ids"]]
        candidate_ids = [int(value) for value in candidate[key]["token_ids"]]
        first_mismatch = next(
            (
                index
                for index, pair in enumerate(zip(control_ids, candidate_ids))
                if pair[0] != pair[1]
            ),
            None,
        )
        equal = control_ids == candidate_ids
        all_equal = all_equal and equal
        first_token_equal = bool(control_ids and control_ids[0] == candidate_ids[0])
        equal_first_token_count += int(first_token_equal)
        equal_token_position_count += sum(
            int(control_id == candidate_id)
            for control_id, candidate_id in zip(control_ids, candidate_ids)
        )
        total_token_count += max(len(control_ids), len(candidate_ids))
        rows.append(
            {
                "conv_id": key[0],
                "turn_idx": key[1],
                "equal": equal,
                "first_token_equal": first_token_equal,
                "first_mismatch": first_mismatch,
                "control_token_ids": control_ids,
                "candidate_token_ids": candidate_ids,
            }
        )

    result = {
        "schema": "r002aa.output_parity_comparison.v1",
        "all_equal": all_equal,
        "panel_count": len(rows),
        "equal_panel_count": sum(int(row["equal"]) for row in rows),
        "equal_first_token_count": equal_first_token_count,
        "equal_token_position_count": equal_token_position_count,
        "total_token_count": total_token_count,
        "token_position_agreement": (
            equal_token_position_count / total_token_count
            if total_token_count
            else 1.0
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"all_equal={int(all_equal)} "
        f"equal_panels={result['equal_panel_count']}/{len(rows)} "
        f"equal_first={equal_first_token_count}/{len(rows)} "
        f"token_agreement={result['token_position_agreement']:.6f}"
    )
    if not all_equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
