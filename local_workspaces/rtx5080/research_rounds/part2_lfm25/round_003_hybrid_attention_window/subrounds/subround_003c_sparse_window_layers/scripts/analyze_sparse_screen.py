#!/usr/bin/env python3
"""Summarize the R003c sparse-window profile, retrieval and quality screens."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


def load_profile(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    panels = payload["panels"]
    if len(panels) != 6 or not all(panel["success"] for panel in panels):
        raise ValueError(f"invalid profile manifest: {manifest_path}")
    by_turn = {
        str(turn): [
            float(panel["boundary_gpu_ms"])
            for panel in panels
            if int(panel["turn_idx"]) == turn
        ]
        for turn in range(6)
    }
    run_dir = manifest_path.parent
    startup = (run_dir / "startup.log").read_text(encoding="utf-8")
    kv_match = re.search(r"GPU KV cache size: ([0-9,]+) tokens", startup)
    global_path = run_dir / "global_layers.txt"
    return {
        "path": str(run_dir),
        "global_layers": (
            global_path.read_text(encoding="utf-8").strip()
            if global_path.exists()
            else ""
        ),
        "turn_gpu_ms": {
            turn: statistics.fmean(values) for turn, values in by_turn.items()
        },
        "kv_cache_tokens": int(kv_match.group(1).replace(",", ""))
        if kv_match
        else None,
        "all_success": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument(
        "--candidate", action="append", required=True, metavar="NAME=MANIFEST"
    )
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--full-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control = load_profile(args.control)
    candidates: dict[str, Any] = {}
    for raw in args.candidate:
        name, raw_path = raw.split("=", 1)
        profile = load_profile(Path(raw_path))
        profile["turn0_regression"] = (
            profile["turn_gpu_ms"]["0"] / control["turn_gpu_ms"]["0"] - 1.0
        )
        profile["turn5_saving_ms"] = (
            control["turn_gpu_ms"]["5"] - profile["turn_gpu_ms"]["5"]
        )
        candidates[name] = profile

    retrieval = json.loads(args.retrieval.read_text(encoding="utf-8"))
    retrieval_summary = {
        name: {
            "candidate_contains_key": value["candidate_contains_key"],
            "lost_control_pass": value["control_pass_candidate_fail"],
            "gained_control_fail": value["control_fail_candidate_pass"],
            "quality_gate": value["quality_risk_gate"],
        }
        for name, value in retrieval["candidates"].items()
    }
    quality = json.loads(args.quality.read_text(encoding="utf-8"))
    full_replay = json.loads(args.full_replay.read_text(encoding="utf-8"))
    quality_summary = {
        "candidate_accuracy": quality["candidate"]["accuracy"],
        "candidate_correct": quality["candidate"]["correct"],
        "candidate_total": quality["candidate"]["total"],
        "control_accuracy": quality["control"]["accuracy"],
        "control_correct": quality["control"]["correct"],
        "control_total": quality["control"]["total"],
    }
    output = {
        "schema": "r003c.sparse_window_screen.v1",
        "control": control,
        "candidates": candidates,
        "retrieval": retrieval_summary,
        "quality": quality_summary,
        "full_replay_qualification": full_replay["qualification"],
        "full_replay_delta": full_replay["delta_candidate_minus_control"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
