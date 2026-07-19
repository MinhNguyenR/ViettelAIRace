#!/usr/bin/env python3
"""Deterministic tests for R002x interpolation and replay occupancy helpers."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_bracket.py"
SPEC = importlib.util.spec_from_file_location("r002x_compare", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_interpolation() -> None:
    values = {1: 1.0, 2: 2.0, 4: 4.0, 8: 8.0}
    assert MODULE.interpolate(values, 3) == 3.0
    assert MODULE.interpolate(values, 5) == 5.0
    assert MODULE.interpolate(values, 6) == 6.0
    assert MODULE.interpolate(values, 7) == 7.0


def test_occupancy_weights() -> None:
    rows = [
        {
            "in_warmup": False,
            "success": True,
            "output_tokens": 3,
            "tpot_ms": 100.0,
            "completed_at_s": 1.0,
        },
        {
            "in_warmup": False,
            "success": True,
            "output_tokens": 3,
            "tpot_ms": 100.0,
            "completed_at_s": 1.0,
        },
    ]
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "requests.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        result = MODULE.estimate_batch_weights(path)
    assert result["request_count"] == 2
    assert result["token_events"] == 4
    assert result["clamped_events"] == 0
    assert result["weights"]["2"] == 1.0


def test_scheduler_weights() -> None:
    payload = {
        "final": True,
        "classification": "test",
        "total_steps": 12,
        "empty_steps": 2,
        "pure_decode_steps": 8,
        "mixed_steps": 1,
        "prefill_only_steps": 1,
        "pure_decode_rows": 12,
        "mixed_decode_rows": 3,
        "pure_decode_batch_histogram": {"1": 4, "2": 4},
    }
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "telemetry.json"
        path.write_text(json.dumps(payload) + "\n")
        result = MODULE.scheduler_batch_weights(path)
    assert result["pure_decode_step_fraction"] == 0.8
    assert result["eligible_decode_row_fraction"] == 0.8
    assert result["weights"]["1"] == 0.5
    assert result["weights"]["2"] == 0.5


def main() -> None:
    test_interpolation()
    test_occupancy_weights()
    test_scheduler_weights()
    print("r002x_compare_tests=passed")


if __name__ == "__main__":
    main()
