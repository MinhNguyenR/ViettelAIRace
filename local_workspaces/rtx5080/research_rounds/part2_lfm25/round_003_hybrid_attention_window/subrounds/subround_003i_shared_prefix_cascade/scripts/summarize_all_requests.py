#!/usr/bin/env python3
"""Summarize every replay row, including rows tagged as BTC warmup."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("requests", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.requests.read_text().splitlines() if line]
    successful = [row for row in rows if row.get("success")]
    if not successful:
        raise SystemExit("no successful requests")

    ttft = [float(row["ttft_ms"]) for row in successful]
    tpot = [float(row["tpot_ms"]) for row in successful]
    result: dict[str, Any] = {
        "total_count": len(rows),
        "successful_count": len(successful),
        "failed_count": len(rows) - len(successful),
        "ttft_ms": summarize(ttft),
        "tpot_ms": summarize(tpot),
        "ttft_le_400_ms": sum(value <= 400 for value in ttft),
        "tpot_le_10_ms": sum(value <= 10 for value in tpot),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
