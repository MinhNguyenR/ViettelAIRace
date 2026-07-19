#!/usr/bin/env python3
"""Analyze one R002y timing or Kineto panel."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


FAMILY_PATTERNS = (
    ("r002k_output_head", ("partial_greedy", "reduce_greedy", "r002k")),
    ("attention", ("flash_attn", "flashattention", "fmha", "attention")),
    ("short_conv", ("causal_conv", "causalconv", "short_conv", "conv1d")),
    ("fp8_gemm", ("fp8", "f8f8", "e4m3", "scaled_mm")),
    ("bf16_gemm", ("gemm", "gemvx", "cublasgemv", "matmul")),
    ("normalization", ("rms_norm", "rmsnorm", "layer_norm")),
    ("copy", ("memcpy", "copy_kernel")),
    ("elementwise", ("triton_poi", "elementwise", "vectorized_elementwise")),
)


def is_kernel_event(event: dict[str, Any]) -> bool:
    return (
        event.get("ph") == "X"
        and float(event.get("dur", 0.0) or 0.0) > 0.0
        and "kernel" in str(event.get("cat", "")).lower()
    )


def kernel_family(name: str) -> str:
    lowered = name.lower()
    for family, patterns in FAMILY_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return family
    return "other"


def aggregate_rows(
    source: dict[str, dict[str, float | int]],
    *,
    total_us: float,
    top: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        source.items(), key=lambda item: float(item[1]["duration_us"]), reverse=True
    )
    result = []
    for name, values in ordered[:top]:
        duration_ms = float(values["duration_us"]) / 1000.0
        result.append(
            {
                "name": name,
                "count": int(values["count"]),
                "duration_ms": duration_ms,
                "share": duration_ms / (total_us / 1000.0) if total_us else 0.0,
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.panel_dir / "profile_metadata.json").read_text())
    client = json.loads((args.panel_dir / "client.json").read_text())
    if not metadata["valid"] or not client["success"]:
        raise ValueError(f"invalid R002y panel: {args.panel_dir}")

    by_name: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"duration_us": 0.0, "count": 0}
    )
    by_family: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"duration_us": 0.0, "count": 0}
    )
    total_us = 0.0
    kernel_count = 0
    trace_path = args.panel_dir / "trace.json"
    if metadata["kineto"]:
        trace = json.loads(trace_path.read_text())
        for event in trace.get("traceEvents", []):
            if not isinstance(event, dict) or not is_kernel_event(event):
                continue
            name = str(event.get("name", "<unnamed>"))
            duration_us = float(event["dur"])
            family = kernel_family(name)
            by_name[name]["duration_us"] += duration_us
            by_name[name]["count"] += 1
            by_family[family]["duration_us"] += duration_us
            by_family[family]["count"] += 1
            total_us += duration_us
            kernel_count += 1

    output = {
        "schema": "r002y.panel_summary.v1",
        "panel_id": metadata["panel_id"],
        "repeat": int(client["repeat"]),
        "conv_id": int(client["conv_id"]),
        "turn_idx": int(client["turn_idx"]),
        "valid": True,
        "kineto": bool(metadata["kineto"]),
        "prompt_tokens": int(metadata["prompt_tokens"]),
        "initial_computed_tokens": int(metadata["initial_computed_tokens"]),
        "scheduled_prefill_tokens": int(metadata["scheduled_prefill_tokens"]),
        "cache_fraction": float(metadata["cache_fraction"]),
        "step_count": int(metadata["step_count"]),
        "step_rows": metadata["step_rows"],
        "boundary_gpu_ms": float(metadata["boundary_gpu_ms"]),
        "step_gpu_ms": [float(value) for value in metadata["step_gpu_ms"]],
        "cpu_step_ms": [float(value) for value in metadata["cpu_step_ms"]],
        "client_ttft_ms": float(client["ttft_ms"]),
        "kernel_duration_ms": total_us / 1000.0,
        "kernel_event_count": kernel_count,
        "families": aggregate_rows(by_family, total_us=total_us, top=args.top),
        "kernel_names": aggregate_rows(by_name, total_us=total_us, top=args.top),
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "panel_id": output["panel_id"],
                "prompt_tokens": output["prompt_tokens"],
                "computed_tokens": output["initial_computed_tokens"],
                "boundary_gpu_ms": output["boundary_gpu_ms"],
                "client_ttft_ms": output["client_ttft_ms"],
                "kernel_duration_ms": output["kernel_duration_ms"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
