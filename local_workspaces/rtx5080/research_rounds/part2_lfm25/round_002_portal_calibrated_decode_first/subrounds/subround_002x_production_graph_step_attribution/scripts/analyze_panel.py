#!/usr/bin/env python3
"""Analyze one R002x graph-step trace and its event metadata."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


FAMILY_PATTERNS = (
    ("r002w_fused_epilogue", ("short_conv_decode_epilogue",)),
    ("r002k_output_head", ("partial_greedy", "reduce_greedy", "r002k")),
    ("native_causal_conv", ("causal_conv", "causalconv", "conv1d_update")),
    ("native_shortconv_mul", ("binary_internal::mulfunctor",)),
    ("native_shortconv_staging", ("memcpy32_post",)),
    ("attention", ("flash_attn", "flashattention", "paged_attention", "fmha")),
    ("fp8_gemm", ("fp8", "f8f8", "e4m3", "scaled_mm")),
    (
        "short_conv_projection_bf16",
        ("gemvx", "cublasgemv", "gemm_bf16", "tensorop_bf16"),
    ),
    ("gemm", ("cutlass", "gemm", "matmul")),
    ("normalization", ("rms_norm", "rmsnorm", "layer_norm")),
    ("activation", ("silu", "gelu", "activation", "mul_and_silu")),
    ("reduction", ("reduce", "argmax", "softmax")),
)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.panel_dir / "profile_metadata.json").read_text())
    trace = json.loads((args.panel_dir / "trace.json").read_text())
    client = json.loads((args.panel_dir / "client.json").read_text())
    steps = int(metadata["completed_steps"])
    if steps != int(metadata["target_steps"]):
        raise ValueError("incomplete R002x panel")
    if not metadata["all_decode_only"] or not metadata["all_target_batch"]:
        raise ValueError("contaminated R002x panel")

    by_name: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"duration_us": 0.0, "count": 0}
    )
    by_family: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"duration_us": 0.0, "count": 0}
    )
    total_us = 0.0
    kernel_count = 0
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

    def rows(source: dict[str, dict[str, float | int]]) -> list[dict[str, Any]]:
        ordered = sorted(
            source.items(),
            key=lambda item: float(item[1]["duration_us"]),
            reverse=True,
        )
        result = []
        for name, values in ordered[: args.top]:
            duration_ms = float(values["duration_us"]) / 1000.0
            count = int(values["count"])
            result.append(
                {
                    "name": name,
                    "count": count,
                    "count_per_step": count / steps,
                    "duration_ms": duration_ms,
                    "duration_ms_per_step": duration_ms / steps,
                    "share": duration_ms / (total_us / 1000.0) if total_us else 0.0,
                }
            )
        return result

    profiled = client["profiled"]
    failures = sum(
        int(item["completion_tokens"]) != int(profiled["max_tokens"])
        for item in profiled["requests"]
    )
    output = {
        "panel_id": metadata["panel_id"],
        "batch": int(metadata["target_batch"]),
        "steps": steps,
        "valid": True,
        "per_step_synchronize": bool(metadata["per_step_synchronize"]),
        "gpu_step_ms": describe([float(value) for value in metadata["gpu_step_ms"]]),
        "cpu_step_ms": describe([float(value) for value in metadata["cpu_step_ms"]]),
        "kernel_duration_ms": total_us / 1000.0,
        "kernel_duration_ms_per_step": total_us / 1000.0 / steps,
        "kernel_event_count": kernel_count,
        "kernel_event_count_per_step": kernel_count / steps,
        "families": rows(by_family),
        "kernel_names": rows(by_name),
        "client": {
            "completion_tokens": int(profiled["completion_tokens"]),
            "elapsed_s": float(profiled["elapsed_s"]),
            "tokens_per_second": float(profiled["tokens_per_second"]),
            "failures": failures,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "panel_id": output["panel_id"],
                "gpu_step_mean_ms": output["gpu_step_ms"]["mean"],
                "kernel_ms_per_step": output["kernel_duration_ms_per_step"],
                "cpu_step_mean_ms": output["cpu_step_ms"]["mean"],
                "failures": failures,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
