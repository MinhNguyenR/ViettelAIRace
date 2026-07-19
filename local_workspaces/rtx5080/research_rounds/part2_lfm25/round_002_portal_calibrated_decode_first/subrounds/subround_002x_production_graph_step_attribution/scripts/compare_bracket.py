#!/usr/bin/env python3
"""Compare the R002x control/candidate/control graph-step bracket."""

from __future__ import annotations

import argparse
import bisect
import json
import statistics
from pathlib import Path
from typing import Any


BATCHES = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-a", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control-b", type=Path, required=True)
    parser.add_argument("--replay-requests", type=Path, required=True)
    parser.add_argument("--scheduler-telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_panels(root: Path) -> dict[int, dict[str, Any]]:
    panels = {}
    for batch in BATCHES:
        path = root / "panels" / f"batch{batch}" / "summary.json"
        payload = json.loads(path.read_text())
        if not payload["valid"] or int(payload["batch"]) != batch:
            raise ValueError(f"invalid R002x summary: {path}")
        if int(payload["client"]["failures"]) != 0:
            raise ValueError(f"request failure in R002x summary: {path}")
        panels[batch] = payload
    return panels


def family_value(panel: dict[str, Any], name: str) -> float:
    for row in panel["families"]:
        if row["name"] == name:
            return float(row["duration_ms_per_step"])
    return 0.0


def estimate_batch_weights(path: Path) -> dict[str, Any]:
    requests = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("in_warmup") or not row.get("success"):
                continue
            output_tokens = int(row.get("output_tokens") or 0)
            tpot_ms = float(row.get("tpot_ms") or 0.0)
            completed = float(row.get("completed_at_s") or 0.0)
            if output_tokens <= 1 or tpot_ms <= 0.0:
                continue
            first = completed - (output_tokens - 1) * tpot_ms / 1000.0
            requests.append(
                {
                    "first": first,
                    "end": completed,
                    "output_tokens": output_tokens,
                    "tpot_s": tpot_ms / 1000.0,
                }
            )
    starts = sorted(float(row["first"]) for row in requests)
    ends = sorted(float(row["end"]) for row in requests)
    estimated_steps = {batch: 0.0 for batch in range(1, 9)}
    clamped_events = 0
    token_events = 0
    for row in requests:
        for token_index in range(1, int(row["output_tokens"])):
            timestamp = float(row["first"]) + token_index * float(row["tpot_s"])
            active = bisect.bisect_right(starts, timestamp) - bisect.bisect_left(
                ends, timestamp
            )
            active = max(active, 1)
            if active > 8:
                active = 8
                clamped_events += 1
            estimated_steps[active] += 1.0 / active
            token_events += 1
    total = sum(estimated_steps.values())
    weights = {
        str(batch): estimated_steps[batch] / total if total else 0.0
        for batch in range(1, 9)
    }
    return {
        "method": "graded client decode intervals; token events divided by active count",
        "request_count": len(requests),
        "token_events": token_events,
        "clamped_events": clamped_events,
        "estimated_steps": {str(key): value for key, value in estimated_steps.items()},
        "weights": weights,
    }


def scheduler_batch_weights(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not payload.get("final"):
        raise ValueError("scheduler telemetry is not final")
    histogram = {
        int(key): int(value)
        for key, value in payload["pure_decode_batch_histogram"].items()
    }
    total = sum(histogram.values())
    if total != int(payload["pure_decode_steps"]):
        raise ValueError("pure-decode histogram does not match step count")
    weights = {
        str(batch): histogram.get(batch, 0) / total if total else 0.0
        for batch in range(1, 9)
    }
    active_steps = int(payload["total_steps"]) - int(payload["empty_steps"])
    decode_rows = int(payload["pure_decode_rows"]) + int(payload["mixed_decode_rows"])
    return {
        "method": "GPUModelRunner num_scheduled_tokens histogram",
        "classification": payload["classification"],
        "pure_decode_steps": int(payload["pure_decode_steps"]),
        "active_steps": active_steps,
        "pure_decode_step_fraction": int(payload["pure_decode_steps"])
        / active_steps,
        "pure_decode_rows": int(payload["pure_decode_rows"]),
        "mixed_decode_rows": int(payload["mixed_decode_rows"]),
        "eligible_decode_row_fraction": int(payload["pure_decode_rows"])
        / decode_rows,
        "histogram": {str(key): value for key, value in sorted(histogram.items())},
        "weights": weights,
    }


def interpolate(profiled: dict[int, float], batch: int) -> float:
    if batch in profiled:
        return profiled[batch]
    if batch == 3:
        return profiled[2] + 0.5 * (profiled[4] - profiled[2])
    if 5 <= batch <= 7:
        fraction = (batch - 4) / 4.0
        return profiled[4] + fraction * (profiled[8] - profiled[4])
    raise ValueError(f"cannot interpolate batch {batch}")


def weighted_value(profiled: dict[int, float], weights: dict[str, float]) -> float:
    return sum(
        float(weights[str(batch)]) * interpolate(profiled, batch)
        for batch in range(1, 9)
    )


def metric(panel: dict[str, Any], name: str) -> float:
    mapping = {
        "gpu_step_mean_ms": float(panel["gpu_step_ms"]["mean"]),
        "gpu_step_median_ms": float(panel["gpu_step_ms"]["median"]),
        "kernel_ms_per_step": float(panel["kernel_duration_ms_per_step"]),
        "kernel_count_per_step": float(panel["kernel_event_count_per_step"]),
        "cpu_step_mean_ms": float(panel["cpu_step_ms"]["mean"]),
        "client_tokens_per_second": float(panel["client"]["tokens_per_second"]),
    }
    return mapping[name]


def main() -> None:
    args = parse_args()
    control_a = load_panels(args.control_a)
    candidate = load_panels(args.candidate)
    control_b = load_panels(args.control_b)
    client_occupancy = estimate_batch_weights(args.replay_requests)
    scheduler_occupancy = scheduler_batch_weights(args.scheduler_telemetry)
    metrics = (
        "gpu_step_mean_ms",
        "gpu_step_median_ms",
        "kernel_ms_per_step",
        "kernel_count_per_step",
        "cpu_step_mean_ms",
        "client_tokens_per_second",
    )
    rows = []
    for batch in BATCHES:
        metric_rows = {}
        for name in metrics:
            a = metric(control_a[batch], name)
            b = metric(control_b[batch], name)
            value = metric(candidate[batch], name)
            control_mean = statistics.fmean((a, b))
            lower_is_better = name != "client_tokens_per_second"
            gain = (
                (control_mean - value) / control_mean
                if lower_is_better
                else (value - control_mean) / control_mean
            )
            metric_rows[name] = {
                "control_a": a,
                "candidate": value,
                "control_b": b,
                "control_mean": control_mean,
                "control_drift_fraction": abs(a - b) / control_mean,
                "candidate_gain_fraction": gain,
            }
        rows.append(
            {
                "batch": batch,
                "metrics": metric_rows,
                "control_native_causal_conv_ms_per_step": statistics.fmean(
                    (
                        family_value(control_a[batch], "native_causal_conv"),
                        family_value(control_b[batch], "native_causal_conv"),
                    )
                ),
                "candidate_native_causal_conv_ms_per_step": family_value(
                    candidate[batch], "native_causal_conv"
                ),
                "candidate_r002w_fused_ms_per_step": family_value(
                    candidate[batch], "r002w_fused_epilogue"
                ),
                "control_shortconv_mul_ms_per_step": statistics.fmean(
                    (
                        family_value(control_a[batch], "native_shortconv_mul"),
                        family_value(control_b[batch], "native_shortconv_mul"),
                    )
                ),
                "candidate_shortconv_mul_ms_per_step": family_value(
                    candidate[batch], "native_shortconv_mul"
                ),
                "control_staging_ms_per_step": statistics.fmean(
                    (
                        family_value(
                            control_a[batch], "native_shortconv_staging"
                        ),
                        family_value(
                            control_b[batch], "native_shortconv_staging"
                        ),
                    )
                ),
                "candidate_staging_ms_per_step": family_value(
                    candidate[batch], "native_shortconv_staging"
                ),
            }
        )

    for row in rows:
        direct_saving = (
            row["control_native_causal_conv_ms_per_step"]
            + row["control_shortconv_mul_ms_per_step"]
            - row["candidate_shortconv_mul_ms_per_step"]
            + row["control_staging_ms_per_step"]
            - row["candidate_staging_ms_per_step"]
            - row["candidate_r002w_fused_ms_per_step"]
        )
        row["direct_shortconv_saving_ms_per_step"] = direct_saving
        row["direct_shortconv_saving_fraction_of_control_kernel"] = (
            direct_saving
            / row["metrics"]["kernel_ms_per_step"]["control_mean"]
        )

    weighted = {}
    for name in metrics:
        a_values = {batch: metric(control_a[batch], name) for batch in BATCHES}
        b_values = {batch: metric(control_b[batch], name) for batch in BATCHES}
        candidate_values = {
            batch: metric(candidate[batch], name) for batch in BATCHES
        }
        control_values = {
            batch: statistics.fmean((a_values[batch], b_values[batch]))
            for batch in BATCHES
        }
        weighted_a = weighted_value(a_values, scheduler_occupancy["weights"])
        weighted_b = weighted_value(b_values, scheduler_occupancy["weights"])
        weighted_control = weighted_value(
            control_values, scheduler_occupancy["weights"]
        )
        weighted_candidate = weighted_value(
            candidate_values, scheduler_occupancy["weights"]
        )
        lower_is_better = name != "client_tokens_per_second"
        gain = (
            (weighted_control - weighted_candidate) / weighted_control
            if lower_is_better
            else (weighted_candidate - weighted_control) / weighted_control
        )
        weighted[name] = {
            "control_a": weighted_a,
            "candidate": weighted_candidate,
            "control_b": weighted_b,
            "control_mean": weighted_control,
            "control_drift_fraction": abs(weighted_a - weighted_b)
            / weighted_control,
            "candidate_gain_fraction": gain,
        }

    gpu = weighted["gpu_step_mean_ms"]
    cpu = weighted["cpu_step_mean_ms"]
    row_by_batch = {int(row["batch"]): row for row in rows}
    direct_saving_by_batch = {
        batch: float(row_by_batch[batch]["direct_shortconv_saving_ms_per_step"])
        for batch in BATCHES
    }
    control_kernel_by_batch = {
        batch: float(
            row_by_batch[batch]["metrics"]["kernel_ms_per_step"]["control_mean"]
        )
        for batch in BATCHES
    }
    weighted_direct_saving = weighted_value(
        direct_saving_by_batch, scheduler_occupancy["weights"]
    )
    weighted_control_kernel = weighted_value(
        control_kernel_by_batch, scheduler_occupancy["weights"]
    )
    direct_attribution = {
        "weighted_saving_ms_per_step": weighted_direct_saving,
        "weighted_fraction_of_control_kernel": weighted_direct_saving
        / weighted_control_kernel,
        "m8_saving_ms_per_step": direct_saving_by_batch[8],
    }
    opening_gate = {
        "weighted_gpu_gain_ge_1pct": gpu["candidate_gain_fraction"] >= 0.01,
        "batch4_gpu_gain_positive": row_by_batch[4]["metrics"][
            "gpu_step_mean_ms"
        ]["candidate_gain_fraction"] > 0.0,
        "batch8_gpu_gain_positive": row_by_batch[8]["metrics"][
            "gpu_step_mean_ms"
        ]["candidate_gain_fraction"] > 0.0,
        "weighted_gain_exceeds_control_drift": gpu["candidate_gain_fraction"]
        > gpu["control_drift_fraction"],
        "weighted_cpu_regression_le_2pct": cpu["candidate_gain_fraction"] >= -0.02,
        "all_panels_zero_failure": True,
    }
    opening_gate["passed"] = all(opening_gate.values())
    output = {
        "control_a": str(args.control_a),
        "candidate": str(args.candidate),
        "control_b": str(args.control_b),
        "scheduler_occupancy": scheduler_occupancy,
        "client_occupancy_crosscheck": client_occupancy,
        "rows": rows,
        "weighted": weighted,
        "direct_shortconv_attribution": direct_attribution,
        "opening_gate": opening_gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"weighted": weighted, "opening_gate": opening_gate}, indent=2))


if __name__ == "__main__":
    main()
