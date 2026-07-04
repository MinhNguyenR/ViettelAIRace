from __future__ import annotations

from collections.abc import Iterable
from statistics import mean


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("status") == "ok"]
    latencies = [r.get("latency_ms") for r in ok if r.get("latency_ms") is not None]
    ttfts = [r.get("ttft_ms") for r in ok if r.get("ttft_ms") is not None]
    tpots = [r.get("tpot_ms") for r in ok if r.get("tpot_ms") is not None]
    output_tokens = sum((r.get("output_tokens") or 0) for r in ok)
    total_latency_s = sum((r.get("latency_ms") or 0) for r in ok) / 1000
    starts = [r.get("started_at_epoch") for r in records if r.get("started_at_epoch") is not None]
    finishes = [r.get("finished_at_epoch") for r in records if r.get("finished_at_epoch") is not None]
    wall_duration_s = (max(finishes) - min(starts)) if starts and finishes else None

    return {
        "requests": len(records),
        "success": len(ok),
        "errors": len(records) - len(ok),
        "success_rate": len(ok) / len(records) if records else 0.0,
        "wall_duration_s": wall_duration_s,
        "requests_per_s": len(ok) / wall_duration_s if wall_duration_s else None,
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p90_ms": percentile(latencies, 0.90),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_p99_ms": percentile(latencies, 0.99),
        "latency_avg_ms": mean(latencies) if latencies else None,
        "ttft_p50_ms": percentile(ttfts, 0.50),
        "ttft_p90_ms": percentile(ttfts, 0.90),
        "ttft_p95_ms": percentile(ttfts, 0.95),
        "ttft_p99_ms": percentile(ttfts, 0.99),
        "tpot_p50_ms": percentile(tpots, 0.50),
        "tpot_p95_ms": percentile(tpots, 0.95),
        "output_tokens": output_tokens,
        "wall_tokens_per_s": output_tokens / wall_duration_s if wall_duration_s else None,
        "approx_serial_tokens_per_s": output_tokens / total_latency_s if total_latency_s else None,
    }


def group_counts(records: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
