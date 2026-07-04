from __future__ import annotations

import json
import random
from pathlib import Path


SHORT_PROMPTS = [
    "Explain KV cache in one paragraph.",
    "Write a Python function to merge two sorted lists.",
    "Summarize the trade-off between latency and throughput in LLM serving.",
]

MEDIUM_PROMPTS = [
    "You are optimizing an LLM inference server. Compare continuous batching, "
    "dynamic batching, prefix caching, and paged KV cache. Give practical trade-offs.",
    "Given a production trace with bursty traffic, long prompts, and mixed decode lengths, "
    "propose a benchmark plan that can expose scheduler bottlenecks.",
]

LONG_PROMPT_BASE = (
    "We are designing a production LLM serving stack for a contest environment. "
    "The system must optimize TTFT, TPOT, throughput, GPU utilization, memory stability, "
    "scheduler fairness, and output validity under mixed workloads. "
)


def _long_prompt(repeats: int) -> str:
    return LONG_PROMPT_BASE * repeats + "Return a concrete optimization plan."


PROMPT_CLASSES = ("short", "medium", "long")
DECODE_CLASSES = ("short", "medium", "long")
PATTERNS = ("continuous", "burst", "idle", "mixed")
SCENARIOS = ("chat", "coding", "agent", "long_context")


def _choose_class(
    rng: random.Random,
    forced: str | None,
    values: tuple[str, ...],
    weights: list[float],
) -> str:
    if forced and forced != "mixed":
        if forced not in values:
            raise ValueError(f"Unknown class {forced!r}; expected one of {values}")
        return forced
    return rng.choices(list(values), weights=weights)[0]


def _arrival_delay_ms(idx: int, pattern: str, rng: random.Random) -> int:
    if pattern == "continuous":
        return idx * 40
    if pattern == "burst":
        burst_id = idx // 16
        inside_burst = idx % 16
        return burst_id * 1500 + inside_burst * 4
    if pattern == "idle":
        return idx * 250
    if pattern == "mixed":
        if idx % 23 == 0:
            return idx * 120
        return idx * rng.choice([0, 15, 40, 100])
    raise ValueError(f"Unknown request pattern {pattern!r}; expected one of {PATTERNS}")


def make_record(
    idx: int,
    rng: random.Random,
    *,
    prompt_class: str | None = None,
    decode_class: str | None = None,
    pattern: str = "mixed",
    case_id: str | None = None,
) -> dict:
    prompt_class = _choose_class(rng, prompt_class, PROMPT_CLASSES, [0.45, 0.35, 0.20])
    decode_class = _choose_class(rng, decode_class, DECODE_CLASSES, [0.50, 0.35, 0.15])

    if prompt_class == "short":
        prompt = rng.choice(SHORT_PROMPTS)
    elif prompt_class == "medium":
        prompt = rng.choice(MEDIUM_PROMPTS)
    else:
        prompt = _long_prompt(rng.randint(12, 32))

    max_tokens = {"short": 64, "medium": 256, "long": 768}[decode_class]
    scenario = rng.choice(SCENARIOS)
    request_id = f"mock-{idx:06d}" if case_id is None else f"{case_id}-{idx:06d}"

    return {
        "request_id": request_id,
        "case_id": case_id,
        "scenario": scenario,
        "prompt_class": prompt_class,
        "decode_class": decode_class,
        "request_pattern": pattern,
        "arrival_delay_ms": _arrival_delay_ms(idx, pattern, rng),
        "prompt": prompt,
        "prompt_tokens_hint": max(1, len(prompt) // 4),
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }


def generate_records(
    requests: int,
    seed: int,
    *,
    prompt_class: str | None = None,
    decode_class: str | None = None,
    pattern: str = "mixed",
    case_id: str | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    return [
        make_record(
            idx,
            rng,
            prompt_class=prompt_class,
            decode_class=decode_class,
            pattern=pattern,
            case_id=case_id,
        )
        for idx in range(requests)
    ]


def write_jsonl(output: Path, records: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_workload(
    output: Path,
    requests: int,
    seed: int,
    *,
    prompt_class: str | None = None,
    decode_class: str | None = None,
    pattern: str = "mixed",
    case_id: str | None = None,
) -> None:
    records = generate_records(
        requests,
        seed,
        prompt_class=prompt_class,
        decode_class=decode_class,
        pattern=pattern,
        case_id=case_id,
    )
    write_jsonl(output, records)
    print(f"Wrote {requests} requests to {output}")
