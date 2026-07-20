from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import aiohttp


TTFT_FLOOR_MS = 10.0
TTFT_CEILING_MS = 400.0
TPOT_FLOOR_MS = 1.0
TPOT_CEILING_MS = 10.0
STATEFUL_MODES = {
    "rolling_window",
    "rolling_window_static_assistant",
    "append_only",
    "append_only_static_assistant",
}
OUTPUT_CAPTURE_REQUIRED_MODES = {"rolling_window", "append_only"}


def component(value: float, floor: float, ceiling: float) -> float:
    normalized = max(0.0, min(1.0, (ceiling - value) / (ceiling - floor)))
    return normalized * normalized


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def metric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def load_trace(path: Path) -> list[list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {
                "conv_id",
                "turn_idx",
                "in_warmup",
                "timestamp_ms",
                "think_ms",
                "in_tokens_est",
                "out_tokens_max",
            }
            missing = required.difference(record)
            if missing:
                raise ValueError(f"line {line_number} missing fields: {sorted(missing)}")
            grouped[int(record["conv_id"])].append(record)

    conversations: list[list[dict[str, Any]]] = []
    for conv_id in sorted(grouped):
        turns = sorted(grouped[conv_id], key=lambda item: int(item["turn_idx"]))
        expected_turns = list(range(len(turns)))
        if [int(item["turn_idx"]) for item in turns] != expected_turns:
            raise ValueError(f"conversation {conv_id} has non-contiguous turns")
        if not turns or int(turns[0]["timestamp_ms"]) < 0:
            raise ValueError(f"conversation {conv_id} has invalid start")
        if any(int(item["timestamp_ms"]) != 0 for item in turns[1:]):
            raise ValueError(f"conversation {conv_id} has nonzero later timestamp")
        conversations.append(turns)
    return conversations


def patterned_tokens(token_pool: list[int], *, seed: int, length: int) -> list[int]:
    if length <= 0:
        return []
    if len(token_pool) < 2048:
        raise ValueError("token_pool must contain at least 2048 IDs")
    pool_size = len(token_pool)
    start = (seed * 4099 + 97) % pool_size
    stride = 7919
    return [token_pool[(start + index * stride) % pool_size] for index in range(length)]


@dataclass
class ConversationState:
    history: list[int] = field(default_factory=list)


@dataclass
class PromptBuild:
    token_ids: list[int]
    history_tokens_before_turn: int
    new_turn_tokens: int
    stable_prefix_tokens: int


def build_prompt(
    *,
    mode: str,
    conv_id: int,
    turn_idx: int,
    target_len: int,
    token_pool: list[int],
    state: ConversationState,
    fixed_prefix_tokens: int,
    stable_prefix_tokens: int,
    rolling_new_tokens: int,
) -> PromptBuild:
    if target_len < 2:
        raise ValueError("target_len must be at least two")
    request_seed = 10_000 + conv_id * 64 + turn_idx
    history_before = len(state.history)

    if mode == "independent":
        prompt = patterned_tokens(token_pool, seed=request_seed, length=target_len)
        return PromptBuild(prompt, history_before, target_len, 0)

    if mode == "fixed_prefix":
        prefix_len = min(fixed_prefix_tokens, target_len - 1)
        prefix = patterned_tokens(token_pool, seed=1_000_000 + conv_id, length=prefix_len)
        suffix = patterned_tokens(token_pool, seed=request_seed, length=target_len - prefix_len)
        return PromptBuild(prefix + suffix, history_before, len(suffix), prefix_len)

    if mode == "shared_fixed_prefix":
        prefix_len = min(fixed_prefix_tokens, target_len - 1)
        # Diagnostic-only mode: one trace-independent prefix for every
        # conversation makes the native common-prefix cohort observable.
        prefix = patterned_tokens(token_pool, seed=1_000_000, length=prefix_len)
        suffix = patterned_tokens(token_pool, seed=request_seed, length=target_len - prefix_len)
        return PromptBuild(prefix + suffix, history_before, len(suffix), prefix_len)

    if mode in {"append_only", "append_only_static_assistant"}:
        segment = patterned_tokens(token_pool, seed=request_seed, length=target_len)
        return PromptBuild(
            state.history + segment,
            history_before,
            len(segment),
            0,
        )

    if mode not in {"rolling_window", "rolling_window_static_assistant"}:
        raise ValueError(f"unsupported prompt mode: {mode}")

    prefix_len = min(stable_prefix_tokens, target_len - 1)
    stable_prefix = patterned_tokens(token_pool, seed=2_000_000 + conv_id, length=prefix_len)
    dynamic_budget = target_len - prefix_len
    new_len = min(rolling_new_tokens, dynamic_budget)
    carry_budget = dynamic_budget - new_len
    prior_dynamic = state.history[prefix_len:]
    carry = prior_dynamic[-carry_budget:] if carry_budget else []
    missing = carry_budget - len(carry)
    padding = patterned_tokens(
        token_pool,
        seed=3_000_000 + conv_id,
        length=missing,
    )
    new_segment = patterned_tokens(token_pool, seed=request_seed, length=new_len)
    prompt = stable_prefix + padding + carry + new_segment
    if len(prompt) != target_len:
        raise AssertionError(f"rolling prompt length {len(prompt)} != {target_len}")
    return PromptBuild(prompt, history_before, new_len, prefix_len)


def commit_turn(
    *,
    mode: str,
    state: ConversationState,
    prompt_ids: list[int],
    output_ids: list[int],
) -> None:
    if mode in STATEFUL_MODES:
        state.history = prompt_ids + output_ids


def select_history_output_ids(
    *,
    mode: str,
    generated_output_ids: list[int],
    token_pool: list[int],
    conv_id: int,
    turn_idx: int,
    output_tokens: int,
) -> tuple[list[int], str]:
    if mode not in {
        "rolling_window_static_assistant",
        "append_only_static_assistant",
    }:
        return generated_output_ids, "model_generated"
    return (
        patterned_tokens(
            token_pool,
            seed=4_000_000 + conv_id * 64 + turn_idx,
            length=output_tokens,
        ),
        "deterministic_static",
    )


def parse_sse_event(event: bytes) -> list[str]:
    payloads: list[str] = []
    for raw_line in event.splitlines():
        line = raw_line.strip()
        if line.startswith(b"data:"):
            payloads.append(line[5:].strip().decode("utf-8"))
    return payloads


def token_ids_from_choice(choice: dict[str, Any], capture_mode: str) -> list[int]:
    if capture_mode == "response":
        return [int(token_id) for token_id in choice.get("token_ids") or []]
    if capture_mode != "logprobs":
        return []
    logprobs = choice.get("logprobs") or {}
    token_ids: list[int] = []
    for token in logprobs.get("tokens") or []:
        if not isinstance(token, str) or not token.startswith("token_id:"):
            raise ValueError(f"unexpected token representation: {token!r}")
        token_ids.append(int(token.removeprefix("token_id:")))
    return token_ids


def token_id_digest(token_ids: list[int]) -> str | None:
    if not token_ids:
        return None
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, byteorder="little", signed=False))
    return digest.hexdigest()


async def stream_completion(
    session: "aiohttp.ClientSession",
    *,
    url: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    capture_mode: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    usage_tokens = 0
    cached_tokens: int | None = None
    output_token_ids: list[int] = []
    nonempty_chunks = 0
    error = ""
    status = 0
    buffer = b""

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if capture_mode == "response":
        payload["return_token_ids"] = True
    elif capture_mode == "logprobs":
        payload["logprobs"] = 0
        payload["return_tokens_as_token_ids"] = True

    try:
        async with session.post(url, json=payload) as response:
            status = response.status
            if response.status != 200:
                error = (await response.text())[:1000]
            else:
                async for chunk in response.content.iter_any():
                    buffer += chunk
                    while b"\n\n" in buffer:
                        event, buffer = buffer.split(b"\n\n", 1)
                        for data in parse_sse_event(event):
                            if data == "[DONE]":
                                continue
                            message = json.loads(data)
                            usage = message.get("usage")
                            if usage:
                                usage_tokens = int(usage.get("completion_tokens") or usage_tokens)
                                details = usage.get("prompt_tokens_details") or {}
                                if details.get("cached_tokens") is not None:
                                    cached_tokens = int(details["cached_tokens"])
                            for choice in message.get("choices") or []:
                                text = choice.get("text") or ""
                                choice_ids = token_ids_from_choice(choice, capture_mode)
                                if choice_ids:
                                    now = time.perf_counter()
                                    first_token_at = first_token_at or now
                                    last_token_at = now
                                    output_token_ids.extend(choice_ids)
                                elif text:
                                    now = time.perf_counter()
                                    first_token_at = first_token_at or now
                                    last_token_at = now
                                if text or choice_ids:
                                    nonempty_chunks += 1
    except Exception as exc:  # noqa: BLE001 - benchmark records transport failures
        error = f"{type(exc).__name__}: {exc}"

    finished = time.perf_counter()
    output_tokens = usage_tokens or len(output_token_ids) or nonempty_chunks
    if capture_mode != "none" and output_tokens != len(output_token_ids):
        error = error or (f"captured {len(output_token_ids)} output IDs for {output_tokens} tokens")
    successful = (
        status == 200
        and not error
        and first_token_at is not None
        and last_token_at is not None
        and output_tokens > 0
    )
    ttft_ms = (first_token_at - started) * 1000.0 if first_token_at else None
    if successful and output_tokens > 1:
        tpot_ms = (last_token_at - first_token_at) * 1000.0 / (output_tokens - 1)
    elif successful:
        tpot_ms = 0.0
    else:
        tpot_ms = None

    return {
        "success": successful,
        "status": status,
        "error": error,
        "input_tokens": len(prompt_ids),
        "cached_tokens": cached_tokens,
        "computed_prompt_tokens": (
            len(prompt_ids) - cached_tokens if cached_tokens is not None else None
        ),
        "output_tokens": output_tokens,
        "captured_output_token_ids": len(output_token_ids),
        "output_token_ids_sha256": token_id_digest(output_token_ids),
        "output_token_ids_prefix": output_token_ids[:8],
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": (finished - started) * 1000.0,
        "_output_token_ids": output_token_ids,
    }


async def run_benchmark(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import aiohttp
    from transformers import AutoTokenizer

    conversations = load_trace(args.trace)
    if args.conversation_limit is not None:
        conversations = conversations[: args.conversation_limit]
    expected_requests = sum(len(turns) for turns in conversations)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    special_ids = set(tokenizer.all_special_ids)
    token_pool = [token_id for token_id in range(len(tokenizer)) if token_id not in special_ids]
    if len(token_pool) < 2048:
        raise ValueError("tokenizer does not expose enough non-special IDs")

    state = {"active": 0, "peak_active": 0, "completed": 0}
    results: list[dict[str, Any]] = []
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=0, force_close=True)
    wall_started = time.perf_counter()
    schedule_base = wall_started + args.start_delay

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        if args.prime_shared_prefix:
            prime_prompt = build_prompt(
                mode="shared_fixed_prefix",
                conv_id=999_999,
                turn_idx=0,
                target_len=args.prime_prompt_tokens,
                token_pool=token_pool,
                state=ConversationState(),
                fixed_prefix_tokens=args.fixed_prefix_tokens,
                stable_prefix_tokens=args.stable_prefix_tokens,
                rolling_new_tokens=args.rolling_new_tokens,
            )
            prime = await stream_completion(
                session,
                url=f"{args.base_url.rstrip('/')}/v1/completions",
                model=args.model,
                prompt_ids=prime_prompt.token_ids,
                max_tokens=args.prime_output_tokens,
                capture_mode="none",
            )
            print(
                f"prime_shared_prefix success={int(prime['success'])} "
                f"prompt_tokens={len(prime_prompt.token_ids)}",
                flush=True,
            )

        async def run_conversation(turns: list[dict[str, Any]]) -> None:
            conversation_state = ConversationState()
            first = turns[0]
            due = schedule_base + int(first["timestamp_ms"]) / 1000.0 / args.time_scale
            await asyncio.sleep(max(0.0, due - time.perf_counter()))

            for index, record in enumerate(turns):
                conv_id = int(record["conv_id"])
                turn_idx = int(record["turn_idx"])
                declared_turn_tokens = int(record["in_tokens_est"])
                build_started = time.perf_counter()
                prompt = build_prompt(
                    mode=args.prompt_mode,
                    conv_id=conv_id,
                    turn_idx=turn_idx,
                    target_len=declared_turn_tokens,
                    token_pool=token_pool,
                    state=conversation_state,
                    fixed_prefix_tokens=args.fixed_prefix_tokens,
                    stable_prefix_tokens=args.stable_prefix_tokens,
                    rolling_new_tokens=args.rolling_new_tokens,
                )
                build_ms = (time.perf_counter() - build_started) * 1000.0
                max_tokens = int(record["out_tokens_max"])
                if args.output_token_limit is not None:
                    max_tokens = min(max_tokens, args.output_token_limit)

                state["active"] += 1
                state["peak_active"] = max(state["peak_active"], state["active"])
                try:
                    measured = await stream_completion(
                        session,
                        url=f"{args.base_url.rstrip('/')}/v1/completions",
                        model=args.model,
                        prompt_ids=prompt.token_ids,
                        max_tokens=max_tokens,
                        capture_mode=args.capture_output_token_ids,
                    )
                finally:
                    state["active"] -= 1

                output_token_ids = measured.pop("_output_token_ids")
                history_output_ids, history_output_source = select_history_output_ids(
                    mode=args.prompt_mode,
                    generated_output_ids=output_token_ids,
                    token_pool=token_pool,
                    conv_id=conv_id,
                    turn_idx=turn_idx,
                    output_tokens=max_tokens,
                )
                commit_turn(
                    mode=args.prompt_mode,
                    state=conversation_state,
                    prompt_ids=prompt.token_ids,
                    output_ids=history_output_ids,
                )
                measured.update(
                    {
                        "conv_id": conv_id,
                        "turn_idx": turn_idx,
                        "in_warmup": bool(record["in_warmup"]),
                        "declared_start_ms": int(first["timestamp_ms"]),
                        "declared_turn_tokens": declared_turn_tokens,
                        "history_tokens_before_turn": prompt.history_tokens_before_turn,
                        "new_turn_tokens": prompt.new_turn_tokens,
                        "stable_prefix_tokens": prompt.stable_prefix_tokens,
                        "history_output_source": history_output_source,
                        "history_output_tokens": len(history_output_ids),
                        "prompt_build_ms": build_ms,
                        "completed_at_s": time.perf_counter() - wall_started,
                    }
                )
                results.append(measured)
                state["completed"] += 1
                if state["completed"] % 30 == 0 or state["completed"] == expected_requests:
                    print(
                        f"progress={state['completed']}/{expected_requests} "
                        f"active={state['active']} peak={state['peak_active']}",
                        flush=True,
                    )

                if index + 1 < len(turns):
                    think_ms = int(record["think_ms"]) + args.extra_think_ms
                    await asyncio.sleep(think_ms / 1000.0 / args.time_scale)

        await asyncio.gather(*(run_conversation(turns) for turns in conversations))

    if len(results) != expected_requests:
        raise RuntimeError(
            f"benchmark produced {len(results)} of {expected_requests} expected requests"
        )
    wall_duration = time.perf_counter() - wall_started
    results.sort(key=lambda item: (item["conv_id"], item["turn_idx"]))
    graded = [item for item in results if not item["in_warmup"]]
    successful = [item for item in graded if item["success"]]
    scores: list[float] = []
    for item in graded:
        if not item["success"] or item["output_tokens"] <= 0:
            scores.append(0.0)
            continue
        assert item["ttft_ms"] is not None and item["tpot_ms"] is not None
        scores.append(
            0.5 * component(item["ttft_ms"], TTFT_FLOOR_MS, TTFT_CEILING_MS)
            + 0.5 * component(item["tpot_ms"], TPOT_FLOOR_MS, TPOT_CEILING_MS)
        )

    ttfts = [item["ttft_ms"] for item in successful if item["ttft_ms"] is not None]
    tpots = [item["tpot_ms"] for item in successful if item["tpot_ms"] is not None]
    cached = [
        float(item["cached_tokens"]) for item in successful if item["cached_tokens"] is not None
    ]
    computed = [
        float(item["computed_prompt_tokens"])
        for item in successful
        if item["computed_prompt_tokens"] is not None
    ]
    cache_ratios = [
        float(item["cached_tokens"]) / float(item["input_tokens"])
        for item in successful
        if item["cached_tokens"] is not None and item["input_tokens"] > 0
    ]

    by_turn: dict[str, Any] = {}
    turn_count = max((int(item["turn_idx"]) for item in results), default=-1) + 1
    for turn_idx in range(turn_count):
        cohort = [item for item in successful if item["turn_idx"] == turn_idx]
        turn_cached = [
            float(item["cached_tokens"]) for item in cohort if item["cached_tokens"] is not None
        ]
        turn_computed = [
            float(item["computed_prompt_tokens"])
            for item in cohort
            if item["computed_prompt_tokens"] is not None
        ]
        by_turn[str(turn_idx)] = {
            "count": len(cohort),
            "input_tokens": metric_summary([float(item["input_tokens"]) for item in cohort]),
            "cached_tokens": metric_summary(turn_cached),
            "computed_prompt_tokens": metric_summary(turn_computed),
            "ttft_ms": metric_summary(
                [float(item["ttft_ms"]) for item in cohort if item["ttft_ms"] is not None]
            ),
            "tpot_ms": metric_summary(
                [float(item["tpot_ms"]) for item in cohort if item["tpot_ms"] is not None]
            ),
        }

    summary = {
        "schema_version": 1,
        "label": args.label,
        "prompt_mode": args.prompt_mode,
        "capture_output_token_ids": args.capture_output_token_ids,
        "time_scale": args.time_scale,
        "extra_think_ms": args.extra_think_ms,
        "output_token_limit": args.output_token_limit,
        "conversation_count": len(conversations),
        "wall_duration_s": wall_duration,
        "total_count": len(results),
        "warmup_count": sum(1 for item in results if item["in_warmup"]),
        "graded_count": len(graded),
        "successful_graded": len(successful),
        "failed_graded": len(graded) - len(successful),
        "peak_active_requests": state["peak_active"],
        "output_sequence_sha256": hashlib.sha256(
            "".join(str(item["output_token_ids_sha256"]) for item in results).encode("ascii")
        ).hexdigest(),
        "ers": statistics.fmean(scores) if scores else 0.0,
        "score_points": 100.0 * statistics.fmean(scores) if scores else 0.0,
        "input_tokens": metric_summary([float(item["input_tokens"]) for item in successful]),
        "cached_tokens": metric_summary(cached),
        "computed_prompt_tokens": metric_summary(computed),
        "cache_hit_ratio": metric_summary(cache_ratios),
        "ttft_ms": metric_summary([float(value) for value in ttfts]),
        "tpot_ms": metric_summary([float(value) for value in tpots]),
        "ttft_le_10_ms": sum(value <= TTFT_FLOOR_MS for value in ttfts),
        "ttft_le_400_ms": sum(value <= TTFT_CEILING_MS for value in ttfts),
        "tpot_le_1_ms": sum(value <= TPOT_FLOOR_MS for value in tpots),
        "tpot_le_10_ms": sum(value <= TPOT_CEILING_MS for value in tpots),
        "by_turn": by_turn,
    }
    return results, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="LFM2.5-1.2B-Instruct")
    parser.add_argument("--tokenizer", default="/model")
    parser.add_argument(
        "--prompt-mode",
        choices=(
            "independent",
            "fixed_prefix",
            "shared_fixed_prefix",
            "rolling_window",
            "rolling_window_static_assistant",
            "append_only",
            "append_only_static_assistant",
        ),
        required=True,
    )
    parser.add_argument(
        "--capture-output-token-ids",
        choices=("none", "logprobs", "response"),
        default="logprobs",
    )
    parser.add_argument("--fixed-prefix-tokens", type=int, default=3000)
    parser.add_argument("--stable-prefix-tokens", type=int, default=512)
    parser.add_argument("--rolling-new-tokens", type=int, default=1000)
    parser.add_argument("--conversation-limit", type=int)
    parser.add_argument("--output-token-limit", type=int)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--extra-think-ms", type=float, default=0.0)
    parser.add_argument("--start-delay", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--prime-shared-prefix", action="store_true")
    parser.add_argument("--prime-prompt-tokens", type=int, default=4000)
    parser.add_argument("--prime-output-tokens", type=int, default=1)
    parser.add_argument("--label", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.time_scale <= 0:
        parser.error("--time-scale must be positive")
    if args.extra_think_ms < 0:
        parser.error("--extra-think-ms must be non-negative")
    if args.prime_prompt_tokens < 2 or args.prime_output_tokens <= 0:
        parser.error("prime token counts are invalid")
    if args.conversation_limit is not None and args.conversation_limit <= 0:
        parser.error("--conversation-limit must be positive")
    if args.output_token_limit is not None and args.output_token_limit <= 0:
        parser.error("--output-token-limit must be positive")
    if (
        min(
            args.fixed_prefix_tokens,
            args.stable_prefix_tokens,
            args.rolling_new_tokens,
        )
        < 0
    ):
        parser.error("prompt token controls must be non-negative")
    if (
        args.prompt_mode in OUTPUT_CAPTURE_REQUIRED_MODES
        and args.capture_output_token_ids == "none"
    ):
        parser.error("stateful prompt modes require exact output token capture")
    return args


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, summary = asyncio.run(run_benchmark(args))

    with (args.output_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed_graded"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
