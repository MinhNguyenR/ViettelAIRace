#!/usr/bin/env python3
"""Drive isolated six-turn append-only R002y profiling panels."""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any

import aiohttp


def load_trace(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                grouped[int(row["conv_id"])].append(row)
    for conv_id, turns in grouped.items():
        turns.sort(key=lambda row: int(row["turn_idx"]))
        if [int(row["turn_idx"]) for row in turns] != list(range(6)):
            raise ValueError(f"conversation {conv_id} does not contain six turns")
    return dict(grouped)


def patterned_tokens(token_pool: list[int], *, seed: int, length: int) -> list[int]:
    pool_size = len(token_pool)
    start = (seed * 4099 + 97) % pool_size
    stride = 7919
    return [token_pool[(start + index * stride) % pool_size] for index in range(length)]


def parse_sse_event(event: bytes) -> list[str]:
    payloads: list[str] = []
    for raw_line in event.splitlines():
        line = raw_line.strip()
        if line.startswith(b"data:"):
            payloads.append(line[5:].strip().decode("utf-8"))
    return payloads


async def stream_first_token(
    session: aiohttp.ClientSession,
    *,
    url: str,
    model: str,
    prompt_ids: list[int],
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": 1,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    started = time.perf_counter()
    first_token_at: float | None = None
    status = 0
    error = ""
    output_tokens = 0
    buffer = b""
    try:
        async with session.post(url, json=payload) as response:
            status = response.status
            if status != 200:
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
                            usage = message.get("usage") or {}
                            output_tokens = int(
                                usage.get("completion_tokens") or output_tokens
                            )
                            for choice in message.get("choices") or []:
                                if choice.get("text") or choice.get("token_ids"):
                                    first_token_at = first_token_at or time.perf_counter()
    except Exception as exc:  # noqa: BLE001 - profiler records transport failure
        error = f"{type(exc).__name__}: {exc}"
    finished = time.perf_counter()
    return {
        "success": status == 200 and not error and first_token_at is not None,
        "status": status,
        "error": error,
        "output_tokens": output_tokens,
        "ttft_ms": (
            (first_token_at - started) * 1000.0 if first_token_at else None
        ),
        "response_ms": (finished - started) * 1000.0,
    }


async def flush_profile(
    session: aiohttp.ClientSession,
    *,
    url: str,
    model: str,
    token_ids: list[int],
) -> None:
    payload = {
        "model": model,
        "prompt": token_ids,
        "max_tokens": 1,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }
    async with session.post(url, json=payload) as response:
        if response.status != 200:
            raise RuntimeError(
                f"profile flush failed: {response.status} {(await response.text())[:500]}"
            )
        await response.read()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


async def wait_for_done(path: Path, timeout_s: float = 120.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get("valid"):
                return payload
            raise RuntimeError(f"invalid profiler panel: {payload}")
        await asyncio.sleep(0.05)
    raise TimeoutError(f"timed out waiting for {path}")


async def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    trace = load_trace(args.trace)
    conv_ids = [int(value) for value in args.conversation_ids.split(",")]
    if len(conv_ids) != args.repeats:
        raise ValueError("conversation ID count must equal repeats")
    if any(conv_id not in trace for conv_id in conv_ids):
        raise ValueError("unknown conversation ID")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    special_ids = set(tokenizer.all_special_ids)
    token_pool = [
        token_id for token_id in range(len(tokenizer)) if token_id not in special_ids
    ]
    if len(token_pool) < 2048:
        raise ValueError("tokenizer does not expose enough non-special IDs")

    args.output_root.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=180.0)
    connector = aiohttp.TCPConnector(limit=1, force_close=True)
    manifest: list[dict[str, Any]] = []
    warmup_results: list[dict[str, Any]] = []
    completion_url = f"{args.base_url.rstrip('/')}/v1/completions"
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        warmup_conv_ids = [
            conv_id
            for conv_id in sorted(trace)
            if all(bool(turn["in_warmup"]) for turn in trace[conv_id])
        ][: args.warmup_conversation_count]
        if len(warmup_conv_ids) != args.warmup_conversation_count:
            raise ValueError("trace does not contain enough warmup conversations")
        for warmup_conv_id in warmup_conv_ids:
            warmup_history: list[int] = []
            for turn in trace[warmup_conv_id]:
                turn_idx = int(turn["turn_idx"])
                warmup_segment = patterned_tokens(
                    token_pool,
                    seed=10_000 + warmup_conv_id * 64 + turn_idx,
                    length=int(turn["in_tokens_est"]),
                )
                warmup_prompt = warmup_history + warmup_segment
                measured = await stream_first_token(
                    session,
                    url=completion_url,
                    model=args.model,
                    prompt_ids=warmup_prompt,
                )
                if not measured["success"]:
                    raise RuntimeError(
                        "shape warmup failed: "
                        f"conv={warmup_conv_id} turn={turn_idx} {measured}"
                    )
                warmup_results.append(
                    {
                        "conv_id": warmup_conv_id,
                        "turn_idx": turn_idx,
                        "prompt_tokens": len(warmup_prompt),
                        **measured,
                    }
                )
                warmup_history = warmup_prompt + patterned_tokens(
                    token_pool,
                    seed=4_000_000 + warmup_conv_id * 64 + turn_idx,
                    length=int(turn["out_tokens_max"]),
                )

        for repeat, conv_id in enumerate(conv_ids):
            history: list[int] = []
            for turn in trace[conv_id]:
                turn_idx = int(turn["turn_idx"])
                segment = patterned_tokens(
                    token_pool,
                    seed=10_000 + conv_id * 64 + turn_idx,
                    length=int(turn["in_tokens_est"]),
                )
                prompt = history + segment
                panel_id = f"r{repeat:02d}-c{conv_id:03d}-t{turn_idx}"
                panel_dir = args.output_root / "panels" / panel_id
                panel_dir.mkdir(parents=True, exist_ok=True)
                command = {
                    "panel_id": panel_id,
                    "expected_prompt_tokens": len(prompt),
                    "kineto": args.kineto,
                }
                write_json_atomic(args.arm, command)
                measured = await stream_first_token(
                    session,
                    url=completion_url,
                    model=args.model,
                    prompt_ids=prompt,
                )
                if not measured["success"]:
                    raise RuntimeError(f"target request failed: {panel_id} {measured}")

                args.finalize.touch()
                flush_ids = patterned_tokens(
                    token_pool,
                    seed=9_000_000 + repeat * 16 + turn_idx,
                    length=8,
                )
                await flush_profile(
                    session,
                    url=completion_url,
                    model=args.model,
                    token_ids=flush_ids,
                )
                await wait_for_done(panel_dir / "done.json")
                profile = json.loads((panel_dir / "profile_metadata.json").read_text())
                client = {
                    "panel_id": panel_id,
                    "repeat": repeat,
                    "conv_id": conv_id,
                    "turn_idx": turn_idx,
                    "declared_new_tokens": int(turn["in_tokens_est"]),
                    "declared_assistant_tokens": int(turn["out_tokens_max"]),
                    "prompt_tokens": len(prompt),
                    **measured,
                }
                write_json_atomic(panel_dir / "client.json", client)
                manifest.append(
                    {
                        **client,
                        "initial_computed_tokens": profile[
                            "initial_computed_tokens"
                        ],
                        "scheduled_prefill_tokens": profile[
                            "scheduled_prefill_tokens"
                        ],
                        "boundary_gpu_ms": profile["boundary_gpu_ms"],
                        "step_count": profile["step_count"],
                    }
                )
                print(
                    f"panel={panel_id} prompt={len(prompt)} "
                    f"computed={profile['initial_computed_tokens']} "
                    f"scheduled={profile['scheduled_prefill_tokens']} "
                    f"gpu_ms={profile['boundary_gpu_ms']:.6f} "
                    f"ttft_ms={measured['ttft_ms']:.6f}",
                    flush=True,
                )
                history = prompt + patterned_tokens(
                    token_pool,
                    seed=4_000_000 + conv_id * 64 + turn_idx,
                    length=int(turn["out_tokens_max"]),
                )

    write_json_atomic(
        args.output_root / "manifest.json",
        {
            "schema": "r002y.panel_manifest.v1",
            "kineto": args.kineto,
            "repeats": args.repeats,
            "conversation_ids": conv_ids,
            "warmup_conversation_ids": warmup_conv_ids,
            "warmup_results": warmup_results,
            "panels": manifest,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="LFM2.5-1.2B-Instruct")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--finalize", type=Path, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--conversation-ids", required=True)
    parser.add_argument("--warmup-conversation-count", type=int, default=15)
    parser.add_argument("--kineto", action="store_true")
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup_conversation_count <= 0:
        parser.error("--warmup-conversation-count must be positive")
    return args


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
