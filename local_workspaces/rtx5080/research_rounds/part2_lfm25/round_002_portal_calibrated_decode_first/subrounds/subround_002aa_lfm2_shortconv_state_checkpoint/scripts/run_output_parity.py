#!/usr/bin/env python3
"""Capture deterministic multi-token outputs for R002aa state parity."""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import json
from pathlib import Path
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


async def complete(
    session: aiohttp.ClientSession,
    *,
    url: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
        "return_token_ids": True,
    }
    async with session.post(url, json=payload) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"completion failed: {response.status} {body[:1000]}")
    message = json.loads(body)
    choices = message.get("choices") or []
    if len(choices) != 1:
        raise RuntimeError(f"expected one completion choice, got {len(choices)}")
    token_ids = choices[0].get("token_ids")
    if not isinstance(token_ids, list) or len(token_ids) != max_tokens:
        raise RuntimeError(
            f"expected {max_tokens} returned token IDs, got {token_ids!r}"
        )
    return {
        "token_ids": [int(token_id) for token_id in token_ids],
        "text": str(choices[0].get("text") or ""),
        "finish_reason": choices[0].get("finish_reason"),
    }


async def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    trace = load_trace(args.trace)
    conversation_ids = [int(value) for value in args.conversation_ids.split(",")]
    if any(conv_id not in trace for conv_id in conversation_ids):
        raise ValueError("unknown target conversation ID")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    special_ids = set(tokenizer.all_special_ids)
    token_pool = [
        token_id for token_id in range(len(tokenizer)) if token_id not in special_ids
    ]
    if len(token_pool) < 2048:
        raise ValueError("tokenizer does not expose enough non-special IDs")

    warmup_conv_ids = [
        conv_id
        for conv_id in sorted(trace)
        if all(bool(turn["in_warmup"]) for turn in trace[conv_id])
    ][: args.warmup_conversation_count]
    if len(warmup_conv_ids) != args.warmup_conversation_count:
        raise ValueError("trace does not contain enough warmup conversations")

    completion_url = f"{args.base_url.rstrip('/')}/v1/completions"
    timeout = aiohttp.ClientTimeout(total=180.0)
    connector = aiohttp.TCPConnector(limit=1, force_close=True)
    rows: list[dict[str, Any]] = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for warmup_conv_id in warmup_conv_ids:
            history: list[int] = []
            for turn in trace[warmup_conv_id]:
                turn_idx = int(turn["turn_idx"])
                segment = patterned_tokens(
                    token_pool,
                    seed=10_000 + warmup_conv_id * 64 + turn_idx,
                    length=int(turn["in_tokens_est"]),
                )
                prompt = history + segment
                await complete(
                    session,
                    url=completion_url,
                    model=args.model,
                    prompt_ids=prompt,
                    max_tokens=1,
                )
                history = prompt + patterned_tokens(
                    token_pool,
                    seed=4_000_000 + warmup_conv_id * 64 + turn_idx,
                    length=int(turn["out_tokens_max"]),
                )

        for conv_id in conversation_ids:
            history = []
            for turn in trace[conv_id]:
                turn_idx = int(turn["turn_idx"])
                segment = patterned_tokens(
                    token_pool,
                    seed=10_000 + conv_id * 64 + turn_idx,
                    length=int(turn["in_tokens_est"]),
                )
                prompt = history + segment
                completion = await complete(
                    session,
                    url=completion_url,
                    model=args.model,
                    prompt_ids=prompt,
                    max_tokens=args.max_tokens,
                )
                row = {
                    "conv_id": conv_id,
                    "turn_idx": turn_idx,
                    "prompt_tokens": len(prompt),
                    **completion,
                }
                rows.append(row)
                print(
                    f"conv={conv_id} turn={turn_idx} prompt={len(prompt)} "
                    f"output_ids={completion['token_ids']}",
                    flush=True,
                )
                history = prompt + patterned_tokens(
                    token_pool,
                    seed=4_000_000 + conv_id * 64 + turn_idx,
                    length=int(turn["out_tokens_max"]),
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "r002aa.output_parity.v1",
                "max_tokens": args.max_tokens,
                "warmup_conversation_ids": warmup_conv_ids,
                "conversation_ids": conversation_ids,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="LFM2.5-1.2B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conversation-ids", required=True)
    parser.add_argument("--warmup-conversation-count", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()
    if args.warmup_conversation_count <= 0:
        parser.error("--warmup-conversation-count must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    return args


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
