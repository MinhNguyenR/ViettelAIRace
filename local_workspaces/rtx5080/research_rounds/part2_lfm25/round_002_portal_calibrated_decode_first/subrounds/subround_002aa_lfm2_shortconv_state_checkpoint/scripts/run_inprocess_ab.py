#!/usr/bin/env python3
"""Alternate native align and sparse R002aa panels inside one engine process."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

import aiohttp


def set_mode(path: Path, *, sparse: bool) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text("1\n" if sparse else "0\n")
    temporary.replace(path)


async def run(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.r002y_scripts))
    from run_prefill_panels import (  # type: ignore[import-not-found]
        flush_profile,
        load_trace,
        patterned_tokens,
        stream_first_token,
        wait_for_done,
        write_json_atomic,
    )
    from transformers import AutoTokenizer

    trace = load_trace(args.trace)
    conv_ids = [int(value) for value in args.conversation_ids.split(",")]
    if len(conv_ids) != args.pairs * 2:
        raise ValueError("conversation ID count must equal two times --pairs")
    if any(conv_id not in trace for conv_id in conv_ids):
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

    assignments: list[list[tuple[str, int]]] = []
    for pair in range(args.pairs):
        first_id, second_id = conv_ids[pair * 2 : pair * 2 + 2]
        if pair % 2 == 0:
            assignments.append([("control", first_id), ("sparse", second_id)])
        else:
            assignments.append([("sparse", first_id), ("control", second_id)])

    completion_url = f"{args.base_url.rstrip('/')}/v1/completions"
    timeout = aiohttp.ClientTimeout(total=180.0)
    connector = aiohttp.TCPConnector(limit=1, force_close=True)
    histories: dict[int, list[int]] = {conv_id: [] for conv_id in conv_ids}
    manifest: list[dict[str, Any]] = []
    set_mode(args.mode_file, sparse=False)

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
                measured = await stream_first_token(
                    session,
                    url=completion_url,
                    model=args.model,
                    prompt_ids=prompt,
                )
                if not measured["success"]:
                    raise RuntimeError(
                        "shape warmup failed: "
                        f"conv={warmup_conv_id} turn={turn_idx} {measured}"
                    )
                history = prompt + patterned_tokens(
                    token_pool,
                    seed=4_000_000 + warmup_conv_id * 64 + turn_idx,
                    length=int(turn["out_tokens_max"]),
                )

        # Compile the sparse/APC branch and its padded shape outside the panels.
        # The regular warmup above intentionally exercises only the native control.
        set_mode(args.mode_file, sparse=True)
        sparse_warmup_history: list[int] = []
        for turn in trace[conv_ids[0]][:2]:
            turn_idx = int(turn["turn_idx"])
            segment = patterned_tokens(
                token_pool,
                seed=8_000_000 + turn_idx,
                length=int(turn["in_tokens_est"]),
            )
            prompt = sparse_warmup_history + segment
            measured = await stream_first_token(
                session,
                url=completion_url,
                model=args.model,
                prompt_ids=prompt,
            )
            if not measured["success"]:
                raise RuntimeError(f"sparse branch warmup failed: {measured}")
            sparse_warmup_history = prompt + patterned_tokens(
                token_pool,
                seed=8_100_000 + turn_idx,
                length=int(turn["out_tokens_max"]),
            )
        set_mode(args.mode_file, sparse=False)

        for turn_idx in range(6):
            for pair, ordered_assignments in enumerate(assignments):
                shape_turn = trace[ordered_assignments[0][1]][turn_idx]
                for mode, conv_id in ordered_assignments:
                    segment = patterned_tokens(
                        token_pool,
                        seed=10_000 + conv_id * 64 + turn_idx,
                        length=int(shape_turn["in_tokens_est"]),
                    )
                    prompt = histories[conv_id] + segment
                    panel_id = f"p{pair:02d}-{mode}-c{conv_id:03d}-t{turn_idx}"
                    panel_dir = args.output_root / "panels" / panel_id
                    panel_dir.mkdir(parents=True, exist_ok=True)
                    set_mode(args.mode_file, sparse=mode == "sparse")
                    write_json_atomic(
                        args.arm,
                        {
                            "panel_id": panel_id,
                            "expected_prompt_tokens": len(prompt),
                            "kineto": args.kineto,
                        },
                    )
                    measured = await stream_first_token(
                        session,
                        url=completion_url,
                        model=args.model,
                        prompt_ids=prompt,
                    )
                    if not measured["success"]:
                        raise RuntimeError(
                            f"target request failed: {panel_id} {measured}"
                        )

                    args.finalize.touch()
                    flush_ids = patterned_tokens(
                        token_pool,
                        seed=9_000_000 + pair * 64 + turn_idx * 2 + int(mode == "sparse"),
                        length=8,
                    )
                    await flush_profile(
                        session,
                        url=completion_url,
                        model=args.model,
                        token_ids=flush_ids,
                    )
                    await wait_for_done(panel_dir / "done.json")
                    profile = json.loads(
                        (panel_dir / "profile_metadata.json").read_text()
                    )
                    row = {
                        "panel_id": panel_id,
                        "pair": pair,
                        "mode": mode,
                        "conv_id": conv_id,
                        "turn_idx": turn_idx,
                        "prompt_tokens": len(prompt),
                        **measured,
                        "initial_computed_tokens": profile[
                            "initial_computed_tokens"
                        ],
                        "scheduled_prefill_tokens": profile[
                            "scheduled_prefill_tokens"
                        ],
                        "boundary_gpu_ms": profile["boundary_gpu_ms"],
                        "step_count": profile["step_count"],
                    }
                    write_json_atomic(panel_dir / "client.json", row)
                    manifest.append(row)
                    print(
                        f"panel={panel_id} prompt={len(prompt)} "
                        f"computed={profile['initial_computed_tokens']} "
                        f"steps={profile['step_count']} "
                        f"gpu_ms={profile['boundary_gpu_ms']:.6f} "
                        f"ttft_ms={measured['ttft_ms']:.6f}",
                        flush=True,
                    )
                    histories[conv_id] = prompt + patterned_tokens(
                        token_pool,
                        seed=4_000_000 + conv_id * 64 + turn_idx,
                        length=int(shape_turn["out_tokens_max"]),
                    )

    set_mode(args.mode_file, sparse=False)
    write_json_atomic(
        args.output_root / "manifest.json",
        {
            "schema": "r002aa.inprocess_ab.v1",
            "pairs": args.pairs,
            "conversation_ids": conv_ids,
            "assignments": assignments,
            "warmup_conversation_ids": warmup_conv_ids,
            "panels": manifest,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--r002y-scripts", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="LFM2.5-1.2B-Instruct")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--finalize", type=Path, required=True)
    parser.add_argument("--mode-file", type=Path, required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--conversation-ids", required=True)
    parser.add_argument("--warmup-conversation-count", type=int, default=15)
    parser.add_argument("--kineto", action="store_true")
    args = parser.parse_args()
    if args.pairs <= 0:
        parser.error("--pairs must be positive")
    return args


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
