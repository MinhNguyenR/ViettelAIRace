#!/usr/bin/env python3
"""Run M1/M2/M4/M8 steady-decode panels against one clean server."""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


BATCHES = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--tail-tokens", type=int, default=32)
    return parser.parse_args()


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def run_batch(
    base_url: str,
    batch: int,
    max_tokens: int,
    phase: str,
) -> dict[str, Any]:
    barrier = threading.Barrier(batch)
    started = time.perf_counter()

    def worker(index: int) -> dict[str, Any]:
        prompt = (
            "Describe graph-safe GPU inference scheduling with concrete steps. "
            + "Continue with implementation-level details and stable terminology. "
            * 64
            + f"Unique R002x marker {phase}-{batch}-{index}."
        )
        body = {
            "model": "LFM2.5-1.2B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "stream": False,
        }
        barrier.wait()
        request_started = time.perf_counter()
        response = post_json(f"{base_url}/v1/chat/completions", body)
        usage = response.get("usage") or {}
        return {
            "index": index,
            "elapsed_s": time.perf_counter() - request_started,
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
        }

    with ThreadPoolExecutor(max_workers=batch) as executor:
        requests = list(executor.map(worker, range(batch)))
    elapsed = time.perf_counter() - started
    tokens = sum(item["completion_tokens"] for item in requests)
    return {
        "phase": phase,
        "batch": batch,
        "max_tokens": max_tokens,
        "elapsed_s": elapsed,
        "completion_tokens": tokens,
        "tokens_per_second": tokens / elapsed if elapsed else None,
        "requests": requests,
    }


def write_arm(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def wait_for_done(path: Path, timeout_s: float = 180.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text())
        time.sleep(0.1)
    raise RuntimeError(f"profile panel did not finish: {path}")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    arm_path = args.output_root / "arm.json"
    arm_path.unlink(missing_ok=True)
    manifest: dict[str, Any] = {
        "steps": args.steps,
        "warmup_tokens": args.warmup_tokens,
        "tail_tokens": args.tail_tokens,
        "panels": [],
    }

    for batch in BATCHES:
        panel_id = f"batch{batch}"
        panel_dir = args.output_root / "panels" / panel_id
        panel_dir.mkdir(parents=True, exist_ok=True)
        warmup = run_batch(
            args.base_url,
            batch,
            args.warmup_tokens,
            f"warmup-{panel_id}",
        )
        write_arm(
            arm_path,
            {
                "panel_id": panel_id,
                "target_batch": batch,
                "target_steps": args.steps,
            },
        )
        profiled = run_batch(
            args.base_url,
            batch,
            args.steps + args.tail_tokens,
            f"profile-{panel_id}",
        )
        done = wait_for_done(panel_dir / "done.json")
        if not done.get("valid"):
            raise RuntimeError(f"invalid R002x profile panel: {done}")
        client = {
            "panel_id": panel_id,
            "batch": batch,
            "warmup": warmup,
            "profiled": profiled,
            "done": done,
        }
        (panel_dir / "client.json").write_text(
            json.dumps(client, indent=2) + "\n"
        )
        manifest["panels"].append(
            {
                "panel_id": panel_id,
                "batch": batch,
                "profiled_tokens_per_second": profiled["tokens_per_second"],
                "request_failures": sum(
                    item["completion_tokens"] != args.steps + args.tail_tokens
                    for item in profiled["requests"]
                ),
            }
        )
        print(
            f"R002x panel complete: batch={batch} "
            f"throughput={profiled['tokens_per_second']:.3f}",
            flush=True,
        )

    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
