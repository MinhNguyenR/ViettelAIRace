from __future__ import annotations

import concurrent.futures
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                if limit and len(records) >= limit:
                    break
    return records


def _dry_result(record: dict[str, Any]) -> dict[str, Any]:
    prompt_factor = {"short": 1.0, "medium": 2.5, "long": 8.0}.get(record["prompt_class"], 1.0)
    decode_factor = {"short": 1.0, "medium": 3.0, "long": 7.0}.get(record["decode_class"], 1.0)
    jitter = random.Random(record["request_id"]).uniform(0.85, 1.25)
    ttft_ms = 80.0 * prompt_factor * jitter
    latency_ms = ttft_ms + 12.0 * decode_factor * record["max_tokens"] / 16
    return {
        "request_id": record["request_id"],
        "case_id": record.get("case_id"),
        "scenario": record["scenario"],
        "prompt_class": record["prompt_class"],
        "decode_class": record["decode_class"],
        "request_pattern": record.get("request_pattern"),
        "arrival_delay_ms": record.get("arrival_delay_ms", 0),
        "prompt_tokens_hint": record["prompt_tokens_hint"],
        "max_tokens": record["max_tokens"],
        "status": "ok",
        "ttft_ms": round(ttft_ms, 3),
        "latency_ms": round(latency_ms, 3),
        "output_tokens": record["max_tokens"],
        "error": "",
        "dry_run": True,
    }


def _post_openai_chat(
    api_base: str,
    model: str,
    record: dict[str, Any],
    timeout: float,
    stream: bool,
) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "messages": [{"role": "user", "content": record["prompt"]}],
        "temperature": record.get("temperature", 0.0),
        "max_tokens": record["max_tokens"],
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}

    started = time.perf_counter()
    ttft = None
    output_chunks = 0
    output_tokens = None
    status = "ok"
    error = ""

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if stream:
                while True:
                    line = response.readline()
                    if not line:
                        break
                    if not line.startswith(b"data:"):
                        continue
                    payload = line.removeprefix(b"data:").strip()
                    if payload == b"[DONE]":
                        break
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    output_chunks += 1
                    try:
                        event = json.loads(payload)
                        usage = event.get("usage") or {}
                        if usage.get("completion_tokens") is not None:
                            output_tokens = usage["completion_tokens"]
                    except json.JSONDecodeError:
                        pass
            else:
                payload = json.loads(response.read().decode("utf-8"))
                usage = payload.get("usage") or {}
                output_tokens = usage.get("completion_tokens")
                ttft = time.perf_counter() - started
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        status = "error"
        error = str(exc)

    latency = time.perf_counter() - started
    return {
        "request_id": record["request_id"],
        "case_id": record.get("case_id"),
        "scenario": record["scenario"],
        "prompt_class": record["prompt_class"],
        "decode_class": record["decode_class"],
        "request_pattern": record.get("request_pattern"),
        "arrival_delay_ms": record.get("arrival_delay_ms", 0),
        "prompt_tokens_hint": record["prompt_tokens_hint"],
        "max_tokens": record["max_tokens"],
        "status": status,
        "ttft_ms": round(ttft * 1000, 3) if ttft is not None else None,
        "latency_ms": round(latency * 1000, 3),
        "output_tokens": output_tokens if output_tokens is not None else output_chunks,
        "error": error,
        "dry_run": False,
    }


def _with_timing(record: dict[str, Any], worker, time_scale: float) -> dict[str, Any]:
    delay_ms = float(record.get("arrival_delay_ms") or 0) * max(0.0, time_scale)
    if delay_ms > 0:
        time.sleep(delay_ms / 1000)
    started_at = time.time()
    result = worker(record)
    finished_at = time.time()
    result["started_at_epoch"] = round(started_at, 6)
    result["finished_at_epoch"] = round(finished_at, 6)
    result["wall_time_ms"] = round((finished_at - started_at) * 1000, 3)
    output_tokens = result.get("output_tokens") or 0
    latency_ms = result.get("latency_ms") or 0
    result["tpot_ms"] = round(latency_ms / output_tokens, 6) if output_tokens else None
    return result


def run_benchmark(
    workload_path: Path,
    output_path: Path,
    api_base: str,
    model: str,
    concurrency: int,
    limit: int,
    timeout: float,
    stream: bool,
    dry_run: bool,
    time_scale: float = 1.0,
) -> dict[str, Any]:
    records = _load_jsonl(workload_path, limit=limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    worker = _dry_result if dry_run else lambda r: _post_openai_chat(api_base, model, r, timeout, stream)
    started_at = time.time()

    with output_path.open("w", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(_with_timing, record, worker, time_scale) for record in records]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

    finished_at = time.time()
    print(f"Wrote {len(records)} benchmark results to {output_path}")
    return {
        "workload_path": str(workload_path),
        "output_path": str(output_path),
        "requests": len(records),
        "concurrency": concurrency,
        "dry_run": dry_run,
        "started_at_epoch": round(started_at, 6),
        "finished_at_epoch": round(finished_at, 6),
        "duration_s": round(finished_at - started_at, 6),
    }
