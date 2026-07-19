#!/usr/bin/env python3
"""Probe deterministic passkey recall across long-context distances.

The client intentionally uses plain strings. The server's own tokenizer
reports the actual prompt token count, so this probe does not load a second
copy of Transformers or the model tokenizer in the client container.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib import error, request


FILLER = (
    "The archive contains ordinary neutral background details about schedules, "
    "materials, locations, and routine observations. "
)
CHARS_PER_TOKEN = 6.7
PREFIX_TEMPLATE = (
    "This is a memory test. The secret passkey is {key}. Remember this exact "
    "passkey until the final question.\n\n"
)
SUFFIX = (
    "\n\nFinal question: What is the secret passkey? Return only the passkey "
    "with no explanation."
)


def normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def build_prompt(key: str, target_tokens: int) -> dict[str, Any]:
    # This filler was calibrated against the server tokenizer at about 6.7
    # characters per token. The server still records the exact count.
    target_chars = max(512, round(target_tokens * CHARS_PER_TOKEN))
    body_chars = max(len(FILLER), target_chars - len(PREFIX_TEMPLATE) - len(SUFFIX))
    repetitions = (body_chars + len(FILLER) - 1) // len(FILLER)
    body = (FILLER * repetitions)[:body_chars]
    prompt = PREFIX_TEMPLATE.format(key=key) + body + SUFFIX
    return {
        "prompt": prompt,
        "target_tokens": target_tokens,
        "prompt_chars": len(prompt),
        "approx_secret_to_query_chars": len(body),
    }


def complete(base_url: str, model: str, prompt: str) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 24,
            "temperature": 0,
            "stream": False,
        }
    ).encode("utf-8")
    req = request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    status = 0
    message: dict[str, Any] = {}
    failure = ""
    try:
        with request.urlopen(req, timeout=180) as response:
            status = int(response.status)
            message = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        status = int(exc.code)
        failure = exc.read().decode("utf-8", errors="replace")[:2000]
    except Exception as exc:  # noqa: BLE001 - preserve probe failure
        failure = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    choices = message.get("choices") or []
    output = str(choices[0].get("text") or "") if choices else ""
    usage = message.get("usage") or {}
    return {
        "success": status == 200 and bool(choices),
        "status": status,
        "error": failure,
        "output": output,
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "actual_prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "elapsed_ms": elapsed_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    # Kept for a stable runner interface; tokenization is intentionally server-side.
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="LFM2.5-1.2B-Instruct")
    parser.add_argument("--label", required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument(
        "--targets", default="3072,5120,7168,9216,12288,17408,22528"
    )
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    targets = [int(value) for value in args.targets.split(",") if value]
    rows: list[dict[str, Any]] = []
    for target in targets:
        for replicate in range(args.replicates):
            digest = hashlib.sha256(
                f"r003a:{target}:{replicate}".encode("ascii")
            ).hexdigest()[:10].upper()
            key = f"K{digest}"
            built = build_prompt(key, target)
            response = (
                {
                    "success": True,
                    "status": 0,
                    "error": "",
                    "output": "",
                    "output_tokens": 0,
                    "actual_prompt_tokens": 0,
                    "elapsed_ms": 0.0,
                }
                if args.build_only
                else complete(args.base_url, args.model, built["prompt"])
            )
            normalized_output = normalize(response["output"])
            expected = normalize(key)
            row = {
                "target_tokens": target,
                "replicate": replicate,
                "key": key,
                "expected_normalized": expected,
                "prompt_chars": built["prompt_chars"],
                "approx_secret_to_query_chars": built[
                    "approx_secret_to_query_chars"
                ],
                **response,
                "normalized_output": normalized_output,
                "exact_match": normalized_output == expected,
                "contains_key": expected in normalized_output,
                "output_sha256": hashlib.sha256(
                    response["output"].encode("utf-8")
                ).hexdigest(),
            }
            rows.append(row)
            print(
                f"target={target} actual={row['actual_prompt_tokens']} "
                f"rep={replicate} ok={int(row['success'])} "
                f"contains={int(row['contains_key'])} "
                f"output={response['output']!r}",
                flush=True,
            )

    by_target = {}
    for target in targets:
        selected = [row for row in rows if row["target_tokens"] == target]
        by_target[str(target)] = {
            "cases": len(selected),
            "success": sum(bool(row["success"]) for row in selected),
            "exact_match": sum(bool(row["exact_match"]) for row in selected),
            "contains_key": sum(bool(row["contains_key"]) for row in selected),
            "actual_prompt_tokens": [
                int(row["actual_prompt_tokens"]) for row in selected
            ],
        }
    payload = {
        "schema": "r003a.long_context_retrieval.v2",
        "label": args.label,
        "window": args.window,
        "build_only": args.build_only,
        "total_cases": len(rows),
        "successful_cases": sum(bool(row["success"]) for row in rows),
        "exact_matches": sum(bool(row["exact_match"]) for row in rows),
        "contains_key": sum(bool(row["contains_key"]) for row in rows),
        "by_target": by_target,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
