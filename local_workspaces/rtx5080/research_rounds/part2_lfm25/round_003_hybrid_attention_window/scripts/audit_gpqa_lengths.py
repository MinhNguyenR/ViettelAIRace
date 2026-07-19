#!/usr/bin/env python3
"""Report aggregate exact-token lengths for the fixed GPQA prompt contract."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def build_prompt(item: dict[str, Any]) -> str:
    choices = item["choices"]
    return (
        "Question:\n"
        f"{item['question']}\n\n"
        "Choices:\n"
        f"A. {choices['A']}\n"
        f"B. {choices['B']}\n"
        f"C. {choices['C']}\n"
        f"D. {choices['D']}\n\n"
        "Answer with exactly one capital letter: A, B, C, or D."
    )


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    lengths: list[int] = []
    records = 0
    with args.dataset.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are solving a multiple-choice science question. "
                        "Return exactly one capital letter: A, B, C, or D. "
                        "Do not explain."
                    ),
                },
                {"role": "user", "content": build_prompt(item)},
            ]
            if getattr(tokenizer, "chat_template", None):
                ids = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True
                )
                if hasattr(ids, "input_ids"):
                    ids = ids.input_ids
                elif isinstance(ids, dict):
                    ids = ids["input_ids"]
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
            else:
                text = messages[0]["content"] + "\n" + messages[1]["content"]
                ids = tokenizer(text, add_special_tokens=True)["input_ids"]
            lengths.append(len(ids))
            records += 1

    if not lengths:
        raise RuntimeError("dataset is empty")
    payload = {
        "schema": "r003.gpqa_lengths.v1",
        "records": records,
        "output_budget": 8,
        "input_tokens": {
            "min": min(lengths),
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
            "mean": statistics.fmean(lengths),
        },
        "window_8192_inactive_for_input_plus_output": max(lengths) + 8 <= 8192,
        "window_4096_inactive_for_input_plus_output": max(lengths) + 8 <= 4096,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
