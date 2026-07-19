#!/usr/bin/env python3
"""Test append-only prompt progression without loading a tokenizer."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_prefill_panels.py"
SPEC = importlib.util.spec_from_file_location("r002y_driver", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    token_pool = list(range(4096))
    first = MODULE.patterned_tokens(token_pool, seed=1, length=4000)
    assistant = MODULE.patterned_tokens(token_pool, seed=2, length=200)
    second = first + assistant + MODULE.patterned_tokens(
        token_pool, seed=3, length=3999
    )
    assert len(first) == 4000
    assert len(second) == 8199
    assert second[:4000] == first
    assert first != MODULE.patterned_tokens(token_pool, seed=4, length=4000)
    print("r002y_panel_driver_tests=passed")


if __name__ == "__main__":
    main()
