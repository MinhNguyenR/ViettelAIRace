#!/usr/bin/env python3
"""Unit checks for the R003g layer and short-path policy."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "overlay"))

from policy import (
    FULL_LAYERS,
    effective_split_prefix_tokens,
    parse_window_map,
)


def test_complete_map() -> None:
    parsed = parse_window_map("2:4096,5:4096,8:4096,10:4096,12:4096,14:4096")
    assert set(parsed) == FULL_LAYERS
    assert set(parsed.values()) == {4096}


def test_incomplete_map_rejected() -> None:
    try:
        parse_window_map("2:4096")
    except ValueError:
        return
    raise AssertionError("incomplete layer map was accepted")


def test_effective_split_prefix_tokens() -> None:
    common = {
        "window_size": (4095, 0),
        "configured_prefix_tokens": 64,
        "short_native_enabled": True,
    }
    assert effective_split_prefix_tokens(max_seqlen_k=1, **common) == 0
    assert effective_split_prefix_tokens(max_seqlen_k=4096, **common) == 0
    assert effective_split_prefix_tokens(max_seqlen_k=4097, **common) == 64
    assert (
        effective_split_prefix_tokens(
            max_seqlen_k=4096,
            window_size=(4095, 0),
            configured_prefix_tokens=64,
            short_native_enabled=False,
        )
        == 64
    )
    assert (
        effective_split_prefix_tokens(
            max_seqlen_k=32,
            window_size=(-1, -1),
            configured_prefix_tokens=64,
            short_native_enabled=True,
        )
        == 64
    )


if __name__ == "__main__":
    test_complete_map()
    test_incomplete_map_rejected()
    test_effective_split_prefix_tokens()
