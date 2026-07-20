#!/usr/bin/env python3
"""Unit checks for compact split-range tile mapping."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "overlay"))

from split_range_reference import allowed_key_indices, compact_tile_ids


def test_discarded_middle_tiles_are_absent() -> None:
    tiles = compact_tile_ids(
        context_len=24_000,
        query_lo=0,
        query_hi=3,
        seq_len=24_004,
        tile_size=32,
        window_tokens=4096,
        prefix_tokens=128,
    )
    assert tiles[:4] == [0, 1, 2, 3]
    assert tiles[4] == 622
    assert tiles[-1] == 750
    assert set(range(4, 622)).isdisjoint(tiles)
    assert len(tiles) == len(set(tiles))


def test_overlap_emits_each_tile_once() -> None:
    tiles = compact_tile_ids(
        context_len=48,
        query_lo=0,
        query_hi=3,
        seq_len=52,
        tile_size=32,
        window_tokens=64,
        prefix_tokens=48,
    )
    assert tiles == [0, 1]


def test_token_mask_is_causal_prefix_or_recent() -> None:
    keys = allowed_key_indices(99, window_tokens=16, prefix_tokens=8)
    assert keys == [*range(8), *range(84, 100)]
    assert 8 not in keys
    assert 83 not in keys
    assert 100 not in keys


if __name__ == "__main__":
    test_discarded_middle_tiles_are_absent()
    test_overlap_emits_each_tile_once()
    test_token_mask_is_causal_prefix_or_recent()
