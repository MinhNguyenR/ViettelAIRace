#!/usr/bin/env python3
"""Unit checks for the R003f layer policy."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "overlay"))

from policy import FULL_LAYERS, parse_window_map


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


if __name__ == "__main__":
    test_complete_map()
    test_incomplete_map_rejected()
