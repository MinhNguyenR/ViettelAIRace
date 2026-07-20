"""Pure policy helpers for the R003f split-range attention overlay."""

from __future__ import annotations


FULL_LAYERS = frozenset({2, 5, 8, 10, 12, 14})


def parse_window_map(raw: str) -> dict[int, int]:
    if not raw.strip():
        raise ValueError("R003F_WINDOW_MAP is required")
    result: dict[int, int] = {}
    for item in raw.split(","):
        pieces = item.strip().split(":")
        if len(pieces) != 2:
            raise ValueError(f"invalid map item: {item!r}")
        layer, window = (int(piece.strip()) for piece in pieces)
        if layer in result or window <= 0:
            raise ValueError(f"invalid map item: {item!r}")
        result[layer] = window
    if set(result) != FULL_LAYERS:
        raise ValueError(
            f"map must cover {sorted(FULL_LAYERS)}, observed={sorted(result)}"
        )
    return result
