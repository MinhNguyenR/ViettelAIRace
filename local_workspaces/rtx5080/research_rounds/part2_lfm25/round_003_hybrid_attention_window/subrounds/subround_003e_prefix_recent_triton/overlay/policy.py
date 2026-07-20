"""Pure policy helpers for the R003e prefix-plus-recent overlay."""

from __future__ import annotations

FULL_LAYERS = frozenset({2, 5, 8, 10, 12, 14})


def parse_window_map(raw: str) -> dict[int, int]:
    if not raw.strip():
        raise ValueError("R003E_WINDOW_MAP is required")
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


def prefix_ranges(num_requests: int, prefix_tokens: int) -> dict[int, list[tuple[int, int]]]:
    if num_requests < 0 or prefix_tokens <= 0:
        raise ValueError("num_requests must be non-negative and prefix_tokens positive")
    end = prefix_tokens - 1
    return {index: [(0, end)] for index in range(num_requests)}
