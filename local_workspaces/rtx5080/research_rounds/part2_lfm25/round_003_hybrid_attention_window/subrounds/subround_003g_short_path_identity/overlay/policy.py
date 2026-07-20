"""Pure policy helpers for the R003g split-range attention overlay."""

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


def effective_split_prefix_tokens(
    *,
    max_seqlen_k: int,
    window_size: tuple[int, int],
    configured_prefix_tokens: int,
    short_native_enabled: bool,
) -> int:
    """Select native specialization when the recent window covers every key."""
    if max_seqlen_k <= 0:
        raise ValueError("max_seqlen_k must be positive")
    if len(window_size) != 2:
        raise ValueError("window_size must contain left and right bounds")
    if configured_prefix_tokens < 0:
        raise ValueError("configured_prefix_tokens must be non-negative")
    if not short_native_enabled or configured_prefix_tokens == 0:
        return configured_prefix_tokens

    left_window = int(window_size[0])
    window_tokens = left_window + 1 if left_window >= 0 else 0
    if window_tokens > 0 and max_seqlen_k <= window_tokens:
        return 0
    return configured_prefix_tokens
