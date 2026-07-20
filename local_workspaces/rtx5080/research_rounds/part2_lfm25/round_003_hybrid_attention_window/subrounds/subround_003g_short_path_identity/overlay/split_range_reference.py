"""Pure reference helpers for the R003f causal split-range mask."""

from __future__ import annotations


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def compact_tile_ids(
    *,
    context_len: int,
    query_lo: int,
    query_hi: int,
    seq_len: int,
    tile_size: int,
    window_tokens: int,
    prefix_tokens: int,
) -> list[int]:
    """Return physical tiles visited by one compact query-block loop."""
    if not (0 <= query_lo <= query_hi and query_hi < seq_len - context_len):
        raise ValueError("invalid query range")
    if tile_size <= 0 or window_tokens <= 0 or prefix_tokens < 0:
        raise ValueError("invalid split-range dimensions")

    causal_end = min(context_len + query_hi + 1, seq_len)
    recent_lo = max(0, (context_len + query_lo - window_tokens + 1) // tile_size)
    recent_hi = min(
        (context_len + query_hi) // tile_size + 1,
        ceil_div(causal_end, tile_size),
    )
    prefix_hi = min(
        ceil_div(min(prefix_tokens, causal_end), tile_size),
        recent_lo,
    )
    return [*range(prefix_hi), *range(recent_lo, recent_hi)]


def allowed_key_indices(
    query_abs_pos: int,
    *,
    window_tokens: int,
    prefix_tokens: int,
) -> list[int]:
    """Return the exact causal prefix-or-recent key set for one query."""
    if query_abs_pos < 0 or window_tokens <= 0 or prefix_tokens < 0:
        raise ValueError("invalid split-range dimensions")
    recent_start = max(0, query_abs_pos - window_tokens + 1)
    return [
        key
        for key in range(query_abs_pos + 1)
        if key < prefix_tokens or key >= recent_start
    ]
