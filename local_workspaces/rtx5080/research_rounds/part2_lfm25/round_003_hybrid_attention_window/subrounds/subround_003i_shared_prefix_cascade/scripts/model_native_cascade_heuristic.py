#!/usr/bin/env python3
"""Reproduce vLLM 0.22.1's native FlashAttention cascade heuristic."""

from __future__ import annotations

import argparse
import math


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def evaluate(
    *,
    common_prefix_len: int,
    batch_size: int,
    query_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    num_sms: int,
) -> tuple[bool, int, int, str]:
    if common_prefix_len < 256:
        return False, 0, 0, "prefix_lt_256"
    if batch_size < 8:
        return False, 0, 0, "batch_lt_8"

    queries_per_kv = num_query_heads // num_kv_heads
    use_flash_decoding = queries_per_kv > 1 and query_len == 1
    if not use_flash_decoding:
        return True, 0, 0, "non_decode_query"

    prefix_tiles = ceil_div(common_prefix_len, 128)
    cascade_ctas = num_query_heads * ceil_div(batch_size, 128)
    cascade_time = ceil_div(cascade_ctas, num_sms) * prefix_tiles

    flash_decoding_ctas = batch_size * num_kv_heads
    flash_decoding_ctas *= ceil_div(queries_per_kv, 128)
    flash_decoding_ctas *= prefix_tiles
    flash_decoding_time = ceil_div(flash_decoding_ctas, num_sms)
    return (
        cascade_time < flash_decoding_time,
        cascade_time,
        flash_decoding_time,
        "performance_model",
    )


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-tokens", type=int, default=8192)
    parser.add_argument("--batches", type=parse_ints, default=parse_ints("1,4,8,12,16,20"))
    parser.add_argument("--num-sms", type=parse_ints, default=parse_ints("16,18,19,84,132"))
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--num-query-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    args = parser.parse_args()

    if args.prefix_tokens < 0 or args.query_len <= 0:
        parser.error("token counts must be non-negative and query-len must be positive")
    if not args.batches or min(args.batches) <= 0:
        parser.error("batches must contain positive integers")
    if not args.num_sms or min(args.num_sms) <= 0:
        parser.error("num-sms must contain positive integers")
    if args.num_query_heads % args.num_kv_heads:
        parser.error("num-query-heads must be divisible by num-kv-heads")

    print(
        "prefix_tokens,batch_size,num_sms,cascade_time,"
        "flash_decoding_time,use_cascade,reason"
    )
    for num_sms in args.num_sms:
        for batch_size in args.batches:
            decision, cascade_time, flash_time, reason = evaluate(
                common_prefix_len=args.prefix_tokens,
                batch_size=batch_size,
                query_len=args.query_len,
                num_query_heads=args.num_query_heads,
                num_kv_heads=args.num_kv_heads,
                num_sms=num_sms,
            )
            print(
                f"{args.prefix_tokens},{batch_size},{num_sms},{cascade_time},"
                f"{flash_time},{int(decision)},{reason}"
            )


if __name__ == "__main__":
    main()
