#!/usr/bin/env python3
"""Probe FlashInfer's native multi-level cascade on page-aligned decode data."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import flashinfer


def _pack_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    block_size: int,
) -> tuple[torch.Tensor, list[list[int]]]:
    batch, seq_len, num_kv_heads, head_dim = key.shape
    pages_per_request = math.ceil(seq_len / block_size)
    total_pages = batch * pages_per_request
    cache = torch.empty(
        (total_pages, 2, block_size, num_kv_heads, head_dim),
        device=key.device,
        dtype=key.dtype,
    )
    page_map: list[list[int]] = []
    for request in range(batch):
        pages: list[int] = []
        for page in range(pages_per_request):
            physical = request * pages_per_request + page
            start = page * block_size
            stop = min(start + block_size, seq_len)
            cache[physical, 0].zero_()
            cache[physical, 1].zero_()
            cache[physical, 0, : stop - start] = key[request, start:stop]
            cache[physical, 1, : stop - start] = value[request, start:stop]
            pages.append(physical)
        page_map.append(pages)
    return cache, page_map


def _flat_tables(
    page_map: list[list[int]],
    ranges: list[tuple[int, int]],
    *,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices: list[int] = []
    indptr = [0]
    last_lengths: list[int] = []
    for pages, (start_page, end_page) in zip(page_map, ranges, strict=True):
        selected = pages[start_page:end_page]
        indices.extend(selected)
        indptr.append(len(indices))
        last_lengths.append(block_size)
    return (
        torch.tensor(indptr, dtype=torch.int32, device=device),
        torch.tensor(indices, dtype=torch.int32, device=device),
        torch.tensor(last_lengths, dtype=torch.int32, device=device),
    )


def _dense_selected(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    prefix_tokens: int,
    recent_tokens: int,
    scale: float,
) -> torch.Tensor:
    # This probe is decode-only, so each row sees the complete selected set.
    key_selected = torch.cat(
        [key[:, :prefix_tokens], key[:, -recent_tokens:]], dim=1
    ).float()
    value_selected = torch.cat(
        [value[:, :prefix_tokens], value[:, -recent_tokens:]], dim=1
    ).float()
    batch, num_heads, head_dim = query.shape
    num_kv_heads = key.shape[2]
    queries_per_kv = num_heads // num_kv_heads
    query_f = query.float()
    output = torch.empty_like(query_f)
    for request in range(batch):
        for head in range(num_heads):
            kv_head = head // queries_per_kv
            scores = query_f[request, head] @ key_selected[request, :, kv_head].T
            probs = torch.softmax(scores * scale, dim=-1)
            output[request, head] = probs @ value_selected[request, :, kv_head]
    return output


def _measure(fn, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / repeats


def run_case(
    dtype_name: str,
    *,
    batch: int,
    seq_len: int,
    prefix_tokens: int,
    recent_tokens: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    torch.manual_seed(7300 + (0 if dtype_name == "bf16" else 1))
    device = torch.device("cuda")
    block_size = 16
    num_heads = 8
    num_kv_heads = 2
    head_dim = 64
    scale = head_dim**-0.5

    query = torch.randn(
        batch, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    key_bf16 = torch.randn(
        batch, seq_len, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    value_bf16 = torch.randn_like(key_bf16)
    if dtype_name == "bf16":
        key = key_bf16
        value = value_bf16
        kv_dtype: str | torch.dtype = torch.bfloat16
    elif dtype_name == "fp8":
        key = key_bf16.to(torch.float8_e4m3fn)
        value = value_bf16.to(torch.float8_e4m3fn)
        kv_dtype = torch.float8_e4m3fn
    else:
        raise ValueError(dtype_name)

    cache, page_map = _pack_cache(key, value, block_size=block_size)
    pages_per_request = seq_len // block_size
    prefix_pages = prefix_tokens // block_size
    recent_pages = recent_tokens // block_size
    full_indptr, full_indices, full_last = _flat_tables(
        page_map,
        [(0, pages_per_request)] * batch,
        block_size=block_size,
        device=device,
    )
    prefix_tables = _flat_tables(
        page_map,
        [(0, prefix_pages)] * batch,
        block_size=block_size,
        device=device,
    )
    recent_tables = _flat_tables(
        page_map,
        [(pages_per_request - recent_pages, pages_per_request)] * batch,
        block_size=block_size,
        device=device,
    )
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    full = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    full.plan(
        full_indptr,
        full_indices,
        full_last,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        q_data_type=torch.bfloat16,
        kv_data_type=kv_dtype,
        sm_scale=scale,
        window_left=-1,
    )
    recent = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    recent.plan(
        recent_tables[0],
        recent_tables[1],
        recent_tables[2],
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        q_data_type=torch.bfloat16,
        kv_data_type=kv_dtype,
        sm_scale=scale,
        window_left=-1,
    )
    cascade = flashinfer.MultiLevelCascadeAttentionWrapper(2, workspace, "NHD")
    cascade.plan(
        [qo_indptr, qo_indptr],
        [prefix_tables[0], recent_tables[0]],
        [prefix_tables[1], recent_tables[1]],
        [prefix_tables[2], recent_tables[2]],
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        causal=True,
        sm_scale=scale,
        window_left=-1,
        q_data_type=torch.bfloat16,
        kv_data_type=kv_dtype,
    )

    cascade_out = cascade.run(query, cache)
    torch.cuda.synchronize()
    dense = _dense_selected(
        query,
        key,
        value,
        prefix_tokens=prefix_tokens,
        recent_tokens=recent_tokens,
        scale=scale,
    )
    error = (cascade_out.float() - dense).abs()
    full_out = full.run(query, cache)
    recent_out = recent.run(query, cache)
    torch.cuda.synchronize()
    cascade_ms = _measure(lambda: cascade.run(query, cache), warmup=warmup, repeats=repeats)
    full_ms = _measure(lambda: full.run(query, cache), warmup=warmup, repeats=repeats)
    recent_ms = _measure(lambda: recent.run(query, cache), warmup=warmup, repeats=repeats)
    return {
        "dtype": dtype_name,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "flashinfer": getattr(flashinfer, "__version__", "unknown"),
        "batch": batch,
        "seq_len": seq_len,
        "prefix_tokens": prefix_tokens,
        "recent_tokens": recent_tokens,
        "block_size": block_size,
        "cascade_max_abs": float(error.max().item()),
        "cascade_mean_abs": float(error.mean().item()),
        "full_vs_dense_selected_max_abs": float(
            (full_out.float() - dense).abs().max().item()
        ),
        "recent_vs_dense_selected_max_abs": float(
            (recent_out.float() - dense).abs().max().item()
        ),
        "cascade_ms": cascade_ms,
        "full_ms": full_ms,
        "recent_ms": recent_ms,
        "cascade_speedup_vs_full": full_ms / cascade_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--recent-tokens", type=int, default=4096)
    parser.add_argument(
        "--dtype", choices=("bf16", "fp8", "both"), default="both"
    )
    args = parser.parse_args()
    if any(
        value % 16
        for value in (args.seq_len, args.prefix_tokens, args.recent_tokens)
    ):
        parser.error(
            "seq-len, prefix-tokens and recent-tokens must be multiples of 16"
        )
    dtype_names = ("bf16", "fp8") if args.dtype == "both" else (args.dtype,)
    results: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for dtype_name in dtype_names:
        try:
            results.append(
                run_case(
                    dtype_name,
                    batch=args.batch,
                    seq_len=args.seq_len,
                    prefix_tokens=args.prefix_tokens,
                    recent_tokens=args.recent_tokens,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            )
        except Exception as exc:  # Record unsupported dtype/API as a result.
            errors.append(
                {
                    "dtype": dtype_name,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    payload = {
        "schema": "r003h.flashinfer_cascade_probe.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
        "errors": errors,
        "decision": "pending",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
