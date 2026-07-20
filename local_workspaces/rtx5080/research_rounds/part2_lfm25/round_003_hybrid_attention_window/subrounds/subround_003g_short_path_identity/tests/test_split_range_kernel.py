#!/usr/bin/env python3
"""GPU numerical and native-identity checks for the R003g attention path."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "overlay"))

from split_range_unified_attention import unified_attention as copied_attention
from vllm.v1.attention.ops.triton_unified_attention import (
    unified_attention as native_attention,
)
from vllm.v1.kv_cache_interface import KVQuantMode


def dense_reference(
    query: torch.Tensor,
    logical_key: torch.Tensor,
    logical_value: torch.Tensor,
    *,
    context_len: int,
    window_tokens: int,
    prefix_tokens: int,
    scale: float,
) -> torch.Tensor:
    query_f = query.float()
    key_f = logical_key.float()
    value_f = logical_value.float()
    num_query_heads = query.shape[1]
    num_kv_heads = logical_key.shape[1]
    queries_per_kv = num_query_heads // num_kv_heads
    output = torch.empty_like(query_f)

    for query_idx in range(query.shape[0]):
        query_abs = context_len + query_idx
        positions = torch.arange(logical_key.shape[0], device=query.device)
        allowed = (positions <= query_abs) & (
            (positions < prefix_tokens)
            | ((query_abs - positions) < window_tokens)
        )
        for query_head in range(num_query_heads):
            kv_head = query_head // queries_per_kv
            scores = (
                query_f[query_idx, query_head]
                @ key_f[:, kv_head].transpose(0, 1)
            ) * scale
            probs = torch.softmax(scores.masked_fill(~allowed, -torch.inf), dim=0)
            output[query_idx, query_head] = probs @ value_f[:, kv_head]
    return output


def pack_paged(
    logical: torch.Tensor,
    *,
    block_size: int,
    physical_order: torch.Tensor,
) -> torch.Tensor:
    seq_len, num_heads, head_size = logical.shape
    num_blocks = math.ceil(seq_len / block_size)
    paged = torch.zeros(
        (num_blocks, block_size, num_heads, head_size),
        dtype=logical.dtype,
        device=logical.device,
    )
    for logical_block in range(num_blocks):
        start = logical_block * block_size
        stop = min(start + block_size, seq_len)
        physical_block = int(physical_order[logical_block])
        paged[physical_block, : stop - start] = logical[start:stop]
    return paged


def run_case(
    *,
    dtype_name: str,
    seq_len: int,
    query_len: int,
    window_tokens: int,
    prefix_tokens: int,
    use_3d: bool = False,
    compare_native: bool = False,
) -> dict[str, bool | float | int | str]:
    torch.manual_seed(1701 + seq_len + prefix_tokens)
    device = torch.device("cuda")
    num_query_heads = 8
    num_kv_heads = 2
    head_size = 64
    block_size = 16
    context_len = seq_len - query_len
    scale = head_size**-0.5

    query = torch.randn(
        query_len,
        num_query_heads,
        head_size,
        device=device,
        dtype=torch.bfloat16,
    )
    logical_key_bf16 = torch.randn(
        seq_len,
        num_kv_heads,
        head_size,
        device=device,
        dtype=torch.bfloat16,
    )
    logical_value_bf16 = torch.randn_like(logical_key_bf16)

    if dtype_name == "bf16":
        logical_key = logical_key_bf16
        logical_value = logical_value_bf16
        kv_quant_mode = KVQuantMode.NONE
        atol = 0.035
    elif dtype_name == "fp8":
        logical_key = logical_key_bf16.to(torch.float8_e4m3fn)
        logical_value = logical_value_bf16.to(torch.float8_e4m3fn)
        kv_quant_mode = KVQuantMode.FP8_PER_TENSOR
        atol = 0.08
    else:
        raise ValueError(dtype_name)

    num_blocks = math.ceil(seq_len / block_size)
    physical_order = torch.randperm(num_blocks, device=device, dtype=torch.int64)
    key_cache = pack_paged(
        logical_key, block_size=block_size, physical_order=physical_order
    )
    value_cache = pack_paged(
        logical_value, block_size=block_size, physical_order=physical_order
    )
    block_table = physical_order.to(torch.int32).unsqueeze(0)
    cu_seqlens_q = torch.tensor([0, query_len], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([seq_len], device=device, dtype=torch.int32)
    one = torch.ones(1, device=device, dtype=torch.float32)
    output = torch.empty_like(query)
    segment_count = 4
    if use_3d:
        segment_output = torch.empty(
            (8, num_query_heads, segment_count, head_size),
            device=device,
            dtype=torch.float32,
        )
        segment_max = torch.empty(
            (8, num_query_heads, segment_count), device=device, dtype=torch.float32
        )
        segment_expsum = torch.empty_like(segment_max)
    else:
        segment_output = segment_max = segment_expsum = None

    common_kwargs = dict(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=query_len,
        seqused_k=seqused_k,
        max_seqlen_k=seq_len,
        softmax_scale=scale,
        causal=True,
        window_size=(window_tokens - 1, 0),
        block_table=block_table,
        softcap=0.0,
        q_descale=one,
        k_descale=one,
        v_descale=one,
        seq_threshold_3D=8 if use_3d else None,
        num_par_softmax_segments=segment_count if use_3d else None,
        softmax_segm_output=segment_output,
        softmax_segm_max=segment_max,
        softmax_segm_expsum=segment_expsum,
        kv_quant_mode=kv_quant_mode,
    )

    native_output = torch.empty_like(query)
    copied_zero_output = torch.empty_like(query)
    if compare_native:
        if window_tokens < seq_len:
            raise ValueError("native comparison requires a full-overlap window")
        native_attention(out=native_output, **common_kwargs)
        copied_attention(
            out=copied_zero_output,
            split_prefix_tokens=0,
            **common_kwargs,
        )
    copied_attention(
        out=output,
        split_prefix_tokens=prefix_tokens,
        **common_kwargs,
    )
    torch.cuda.synchronize()

    reference = dense_reference(
        query,
        logical_key,
        logical_value,
        context_len=context_len,
        window_tokens=window_tokens,
        prefix_tokens=prefix_tokens,
        scale=scale,
    )
    error = (output.float() - reference).abs()
    max_abs = float(error.max().item())
    mean_abs = float(error.mean().item())
    if not torch.isfinite(output).all():
        raise AssertionError(f"{dtype_name}: non-finite kernel output")
    if max_abs > atol:
        raise AssertionError(
            f"{dtype_name}: max_abs={max_abs:.6f} exceeds atol={atol:.6f}"
        )
    result: dict[str, bool | float | int | str] = {
        "dtype": dtype_name,
        "seq_len": seq_len,
        "query_len": query_len,
        "window_tokens": window_tokens,
        "prefix_tokens": prefix_tokens,
        "use_3d": use_3d,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "atol": atol,
    }
    if compare_native:
        native_zero_error = (native_output.float() - copied_zero_output.float()).abs()
        native_split_error = (native_output.float() - output.float()).abs()
        native_zero_exact = bool(torch.equal(native_output, copied_zero_output))
        result.update(
            {
                "native_zero_exact": native_zero_exact,
                "native_zero_max_abs": float(native_zero_error.max().item()),
                "native_split_exact": bool(torch.equal(native_output, output)),
                "native_split_max_abs": float(native_split_error.max().item()),
            }
        )
        if not native_zero_exact:
            raise AssertionError(
                f"{dtype_name}: copied split=0 path is not bit-identical to native"
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = [
        run_case(
            dtype_name=dtype_name,
            seq_len=257,
            query_len=17,
            window_tokens=64,
            prefix_tokens=16,
        )
        for dtype_name in ("bf16", "fp8")
    ]
    cases.append(
        run_case(
            dtype_name="bf16",
            seq_len=52,
            query_len=4,
            window_tokens=64,
            prefix_tokens=48,
            compare_native=True,
        )
    )
    cases.append(
        run_case(
            dtype_name="fp8",
            seq_len=257,
            query_len=17,
            window_tokens=512,
            prefix_tokens=64,
            compare_native=True,
        )
    )
    cases.append(
        run_case(
            dtype_name="bf16",
            seq_len=257,
            query_len=1,
            window_tokens=512,
            prefix_tokens=64,
            use_3d=True,
            compare_native=True,
        )
    )
    cases.append(
        run_case(
            dtype_name="bf16",
            seq_len=257,
            query_len=1,
            window_tokens=64,
            prefix_tokens=16,
            use_3d=True,
        )
    )
    result = {
        "schema": "r003g.native_identity.v1",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "cases": cases,
        "decision": "pass",
    }
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
