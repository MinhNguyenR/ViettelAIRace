# R003f Source Audit

Status: exact-image and implemented integration contracts complete.

Questions to answer:

1. Which Triton kernel and launch wrapper consume `mm_prefix_range_tensor`?
2. Is the key loop bounded by the recent window, or by total context length?
3. Does prefix handling add a second bounded loop or a mask in the full loop?
4. Are output LSE/max statistics available to the Python/custom-op boundary?
5. Can paged FP8 KV be sliced into two logical ranges without copying pages?
6. Which code path is captured for prefill, mixed prefill/decode and decode?
7. Can a custom implementation preserve vLLM prefix-cache block tables and
   hybrid cache coordination without changing scheduler-visible state?

## Confirmed Exact-Image Behavior

Image: `localhost/part2-lfm25-r002k:fused-bf16-greedy`, vLLM `0.22.1`.

Relevant files:

- `vllm/v1/attention/ops/triton_attention_helpers.py`
- `vllm/v1/attention/ops/triton_unified_attention.py`
- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/worker/gpu_model_runner.py`

`compute_kv_seq_mask` implements:

```text
(causal AND sliding_window) OR (query_in_range AND key_in_range)
```

This is multimodal/document bidirectionality, not persistent prefix-key
visibility. It also allows future keys inside the range, unlike the causal
sink mask required by R003f.

`compute_tile_loop_bounds` expands `max_seq_prefix_len` to `seq_len` whenever
`USE_MM_PREFIX` is true. Sliding-window tile pruning is guarded by
`SLIDING_WINDOW > 0 and not USE_MM_PREFIX`. The kernel consequently loads K/V
tiles across the full sequence and only then applies the range/window mask.

The wrapper selects the 2D kernel whenever `max_seqlen_q > 1`, which covers
the measured prefill panels. That path normalizes `acc / L` and stores only
the final output. Segment output/max/expsum buffers and `reduce_segments` are
used only by the 3D path selected for small-query decode. Therefore two calls
to the existing prefill wrapper cannot be merged exactly: their LSE state is
not exposed.

The exact online-softmax merge formula already exists in `reduce_segments`,
so the math can be reused. The smallest viable implementation is likely a
forked prefill kernel that either runs two pruned loops in one program or
stores two partial `(acc, max, expsum)` segments and invokes the existing
reduction pattern.

FP8 paged KV is handled inside the unified kernel through `KV_QUANT_MODE`; a
fork can preserve block-table addressing and per-tensor scales without
copying KV pages. Prefix caching and hybrid cache ownership can remain
scheduler-visible and unchanged if only the attention read loop changes.

## Implemented Contract

The fork preserves the exact native wrapper signature and adds one internal
`split_prefix_tokens` argument. For a non-zero prefix it computes a compact
virtual tile count:

```text
prefix_tiles_before_recent + recent_tiles
```

Each virtual tile maps either to the prefix span or the recent span before
the physical block-table entry and K/V cache pointers are loaded. Prefix
tiles overlapping the recent span are emitted once. The per-token causal mask
is exactly:

```text
causal AND (key_position < prefix_tokens OR key_in_recent_window)
```

The 2D prefill path retains one online-softmax accumulator. The 3D decode path
partitions the compact virtual list across native segments; `reduce_segments`
recomputes the same virtual-tile count and reduces only active segments.
Paged BF16/FP8 KV modes, tensor scales, sinks, ALiBi, QQ bias, output scaling,
block tables and CUDA Graph call shapes remain wired as in vLLM 0.22.1.

The model overlay forces this backend only for full-attention layers
`[2,5,8,10,12,14]`. It does not alter the ShortConv state manager, prefix
cache ownership, scheduler, request metadata or tokenizer. Native
`mm_prefix_range` is rejected to prevent combining incompatible semantics.

## Remaining Portability Boundary

The local image runs CUDA 13 on RTX 5080 SM120. Numerical and service tests do
not prove compilation, register pressure or speed on BTC's CUDA 12/H200 SM90
path. The source fork is therefore a local research implementation until an
SM90 build/run gate is available.
