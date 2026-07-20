# Subround 003f Plan

Status: Phases A-B complete; Phase C in progress.

## Phase A - Exact Source Contract

- Locate vLLM 0.22.1 Triton prefill/decode kernels and the exact
  `mm_prefix_range` branch used by R003e.
- Determine whether block pruning skips the discarded middle span or only
  masks it after loading.
- Determine whether the backend exposes per-query max/log-sum-exp statistics
  required for an exact two-call merge.
- Map every required integration point for paged FP8 KV, prefix caching,
  CUDA Graph and hybrid-layer metadata.

Answer: native mm-prefix scans the full sequence and has the wrong sink
semantics. Prefill uses the 2D kernel and exposes only normalized output;
max/expsum buffers are internal to the decode-oriented 3D path. Two unmodified
prefill calls therefore cannot be merged exactly through the current wrapper.

## Phase B - Opportunity Bound

- Compute query/key block work for unbounded, recent-only, native mm-prefix
  and ideal split-range masks at the six measured turn shapes.
- Use R003e timings to bound fixed launch cost and avoid an optimistic kernel
  project with less than 10ms turn-5 opportunity.
- Reject if the ideal projection cannot beat the `114.735ms` unbounded
  control with at least 10 percent margin.

Answer: turn-5 ideal tile work is only `1.031x` recent-only versus `6.054x`
for native mm-prefix. Projected time is `75.914ms`, or `85.914ms` with a
10ms merge stress. The static gate passes and opens one primitive prototype.

## Phase C - Primitive Prototype

- Prefer two existing kernels plus exact online-softmax merge only if both
  partial calls return stable LSE values without materializing dense scores.
- Otherwise fork the smallest Triton kernel surface and prune key blocks
  before K/V loads.
- Test BF16 numerical agreement against a dense PyTorch union-mask reference,
  then test FP8 paged KV shapes used by LFM2.

## Phase D - Engine Integration

- Apply the primitive only to the six full-attention layers.
- Preserve native fallback for unsupported shapes and fail startup on silent
  backend fallback.
- Reuse the R002y six-turn panel before retrieval or full replay.

## Gates

1. Static source audit proves the middle span is skipped before memory loads.
2. Primitive max absolute error is within the existing backend's BF16/FP8
   tolerance on short deterministic tensors.
3. Turn-0 complete GPU regression is at most 15 percent versus unbounded.
4. Turn-5 complete GPU time is below `103.261ms` (10 percent below control).
5. Only then run the 14-case retrieval gate; only zero-loss candidates open
   two full replays and the short GPQA proxy.

No image, Compose or Portal artifact opens from source analysis alone.
