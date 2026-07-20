# Subround 003f Plan

Status: Phases A-E complete; branch retained as research-only evidence.

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

Answer: a single compact virtual-tile loop was smaller and safer than two
kernel calls plus a new public LSE interface. It skips the middle span before
block-table/K/V access and preserves both native 2D prefill and segmented 3D
decode. BF16, FP8 paged-KV, overlap and 3D numerical cases pass with maximum
absolute error `0.002719`.

## Phase D - Engine Integration

- Apply the primitive only to the six full-attention layers.
- Preserve native fallback for unsupported shapes and fail startup on silent
  backend fallback.
- Reuse the R002y six-turn panel before retrieval or full replay.

Answer: startup verifies six Triton full-attention layers, ten unchanged
ShortConv layers, no native prefix metadata, R002k fused greedy and CUDA
Graph. P64/W4096 is the profile incumbent at `61.607/75.290ms` for turns 0/5.
All P32/P64/P128 candidates pass the profile gate and improve paired retrieval
from `5/14` to `7/14` without losing a control-pass case.

## Phase E - Replay, Quality And Decision

- Keep native W4096 and safe8192 as registered score/quality references.
- Require two valid P64 full replays, no failed graded requests and no TPOT
  regression caused by disabling segmented decode.
- Run the 100-case short GPQA proxy only after profile and retrieval pass.
- Package only if the candidate is score-competitive, quality-equivalent and
  target-backend qualified.

Answer: forcing 2D decode scored `54.9403` and was rejected. Restoring compact
3D decode raised P64 to `63.1183`; its valid repeat scored `60.3961`. The mean
`61.7572` is effectively tied with but slightly below safe8192 (`61.8876`) and
below native W4096 (`64.5504`). GPQA proxy repeated at `20/100` versus control
`21..22/100`, failing strict equivalence. H200 SM90 remains untested. Keep the
primitive as a research Pareto point and create no image or submission.

## Gates

1. Static source audit proves the middle span is skipped before memory loads.
2. Primitive max absolute error is within the existing backend's BF16/FP8
   tolerance on short deterministic tensors.
3. Turn-0 complete GPU regression is at most 15 percent versus unbounded.
4. Turn-5 complete GPU time is below `103.261ms` (10 percent below control).
5. Only then run the 14-case retrieval gate; only zero-loss candidates open
   two full replays and the short GPQA proxy.

All dependency gates through replay were exercised. The final promotion gate
failed on aggregate score, quality equivalence and target qualification. No
image, Compose or Portal artifact opens.
