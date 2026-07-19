# Subround 003b - Selective Global Attention

Status: complete local research; rejected for full replay and promotion.

## Question

R003a showed that an all-layer `4096` causal window is the local score maximum,
but it loses two paired long-context retrieval cases. This subround tests
whether keeping one or two full-attention layers globally causal can recover
those cases while retaining most of the latency gain.

## Fixed Contract

- Exact `localhost/part2-lfm25-r002k:fused-bf16-greedy` image and model mount.
- vLLM `0.22.1`, RTX 5080, 3 CPU cores, 8 GiB RAM, GPU utilization `0.80`.
- Full-attention layers are `[2, 5, 8, 10, 12, 14]`.
- Ten ShortConv layers `[0, 1, 3, 4, 6, 7, 9, 11, 13, 15]` are untouched.
- Every non-global full-attention layer uses `per_layer_sliding_window=4096`.
- R002k precision, prefix cache, b16384, scheduler, CUDA Graph policy and
  fused greedy head are unchanged.
- The overlay is trace-independent and fail-fast: an invalid layer or prefix
  topology aborts startup rather than silently falling back.

## Candidates

| Label | Global full-attention layers | Windowed full-attention layers |
|---|---|---|
| `global14` | `[14]` | `[2, 5, 8, 10, 12]` |
| `global2` | `[2]` | `[5, 8, 10, 12, 14]` |
| `global8` | `[8]` | `[2, 5, 10, 12, 14]` |
| `global2_14` | `[2, 14]` | `[5, 8, 10, 12]` |

## Measurement

Each candidate received one isolated six-turn profile and the same calibrated
14-case paired retrieval probe. The profile gate required turn-0 regression at
most 5%, turn-5 saving at least 30 ms, exact cache progression and zero
failures. The retrieval gate required no lost control-pass case.

All four profiles passed the performance gate and all retrieval requests
completed. Every candidate nevertheless lost the same two control-pass cases,
at actual prompt lengths 5084 and 7109 tokens. No candidate gained a case.

## Decision

Selective placement does not repair the quality boundary created by a 4096
window in the other full-attention layers. The strongest profile is `global8`
(`60.453 -> 76.802 ms`, `37.932 ms` turn-5 saving), but it still loses `2/14`
retrieval cases. `global2_14` consumes more cache and is slower without a
quality benefit.

Do not open a 420-request replay, build an image or create a Compose from this
branch. Keep R002k as the submission incumbent, R003a `4096` as a score upper
bound only and R003 `8192` as the balanced research control. The next useful
attention study must preserve information selectively, rather than merely
moving a global layer.

## Reproduction

- `overlay/selective_global_attention.py`: exact layer routing and assertions.
- `tests/test_selective_global_attention.py`: static topology tests.
- `scripts/run_profile.sh`: six-turn GPU profile.
- `scripts/run_retrieval.sh`: calibrated retrieval screen.
- `scripts/analyze_sparse_screen.py`: compact profile/retrieval summary.
- `analysis/profile_screen.json`: measured profile frontier.
- `analysis/retrieval_screen.json`: paired retrieval outcomes.
