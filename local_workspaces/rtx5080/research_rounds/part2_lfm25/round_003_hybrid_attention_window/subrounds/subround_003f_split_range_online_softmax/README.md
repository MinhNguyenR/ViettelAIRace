# Subround 003f - Split-Range Online-Softmax Attention

Status: complete local primitive and engine screen; retained as a research
Pareto point, not promoted or packaged.

## Question

Can prefix-plus-recent attention preserve the first request-local tokens
without paying the full-span cost measured in R003e?

For each query, the intended key set is:

```text
[0, min(P, q+1)) UNION [max(0, q-W+1), q+1)
```

The discarded middle span must not be loaded or masked. Prefix and recent
partial attention results must share one exact softmax normalization, either
inside one sparse kernel or through an online-softmax merge of two kernels.

Native `mm_prefix_range` cannot implement this mask. It requires both query
and key inside the range and disables sliding-window tile pruning, which is
the exact behavior rejected in R003e.

## Constraints

- One model, one GPU and the existing hybrid KV/Mamba state pool.
- Preserve FP8 E4M3 KV, R002k fused greedy decode head, scheduler and CUDA
  Graph policy unless a measured dependency requires a change.
- No prompt-content routing, trace identity, answer cache or model-output
  modification.
- RTX 5080 evidence is local only; H200 SM90 requires a separate backend gate.

## References

- R003e no-prefix Triton/window turn-5 floor: `75.222595ms`.
- Registered unbounded turn-5 control: `114.734528ms`.
- R003e native non-empty prefix turn-5 range: `187.058655..187.862274ms`.
- R003d safe8192 remains the long-context quality-oriented comparison branch.

## Static Opportunity

At turn 5, the exact launch-shape model counts `822,394` native mm-prefix K/V
tile visits per KV-head/layer versus `135,837` recent-only and `140,049` for
an ideal causal prefix128+recent union. The ideal mask adds only `3.10%` tile
work over recent-only instead of the native path's `6.05x` total ratio.

Linear interpolation from the matched R003e measurements projects
`75.914ms`. A deliberately large `10ms` merge stress projects `85.914ms`,
below the registered `103.261ms` opening gate. Turn 0 has complete range
overlap and adds no prefix loop. This is an opportunity bound, not a measured
kernel result; it opens Phase C only.

## Implemented Path

The exact vLLM 0.22.1 Triton unified-attention source is retained under its
Apache-2.0 header and patched with a compact virtual key-tile list. Prefix
tiles and recent-window tiles are de-duplicated, then mapped to physical
paged-KV positions before the first block-table or K/V load. Prefill keeps the
native 2D launch shape; decode keeps the native segmented 3D path and reduces
online-softmax `(max, expsum, acc)` state exactly.

The overlay applies only to full-attention layers `[2,5,8,10,12,14]`, keeps
all ten ShortConv/Mamba layers and R002k fused greedy logic, rejects native
`mm_prefix_range`, and fails startup if the expected Triton topology is not
installed.

## Result Summary

- BF16, FP8 paged-KV, overlap and segmented-3D numerical cases pass; maximum
  absolute error is `0.002719`.
- P64/W4096 reaches `61.607/75.290ms` at turns 0/5, only `0.068ms` above the
  recent-only turn-5 floor and `39.444ms` below unbounded.
- P32/P64/P128 all preserve the 5 control retrieval successes and improve
  the probe from `5/14` to `7/14`.
- Two valid P64 replays score `63.1183` and `60.3961`, mean `61.7572`. This is
  `0.1304` below safe8192 and `2.7932` below native W4096.
- P64 GPQA proxy repeats at `20/100`; controls are `21/100` and `22/100`.
  Since all proxy prompts are below 847 tokens, this is a numerical or
  implementation-risk signal rather than a long-context mask effect.

The branch proves that exact split-range pruning is technically viable and
eliminates R003e's full-span cost. It does not beat the existing score branch
and fails strict local quality equivalence. H200 SM90 compilation and
performance are also unverified, so no image, Compose or Portal artifact is
created.

## Files

- `results.md`: measured primitive, profile, retrieval, replay and quality
  evidence.
- `round_plan.md`: completed phase answers and final decision.
- `source_audit.md`: exact vLLM source and integration contract.
- `failed_actions.md`: invalid harness runs and rejected designs.
- `analysis/*.json`: compact registered evidence; raw logs remain ignored.
- `overlay/`: kill-switchable split-range Triton implementation.
- `tests/` and `scripts/`: numerical, profile, retrieval, replay and quality
  reproduction paths.
