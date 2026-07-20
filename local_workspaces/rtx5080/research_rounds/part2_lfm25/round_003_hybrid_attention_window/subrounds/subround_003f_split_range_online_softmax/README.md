# Subround 003f - Split-Range Online-Softmax Attention

Status: source and opportunity audit in progress.

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
