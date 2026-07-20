# R003e Results - Prefix Plus Recent Triton Retention

Status: complete and rejected at the profile gate.

## Validity

Every canonical run used the exact R002k image, RTX 5080, 3 CPU cores, 8 GiB
RAM and GPU memory utilization `0.80`. Startup proved all six full-attention
layers used `TRITON_ATTN`, all ten ShortConv layers were unchanged, CUDA Graph
capture completed and the requested prefix metadata state matched the label.
All six isolated turn panels were valid.

## Profile Result

The registered unbounded control is `60.781921ms` at turn 0 and
`114.734528ms` at turn 5.

| Candidate | Prefix | Turn 0 GPU ms | Turn 5 GPU ms | Turn 5 saving vs control | Decision |
|---|---:|---:|---:|---:|---|
| `p0w4096` | 0 | 62.364 | 75.223 | +39.512 ms | diagnostic only |
| `p64w4096` | 64 | 69.901 | 187.100 | -72.365 ms | reject |
| `p128w4096` | 128 | 70.304 | 187.862 | -73.128 ms | reject |
| `p256w4096` | 256 | 70.224 | 187.059 | -72.324 ms | reject |

The no-prefix Triton/window control is fast, so explicit Triton selection is
not the dominant regression. Any non-empty prefix range adds about `112ms` at
turn 5 versus `p0`, and changing the retained prefix from 64 to 256 tokens has
almost no effect. The evidence therefore isolates the regression to the
non-empty disjoint prefix-mask execution path, not prefix size.

Exact post-profile source audit confirms the mechanism. When
`USE_MM_PREFIX=True`, `compute_tile_loop_bounds` expands every query block to
the full sequence and disables sliding-window tile pruning. K/V tiles in the
discarded middle are loaded before the per-token mask is applied.

The same audit also corrects the semantic interpretation: native
`mm_prefix_range` uses `query_in_range AND key_in_range`. Later queries outside
the range cannot attend to prefix keys. R003e therefore did not implement a
persistent prefix plus recent window; it screened and rejected the wrong
native API before any quality claim was made.

## Decision

All semantic candidates fail the required turn-5 saving gate by more than
`72ms`, so retrieval, full replay, GPQA, image and Compose work are skipped.
`p0w4096` is not a prefix-retention candidate and inherits the known semantic
risk of a plain 4096 window. The non-empty variants also fail the intended
retention contract, independently of their latency failure.

Do not retry scalar prefix sizes on this native path. A future branch must
avoid the full-span disjoint-mask cost, for example by computing prefix and
recent ranges separately and merging their online-softmax statistics inside
a target-qualified kernel.
