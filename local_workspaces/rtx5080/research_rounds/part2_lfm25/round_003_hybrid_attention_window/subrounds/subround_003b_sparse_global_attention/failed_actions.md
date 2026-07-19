# Subround 003b Failed Actions

## Selective Global Placements Failed The Quality Gate

Date: 2026-07-19

Action:

- Kept layer `14`, `2`, `8`, or the pair `2,14` globally causal.
- Applied a `4096` window to every other full-attention layer.

Expected:

- Recover the two long-context retrieval cases lost by the all-layer `4096`
  candidate while retaining at least 30 ms of turn-5 GPU savings.

Observed:

- All four profiles passed the performance gate and completed all requests.
- Every retrieval candidate answered only `2/14` key cases versus `4/14` for
  the unbounded control.
- Every candidate lost the same `5084`- and `7109`-token control-pass cases;
  no candidate gained a case.

Root cause / best hypothesis:

- A later global layer cannot reconstruct information that earlier windowed
  layers have already discarded. The quality loss is distributed across the
  bounded stack rather than localized to one layer.

Decision:

- Reject all placements and skip full replay, image creation and Compose work.

Do not repeat unless:

- The retention rule changes to preserve old information in every affected
  layer, for example a validated sink/prefix token set or another
  semantics-preserving sparse pattern.

Related files:

```text
README.md
results.md
analysis/profile_screen.json
analysis/retrieval_screen.json
```

## Harness Compatibility Note

The first compact profile summary was accidentally written to a legacy
`round_003b_sparse_global_attention/` path, and the analyzer initially assumed
the old control manifest had `global_layers.txt`. The analyzer now treats a
missing control marker as `unbounded-control`; the canonical summary is under
this subround's `analysis/` directory. This was a bookkeeping issue, not a
benchmark result, and the misplaced generated JSON is intentionally excluded
from the commit.
