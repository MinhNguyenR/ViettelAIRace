# R002z Results - H200 FA3 Exactness Boundary

Status: complete; no candidate promoted.

## Exact-Image Findings

| Mechanism | Exact result | Decision |
|---|---|---|
| Automatic backend | SM90 selects FA3 with FP8 KV | Keep automatic |
| FA3 AOT schedule | Enabled automatically on FA3 | Already active |
| Long-prefill splits | `num_splits=0`, backend heuristic | Do not override blind |
| Graph split cap | Applies only inside captured token range | Irrelevant to 4k prefill |
| KV block size | Default `16`; FA3 requires a multiple of 16 | Already optimal contract |
| Prefill/decode split | Config declaration has no runtime consumer | Unusable |
| Batch invariance | Disables AOT/forces split behavior and affects decode | Reject |
| TRT-LLM attention | Requires SM100 | Reject on H200 |

## Decision

No exact-image scalar can isolate and improve the H200 long-prefill path with a
local or source-backed gain. The registered implementation gate fails. R002k
remains immutable and R002z consumes no GPU or Portal budget.
