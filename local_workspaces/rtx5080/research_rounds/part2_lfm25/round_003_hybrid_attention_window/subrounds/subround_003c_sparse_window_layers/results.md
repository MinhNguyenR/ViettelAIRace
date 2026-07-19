# Subround 003c Results - Sparse Windowed Layers

Status: complete; early2 is retained as the research candidate and is not a
submission artifact.

## Fixed Control

The registered unbounded profile measured 60.782 ms at turn 0 and 114.735 ms
at turn 5. The R002k image, model, precision, cache policy, graph policy and
resource envelope were unchanged. Only the global complement of the six
full-attention layers varied.

## Profile Screen

| Candidate | Windowed layers | Turn 0 ms | Turn 5 ms | Saving ms | KV tokens | Gate |
|---|---|---:|---:|---:|---:|---|
| late2 | 12,14 | 59.392 | 96.711 | 18.024 | 1,695,255 | pass |
| late3 | 10,12,14 | 58.620 | 88.628 | 26.106 | 1,908,252 | pass |
| spread3 | 2,8,14 | 59.209 | 89.431 | 25.304 | 1,908,252 | pass |
| early3 | 2,5,8 | 59.322 | 89.466 | 25.269 | 1,908,252 | pass |
| early2 | 2,5 | 59.318 | 97.011 | 17.724 | 1,695,255 | pass |
| first1 | 2 | 59.609 | 105.778 | 8.957 | 1,335,461 | pass |

All six profiles completed successfully and retained the expected computed
and scheduled token progression. Every candidate passed the opening
performance gate.

## Retrieval Screen

The fresh control contained 5 of 14 key cases. Earlier controls contained 4
of 14, so absolute counts are noisy. The paired signal that matters is lost
control-pass cases:

| Candidate | Successful | Candidate key cases | Lost control-pass | Gained cases | Gate |
|---|---:|---:|---:|---:|---|
| late2 | 14/14 | 2/14 | 3 | 0 | reject |
| late3 | 14/14 | 2/14 | 3 | 0 | reject |
| spread3 | 14/14 | 2/14 | 3 | 0 | reject |
| early3 | 14/14 | 3/14 | 2 | 0 | reject |
| early2 repeat A | 14/14 | 11/14 | 0 | 6 | pass |
| early2 repeat B | 14/14 | 10/14 | 0 | 5 | pass |
| first1 | 14/14 | 3/14 | 3 | 1 | reject |

early2 is the only placement that lost zero control-pass cases in both
repeats. The other placements are rejected before full replay; their profile
latency is mechanism evidence, not a quality-safe result.

## Full Replay

Only early2 opened the full replay gate. Both runs completed 420/420 requests
with zero failures:

| Variant | Score mean | TTFT p50/p95 ms | TPOT p50/p95 ms | At or below 400 ms |
|---|---:|---:|---:|---:|
| Unbounded control, two runs | 56.5450 | 114.26 / 315.56 | 2.571 / 4.159 | 323.5 |
| early2, two runs | 58.0922 | 105.45 / 287.42 | 2.547 / 4.094 | 326.5 |

Candidate score delta is +1.5472 points, or +2.74 percent. The candidate
minimum score is above the control maximum. The gain is mainly TTFT
(+1.5060 score points); TPOT contributes only +0.0411 points.

## Quality Proxy

The early2 GPQA proxy scored 21/100 with 11 parse failures. The repeated
control scored 22/100, while the first control scored 21/100. This is neutral
within a 21..22/100 process range. The prompts are below the 4096 horizon, so
this proxy cannot certify long-context semantic safety.

## Decision

Retain early2 as the best sparse-placement branch for further research. Do
not package or submit it. It shows that windowing layers 2 and 5 can improve
the local long-history score without the observed retrieval loss, but it does
not establish equivalence beyond 4096 tokens and does not beat the stronger
8192 branch's aggregate local score.

Scalar layer-placement permutations are closed. A continuation must preserve
selected old positions in every affected layer, use the same paired retrieval
gate, and remain separate from the R002k submission incumbent.

## Artifacts

- screen_summary.json is the canonical cross-signal summary.
- profile manifests and raw run directories are retained for audit.
- retrieval_screen.json contains the paired case-level comparison.
- full_replay_comparison.json contains score-component and per-turn deltas.
- quality_summary.json records the candidate and both control observations.
