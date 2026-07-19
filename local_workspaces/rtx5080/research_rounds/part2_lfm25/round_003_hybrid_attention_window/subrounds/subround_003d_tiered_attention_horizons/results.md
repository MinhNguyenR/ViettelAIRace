# Subround 003d Results - Tiered Attention Horizons

Status: complete. `safe8192` is the retained local research candidate; no
submission package was created.

## Profile Screen

| Candidate | Turn 0 ms | Turn 5 ms | Saving ms | KV tokens | Gate |
|---|---:|---:|---:|---:|---|
| safe8192 | 59.407 | 73.928 | 40.807 | larger than control | pass |
| mid6144 | 58.821 | 71.625 | 43.110 | larger than control | pass |
| front3 | 59.552 | 72.716 | 42.018 | larger than control | pass |

## Retrieval Screen

| Candidate | Successful | Key cases | Lost control-pass | Gained | Gate |
|---|---:|---:|---:|---:|---|
| safe8192 | 14/14 | 6/14 | 0 | 1 | pass |
| mid6144 | 14/14 | 4/14 | 1 | 0 | reject |
| front3 | 14/14 | 3/14 | 2 | 0 | reject |

## Full Replay

Only `safe8192` passed the profile and retrieval gates. Two 420-request runs
were compared with two unbounded controls under the same local envelope:

| Metric | Control mean | Candidate mean | Delta |
|---|---:|---:|---:|
| Score points | 56.544982 | 61.887592 | +5.342610 |
| TTFT p50 | 114.261750 ms | 90.966193 ms | -23.295557 ms |
| TTFT p95 | 315.559762 ms | 245.762747 ms | -69.797014 ms |
| TPOT p50 | 2.571442 ms | 2.408539 ms | -0.162903 ms |
| TPOT p95 | 4.159323 ms | 3.612358 ms | -0.546965 ms |
| Graded failures | 0 | 0 | 0 |
| Graded requests <=400 ms | 323.5/330 | 329.5/330 | +6 |

Candidate scores were `61.873738` and `61.901446`; the repeat range was
`0.027708` points. The machine-readable comparison is
`analysis/full_replay_comparison.json`.

## Quality Screen

The short GPQA proxy returned `23/100`, `89/100` parsed and `11` parse
failures. Existing controls returned `21/100` and `22/100` with 11 and 10
parse failures. Since the proxy input is below 8192 tokens, this only rules out
an obvious short-prompt regression. It does not validate long-context
semantics. The compact quality record is `analysis/quality_summary.json`.

## Decision

Retain `safe8192` as the best balanced local branch. Close `mid6144` and
`front3` because they lose control-pass retrieval cases. Do not promote the
map to a Portal image or Compose: it changes effective model semantics beyond
the horizon and its score includes increased KV capacity at the same memory
fraction. The next experiment must test a semantics-preserving retention
mechanism or a target-qualified attention implementation.
