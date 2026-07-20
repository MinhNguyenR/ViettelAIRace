# R003f Results - Split-Range Online-Softmax Attention

Status: complete local research branch; promotion rejected.

## Fixed Setup

- Base image: `localhost/part2-lfm25-r002k:fused-bf16-greedy`, vLLM 0.22.1.
- Local envelope: RTX 5080, 3 CPU cores, 8 GiB RAM, memory fraction 0.80.
- Model policy: FP8 per-tensor weights, FP8 E4M3 attention KV, BF16 aligned
  ShortConv state, prefix caching, b16384, max sequences 8, async scheduling,
  FULL_AND_PIECEWISE CUDA Graph and R002k fused greedy head.
- Split-range policy: recent window 4096 on all six full-attention layers and
  a fixed request-local prefix of 32, 64 or 128 tokens.

## Primitive Correctness

The dense reference and paged Triton primitive agree on four registered GPU
cases:

| KV/path | sequence/query | window/prefix | max abs error |
|---|---:|---:|---:|
| BF16 2D | 257 / 17 | 64 / 16 | `0.002125` |
| FP8 2D | 257 / 17 | 64 / 16 | `0.002473` |
| BF16 overlap | 52 / 4 | 64 / 48 | `0.002719` |
| BF16 segmented 3D | 257 / 1 | 64 / 16 | `0.002084` |

All are finite and below their registered tolerances. Static reference tests
also prove exact causal union membership and de-duplication of overlapping
prefix/recent tiles.

## Isolated Profile

Registered references are `60.782/114.735ms` for unbounded turns 0/5 and
`62.364/75.223ms` for native recent-only W4096.

| prefix | turn 0 GPU ms | turn 5 GPU ms | turn 0 vs unbounded | turn 5 vs recent |
|---:|---:|---:|---:|---:|
| 32 | `61.636` | `75.683` | `+1.406%` | `+0.460ms` |
| 64 | `61.607` | `75.290` | `+1.358%` | `+0.068ms` |
| 128 mean | `64.052` | `79.198` | `+5.381%` | `+3.976ms` |

P64 is the profile incumbent. Its turn-5 result is `39.444ms` faster than
unbounded and confirms that the discarded middle is not scanned.

## Long-Context Retrieval

The repeated unbounded control contains the key in `5/14` cases. P32, P64 and
P128 each contain it in `7/14`, lose zero control-pass cases and gain the same
cases at 7,109 and 22,289 actual prompt tokens. This passes the paired
retrieval gate, but the small probe is not an accuracy benchmark.

## Full Replay

Every valid listed run completes all 330 graded requests without failure.

| candidate | valid runs | score points | TTFT p50/p95 ms | TPOT p50/p95 ms |
|---|---:|---:|---:|---:|
| native W4096 | 2 | `64.5121`, `64.5886` | mean `84.49/234.29` | mean `2.318/3.342` |
| safe8192 | 2 | `61.8737`, `61.9014` | mean `90.97/245.76` | mean `2.409/3.612` |
| split P32/W4096 | 1 | `63.0616` | `90.82/278.68` | `2.255/3.421` |
| split P64/W4096 | 2 | `63.1183`, `60.3961` | mean `93.24/274.64` | mean `2.326/3.469` |
| split P128/W4096 | 1 | `61.2771` | `93.14/299.88` | `2.399/3.612` |

P64 mean is `61.7572`, `0.1304` below safe8192 and `2.7932` below native
W4096. The first P64 run is promising, but the repeat demonstrates materially
wider process variance than the registered native/safe branches.

The initial P128 prototype forced 2D decode and scored only `54.9403` with
TPOT p50 `3.021ms`. Compact segmented 3D decode repaired TPOT and raised it to
`61.2771`; preserving the native decode launch policy is mandatory.

## Short Quality Proxy

All 100 GPQA prompts contain only 104 to 847 input tokens, so W4096 never
removes context. P64 repeats at `20/100` in both runs with the same set of
correct cases. Unbounded controls are `21/100` and `22/100`; safe8192 is
`23/100`. Relative to the control repeat, P64 loses correct cases 023 and 036
and gains none.

This cannot be attributed to split-range truncation because the window is
inactive. It is evidence of numerical sensitivity in the forked kernel or
FP8 greedy path. The official quality impact is unknown, but strict local
quality equivalence fails.

## Decision

- Accept the source, numerical and performance mechanism: compact split-range
  attention avoids R003e's full-span cost and can preserve useful old tokens.
- Retain P64 as a research Pareto point, not as the score or quality incumbent.
- Do not package or submit: replay mean does not beat safe8192/native W4096,
  GPQA proxy is two points below the control repeat, and SM90 is unverified.
- The next technical step, if this family is reopened, is numerical identity
  for the full-overlap case plus target SM90 qualification before more prefix
  lengths or scheduler tuning.

Machine-readable evidence is in `analysis/primitive_numerics.json`,
`profile_screen.json`, `retrieval_comparison.json`,
`full_replay_comparison.json` and `quality_comparison.json`.
