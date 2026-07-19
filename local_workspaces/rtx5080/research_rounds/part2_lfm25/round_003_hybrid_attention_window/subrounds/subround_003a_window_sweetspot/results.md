# Subround 003a Results

Status: complete. The measured outcome is a performance/semantic-risk frontier,
not a submission promotion.

## Profile Sweep

| window | repeats | turn 0 GPU ms | turn 5 GPU ms | turn 5 saving vs control | gate |
|---:|---:|---:|---:|---:|---|
| unbounded | 1 registered | `60.782` | `114.735` | baseline | control |
| 16384 | 2 | `61.905` | `101.264` | `13.471ms` | fail saving/retention |
| 12288 | 2 | `62.783` | `87.440` | `27.294ms` | pass |
| 8192 | 2 registered | `62.552` | `81.448` | `33.286ms` | pass |
| 6144 | 2 valid | `61.945` | `79.483` | `35.251ms` | pass |
| 4096 | 2 | `62.587` | `71.141` | `43.593ms` | pass |

All valid candidates retain the exact cache progression
`(0,4000), (4000,4199), (8192,4206), (12384,4213), (16592,4204),
(20784,4210)`. At the same memory fraction, the startup KV pool grows from
`1,649,210` tokens unbounded to `2,177,934` at `8192` and approximately
`2,622,715` at `4096`. The gain therefore combines less attention work with
more admission capacity.

One additional `6144` profile process was excluded before aggregation. Its
turn-0 time rose to about `81.9ms` while the window was inactive, identifying
process contamination rather than a window effect. The retained A/C pair is
recorded in the compact profile artifact.

## Long-Context Retrieval

The calibrated deterministic probe covers actual server-reported prompt
lengths of approximately `3,058`, `5,084`, `7,109`, `9,130`, `12,171`,
`17,228` and `22,290` tokens, with two keys per length.

| window | pass / 14 | lost control-pass cases | gained cases | risk gate |
|---:|---:|---:|---:|---|
| unbounded | `4` | baseline | baseline | control |
| 4096 | `2` | `2` | `0` | fail |
| 6144 | `3` | `1` | `0` | pass, semantic trade-off |
| 8192 | `4` | `0` | `0` | pass |
| 12288 | `4` | `0` | `0` | pass |
| 16384 | `5` | `0` | `1` | pass; gain treated as variance |

The control itself fails most very-long cases, so this is a paired regression
probe, not an accuracy estimate. GPQA length evidence is inherited from R003:
all 100 inputs are below 1k tokens and cannot discriminate these windows.

## Full Replay

All full runs use the exact R002k image and command, 3 CPU cores, 8 GiB RAM,
one RTX 5080, GPU utilization `0.80`, 90 primer requests and 330 graded
append-only requests.

| window | runs | score mean | score range | TTFT p50/p95 ms | TPOT p50/p95 ms | failures |
|---:|---:|---:|---:|---:|---:|---:|
| unbounded | 2 | `56.5450` | `55.9491..57.1409` | `114.26 / 315.56` | `2.571 / 4.159` | `0` |
| 4096 | 2 | `64.5504` | `64.5121..64.5886` | `84.49 / 234.29` | `2.318 / 3.342` | `0` |
| 6144 | 1 | `61.7945` | single run | `90.71 / 238.94` | `2.433 / 3.701` | `0` |
| 8192 | 2 | `61.1708` | `60.8540..61.4875` | `95.35 / 269.89` | `2.408 / 3.661` | `0` |
| 12288 | 1 | `57.0213` | single run | `110.73 / 300.81` | `2.477 / 3.835` | `0` |

The repeated `4096` mean is `+8.0054` points over the repeated control mean.
Its minimum is above both the control maximum and both `8192` runs. It reduces
TTFT p50/p95 by `29.78/81.27ms`, TPOT p50/p95 by `0.253/0.818ms`, and moves
all `330/330` graded requests under the `400ms` TTFT ceiling. Its two-run score
range is only `0.0766` points.

The benefit is depth-causal rather than a one-turn artifact. Candidate minus
control score by turn is `+1.180`, `+3.523`, `+7.028`, `+10.473`, `+12.313`
and `+13.514` points for turns zero through five.

## Decision

- `4096` is the reproducible local score maximum, but fails the registered
  paired retrieval gate with two control-pass losses. Keep it research-only.
- `6144` sits between score and semantic risk, but one run is insufficient and
  it offers only `+0.624` point over the repeated `8192` mean while losing a
  retrieval case. Do not carry it forward.
- `8192` remains the balanced research branch: repeated performance gain and
  zero paired retrieval losses. This does not prove full semantic equivalence.
- `12288` is dominated by `8192` on measured score with no retrieval advantage.
- `16384` fails the profile opening gate, so a full replay is unnecessary.
- R002k remains the only submission incumbent. No image or Compose is created
  from R003a pending BTC driver semantics and rules review.
