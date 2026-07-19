# Round 003 - Hybrid Full-Attention Window

Status: complete local research family; not promoted to a submission. Follow-up
R003a establishes `4096` as the score-max branch and keeps `8192` as the
semantics-balanced branch.

## Question

Can the six LFM2.5 full-attention layers use a bounded causal window while the
ten ShortConv layers, hybrid cache lifecycle, R002k fused head and decode
policy remain unchanged?

R002y found that a cached turn still schedules only about 4.2k new tokens,
while attention cost grows from `4.825ms` to `54.742ms` as retained history
approaches 25k tokens. That makes bounded attention a direct hypothesis for
the observed first-token tail.

## Candidate

The candidate injects a trace-independent `8192`-token
`per_layer_sliding_window` into exactly the six full-attention layers
`[2, 5, 8, 10, 12, 14]`. It does not inspect request IDs, prompt contents,
turn numbers, arrival times or endpoint paths. ShortConv layers
`[0, 1, 3, 4, 6, 7, 9, 11, 13, 15]` are untouched.

The R002k runtime policy remains fixed: FP8 per-tensor weights, FP8 E4M3 KV,
BF16 aligned ShortConv state, prefix caching, b16384, async/interactivity
scheduling and the fused BF16 greedy head.

## Result

On the inferred append-only closed-loop replay, two candidate runs average
`61.1708` ERS points versus two matched controls at `56.5450` (`+4.6258`
points). TTFT p50/p95 improve from `114.26/315.56ms` to
`95.35/269.89ms`; TPOT p95 improves from `4.159ms` to `3.661ms`. All four
full runs complete `420/420` requests.

The gain grows with turn depth: `-0.124, +1.328, +3.082, +6.274, +7.821,
+9.373` score points for turns 0 through 5. An independent 4k-prompt check is
neutral/slightly negative, so the technique is specifically a long-history
optimization rather than a generic 4k prefill shortcut.

## Risk Boundary

The window changes model semantics for contexts beyond 8192 tokens. The short
GPQA proxy passes, but it cannot validate long-context reasoning. The public
trace exposes per-turn lengths and the contract labels closed-loop semantics
as inferred; BTC has not supplied the exact request-body construction. The
candidate therefore remains research-only until these two uncertainties are
resolved.

## R003a Frontier

The follow-up window sweep confirms that the performance effect scales with a
smaller window. Two full `4096` replays score `64.5121/64.5886`, a mean gain of
`+8.0054` points over the repeated unbounded control. However, `4096` loses two
paired long-context retrieval cases. Window `8192` loses none of the four
control-pass cases and remains the balanced branch at a repeated `61.1708`
mean. Windows `12288` and `16384` do not retain enough score opportunity;
`6144` offers an intermediate but weaker quality trade-off.

These results strengthen the mechanism diagnosis but do not remove the rules
or semantic block. R002k remains the submission incumbent.

## Files

- `round_plan.md`: gates, result and follow-up 003a.
- `results.md`: complete measurements and interpretation.
- `overlay/`: kill-switchable R003 hook.
- `scripts/run_prefill_process.sh`: isolated turn-depth profile.
- `scripts/run_quality_process.sh`: 100-case GPQA proxy.
- `scripts/run_full_replay.sh`: full/bounded append-only or independent replay.
- `scripts/analyze_full_replay.py`: score-component and per-turn analysis.
- `analysis/full_replay_comparison.json`: four-run full bracket.
- `subrounds/subround_003a_window_sweetspot/`: complete score/quality frontier
  for windows `4096..16384`.
- `failed_actions.md`: invalid probes and unresolved boundaries.

## Follow-Up R003b - Selective Global Placement

R003b tested whether leaving one or two of the six full-attention layers
unbounded could repair the two retrieval losses seen at window `4096`. The
four placements (`global14`, `global2`, `global8`, `global2_14`) all passed the
isolated performance gate, saving `32.443..37.932ms` at turn 5 with zero
request failures. All four still lost the same two control-pass retrieval
cases at actual prompt lengths `5084` and `7109` tokens.

This closes the placement hypothesis without a full replay. A global layer
cannot reconstruct information discarded by earlier windowed layers. The next
candidate must preserve selected old information in each affected layer, not
just add a global layer after the truncation.

See `subrounds/subround_003b_sparse_global_attention/` for the exact overlay,
compact measurements and rejection record.

## Follow-Up R003c - Sparse Windowed Layer Placement

R003c tested six placements that windowed only two or three of the six
full-attention layers at 4096 while leaving the complement global. All six
profiles passed the opening latency gate. The quality boundary was selective:
late2, late3, spread3, early3 and first1 lost two or three control-pass cases
in the paired retrieval probe, while early2 (windowed layers 2 and 5) lost
zero cases in two repeats.

early2 completed two 420-request replays with zero failures and scored
58.0922 versus 56.5450 for the matched unbounded control. TTFT p50/p95
improved from 114.26/315.56 ms to 105.45/287.42 ms, while TPOT p95 improved
from 4.159 ms to 4.094 ms. The GPQA proxy was 21/100 versus a 21..22/100
control range.

This is a useful local research branch, not a submission candidate. The
short quality proxy does not test information beyond 4096 tokens, and the
branch remains below the aggregate score of the 8192 research candidate.
R003c closes scalar placement permutations. The next bounded-attention idea
must retain selected old positions in every affected layer rather than add
another globally causal layer after information has already been discarded.

## Follow-Up R003d - Tiered Attention Horizons

R003d tested a fixed per-layer horizon map to interpolate between the score-max
`4096` branch and the quality-balanced `8192` branch:

```text
safe8192: 2:4096, 5:4096, 8:8192, 10:8192, 12:8192, 14:8192
mid6144:  2:4096, 5:4096, 8:6144, 10:6144, 12:6144, 14:6144
front3:   2:4096, 5:4096, 8:4096, 10:8192, 12:8192, 14:8192
```

All three profiles passed the latency gate. The paired retrieval gate retained
only `safe8192`: it lost zero control-pass cases in two repeats and gained one
case at 7,109 actual prompt tokens. `mid6144` lost one control-pass case and
`front3` lost two.

Two full replays of `safe8192` averaged `61.8876` points versus `56.5450` for
two unbounded controls. Mean TTFT p50/p95 improved from `114.262/315.560ms`
to `90.966/245.763ms`; TPOT p95 improved from `4.159ms` to `3.612ms`, with
zero graded failures in all four runs. The short GPQA proxy was `23/100`
versus controls at `21..22/100`, but its prompts are shorter than the horizon.

R003d closes scalar horizon interpolation locally. Keep `safe8192` as a
research reference only: the gain combines bounded attention with a larger KV
pool at the same memory fraction, and the horizon changes model semantics for
long contexts. No image, Compose or Portal artifact follows. The next bounded
attention branch must preserve selected old information rather than simply
truncating it.
