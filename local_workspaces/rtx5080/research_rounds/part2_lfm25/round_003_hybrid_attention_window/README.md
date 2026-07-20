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

## Follow-Up R003h - FlashInfer Cascade Prefix/Recent Probe

R003h verified FlashInfer's `MultiLevelCascadeAttentionWrapper` on BF16 and
FP8 paged KV and then exercised a decode-only overlay on the native FlashInfer
backend. The primitive can be substantially faster when the selected key set
is only `prefix OR recent`, but that layout discards the middle of a long
sequence and is therefore an approximate sparse-attention policy. It is not an
exact replacement for the original full-attention model.

The candidate completed the local replay without failures, but the matched
cascade-off control scored `64.7741` versus `60.6272` for the candidate. The
candidate also had a higher TPOT median and a different output hash. R003h is
closed and rejected for submission. The next cascade experiment must target
the exact shared-prefix case already represented by vLLM metadata, where all
non-shared tokens remain visible.

See `subrounds/subround_003h_flashinfer_cascade_probe/` for the primitive,
smoke, quality and full-replay records.

## Follow-Up R003i - Exact Shared-Prefix Cascade

R003i tested vLLM's exact two-level shared-prefix cascade without dropping any
private suffix token. Forced FlashInfer was operational with BF16 and FP8 KV,
but the public proxy produced zero cascade builds and scored `51.1766` versus
`52.6311` for its same-backend matched control.

The native FlashAttention path was then bracketed with global cascade on/off.
At batch 8, candidate/control scores were `35.6212/41.1370`; at batch 20 they
were `19.8181/38.2097`. Native cascade increased both TTFT and TPOT when it
activated. The runtime-SM model predicts more frequent activation on H200
`1g.18gb`, but there is no positive target measurement. R003i is closed: keep
R002k's default cascade setting and create no image, Compose or Portal attempt.

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

## Follow-Up R003e - Native Prefix Plus Recent Mask

R003e forced vLLM 0.22.1's Triton attention backend on all six attention
layers and used its native `mm_prefix_range` tensor to retain the first
`64/128/256` tokens in addition to a 4096-token recent window. Startup proved
the requested range reached all six Triton metadata objects.

The no-prefix Triton/window diagnostic was fast at `62.364/75.223ms` for
turns 0/5. Every non-empty prefix was slow at about `70/187ms`; prefix size
had no meaningful effect. This isolates the regression to the native
disjoint-range execution path. R003e closes before retrieval and full replay.
A future implementation must compute only the two retained ranges and merge
their softmax statistics rather than scan the discarded middle span.

Post-profile source correction: native `mm_prefix_range` is bidirectional only
when both query and key are inside the range. It does not preserve prefix keys
for later queries, and its loop helper disables sliding-window tile pruning.
R003e is therefore a rejected API screen, not a prefix-retention candidate.

## Follow-Up R003f - Compact Split-Range Triton Attention

R003f forks the exact Apache-2.0 vLLM 0.22.1 Triton unified-attention kernel
and implements the intended causal `prefix OR recent` key set. A compact
virtual tile list skips the discarded middle before block-table/K/V loads,
while native 2D prefill and segmented 3D decode launch policies are retained.

BF16/FP8 paged-KV numerical tests pass with maximum absolute error `0.002719`.
P64/W4096 profiles at `61.607/75.290ms` for turns 0/5, only `0.068ms` above
the recent-only floor. P32/P64/P128 all improve the paired retrieval probe
from `5/14` to `7/14` with no control-pass loss.

Two valid P64 full replays score `63.1183` and `60.3961`, mean `61.7572`:
`0.1304` below safe8192 and `2.7932` below native W4096. The short GPQA proxy
repeats at `20/100` versus controls at `21..22/100`, despite all prompts being
inside the window. Keep R003f as a research Pareto point only. It is neither
the score nor quality incumbent, and CUDA 13.x/H200 SM90 remains unverified;
no image, Compose or Portal artifact is created.
