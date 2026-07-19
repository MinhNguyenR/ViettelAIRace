# Round 002 Results

Status: active research record. The native b16384/FP8-KV/async branch is the
Portal rollback; subround 002k is the H200-validated incumbent with exact-run
ERS `60.75/58.94` (best `60.75`, mean `59.845`). Subround 002n is a local
ShortConv projection study; it has no submission artifact.

## Portal Control

```text
Attempt 01 ERS:          56.53
TTFT p50/p95:            73 / 116 ms
TBT median:               4 ms
graded / primer:        330 / 90
failures:                 0
```

## Experiment Table

| Candidate | Process | ERS | TTFT p50/p95 | TPOT p50/p95 | Peak active | Decision |
|---|---:|---:|---:|---:|---:|---|
| control seq8/b8192 | A1, externally prewarmed | 67.326 | 67.360/179.598 | 2.389/3.286 | 7 | Excluded: one client disconnect |
| seq8/b4096 | B1, game running | 52.492 | 92.070/252.720 | 3.326/4.268 | 15 | Contaminated; never use for tuning |
| seq8/b4096 | B2, clean cold start | 66.421 | 69.848/182.728 | 2.506/3.383 | 11 | Valid; statistically tied with control |
| control seq8/b8192 | A2, clean cold start | 66.335 | 69.947/185.914 | 2.471/3.395 | 11 | Valid control; retain as incumbent |
| seq8/b2048 | C, clean cold start | 66.306 | 71.619/175.530 | 2.479/3.344 | 11 | Valid; no score gain |
| seq6/b8192 | D, clean cold start | 66.531 | 69.744/181.363 | 2.521/3.371 | 11 | Valid; small gain, needs stronger evidence |
| seq4/b8192 | E, clean cold start | 66.888 | 70.016/131.485 | 2.493/3.295 | 8 | Promising; two TTFT requests exceeded 400ms |
| seq5/b8192 | F, clean cold start | 64.953 | 72.373/190.046 | 2.615/3.459 | 11 | Rejected; worse TTFT and TPOT |
| seq3/b8192 | G, clean cold start | 66.264 | 71.680/139.264 | 2.561/3.146 | 8 | Rejected; five TTFT requests exceeded 400ms |
| seq4/b8192 | E2 repeat, clean cold start | 65.721 | 70.906/137.932 | 2.616/3.360 | 9 | Rejected; first-run gain did not repeat |
| seq8 generic FP8 | H, clean cold start | 65.909 | 70.607/191.072 | 2.546/3.491 | 10 | Rejected; worse TTFT and TPOT |
| seq8 FP8 KV + requested scale calc | I1, clean cold start | 66.937 | 69.302/183.150 | 2.367/2.985 | 7 | Promising; runtime disabled scale calculation |
| seq8 FP8 KV | I2 repeat, clean cold start | 66.955 | 69.355/183.079 | 2.368/3.119 | 7 | Performance promoted; quality gate required |
| seq8 async scheduling | J, clean cold start | 66.856 | 69.136/181.737 | 2.449/3.355 | 11 | Positive but below repeat threshold; test interaction |
| seq8 FP8 KV + async | K1, clean cold start | 67.455 | 68.626/184.846 | 2.328/3.194 | 7 | New incumbent; cold repeat required |
| seq8 FP8 KV + async | K2 repeat, clean cold start | 67.067 | 68.976/188.137 | 2.364/3.178 | 7 | Repeated positive interaction; incumbent |
| incumbent, 3000-token session prefix | L1, clean cold start | 65.655 | 68.658/227.925 | 2.435/3.779 | 9 | Transfer diagnostic; cache path did not improve aggregate ERS |
| incumbent + `xxhash`, session prefix | L2, clean cold start | 0.000 | N/A | N/A | 2 | Rejected: all requests failed because image lacks `xxhash` |
| incumbent, prefix cache off / Mamba none | M1, clean cold start | 62.816 | 72.578/219.514 | 2.653/3.995 | 9 | Rejected: both TTFT tail and TPOT regressed materially |
| incumbent + `balanced` mode | N1 screen, 90 graded | 67.482 | 72.854/132.229 | 2.463/3.056 | 5 | Screen positive, full confirmation required |
| incumbent + `throughput` mode | N2 screen, 90 graded | 70.302 | 67.662/123.175 | 2.258/2.829 | 6 | Screen positive, full confirmation required |
| incumbent + `throughput` mode | N3 full, 330 graded | 68.029 | 73.370/196.153 | 2.508/3.335 | 6 | Not promoted; screen gain did not repeat |
| incumbent + Mamba `float16` state | O1 screen, 90 graded | 66.071 | 71.191/167.321 | 2.521/3.547 | 9 | Rejected; worse tail and no decode gain |
| seq10 incumbent, no-reuse | P1 screen, 90 graded | 70.289 | 67.749/123.338 | 2.261/2.836 | 6 | No incremental gain; graph/capacity cost |
| seq10 incumbent, session prefix | P2 screen, 90 graded | 70.213 | 67.486/123.563 | 2.249/2.810 | 6 | No incremental gain; no full run |
| graph capture `[1,2,4,8]` | Q0 screen, 90 graded | 0.000 | N/A | N/A | 5 | Rejected: runtime CUDA OOM after lazy JIT |
| incumbent + FP8 E5M2 KV | Q1 screen, 90 graded | 64.539 | 74.013/166.221 | 2.494/3.529 | 9 | Rejected; worse score, TPOT and startup |
| incumbent + compile size `4096` | Q2 screen, 90 graded | 68.857 | 69.307/123.686 | 2.325/3.084 | 6 | Rejected: four helper-kernel JIT warnings remained; screen-only gain |
| incumbent + `throughput` mode | Q3 full, 330 graded | 66.490 | 70.510/187.840 | 2.446/3.298 | 7 | Rejected; second full run did not repeat 68.029 |
| seq10 incumbent | Q4 full, 330 graded | 64.789 | 72.714/207.975 | 2.565/3.391 | 9 | Rejected; extra active slot worsened full-trace ERS |
| incumbent, GPU utilization `.86` | Q5 startup | N/A | N/A | N/A | N/A | Rejected: local free VRAM was below requested reservation |
| incumbent, GPU utilization `.83` | Q6 screen, 90 graded | 67.782 | 71.312/134.030 | 2.388/2.939 | 6 | Rejected: more cache did not improve screen ERS or active count |
| incumbent b8192 | R1 control screen, 90 graded | 68.275 | 72.331/133.813 | 2.439/2.893 | 6 | Opening bracket control |
| incumbent b4096 | R2 screen, 90 graded | 66.615 | 73.473/137.977 | 2.491/3.144 | 7 | Rejected; both components worsened |
| incumbent b16384 | R3 screen, 90 graded | 67.710 | 71.017/137.631 | 2.367/2.947 | 6 | Positive after control-drift adjustment; full gate opened |
| incumbent b8192 | R4 control repeat, 90 graded | 66.464 | 73.286/142.980 | 2.559/3.185 | 6 | Closing control; confirms 1.81-point process drift |
| incumbent b16384 | R5 full, 330 graded | 67.879 | 67.628/200.020 | 2.296/3.112 | 7 | Promising; cold full repeat required |
| incumbent b16384 | R6 full repeat, 330 graded | 68.740 | 66.042/219.206 | 2.239/2.805 | 7 | Promoted; repeated above incumbent mean with zero failures |

## Driver Qualification

The first full control process A1 scored `67.326` but is excluded from
promotion because one graded request failed before HTTP response with
`ServerDisconnectedError` at `0.39ms`. The server remained healthy and emitted
no corresponding error, OOM or traceback. The sparse client had reused a
keep-alive socket across think gaps; subsequent processes force a fresh HTTP
connection per request. A2 completed with `330/330` graded successes, so the
driver repair is qualified.

## GPU Contamination Check

The first `4096` process ran while a game was using the RTX 5080. After the
game was closed, idle inspection showed approximately `1554MiB` display VRAM,
`14255MiB` free VRAM, `2%` GPU utilization and no competing compute process.
The same config was then run again from a cold server:

```text
Metric             game running    clean       change
ERS                    52.492       66.421      +13.929 points
TTFT p50 ms             92.070       69.848      -22.222
TTFT p95 ms            252.720      182.728      -69.992
TPOT p50 ms              3.326        2.506       -0.820
TPOT p95 ms              4.268        3.383       -0.885
peak active             15           11
```

The game therefore caused a material performance regression and changed the
closed-loop arrival pattern. B1 is environmental contamination, not evidence
against a `4096` token budget.

## Clean Budget Comparison

Both valid processes used a cold server, fresh HTTP connections, the same
synthetic `no_reuse` prompts, `330/330` graded successes and peak active count
`11`. The only server argument difference was
`--max-num-batched-tokens=4096` versus `8192`.

```text
ERS delta, 4096 - 8192:       +0.086 points
TTFT p50 delta:               -0.099 ms
TTFT p95 delta:               -3.186 ms
TPOT p50 delta:               +0.035 ms
TPOT p95 delta:               -0.012 ms
```

This is a tie at the precision of one local process. The latency changes are
small and TPOT does not move consistently in one direction. Keep `8192` as the
incumbent because it is the Portal-tested setting and do not promote `4096` as
a sweet spot without repeated-process evidence.

The `2048` boundary process scored `66.306` with zero failures. It improved
TPOT p95 by only `0.051ms` versus the `8192` control while worsening TTFT p50
by `1.672ms`; the per-request ERS did not improve. Across `2048`, `4096` and
`8192`, the complete score range is only `0.115` points. Prefill budget is
therefore frozen at `8192`; further budget tuning is lower priority than the
active decode set.

## Active-Set Sweep

The first `seq6` process improved ERS by only `0.196` points over `seq8` and
did not improve TPOT p50. `seq4` produced a larger `+0.553`-point gain and
lowered TPOT p95 by `0.100ms`, but two requests had TTFT above the `400ms`
ceiling (`439.987ms` maximum). This is a positive active-set signal with a
tail-risk warning, not yet a promotion. Test `seq5` as the likely compromise
before deciding whether the lower `seq3` boundary is necessary.

`seq5` rejected the simple monotonic active-set hypothesis: it scored `64.953`
and regressed both TTFT and TPOT. Startup logs show that vLLM captured FULL
decode graphs through batch 4, 5, 6 and 8 respectively for those active-set
limits, while PIECEWISE capture extends to twice the limit. The isolated
`seq4` gain may therefore be a favorable CUDA-graph/kernel batch shape rather
than a generic benefit from reducing concurrency. Measure `seq3`, then repeat
`seq4` before promotion.

`seq3` lowered TPOT p95 to `3.146ms`, but five requests crossed the TTFT
ceiling and the maximum reached `587.409ms`; ERS fell to `66.264`. This closes
the lower boundary. Do not test `seq2`: the extra admission tail has already
outweighed the TPOT gain at `seq3`.

The mandatory `seq4` repeat scored only `65.721`, `1.167` points below its
first process. Both runs had two TTFT ceiling violations, and their mean ERS
is `66.304`, effectively the same as the `seq8` control (`66.335`). The active
set sweep is therefore closed with no promotion. Keep `seq8` for precision
experiments and require a gain larger than this observed process jitter before
packaging.

## Precision Sweep

Generic `--quantization=fp8` selected a different runtime path from
`fp8_per_tensor`: the former logged `quantization_config=None`, while the
incumbent uses static FP8 E4M3 weights and BF16 activations. Generic FP8 scored
only `65.909` and regressed both TTFT p95 (`191.072ms`) and TPOT p95
(`3.491ms`). Reject it without spending a quality-evaluation run.

FP8 E4M3 attention KV with BF16 recurrent state scored `66.937`, improving
TPOT p50/p95 from `2.471/3.395ms` to `2.367/2.985ms` without a TTFT ceiling
violation. vLLM explicitly disabled requested KV-scale calculation because
LFM2 is hybrid and recurrent state makes its calibration pass unreliable; the
run therefore used scale `1.0`. Repeat the exact effective config without the
deprecated no-op flag, then require a paired quality gate before promotion.

The exact effective config repeated at `66.955`, only `0.018` points from I1,
with the same peak active count and no failure/ceiling violation. Mean ERS is
`66.946`; this is the first performance factor in Round 002 to reproduce.

### FP8 KV Quality Proxy

The same fixed 100-question local GPQA subset and deterministic decoding were
run once on BF16 KV and once on FP8 KV:

```text
BF16 KV accuracy:               21/100
FP8 KV accuracy:                21/100
paired accuracy drop:                0
control-correct -> FP8-wrong:        0
control-wrong -> FP8-correct:        0
changed parsed prediction:          13
```

All 13 prediction movements remained within the already-incorrect/unparsed
cohort. FP8 KV passes this local relative gate, but the low absolute accuracy
and parser compliance show that this old Qwen-oriented prompt is only a drift
proxy for LFM2.5. Hidden BTC quality remains an external risk.

## Runtime Scheduling

Async scheduling on BF16 KV scored `66.856`, a `+0.521`-point single-process
gain with no failure or TTFT ceiling violation. This is positive but smaller
than the observed active-set process jitter, so it is not independently
promoted. Test its interaction with the already-reproduced FP8 KV factor; keep
async only if the combination beats the FP8 KV two-run band.

The combination scored `67.455`, beating the reproduced FP8 KV band by at
least `0.500` points and the BF16-KV control by `1.120` points. It preserved
zero failures and zero TTFT ceiling violations while lowering TPOT p50 to
`2.328ms`. This is a positive interaction; require one cold repeat before
promotion.

The cold repeat scored `67.067`; both combination runs are above `67`, with a
mean of `67.261`. This is `+0.315` points over the two-run FP8-KV-only mean.
The gain is moderate but directionally repeated and async scheduling adds no
lossy arithmetic, so the combination becomes the current performance
incumbent. Qualify it under session-prefix reuse before packaging.

The 3000-token per-conversation shared-prefix proxy scored only `65.655` and
did not show lower aggregate TTFT on later turns. Closed-loop acceleration
raised peak overlap from 7 to 9, while hybrid prefix-cache scheduling and
decode contention increased the tails. This does not invalidate prefix caching
for hidden prompts, but it rejects any assumption that reuse is a free score
gain. Compare hash algorithms only on this reuse workload and retain the
default unless `xxhash` wins materially.

The `xxhash` process never reached inference. Healthcheck passed, but the
optional module is imported only when the first prefix-cache key is built;
the fixed vLLM image then raised `ModuleNotFoundError` for every request.
`420/420` requests failed preprocessing and all 330 graded requests scored
zero. This is a packaging incompatibility, not a performance result. Retain
the default SHA-256 hash and require one real-request smoke test for future
lazy runtime options.

### Prefix Cache Ablation

Disabling prefix caching and explicitly selecting `--mamba-cache-mode=none`
reduced the logged Mamba page-size padding from `300%` to `100%`, but available
cache capacity changed only from about `1.720M` to `1.708M` tokens. The cold
full trace then scored:

```text
ERS:                    62.816
TTFT p50/p95:           72.578 / 219.514 ms
TPOT p50/p95:            2.653 /   3.995 ms
TTFT over 400 ms:        0
failed graded:           0
```

This is `-4.445` points against the two-run incumbent mean of `67.261`.
Although the synthetic `no_reuse` prompts deliberately remove cross-request
prefix reuse, disabling the hybrid cache still worsens both score components.
The likely cost is the non-prefix Mamba-state/block lifecycle rather than
hashing alone. Reject this mode without a session-prefix follow-up; it cannot
recover the already-observed no-reuse loss robustly enough to become the
submission default.

### Performance Mode Screen

The fixed image exposes `balanced`, `interactivity` and `throughput`. The
incumbent is `interactivity`; the new modes were tested with the same FP8
weights, FP8 attention KV, BF16 Mamba state, async scheduling, seq8 and 8192
token budget.

The 30-conversation screens (90 graded requests each) were:

```text
mode          ERS       TTFT p50/p95       TPOT p50/p95       peak
balanced      67.482    72.854/132.229    2.463/3.056          5
throughput    70.302    67.662/123.175    2.258/2.829          6
```

Throughput looked promising and was run over the complete 70-conversation
trace. It first scored `68.029`, with `73.370/196.153ms` TTFT and
`2.508/3.335ms` TPOT. A second cold full repeat scored `66.490`, with
`70.510/187.840ms` TTFT and `2.446/3.298ms` TPOT. The screen gain and first
full gain therefore do not repeat; reject throughput for packaging.

### Mamba State Dtype Screen

The fixed vLLM image accepts only `auto`, `bfloat16`, `float16` and `float32`
for `--mamba-cache-dtype`; it has no FP8 recurrent-state implementation. The
only plausible lower-precision trial, `float16`, started cleanly but scored
`66.071/90 graded` versus the incumbent's comparable screen and increased
TPOT p95 to `3.547ms`. It also lowered the reported cache capacity slightly.
Reject it without a quality run and retain BF16 Mamba/short-convolution state.

### Active-Set Extension

Increasing `max-num-seqs` from 8 to 10 caused vLLM to capture graphs through
batch 20 and reduced local KV capacity to `1,699,097` tokens. The two 90-graded
screens were effectively identical to the throughput-mode screen and did not
increase observed peak active requests:

```text
workload        seq10 ERS   TTFT p50/p95       TPOT p50/p95       peak
no_reuse        70.289      67.749/123.338    2.261/2.836          6
session_prefix  70.213      67.486/123.563    2.249/2.810          6
```

The apparent improvement is process-order variance, not an active-set effect;
there is no full-trace justification for paying the additional graph capture
and cache cost. Keep `max-num-seqs=8`.

### KV FP8 Format Screen

`fp8_e5m2` is executable, but it is not a viable replacement for the promoted
`fp8_e4m3` KV path. Its 90-graded screen scored `64.539`, with TTFT
`74.013/166.221ms` and TPOT `2.494/3.529ms`; startup also took `62s` versus
about `40s` for the incumbent. Reject E5M2 before quality testing and retain
E4M3 with its paired zero-drop local quality result.

### Compile-Size Warmup Probe

Adding `--compilation-config={"compile_sizes":[4096]}` increased cold startup
from about 40 seconds to `86s` and reported a valid `1,705,357`-token KV
capacity. The 90-graded screen scored `68.857`, but the server still logged
the same four first-use helper compilations:

```text
_zero_kv_blocks_kernel
_compute_slot_mapping_kernel
_causal_conv1d_fwd_kernel
batch_memcpy_kernel
```

The extra compile shape did not cover the real cache/block helper paths. Since
the official trace already contains 90 primer requests before the 330 graded
requests, paying the startup cost does not remove a graded-path bottleneck.
Reject the compile-size override and keep automatic compilation.

### Full Active-Set Confirmation

The two short seq10 screens were not representative. A cold full seq10 run
reached peak active `9` but scored only `64.789`, versus the seq8 incumbent
band around `67.261`. Its TTFT p95 rose to `207.975ms` and TPOT p95 to
`3.391ms`. The larger graph inventory and extra contention do not pay back on
the full arrival trace. Keep `max-num-seqs=8`.

### Incumbent Prefill-Budget Bracket

Because the earlier 2048/4096/8192 sweep predated FP8 KV plus async scheduling,
the promoted precision/runtime stack was re-bracketed at 4096/8192/16384. The
90-graded A/B/C/A screen sequence was:

```text
b8192 A1:  68.275
b4096:     66.615
b16384:    67.710
b8192 A2:  66.464
```

The `1.811`-point control drift makes raw screen ranking unsafe. Linear control
interpolation still leaves b4096 about `1.06` points negative and b16384 about
`0.64` points positive. b4096 was closed; b16384 advanced to two independent
cold full traces:

```text
run       ERS      TTFT p50/p95       TPOT p50/p95       failed
R5        67.879   67.628/200.020     2.296/3.112         0
R6        68.740   66.042/219.206     2.239/2.805         0
mean      68.310
```

Both runs beat the prior b8192 incumbent mean `67.261`; all `660` graded
requests succeeded and stayed below both ceilings. The larger budget is
therefore promoted for the exact Part 2 stack. This does not reopen the old
Qwen b16384 result because model, trace, scoring and runtime behavior differ.

## Answer

The promoted Round 002 candidate is `fp8_per_tensor` weights plus FP8 E4M3
attention KV, BF16 Mamba state, async scheduling, prefix caching with Mamba
`align`, chunked prefill, `max-num-seqs=8` and a `16384` token budget. Its two
cold full runs scored `67.879` and `68.740` (mean `68.310`) with `660/660`
graded successes. This exceeds the former b8192 two-run mean of `67.261`.
Generic FP8, lower active sets, `seq10`, `throughput`, explicit compile sizes,
high local cache reservation, `xxhash` and disabling prefix caching are
rejected. Package Attempt 02 from b16384 without adding unmeasured flags.

## Portal Attempt 02/03 Resolution

The Portal resolved the turn-semantics hedge:

```text
Attempt 02 prefix + Mamba align:
  ERS 58.17, TTFT p50/p95 58/82ms, TBT median 4ms, failed 8
Attempt 03 no prefix + Mamba none:
  ERS 47.91, TTFT p50/p95 98/201ms, TBT median 4ms, failed 5
```

Both panels reported `warmup_count=0` and the same visible `config_hash`, even
though the repository artifacts differ in cache policy. The hash therefore
cannot be treated as a complete runtime-policy identity. The score and TTFT
direction agree with the append-only local bracket: retain prefix caching and
Mamba `align`; close the no-prefix hedge. The nonzero failures and primer
accounting remain evaluator uncertainties.

## Output-Head Follow-Up

Subrounds 002d and 002f implemented a separate output-only FP8 copy of the
tied embedding/LM head. On the matched append-only full trace, no-repair tensor
FP8 scored `58.857` versus native `56.729`, improving TPOT p50
`2.584 -> 2.417ms` and TTFT p50 `110.524 -> 105.888ms` with zero failures.

It is not promoted. The deterministic 96-request smoke matched only `80/96`
first tokens and `0/96` complete sequences. Row-wise top-k32 improved the
shortlist to `92/96` first tokens and `29/96` complete sequences but remained
below the accuracy gate. Keep this as a measured kernel ceiling, not a Portal
artifact.

## Non-lossy Fused Greedy Head (Subround 002k)

The exact BF16 tied head was fused into a Triton tiled dot-product followed by
hierarchical argmax. The overlay is strict: empty built-in processor state is
the only additional eligible case; logprobs, grammar, penalties, speculative
decode and unknown processors use native full logits.

```text
native control ERS:       56.7287
fused full replay ERS:    58.3989
delta:                    +1.6702 points
TTFT p50/p95:             106.503 / 336.066 ms
TPOT p50/p95:               2.477 / 3.649 ms
graded failures:          0/330
```

The same-hidden-state audit matched `20/20` audit batches, including M=8,
against the native delegate. Standalone tile sweeps also matched every token.
The clean packaged image starts without source mounts, exercises the native
fallback on logprob requests and completes all smoke requests. Its 100-case
GPQA proxy retains `21/100` accuracy with zero correctness flips versus the
incumbent. Portal Attempt 04 validates H200 transfer:

```text
Attempt 02 native image: ERS 58.17, TTFT 58/82ms, TBT 4ms, failures 8
Attempt 04 R002k image:  ERS 60.75, TTFT 44/80ms, TBT 4ms, failures 6
delta:                  +2.58 ERS, -14/-2ms TTFT, -2 failures
```

The submitted digest predates R002l and contains only the fused greedy overlay.
The displayed online accuracy drop is not final GPQA evidence; post-round
quality selection remains pending.

The unchanged digest repeated at `58.94`, TTFT p50/p95 `55/79ms`, TBT `4ms`
and seven failures, versus `60.75`, `44/80ms`, `4ms` and six failures on the
first run. This produces a `1.81`-point same-artifact ERS range and an 11ms
TTFT-p50 range. Best score remains `60.75`; use mean `59.845` for expected-value
comparisons and reject attribution from a single delta within the observed
range.

## Selective Final RMSNorm (Subround 002l)

The exact source audit found that LFM2 final RMSNorm runs on every scheduled
hidden row before vLLM selects only the logits rows. Deferring RMSNorm until
after selection is bit-exact and improves the isolated large-prefill operation
by a weighted `21.917%`, but the weighted absolute saving is only `0.149ms`.
Decode would regress by about `40.556%`, so the integrated path activates only
at 2048 or more scheduled rows and returns to native fused normalization for
decode.

The runtime audit observed the intended `4000 rows -> deferred`, `1 row ->
native` transition, and all 20 native/fused argmax checks matched. Full output
SHA and native logprob fallback also pass. The full B/A/B bracket is:

```text
R002l candidate 1: 55.8737, TPOT p95 3.746ms, peak 7
R002k control:     53.5857, TPOT p95 3.779ms, peak 7
R002l candidate 2: 52.7145, TPOT p95 4.568ms, peak 11
```

Candidate mean is `+0.7084` points over the middle control and all six
turn-mean deltas are positive. However, the second candidate loses outright
and violates the TPOT guard. The saving is below process variance, so R002l is
closed without packaging or a Portal attempt. Retain it only as a source
primitive for a future larger prefill bundle; keep R002k Attempt 04 immutable.

## Exact Decode-Head V2 (Subround 002m)

R002m measured the remaining R002k output ownership and alternative BF16
weight-access paths. The final run used 16 correctness repeats, nine timing
samples and 100 iterations per sample for M1/M2/M4/M8. Every variant matched
`240/240` native token IDs.

```text
candidate                 weighted GPU delta    weighted wall delta
direct-owned output              +0.011%               +0.430%
unsafe no-clone floor             +0.489%               +0.615%
transposed BF16 (256 MiB)          -1.299%               -0.926%
best manual GEMV                about -154%            about -153%
```

The direct output path only exchanges a tiny clone for allocation/ownership
work. Manual GEMV rereads the tied weight per row and scales to about 8x R002k
latency at M8. None passes the registered 3% head-level screen, so no live
overlay, full trace, image or submission is created. R002k is retained.

## Decode Hotspot Profile and ShortConv FP8 (Subround 002n)

Kineto child-kernel profiles at batches 1, 2, 4 and 8 show that existing FP8
GEMMs consume `58.269%` of decode kernel time, the R002k output head consumes
`16.050%`, and ten ShortConv BF16 projections consume `19.499%`. The latter
passes the registered whole-decode opportunity gate. Source inspection of
vLLM `lfm2.py` and `layers/mamba/short_conv.py` confirms that those projection
constructors do not receive the active FP8 `quant_config`.

An opt-in overlay was implemented with independent `in`, `out` and `all`
modes. Batch-8 profile results were:

```text
control:                    141.598ms / 64 steps
in_proj FP8:                131.718ms / 64 steps
out_proj FP8:               142.546ms / 64 steps
both projections FP8:      130.119ms / 64 steps
```

The all-projection branch did not survive quality/end-to-end gates: its GPQA
proxy was `18/100` twice versus the reference `21/100`, and its full replay
ERS was `58.2706` versus the R002k local anchor `58.3989`. It is closed.

The in-only branch retained proxy accuracy at `22/100` twice and had zero
reference-correct-to-candidate-wrong transitions in both comparisons. Two
exact local full replays scored `58.9590` and `59.4062`; TPOT p50/p95 were
`2.421/3.756ms` and `2.405/4.006ms`. A same-window control scored `57.2978`,
so the immediate paired delta was `+2.1084` points, but the control itself
demonstrates large host variance. Each candidate completed all graded HTTP
requests but had `329/330` below the 400ms TTFT ceiling. First-token matching
against the paired control was `268/330`, so token drift is material.

The local `.95` target could not start: vLLM saw `13.89/15.44 GiB` free and
requested `14.67 GiB`. This is a display-workload limitation, not a candidate
quality result. R002n therefore retains `in_proj` only as a clean-target
research candidate and creates no image, Compose or Portal attempt.

A reversed-order third in-only process was started after this screen, but the
local display workload left insufficient memory for a 20.8k-token piecewise
CUDA-Graph shape. vLLM requested `512MiB` with only `409.69MiB` free and the
engine exited after six requests. Its ERS is invalid and excluded; this does
not alter the two valid candidate replays or the decision.

## Phase-Aware ShortConv Projection (Subround 002o)

R002o tested whether R002n's FP8 projection could be isolated to the decode
phase while preserving native BF16 prefill. The implementation keeps one model
and one scheduler: native and FP8 projection modules initially alias the same
checkpoint parameter, then vLLM post-processing quantizes only the FP8 module.
All ten ShortConv layers passed the resulting storage invariant:

```text
native projection: BF16
decode projection: FP8 E4M3 with scale
shared before quantization: yes
distinct storage after quantization: yes
```

Startup exercised real `prefill_bf16`, `profile_fp8` and `decode_fp8` paths.
PIECEWISE-16 prefill and FULL-8 decode graphs captured successfully. The
candidate used about `0.11 GiB` more model memory and reduced reported KV
capacity by `22,069` tokens (`1.36%`) relative to its paired control.

At batch 8 over 64 clean decode-only steps, the all-layer candidate reduced
non-output-head child-kernel time from the control mean `119.879ms` to
`111.825ms`, a `6.72%` diagnostic gain. The selective layer mask `0,1` reached
only `118.865ms`, or `0.845%`, and failed the registered `1%` profile gate.

The all-layer quality proxy retained `21/100` accuracy in both repeats, with
zero correctness changes and zero reference-correct-to-candidate-wrong
transitions. The authoritative exact full replays were negative:

```text
variant                 ERS      TTFT p50/p95       TPOT p50/p95   <=400ms
phase-aware run 1       53.2023  127.170/379.093    2.665/3.828    316/330
R002k control           55.4200  118.963/326.315    2.599/4.202    323/330
phase-aware run 2       52.9304  121.644/495.384    2.553/8.072    301/330
```

All `990` graded observations completed successfully. Candidate mean was
`53.0663`, `2.3537` points below the immediate control. Dynamic activation
quantization adds ten extra FP8 quantization launches per decode step and the
dual projection consumes extra memory; these mechanisms are consistent with
the request-level tail, although this experiment does not uniquely attribute
their individual costs. Child-kernel sums therefore cannot substitute for a
full replay. Close R002o without an image, Compose or Portal attempt and retain
R002k.

## FP8 Activation Scale Audit (Subround 002p)

R002p measured the dynamic activation side of the 44 existing online-FP8
linears. Clean batch `1/2/4/8` traces put fused activation quantization at a
mean `4.6195%` of child-kernel time. Corrected exact-shape microbenchmarks found
oracle static quant-plus-MM about `6.53-6.60%` faster, but this overestimated
production because vLLM already fuses dynamic quantization with neighboring
RMS/add/SiLU work.

The integrated static CUTLASS path loaded all 44 scales, preserved the R002k
head, captured PIECEWISE-16 and FULL-8 graphs and served valid responses. A
clean control/static/control batch-8 profile measured:

```text
control mean:                2.1928129 ms/step
force-static:               2.1658492 ms/step
gain:                        1.2296%
control drift:               0.4006%
dynamic quant producers:    44 -> 0 per step
```

Graph capture prevented a Python scheduler-state or padded-row selector from
switching only decode to static after compilation. The all-graph diagnostic
then failed quality. Prefill-only scales scored `20/100` and `18/100`; scales
recalibrated over 16 prompts plus 496 decode calls per module scored `19/100`
and `18/100`, versus R002k at `21/100`.

Practical conclusion: keep dynamic per-token activation quantization. The
measured static ceiling is too small to justify the quality loss or a custom
graph-native phase selector. R002p stops before full replay and creates no
image, Compose or Portal artifact.

## Graph-Native FP8 Fusion Audit (Subround 002q)

The exact vLLM image accepts nested `PassConfig` fusion controls. Startup
screening under the 3-CPU/8GB local envelope produced:

```text
norm:      5 non-zero RMSNorm+quant rewrites, graph FULL_AND_PIECEWISE
act:       0 activation-quant rewrites, graph FULL_AND_PIECEWISE
norm+act:  same 5 norm rewrites, activation pass 0
attn:      0 attention/MLA rewrites, graph changed to FULL
```

The only profile candidate, `fuse_norm_quant`, was bracketed with control A/B:

```text
control A:       2.2050282 ms/step
candidate:       2.1898797 ms/step
control B:       2.2010363 ms/step
control mean:    2.2030322 ms/step
drift:            0.1812%
gain:             0.5970%
```

The pass halves the labeled dynamic-quant producer count but moves much of the
time into an `other` family; FP8 GEMM remains 44 calls per step. The complete
gain misses the registered 1% gate, so no quality/replay/package work was
authorized. R002q is closed and R002k remains the only submission artifact.

## FP8 Linear Backend Screen (Subround 002r)

The exact vLLM registry shows that automatic dispatch already selects
`CutlassFP8ScaledMMLinearKernel`. Backend names are not interchangeable for
this representation: FlashInfer requires per-tensor activation scales and
fails with the active dynamic per-token layout; Marlin rejects native-FP8
GPUs without a test-only force variable; Triton and DeepGEMM expose no
matching per-tensor-online class. FlashInfer additionally requires SM100+, so
it cannot target H200 SM90.

Explicit CUTLASS and Torch both retained `FULL_AND_PIECEWISE`, the R002k head,
and completed `8/8` startup requests. Torch selected its channel-wise fallback
and was bracketed at batch 8 over 64 clean decode steps:

```text
CUTLASS A:       2.2056096 ms/step
Torch:           2.2047690 ms/step
CUTLASS B:       2.2021816 ms/step
control mean:    2.2038956 ms/step
drift:            0.1555%
gain:            -0.0396%
```

Torch's two raw GEMM families total `1.103085ms/step`, below CUTLASS FP8 at
`1.129793ms/step`, but its separate FP32 scale/dequant/output-conversion path
consumes the difference. The complete-step 1% gate fails. R002r stops before
quality/replay/package work and keeps automatic/CUTLASS dispatch.

## Online Block-FP8 Target Transfer (Subround 002s)

R002s confirmed that `fp8_per_block` is a genuine online path: BF16 checkpoint
weights become static E4M3 128x128 blocks during load and activations use
dynamic 1x128 groups. All LFM2.5 linear shapes pass the required alignment.
Local SM120 auto selects block CUTLASS; exact SM90 source selects a dynamic
FlashInfer/DeepGEMM wrapper.

All startup variants retained the R002k head, both CUDA Graph modes and `8/8`
responses. The local control/block/control batch-8 profile measured:

```text
per-tensor A:       2.2207708 ms/step
block CUTLASS:      2.3176776 ms/step
per-tensor B:       2.2243989 ms/step
control drift:      0.1632%
candidate gain:    -4.2785%
```

Repeated GPQA proxy accuracy was `18/100` and `18/100` versus R002k `21/100`.
Each candidate process lost four reference-correct cases and gained one. A
diagnostic full replay completed all `330/330` graded requests but scored
`49.3582` versus the R002k anchor `58.3989`; TTFT p50/p95 moved from
`106.503/336.066ms` to `133.855/423.597ms`, TPOT p50/p95 from
`2.477/3.649ms` to `2.749/5.126ms`, and peak active grew `7 -> 11`.

The local speed result applies to the CUTLASS fallback, not the separate H200
kernel. The quality loss applies to the representation and therefore blocks a
target risk trial under the registered gate. Close R002s without an image,
Compose, or Portal artifact and retain per-tensor FP8 in R002k.

## Small-M FP8 Kernel Feasibility (Subround 002t)

R002t preserved R002k's exact online per-tensor FP8 weights, dynamic per-token
activation scaling and BF16 outputs. The clean R002r profile showed Torch raw
GEMMs at `1.1030849ms/step` versus CUTLASS at `1.1297934ms/step`, an apparent
`0.0267084ms` advantage and only `1.2119%` absolute zero-overhead ceiling.
Because the complete Torch path was already slightly slower, more than
`83.07%` of its extra overhead had to disappear to clear a 1% whole-step gate.

Two exact-shape implementations were screened. Direct BF16 `_scaled_mm`
needed a contiguous `[1,N]` expansion of each scalar weight scale for M>1.
The output was bit-exact but the weighted path lost `35.51%`, `117.97%`,
`123.68%` and `132.98%` at M=1/2/4/8. A custom Triton FP32 dequantize/BF16
cast epilogue also matched every tensor and row argmax. Its best M=8 result was:

```text
custom epilogue, 44 calls:       0.320015ms
allowed epilogue budget:         0.004669ms
budget multiple:                 68.53x
custom complete E2E:             1.167806ms
CUTLASS complete E2E:            0.856557ms
custom gain:                    -36.34%
```

Production Inductor already fuses much of the Torch fallback epilogue into
consumers, so an isolated compiled-epilogue comparison cannot be projected
onto live serving. The valid end-to-end and live-budget gates both fail.
R002t stops before server integration, quality, replay, image, Compose or
Portal work. Keep automatic/CUTLASS dispatch and R002k.

## BF16 ShortConv Kernel Screen (Subround 002u)

The ten BF16 ShortConv in/out projection pairs consume `0.434604ms` of a
`2.203896ms` batch-8 decode step (`19.7198%`). This gives a real opportunity:
only `5.0710%` family gain is needed for +1% complete decode.

The opening exact-shape loop found a custom Triton 64x64 tile `11.31-20.50%`
faster than standalone native linear. That result was rejected as an
integration baseline because each cell repeats one 8-24 MiB weight. The
confirmation instead cycles ten 2048->6144 and ten 2048->2048 matrices,
exactly 320 MiB:

```text
M8 cyclic native:                 0.456264ms/20 calls
M8 best Triton:                   0.445971ms/20 calls
same-process family gain:         2.256%
production projection anchor:     0.434604ms/step
projected complete-step gain:    -0.516%
```

Cyclic native is only `4.98%` above the independent production anchor, so the
residency model is credible. Numerical behavior is safe for a screen: worst
Triton NRMSE `0.00024876`, minimum exact fraction `0.997233`, all finite. The
performance gate alone fails. Close R002u without an overlay, quality run,
replay, image, Compose or Portal artifact. A future ShortConv kernel must fuse
projection with a consumer and remove more traffic than GEMM replacement.

## ShortConv in_proj FP8 Target Qualification (Subround 002v)

R002v isolates the only positive R002n factor: use the active online
per-tensor FP8 quantization method for all ten ShortConv `in_proj` matrices and
leave every `out_proj` BF16. The `.84` host-overlay bracket scores
`57.9057/55.6036` versus control `54.4081`, with zero graded failures.

The first package OOMed after six graded requests because a 512 MiB piecewise
allocation could not use fragmented reserved memory. Baking
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` resolves that failure. The
immutable packaged bracket is:

```text
candidate A: 52.6084 ERS, TPOT p50/p95 2.747/3.891ms, 330/330
candidate B: 53.2367 ERS, TPOT p50/p95 2.697/3.829ms, 330/330
control A:   49.5424 ERS, TPOT p50/p95 2.960/5.133ms, 330/330
mean candidate delta: +3.3802 points
```

The fixed GPQA proxy remains `21/100` versus R002k `21/100`; aggregate drop is
zero, but one reference-correct answer is lost and one wrong answer improves.
This is an official-gate pass with explicit numerical drift, not a bit-exact
optimization.

Public digest `sha256:eb9f961172f0668ec62af5ae66e1689670c9ec8d3249d41688f9cb06d399f701`
and Part 2 Attempt 05 are frozen as an H200 probe. The two Portal results are
`60.19` and `59.34`, with TTFT `50/90ms` and `61/97ms`, TBT `4ms` in both
runs, and `6/420` and `5/420` failures. The pair mean is `59.765`, slightly
below R002k's exact-pair mean `59.845`; close the target probe and preserve
R002k Attempt 04 as the incumbent/rollback.

## ShortConv Decode Epilogue Fusion (Subround 002w)

R002w preserves all weights and BF16 state while fusing packed `B*x`, the
sequential FP32 width-three accumulator, recurrent state shift and `C*conv`
into one pure-decode Triton kernel. The native prefill/mixed paths remain
unchanged. Runtime state/index contracts are `(slots,2048,2)` stride
`(8192,1,2048)` and `(M,1)` stride `(1,1)`.

All exact-layout M1/M2/M4/M8 outputs and states are bit-exact over 16 updates.
At M8, ten epilogues fall from `0.285780ms` to `0.091605ms`, projecting
`8.811%` complete-step gain. Ten graph-safe staging buffers total only
`320 KiB`. FULL_AND_PIECEWISE capture passes, and a batch-8 512-token repeat
improves median decode throughput `3331.44 -> 3426.68 token/s` (`+2.86%`).

The complete 420-request bracket is negative:

```text
candidate A: 50.5693 ERS, TTFT p50/p95 132.537/357.351ms, TPOT 2.851/4.412ms
control A:   50.6229 ERS, TTFT p50/p95 132.152/368.959ms, TPOT 2.816/4.274ms
candidate B: 50.1385 ERS, TTFT p50/p95 132.893/411.414ms, TPOT 2.879/4.316ms
```

Every graded request succeeds, but candidate mean is `0.2689` points below
control and neither run is positive. The epilogue saving is real but does not
dominate scheduler/prefill/process variance. Close R002w, skip GPQA and H200
by dependency, publish no image and retain R002k.

## Production Graph-Step Attribution (Subround 002x)

R002x replaced R002w's eager projection with CUDA Events around the complete
`execute_model + sample_tokens` production boundary and concurrent Kineto
child-kernel traces. Control/candidate/control each ran 64 clean pure-decode
steps at M1/M2/M4/M8 with zero request failures and matched
FULL_AND_PIECEWISE capture.

Exact full-replay scheduler telemetry recorded `47,657/48,263` active steps
as pure decode (`98.744%`) and `83,109/83,610` decode rows on the eligible
pure path (`99.401%`). Mixed fallback therefore cannot explain R002w's replay
loss. Scheduler-weighted complete GPU mean is
`2.390551 -> 2.306848ms` (`+3.501%`) with `1.045%` control drift, but the
signal is concentrated at M1/M2:

```text
M1: +3.496% complete GPU, 2.281% drift
M2: +5.100% complete GPU, 1.445% drift
M4: +0.851% complete GPU, 0.851% drift
M8: -1.687% complete GPU, 2.633% drift
```

The candidate removes about 30 kernels per step. Direct M8 ShortConv saving
is only `0.039981ms/step`, or `1.705%` of child-kernel time; the eager
`0.194175ms` estimate was `4.86x` too large because production CUDA Graph
already amortizes most launches. The registered M8-positive gate fails.
Close standalone ShortConv fusion, create no larger fusion/image/Compose/
quality run/Portal attempt, and retain R002k.

## Cached-Turn Prefill Attribution (Subround 002y)

R002y measured the complete first-prefill through first-sample GPU boundary
for all six append-only turns. Three clean processes formed timing A, Kineto
and timing B after all 90 public primer requests exercised every published
shape. Weighted A/B GPU drift is `1.418%`.

Prefix caching behaves correctly. Initial computed tokens progress
`0/4000/8192/12384/16592/20784`, while each request schedules only
`4000/4199/4206/4213/4204/4210` new tokens. Even so, complete GPU time grows
`60.801 -> 114.685ms`. Attention grows `4.825 -> 54.742ms` and explains
`92.64%` of total turn-0 to turn-5 growth; FP8 and BF16 GEMMs are nearly flat.

The weighted family split is `45.87%` FP8 GEMM, `34.69%` attention and
`16.09%` BF16 GEMM. Turns 1-5 also pay a small second aligned-state model step
of `2.662-2.923ms`, a `2.296ms` equal-turn ideal ceiling.

Exact source prevents a false transfer claim. RTX 5080 SM120 cannot run
FlashAttention with FP8 KV and therefore profiles FlashInfer. H200 SM90 has a
packaged FA3 binary, FA3 explicitly supports FP8 KV on family 90, and automatic
priority selects FlashAttention before FlashInfer. TRT-LLM attention requires
SM100 in this image. Complete the attribution, create no serving artifact and
open only H200-specific FA3 prefill qualification; R002k remains incumbent.
