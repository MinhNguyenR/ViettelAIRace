# Effective Technique Watchlist

Purpose: keep every technique with useful signal, even if it is not promoted to
submission yet.

This file is not a roadmap. A technique becomes roadmap work only after it is
linked from `docs/planning/phase1_roadmap.md` or a round plan.

## Current Priority After R033

R021a promoted packaged output-only FP8 tied-head top-k4 into local I007, and
R021b selected seq16. Attempt 12 now validates that exact public artifact on
H200 at score `51.37`, TBT `15ms`, accuracy drop `0` and zero failures. The
otherwise matching I006 seq16 rerun scores `51.59`, so native head remains the
target score baseline.

Attempt 13 R022d scored `49.45`: TTFT p50/p95 improved to `4066/5367ms` and
passed SLO rose to `20/120`, but TBT regressed to `21ms`. The local exact-trace
Score-100 path did not transfer materially and is closed. Native MTP was then
measured and rejected despite real `67.643%` acceptance: ERS fell `15.58%`,
TTFT p50 rose `96.68%`, TPOT p99 rose `15.64%`, and throughput fell `8.90%`.
R023a telemetry found only `2,240` duplicate tokens. R023b admission-only
singleflight was rejected because its corrected hold prevented the follower
cache-publication chain and increased compute. R023-R027 subsequently closed
generic online reuse, scheduler overlays, graph lattices and active-set
interpolation. R028 now closes fixed prefill budget `4096` on exact I006; the
next technique must be a new causal primitive and preserve the very-fast TTFT
cohort, not another scalar sweep. R029 also closes its Tree-Gated derivative:
robust oracle gain was only `+0.351%`, and depth-2 leave-one-wave-out
performance was negative. R030 corrects the metric basis for any next cache
idea: raw prefix query/hit counters include repeated lookup attempts. True
request cache attribution changed only `-0.143%` in R028, so do not open a
cache-loss repair from the raw `-33.87%` hit counter.
R031 then resolves the remaining wall-phase ambiguity: `4096` lowers queue
time but increases scheduled-to-first-token service because fewer scheduler
steps carry much larger peer-prefill batches. The six-run bidirectional gate
rejects fixed and dynamic budget work. The only retained idea is an offline
opportunity audit for final-chunk/service isolation, not an implementation.
R032 ran that audit and failed: half recovery was negative in `3/9` process
pairs, waves 1/5 and the `<=500ms` cohort. Final-chunk isolation is closed;
the next candidate must come from a separate repeated primitive.

Reason:

- I004 repaired wave20 decode with FP8 weights/KV, exact graph20 and BF16
  Mamba state while retaining local quality proxy `1.0`.
- R009 synthetic shape warmup improved ERS `0.581198 -> 0.617657` against the
  control midpoint and removed all observed timed JIT.
- R010 moved the same legal warmup before public readiness; packaged I006
  reached ERS `0.619885` versus midpoint `0.589324`.
- Attempt 11 improved Attempt 10 score `44.09 -> 46.03`, TTFT p50
  `4522 -> 4306ms` and TBT `21 -> 20ms`; warmup transferred but was not enough.
- Packaged top-k4 reached ERS `0.626978` versus control midpoint `0.623107`,
  TPOT p95 `14.492ms`, throughput `819.557 tok/s` and penalty proxy `1`.
- Seq17 added only `0.000747` ERS over a same-session seq16 repeat and consumed
  `0.863ms` more TPOT-p99 margin; keep seq16.
- Attempt 12 improves the fully recorded Attempt 11 score `46.03 -> 51.37`
  and TBT `20 -> 15ms`, while TTFT p95 worsens `5334 -> 7130ms`. Decode and
  hidden quality transferred; TTFT tail/admission is the remaining target.
- I006 seq16 scores `0.22` above I007 seq16. This is only `0.43%` and may be
  Portal jitter, but it is enough to block tied-head promotion without a
  repeat; target optimization returns to native head.
- Brain 1 remains analysis-only; no Tree-Gated serving policy has shown causal
  ERS improvement under the final cgroup/runtime.

R021c answer: head-off seq17 loses ERS `0.530%`, adds zero fast-TTFT requests
and raises TPOT p99 `0.873ms`. Preserve I006 seq16/native head and close the
concurrency-only branch. R022e later proved MTP capability, but R022f rejected
its performance and closed dependent concurrency/tied-head work.

## Technique Status Table

| Technique | Main target | Local signal | Status | Use now? |
|---|---|---|---|---|
| Explicit prefix caching | TTFT, prefill reuse | Strong positive. No-flag/default was much worse. | Promote for vLLM candidates | Yes |
| FP8 E4M3 KV cache | TPOT, KV capacity, TTFT secondary | Attempt 08 official accuracy drop was `0`; performance still depends on concurrency. | Quality-approved, policy-sensitive | Yes with H200 TBT guard |
| High VRAM allocation | KV capacity | Round 012a `0.82 -> 0.90` increased KV tokens +20.24%, but no score gain at `MAX_NUM_SEQS=4`. | Capacity watchlist | Pair with scheduler/concurrency |
| Context `32768` | context/graph range | Exact fit margin `5170`; paired TTFT/TPOT gains `0.496%/0.331%`, no VRAM/ERS gain. | Enabling local baseline | Keep for Score 100 rounds, not standalone Portal |
| Chunked prefill explicit on | Startup stability, long-prefill handling | Positive guardrail; disabling/concurrent partial prefill was bad. | Keep explicit | Yes |
| `MAX_NUM_SEQS=4` safe decode profile | TPOT and ERS preservation | Best score region while TTFT is outside scoring range. | Current safe baseline behavior | Yes as control |
| FP8 KV + `MAX_NUM_SEQS=10/11` | TTFT/admission | Strong RTX paired TTFT gains, but H200 seq8 already produced TBT `21ms`. | Rejected for H200 | No |
| FP8 KV + `MAX_NUM_SEQS=6` | TBT-safe TTFT trade-off | H200 TBT matched projection (`17.94 -> 18ms`), but score `49.98`, TTFT p50 `21496ms`, passed SLO `3/120`. | Valid calibration; rejected winner | No further concurrency-only attempt |
| Wave20 + BF16 weights + FP8 KV | admission/TTFT | TTFT -58.6%, but TPOT p95/p99 `27.6/30.5ms`, 99/120 over floor. | R003 control only | No until weight/kernel speedup |
| Online FP8 linear weights + wave20 + FP8 KV | TTFT/TPOT/throughput | ERS `0.573075`, TTFT p50 `1497ms`, TPOT p95 `19.95ms`; GPQA `0.34`. | Provisional integrated champion | Yes in research; quality/first-wave guard |
| Exact full CUDA Graph size 20 | wave20 TTFT/TPOT | A/B/A: TTFT wins `97/120`, TPOT wins `119/120`, ERS `+0.005212` over controls. | Keep in I003 | Yes with wave20 |
| Online prefix follower hold/release | cold prefill duplication | Native already caches `79.536%`; async overlay saved zero compute and lost ERS `0.002802`. | Rejected | No |
| Native runtime-aligned prefix cache | prompt compute/reuse | Native compute `495,102` exactly matches the `1072`-token block-trie lower bound. | Proven optimal at current block size | Keep |
| BF16 Mamba SSM cache | TTFT, TPOT, cache granularity | Block `1072 -> 560`; ERS `0.586594`; TTFT/TPOT paired wins `99/112`; GPQA flips `0/100`. | I004 promoted | Yes, with FP32 rollback |
| Fixed prefill token budget `4096` | TTFT/throughput | I004 hinted Pareto behavior; exact-I006 R028 confirmed ERS `0.601842` vs `0.610570` midpoint despite fast count `78 -> 84` and lower raw TPOT. | Rejected globally on I006 | No; keep resolved/default 2048 |
| Tree-Gated dynamic `2048/4096` budget | TTFT score shape | Robust oracle only `+0.351%`; stump CV nearly zero and depth2 CV `-0.000224` with negative robust gain. | R029 rejected before GPU | No; classification cannot recover global scheduler interaction |
| Prefix metric source attribution | measurement correctness | R030: lookup hits `-33.87%` with queries `-34.54%`, but actual cache attribution only `-0.143%`; prefill metric is wall phase time. | Required methodology | Yes before any cache/prefill conclusion |
| First-token lifecycle attribution | TTFT mechanism correctness | R031: `4096` queue `-138.716ms`, scheduled-to-first-token `+122.927ms`, ERS `-1.80%`; service penalty in `6/6` waves. | Measurement method retained; candidate rejected | Use observer for diagnosis, not `4096` |
| Final-chunk service isolation | TTFT first-token service | R032 50% net recovery: mean `+0.775%`, worst `-0.497%`, `6/9` pairs; waves 1/5 and `<=500ms` lose. | Offline opportunity rejected | No R032b/GPU implementation |
| Lifecycle dual token budget | cold prefill plus TPOT guard | A/B/A ERS `0.580253` vs midpoint `0.582919`; prompt compute +0.98%, prefill time +8.20%. | Rejected after valid runtime test | No |
| Automatic hybrid backend selection | GDN plus full-attention kernels | Local auto chose Triton/FLA GDN + FlashInfer attention; forced TRITON_ATTN lost all paired TTFT/TPOT requests. | Keep native auto | Yes |
| Forced GDN FlashInfer | Hopper GDN prefill | RTX 5080 startup accepted flag but explicitly fell back to Triton/FLA. | H200-only unresolved; not a local result | Do not force |
| `MAX_NUM_SEQS=12` aggressive queue profile | TTFT reduction | TTFT lower, but `10/120` requests crossed TPOT `20ms` and ERS fell to `0.497559`. | Boundary/negative result | No as default |
| `MAX_NUM_BATCHED_TOKENS=16384` | TPOT center | Seq6 A/B/A improved p50/p95 but worsened TPOT p99, TTFT, throughput and VRAM. | Rejected for Attempt 09 | No with seq6 |
| Synthetic occupancy ramp | cold-start/JIT/allocator warmup | Noisy but sometimes improves TTFT/TPOT tail. | Needs focused repeat | Lab only |
| Endpoint synthetic warmup | TTFT/JIT startup | Local R009/R010 positive; Portal score `44.09 -> 46.03`, TBT `21 -> 20ms`. | H200 validated, below champion | Keep in I006 research base |
| Output-only FP8 tied head + BF16 top-k4 | TPOT and decode bandwidth | R021a ERS `0.626978` vs midpoint `0.623107`; Attempt 12 score `51.37`, TBT `15ms`, accuracy drop `0`; I006 native head scores `51.59`. | H200-safe watchlist, not score winner | No in next control; retain behind feature flag |
| Non-lossy Triton fused BF16 greedy head | TPOT and decode launch/memory traffic | Local ERS `56.7287 -> 58.3989`; H200 Attempt 02/04 `58.17 -> 60.75`, TTFT p50/p95 `58/82 -> 44/80ms`, failures `8 -> 6`; 20 live audit batches exact. | H200-validated incumbent | Yes, behind strict fallback |
| Selective final LFM2 RMSNorm after logits-row selection | long-prefill TTFT | Exact BF16 op improves `21.917%`, only `0.149ms` absolute. Full B/A/B candidate mean is `+0.7084` points, but repeat 2 loses and TPOT p95 reaches `4.568ms`. | Correct but below process variance | No standalone submission; retain only for a larger prefill bundle |
| I006 seq16 active set | TTFT/TPOT balance | Portal score `51.59`; R021c head-off seq17 ERS `0.604517` lost to midpoint `0.607739`, fast bucket unchanged. | Current H200 score champion | Yes; do not continue seq17/18 concurrency tuning |
| Qwen3.5 native MTP-1 | TPOT, decode steps, concurrency headroom | Capability passed with fixed checkpoint; acceptance `67.643%`, but ERS `-15.58%`, TTFT p50 `+96.68%`, TPOT p99 `+15.64%`, throughput `-8.90%`. | R022f rejected | No; graph downgrade and draft/verify overhead dominate |
| Seq18 plus exact graph18 | H200 decode guard | Local ERS `0.593090`; TPOT p99 `18.457ms`; `814.132 tok/s`; calibrated p50 target missed by `0.205ms`. | Conditional fallback | Only after a kernel gain |
| Fused residual/RMSNorm/FP8 quant | launch and HBM traffic | Native pass replaced zero graph patterns. | R014 rejected | No |
| Fused Mamba/GDN decode | TPOT across 18 recurrent layers | Geometry produced only `1.0015x`; native path retained. | R015 rejected | No |
| Mixed FP8 Mamba state | state bandwidth and cache block size | FP8/INT8 halve storage but are slower and drift over recurrent updates. | R017 skipped after R015 | No |
| HybridRadixState | TTFT and hybrid prefix reuse | Ideal block16 saves only `5.89%` compute; dense snapshots exceed VRAM. | R018 rejected | No |
| Lifecycle two-policy scheduler | TTFT before first token, TPOT afterward | Generic dual budget failed and R018 created no new causal state action. | R020 skipped | No |
| Engine-level arrived-request hybrid singleflight | TTFT and duplicate prefill | R023a found only `2240` duplicate tokens; R023b corrected hold raised compute `456542 -> 457662` and duplicate `2240 -> 3360`. | Rejected admission-only path | No; requires invasive in-flight hybrid-state publication for a `0.49%` ceiling |
| ERS-aware marginal lifecycle scheduler | TTFT/TPOT score balance | Planned utility uses only arrived length/cache/wait/decode EWMA; earlier static dual-budget policy was negative. | R023e conditional | Only after stable integrated control |
| Embedded exact-trace preload/render cache | Local TTFT | Local ERS `1.000000`, but Attempt 13 scored `49.45`, TBT `21ms`, below I006 `51.59`. | H200 target-negative and compliance-unresolved | Closed; no derivative |
| Brain 1 RAM/VRAM ranker | policy/risk analysis | Strong classifier/ranker; Round 010 no ERS gain. | Analysis tool | Not serving yet |
| Brain 1 admission policy | TTFT via priority | Proxy reduced TTFT p50 but no ERS gain; still above TTFT ceiling. | Negative/guardrail result | No |
| SSD/offload memory tiering | memory pressure | Environment path uncertain; SSD not guaranteed. | Parked | No |

## Notes By Technique

### Explicit Prefix Caching

Observed effect:

- Keeps shared system/prefix reuse active.
- Avoids a large default/no-flag regression seen in Round 006.

Use when:

- Building any vLLM candidate.
- Running local replay against `trace-round1.jsonl`.

Avoid:

- Treating prefix cache as semantic/answer caching. It is serving-side prefix
  reuse only.

### Online Prefix Follower Hold/Release

Observed effect:

- Native live counters reported `1,924,240` cache-hit tokens and `495,102`
  compute tokens over `2,419,342` prompt tokens.
- The hybrid cache resolves to `1072`-token blocks; the exact trie lower bound
  at that granularity is also `495,102` compute tokens.
- Holding 19 cold followers until a runtime-shared `5360`-token prefix was
  ready did not change either count.
- The corrected async overlay reduced ERS by `0.002802`; the initial sync
  overlay was much worse because it bypassed vLLM AsyncScheduler behavior.

Use when:

- Only as a preserved research harness for a future engine whose measured
  local-compute counter proves duplicate work is actually removed.

Avoid:

- Inferring removable work solely from a tokenizer trie.
- Passing a custom base Scheduler class without preserving AsyncScheduler.
- Holding requests when native prefix publishing already reaches the same
  cache-hit count.

### BF16 Mamba SSM Cache

Observed effect:

- Changes only temporal recurrent state from FP32 to BF16; conv state was
  already BF16.
- Reduces the aligned hybrid cache block from `1072` to `560` tokens and page
  size from `1,097,728` to `573,440` bytes per layer.
- A/B/A improved ERS `3.77%`, TTFT `16.47%` paired mean and TPOT `7.91%`
  paired mean.
- Local prompt compute fell `7.788%`; queue-time sum fell `24.774%`.
- Local 100-question GPQA produced `0/100` prediction flips versus FP32 state
  and repeated deterministically at `34/100`.

Use when:

- Serving Qwen3.5 hybrid with FP8 attention KV under the fixed vLLM 0.22.1
  image.
- Prefix reuse is limited by Mamba-aligned block granularity.

Avoid:

- Assuming local GPQA certifies the hidden BTC subset.
- Combining another lossy state/weight change without a fresh quality A/B.
- Replacing align mode with all mode; final-wave boundary retention exceeds
  the measured local pool estimate.

### FP8 E4M3 KV Cache

Observed effect:

- Improves KV capacity and decode profile.
- Local full replay reached ERS `0.5` with excellent TPOT.
- Local GPQA proxy showed FP8 E4M3 KV accuracy `0.34` versus auto KV `0.38`
  on the public proxy set, still within the no-penalty range against baseline
  `0.4` in that proxy.

Use when:

- Performance target is TPOT/KV pressure.
- Accuracy proxy is rerun after any additional lossy change.

Avoid:

- Combining blindly with weight quantization until GPQA proxy is rerun.

### High VRAM Allocation

Observed effect:

- Round 012a completed a same-session sweep from `GPU_MEMORY_UTILIZATION=0.82`
  to `0.90`.
- KV capacity increased clearly:
  - `0.82`: `883850` KV tokens, `26.97x` max concurrency for 32768-token requests
  - `0.90`: `1062745` KV tokens, `32.43x` max concurrency for 32768-token requests
  - delta: `+178895` KV tokens, about `+20.24%`
- ERS stayed `0.500000` and latency did not improve at `MAX_NUM_SEQS=4`.

Use when:

- Testing H200/MiG capacity assumptions.
- Pairing with higher scheduler concurrency, memory-aware scheduling, or
  long-prefill changes that can use the extra KV budget.

Avoid:

- Raising VRAM blindly as a standalone score optimization.
- Using too-high values locally without checking free VRAM first.

### Chunked Prefill

Observed effect:

- Explicit `chunked prefill on` is safe.
- Disabling chunked prefill or enabling multiple concurrent partial prefills
  failed readiness or behaved badly under local limits.

Use when:

- Serving long-context trace with vLLM.

### Scheduler Concurrency

Observed effect:

- With auto KV, earlier raw concurrency increases lowered TTFT but damaged TPOT
  enough to reduce ERS.
- With FP8 E4M3 KV, the first Attempt 09 local sweep appeared to move the sweet
  spot upward:
  seq10 improved TTFT for `120/120` paired requests versus seq8, and seq11
  improved `120/120` versus seq10 while keeping max TPOT `19.772ms` in its
  controlled bracket.
- Official H200 invalidated this translation: seq8 already reached TBT median
  `21ms` and score `45.29`. Seq10/11 are therefore rejected for H200 despite
  local gains.
- The lower seq4/6/8 bracket selected seq6: local TTFT p50/p95
  `10961.745/17544.772ms`, TPOT p50/p95 `9.816/11.986ms`, with H200 TBT
  projection `17.94ms`.
- Attempt 09 returned TBT `18ms`, validating the decode projection, but TTFT
  p50/p95 stayed `21496/24498ms`, passed SLO was `3/120`, and score `49.98`
  did not beat seq4 Attempt 05 at `50.13`.

Use when:

- Testing TTFT-specific intervention.
- Running hybrid/proxy analysis.

Avoid:

- Promoting higher concurrency without a TPOT guard.
- Copying local memory `0.74/0.77` into H200 Compose; official memory remains
  `0.95`.
- Spending another Portal attempt on seq4/6/8 concurrency alone; the H200
  branch is now measured and closed.

### Exact Batch-20 CUDA Graph

Observed effect:

- Default vLLM captured full graphs only through batch 16 and skipped 20.
- Adding 20 while retaining default sizes resolved a true full graph at 20.
- The same-bracket candidate won TTFT on `97/120` requests and TPOT on
  `119/120`; ERS improved by `0.005212` over the A/B midpoint.
- It did not solve cold wave-1 prefix work: wave 1 still overlapped by
  `3.734s` and had eight TPOT values above `20ms`.

Use when:

- Running the measured `max-num-seqs=20` FP8 integration.

Avoid:

- Treating graph capture as prefix caching or expecting it to remove prefill
  work.
- Adding more graph sizes without observing the corresponding active shape.

### Fixed `MAX_NUM_BATCHED_TOKENS=4096`

Observed effect:

- The early I004 screen showed a Pareto tradeoff but was not enough to settle
  the champion configuration.
- Exact-I006 R028 repeated default `2048`, candidate `4096`, then default
  `2048`. Candidate ERS fell `1.43%` despite six more TTFT values under
  `1500ms`, better p95, better raw TPOT and higher throughput.
- TTFT score fell `0.221140 -> 0.203685` because `4.5` control-midpoint
  requests left the `<=500ms` cohort; every wave lost request score.
- R030 corrected the aggregate mechanism: raw lookup hits fell `33.87%` while
  queries fell `34.54%`, so lookup hit rate rose `0.800pp`. Actual request
  cache attribution fell only `0.143%`; prefill-phase wall time rose `48.14%`
  but is not GPU compute. TPOT score was already saturated at `1.0`.

Decision: reject fixed `4096` globally on exact I006. Keep runtime-resolved
`2048`; do not use fast-count or percentile improvements alone as promotion
evidence.

### Prefix Metric Source Attribution

Effect: prevents false optimization branches. Exact-I006 source shows prefix
query/hit counters are recorded before slot allocation, so a waiting request
can contribute on multiple scheduler iterations. They describe lookup traffic
only when paired as a rate.

How: capture prebenchmark and final Prometheus snapshots, subtract counters,
validate `local_compute + local_cache_hit + external = prompt_total`, and use
`prompt_tokens_by_source_total` for request-level attribution. Treat request
prefill timers as scheduled-to-first-token wall durations unless CUDA/CUPTI
profiling separately measures kernel compute.

Tradeoff: none at runtime; this is measurement discipline. It invalidates the
old R028 cache-loss explanation but not the R028 rejection, because ERS and
request-level frozen gates are unchanged.

### First-Token Lifecycle Attribution

Effect: separates admission queue delay from the service path after a request
first enters a scheduled batch. The R031 observer also counts how many
scheduler steps, peer-prefill tokens and decode tokens share that lifecycle.

How: use a measurement-only scheduler subclass that calls the stock scheduler
and state-update methods unchanged. Compare both run orders and aggregate by
process budget before attributing mechanism. Do not serialize prompt content,
token IDs, outputs or request identity.

Tradeoff and result: `4096` saved mean queue time `138.716ms`, but added
`122.927ms` after first scheduling and about `3,025.714` peer-prefill tokens
to each request's first-token path. ERS fell `1.80%`; service worsened in all
six waves and total score lost in five. Retain the observer as methodology,
reject the budget candidate, and require offline held-out evidence before any
final-chunk isolation experiment.

### Final-Chunk Service Isolation

Effect sought: retain `4096` queue throughput while limiting peer-prefill work
in a request's first-token-critical final prompt step.

Offline result: an ideal full service repair is positive, but this omits the
deferred work that must run elsewhere. With only `50%` net recovery, three of
nine cross-process pairs lose, waves 1 and 5 lose, and the `<=500ms` cohort
loses `1.019%`. All pairs require as much as `66.85%` recovery to clear a
modest `+0.5%` target.

Decision: reject before GPU. Do not collect more step telemetry or implement a
scheduler from this upper bound. Reopen only with a different measured
primitive that removes work rather than deferring it.

### Tree-Gated Dynamic `2048/4096` Selection

Observed effect:

- An optimistic per-request oracle reaches only `+0.457%` midpoint gain and
  `+0.351%` robust gain over exact-I006 controls.
- A legal-feature stump retained `+0.000301` midpoint but only `+0.000046`
  robust ERS in leave-one-wave-out.
- A depth-2 tree fitted `+0.001523` in-sample, then reversed to `-0.000224`
  midpoint and `-0.000764` robust held-out gain.
- Root features changed across folds and only `2/6` waves were positive.

Decision: do not increase model capacity, train XGBoost/C++, or implement a
global scheduler overlay for this decision. vLLM's token budget is global per
iteration, making the already-small per-request oracle unattainable as stated.
Tree-Gated remains useful only when a future causal action has a materially
larger and stable opportunity bound.

### `MAX_NUM_BATCHED_TOKENS=16384`

Observed effect:

- Repeatedly useful for TPOT/tail profile.
- Not a global baseline because it can worsen TTFT/VRAM.
- On seq6+FP8, A/B/A showed it improved TPOT p50/p95 about `7%` but worsened
  TPOT p99 `5.7%`, TTFT p50/p95 `15.7%/8.5%`, throughput `3.2%` and peak VRAM
  `11.2%`. It is rejected for Attempt 09.

Use when:

- TPOT p95/p99 is the active bottleneck.
- Pairing with warmup or runtime changes that already lower TTFT.

### Synthetic Occupancy Ramp

Observed effect:

- Some candidates improved TTFT/TPOT tail, but results were noisy and
  non-monotonic.
- Useful as warmup/JIT hypothesis, not as a proven serving optimization.
- Round 012 with full usable VRAM did not promote pure Torch warmup:
  - 2s warmup was slightly worse than cold 0.83.
  - 10s warmup improved TTFT p50 but worsened TTFT p95 and wall tokens/s.

Next test:

- Do not spend more time on pure Torch-only warmup unless paired with endpoint
  shape warmup or a real startup wrapper.

### Endpoint Synthetic Warmup

Observed effect:

- Round 012 used synthetic streaming `/v1/chat/completions` warmup, not trace
  prompts or precomputed answers.
- At `gpu_memory_utilization=0.82`, same-session cold control:
  - ERS `0.500000`
  - TTFT p50/p95 `18539.6135 / 32264.74145 ms`
- Endpoint warmup result:
  - ERS `0.5006135105048746`
  - TTFT p50/p95 `17849.3315 / 31575.4577 ms`
  - warmup TTFTs `938.286 ms`, `23.262 ms`, `22.321 ms`
- Score 100 R009 then used a stronger shape-specific plan: one synthetic
  `2710`-token chat request followed by twenty concurrent `13`-token requests,
  all with two output tokens. It raised A/B/A ERS to `0.617657` versus midpoint
  `0.581198` and moved five JIT kernels before timed traffic.
- R010 packaged this plan behind readiness. Public port 8000 is not bound until
  the one backend is ready and warmup completes. Packaged ERS was `0.619885`
  versus midpoint `0.589324`; readiness was `102s` and timed JIT count was zero.
- A no-warmup direct/relay/direct bracket found relay ERS `0.594723` versus
  midpoint `0.592591`, so the packaging layer did not consume the gain locally.

Use when:

- Building a healthcheck-gated wrapper where synthetic warmup happens before
  BTC benchmark traffic starts.
- Testing shapes that trigger vLLM/Triton JIT before production trace replay.
- Serving the frozen I006 package for one H200 transfer measurement.

Avoid:

- Using trace prompts/answers for warmup.
- Exposing `/v1/models` before warmup is complete.
- Loading a second model or adding request-aware logic to the transparent relay.
- Assuming the RTX gain transfers to H200 without a Portal result.

### Brain 1 Ranker And Admission

Observed effect:

- Round 009b sweet spot model:
  - request ROC AUC / balanced accuracy / F1:
    `0.972375 / 0.932974 / 0.931900`
  - run ROC AUC / balanced accuracy / F1:
    `0.946675 / 0.909237 / 0.900405`
- Round 010 best policy quality:
  - threshold `p_vram >= 0.15`
  - balanced accuracy `0.976190`
  - VRAM F1 `0.987342`
- Round 010 best hybrid proxy:
  - ERS stays `0.500000`
  - TTFT p50 improves from `18810.979500 ms` to `8016.931000 ms`
  - TTFT remains far above official `1500 ms`

Use when:

- Analyzing request priority, pressure, and potential scheduling rules.
- Designing a future real scheduler intervention.

Avoid:

- Adding Brain 1 admission to serving now.
- Treating Round 010 proxy as proof of official score gain.

### FP8 Tied LM Head With BF16 Top-K Repair

Observed effect:

- R012 measured the tied BF16 output projection at `1.1381ms`, or `28.6%` of
  a pure batch-20 decode step.
- R013 stores a separate output-only E4M3 copy while leaving BF16 embedding
  lookup untouched. Approximate logits select four rows, then those rows are
  recomputed exactly with the tied BF16 weight.
- Real hidden-state replay restored top-1 for `500/500` rows and first-token
  top-1 for `100/100` rows.
- Stable full-trace controls averaged ERS `0.514707`; top-k4 reached
  `0.532424`, with TPOT p95 `23.158ms` versus `24.214ms`.
- Repeated endpoint GPQA was `35/100` control and `33/100` candidate, giving a
  two-point drop and penalty proxy `1`.

Trade-off:

- Persistent FP8 copy costs about `485MiB` and reduced reported KV capacity by
  `6.63%` at the constrained local setting.
- The output-only copy remains a persistent memory cost, but packaged R021a
  measured only `+95MiB` peak VRAM versus the control midpoint in that bracket.
- H200 validation is still required; the package-level A/B/A and repeated
  local quality gates are now complete.

Use when:

- Decode vocabulary projection remains a measured hotspot.
- Exact greedy top-1 behavior is protected with BF16 repair.

Avoid:

- Quantizing tied embedding lookup itself.
- Shipping raw FP8 logits without quality repair.
- Shipping without the feature-off rollback or claiming an H200 gain from the
  RTX 5080 result alone.

### GDN Decode And Recurrent-State Quantization

Observed effect:

- Launch geometry sweep found only `1.0015x` at batch 20; native
  `BV32/warp1/stage3` is already appropriate.
- FP8 state cut storage nearly in half but ran at `0.9554x` and accumulated
  output NRMSE p95 around `0.09` over 200 updates.
- INT8 dynamic row scaling improved quality to cosine `0.999743`, but still
  missed the strict drift gate and every INT8 variant remained slower.

Decision:

- Keep BF16 recurrent state and the native GDN decode kernel.
- Do not open mixed-layer FP8 state until a fundamentally different kernel
  removes conversion/scale overhead.

### GDN Long-Prefill Backend

Observed effect:

- FLA chunk64 remains best; chunk32 was `0.8984x` at 6,931 tokens and chunk128
  is unsupported by the current triangular path.
- CuteDSL fails in the local image because Cutlass DSL CU13 files are
  inconsistent. FlashInfer's packaged GDN launcher rejects SM120 but supports
  target SM90.

Decision:

- Keep `gdn-prefill-backend=auto`. Local RTX 5080 resolves Triton/FLA; H200
  resolves the supported FlashInfer path.
- Do not force a local backend into the H200 Compose.

### R023 Generic Online Optimization Boundaries

Measured effect:

- Native retention is already sufficient: `100/100` later requests have full
  shared state and worst-active pool use is only `32.83%`.
- Exact post-arrival renderer/message reuse has `0/120` hits.
- Deadline-aware holding and `5ms` burst reordering both changed queue order
  without reducing the `456,542` model-compute tokens and lost A/B/A controls.
- One-storage E4M3 tied weights save `484-485MiB` statically, but runtime KV
  capacity and TPOT did not improve; TPOT p99 stayed above `20.5ms`.
- The remaining profile has no untested kernel above the opening threshold.

Use when:

- treating native I006 scheduling, hybrid retention and tied storage as the
  conservative rollback;
- designing future gates that demand causal compute, workspace and A/B/A
  evidence rather than headline probes.

Avoid:

- holding expired requests merely because their TTFT score is already zero;
- shortest-work sorting inside tiny arrival cohorts without a measured gain;
- treating static VRAM savings as usable KV capacity;
- reopening GDN, norm fusion or generic Triton paths without a new profile.

### Deep Cycle 2 DR01 Wave-Local Opportunity Screen

Measured effect:

- the exact R023a control recomputes to ERS `0.549279`, TTFT p50/p95
  `1810.595/4241.179ms` and `48/120` requests at or below `1500ms`;
- all six arrival waves contain 20 requests, but queue p90 remains
  `2926.8-4798.8ms` even after native later-wave cache reuse;
- five generic arrived-request policies were screened at collection windows
  `0/25/50/100/250/500ms` in independent slot-swap and token-work models;
- only `deadline_density + 50ms` beat no-hold FCFS in both while preserving
  fast-count, and its conservative delta was only `+0.000381 ERS` with `0/6`
  per-wave p90 improvements.

Use when:

- justifying why the next work targets capacity, frontend launch gaps or a
  measured kernel rather than another queue-order policy;
- retaining `50ms deadline_density` only as an offline watchlist feature if a
  future primitive creates real prefill/decode headroom.

Avoid:

- opening a live scheduler from a model-only gain below session drift;
- using `250-500ms` collection windows merely because larger cohorts make an
  oracle ranking look better;
- treating an offline reorder as saved model work.

### Exact Post-Arrival Continuation Tokenization

Measured effect:

- exact parent/child reconstruction for 20 public sessions and 100
  continuations, with `120/120` incremental render and token-ID equality;
- tokenizer token-work reduced `77.343%` and tokenizer-only p50 wall time
  reduced `70.125%` under `3 CPU/8GB`;
- integrated vLLM A/B/A ERS improved `0.618001 -> 0.623256` (`+0.850%`)
  while TPOT p99 remained `16.663ms` and reliability stayed `120/120`;
- V2 confirmation reached ERS `0.624348` and TPOT p99 `16.641ms`;
- independent DR20 removal A/B/A reached ERS `0.614390` versus control
  midpoint `0.609855` (`+0.7436%`) with TPOT p99 `17.556ms` and `120/120`;
- quality penalty proxy stayed `1` at GPQA `0.36` versus same-image control
  `0.34`, but the pre-registered repeat found two stable candidate-specific
  changes against a maximum of zero;
- R033 corrected quality interpretation: all five candidate GPQA processes
  retain official penalty `1`, but fresh same-image off/on/off performance
  lost `0.257%` ERS, reduced fast TTFT `83 -> 81`, and had W/T/L `32/35/53`.

Implementation:

- keep only post-arrival stable rendered prefixes and token IDs in process RAM;
- key by exact parent text, use the original tokenizer for suffixes and retain
  unconditional original-path fallback;
- in DR20, return the tokenizer's exact original object when every row misses;
  root conversations are still published to the bounded cache after delegation;
- bound the cache to 64 entries and persist only aggregate counters;
- do not modify chat template, tokenizer vocabulary, request order, engine
  inputs or model state.

Use when:

- serving multi-turn sessions where each new request contains the full prior
  conversation;
- CPU/tokenization is measurable and the tokenizer boundary is append-exact;
- profiling tokenizer/frontend cost in append-exact multi-turn workloads;
- generating a new implementation hypothesis only if it changes the measured
  end-to-end TTFT path beyond the closed DR03b proxy.

Avoid:

- packaging or rerunning unchanged DR03b; R033 removed the image confound and
  failed the frozen same-image performance gate;
- citing saved tokenizer tokens as equivalent to saved model prefill/KV work;
- running quality or building Attempt 17 after the performance dependency has
  already failed;
- claiming GPU prefill/KV savings from this frontend-only mechanism;
- preloading evaluation prompts or keeping hidden trace content;
- removing fallback for unsupported tools, multimodal or custom templates.

### Profile-Gated Kernel Selection

Measured effect:

- fresh I006 seq16 decode is `94.977%` GPU-busy with idle-gap p99 `3.902us`;
- decode GEMM is `80.584%`, while GDN is only `7.373%` and attention `1.169%`;
- category shares differ from canonical batch20 by at most `1.115pp`;
- retained-prefix work hits cache at `96.597%` but is only `53.723%` GPU-busy
  with idle-gap p99 `148.027us`.

Use when:

- applying Amdahl and launch-gap gates before writing Triton/CUDA code;
- directing cache-heavy latency work toward allocator, scheduler and metadata
  paths after model compute has already collapsed;
- retaining CUTLASS/FlashInfer auto dispatch when the profile is dense.

Avoid:

- treating a large category name as proof that a replacement kernel exists;
- reopening GDN decode below the `8%` threshold;
- using one-second `nvidia-smi` utilization samples for sub-two-second cases;
- confusing high cache hit rate with low end-to-end cache-hit latency.

### Native Cache-Control And Pool Attribution

Measured effect:

- exact I006 seq16 observer/control A/B/A preserved `456542/1962800`
  compute/cache tokens and `120/120` success in every run;
- cache lookup/allocation p99 was only `101.063/51.697us`, and all top-level
  cache control was `0.253%` of summed engine TTFT;
- peak physical pool use was `25.323%`, with at least `2026/2713` blocks free;
- true evictions, allocation failures and preemptions were all zero;
- scheduler schedule/update used `2.075%` of runtime.

Use when:

- deciding whether high cache hit plus low GPU occupancy warrants pinning,
  offload, allocator, layout or native scheduler work;
- distinguishing logical hybrid null references from consumed physical
  blocks;
- requiring direct pool and method timing before treating free VRAM as a
  serving optimization.

Avoid:

- pinning already-resident prefixes when pool pressure is absent;
- treating `1260` logical Mamba null references as 1260 stranded blocks;
- rewriting a scheduler whose measured control-plane ceiling is only `2.075%`
  and whose useful action is unchanged;
- inferring fragmentation from sparse completion snapshots; include every
  schedule event when calculating peak occupancy.

### Capability-Gated W4A16 Core Quantization - Rejected On Performance

Measured/static effect:

- W4A16 G128 covers `99.9141%` of the eligible 1.373B text-linear parameters;
- the 36 `[16, 2048]` GDN projections stay BF16 to avoid generic fallback;
- estimated active storage is `1.609GiB`, saving `632.438MiB` versus current
  FP8;
- affected decode CUDA share is only `51.452%`, requiring a `1.102x` W4-core
  kernel win for 5% total decode improvement;
- tied embedding/head W4 is unavailable in the exact Qwen3.5 model path.

R034 completed the missing dynamic proof. Exact GPTQ conversion and native
Marlin startup passed, and GPQA remained penalty `1`, but candidate ERS
`0.542724` lost `10.974%` to the FP8 midpoint. TPOT p99 rose to `23.213ms`,
W/T/L was `0/37/83`, and all waves lost. Available GPU blocks increased
`4.607%`, while sampled utilization almost doubled. The static storage estimate
was real; the required W4 kernel win was not.

Reopen only when:

- exact-image compressed-tensors/GPTQ packages and Marlin selection are proven;
- a separate offline converter preserves tokenizer/config hashes and ignores
  visual, MTP, tied head, conv1d and small-N GDN projections;
- exact-shape microbenchmarks show a real kernel win rather than storage only.

Avoid:

- interpreting smaller weights as more KV capacity when the pool is not full;
- using online INT8 on this dense model, where it does not quantize LinearBase;
- accepting generic AWQ or dequantize-then-BF16 fallback;
- quantizing tied embedding/head without implementing both lookup and output
  projection semantics and rerunning quality.

## I007 Seq19 Active Set

Effect: improves the TTFT score distribution while retaining an almost full
TPOT component. In R025 it reached ERS `0.610256` versus selected I006 control
`0.609243` (`+0.1663%`), increased fast TTFT `77 -> 84`, and kept `119/120`
TPOT values at or below 20ms. The gain was concentrated in TTFT and wave 1.

How: use packaged I007 FP8 tied-head top-k4 with `--max-num-seqs=19` and exact
CUDA graph sizes `1,2,4,8,15,16,17,19,20,24,32,40`. Preserve FP8 weights/KV,
BF16 Mamba state, context32768 and the existing I005 startup warmup.

Tradeoff: local TPOT p99 rose to `19.721ms`, above the preferred 18ms guard,
and the ERS edge is small. Treat it as a score-first H200 trial, not as the
conservative rollback. Attempt 14 is the immutable package.

## I007 Seq18 Active-Set Interpolation

Effect: measurable local decode-tail recovery, but target-negative end to end.
In the frozen seq19/seq18/seq19 bracket, seq18 reduced TPOT p95/p99
`15.909/18.429 -> 15.556/17.899ms` and kept `120/120` values at or below
20ms. It lost one fast-TTFT request and `0.400%` local ERS; TTFT p95 rose
`7.71%` while p50 changed by only `7.2ms`.

How: use the unchanged I007 digest and replace only max-num-seqs 19 with 18
and exact graph19 with graph18. Preserve FP8 weights/KV, BF16 Mamba state,
top-k4 tied head, context32768 and I005 startup warmup.

Tradeoff: Portal Attempt 16 confirmed the decode direction with TBT
`19 -> 18ms` and retained `19/120` SLO passes, but TTFT p95 changed
`6586 -> 8799ms`; score reached only `47.04`. Keep this as evidence that active
set controls the TTFT/TPOT frontier, not as a retained candidate. Close seq18
and do not combine it with graph11.

## Aggressive Long/Decode Startup Warmup

Effect: negative. It reduced I006 ERS `0.609243 -> 0.605986` and seq19 ERS
`0.610256 -> 0.604044`; seq19 fast TTFT remained `84`, while paired mean TTFT
worsened `45.556ms` and TPOT worsened `0.218ms`.

How tested: one synthetic 28,819-token prefill, a full 200-token decode cohort
and a short handoff cohort before readiness. No trace content was used.

Decision: reject this stronger warmup. Existing I005 shape warmup already
captures the useful compile/allocation paths; extra work changes runtime state
without improving the timed first-wave distribution.

## Seq19 Transition Graph18

Status: deprioritized by R026 static screening. The dispatcher pads batch18 to
graph19, but measured active-18 exposure is only `1.088s` (`3.72%`), giving a
rough one-slot padding ceiling of `0.196%` decode wall time.

The broader graph lattice looked more promising statically, but measured
results rejected the upper-bound assumption as a performance predictor.
Graph11's first A/B gain collapsed to `+0.000097` ERS against a warm control,
with worse TPOT p99 and W/T/L `34/28/58`. Graph6+11 lost `0.001479` ERS with
W/T/L `5/28/87`; graph13 was skipped. Keep graph11 only as an optional H200
probe, reject graph6/13, and do not modify frozen Attempt 14.

## Qwen3.5 DS Convolution-State Layout

Effect: valid but too small as a standalone score technique. Exact vLLM 0.22.1
supports `VLLM_SSM_CONV_STATE_LAYOUT=DS`; Qwen3.5 GDN then receives
`(dim, state_len)` state directly instead of a transposed SD view.

Measured ceiling: causal convolution is `1.296%` of pure batch16 decode,
`1.196%` of cold prefill and `1.100%` of a cache-heavy branch. I006's TBT is
already below the 20ms scoring floor, so even impossible full elimination of
this cost does not improve its TPOT component.

Carry-forward: do not allocate a standalone full-trace run. It may be included
as a free companion variable in a future already-qualified non-speculative
candidate, with correctness and memory checks. Do not combine DS with
align-mode speculative rollback in vLLM 0.22.1.

## Gap-Aware First-Token Staging

Effect: conditional offline opportunity, not yet measured serving speed. The
policy keeps one model and state pool, caps active decode at 16, serializes one
staged prefill at a time and may park a request after its first token.

Guarded six-process model:

```text
slots8, queue>=500ms, queue gain 75%, decode interference 10%
mean/worst ERS delta:     +0.016036 / +0.013850
positive panels/waves:    6/6 and 36/36
TPOT p95/p99 maximum:     18.183 / 19.032ms
```

Carry-forward: design a native gap-aware scheduler only. Reserve decode first,
stage from measured slack, retain KV/Mamba state inside the engine, and fail
closed when queue gain drops below 75% or TPOT exceeds `18.5/20.0ms` p95/p99.
Do not substitute native seq24, global `4096`, proxy reordering or repeated
queue mutation.

## Parked Hybrid State And First-Token Swap

Effect: mechanism-valid, score-negative. Exact vLLM 0.22.1 can temporarily
remove a tested post-first-token request from the scheduler running set and
later resume its retained Transformer KV plus Mamba state without a CPU copy.
R040a and R040b both completed `20/20` exact-length outputs with zero
preemption/error/OOM.

Measured R040b result at matched local GPU reservation `0.72`:

```text
candidate/native-midpoint ERS: 0.454380 / 0.466172
TTFT support:                  5/20 / 5/20
candidate/native TPOT p95:     26.491 / 24.434ms
```

Carry-forward: keep the park/resume mechanism as source/runtime knowledge only.
Do not use direct hold or first-token swap, and do not sweep their cap/age.
Late waiting requests gained TTFT but could not cross 1500ms, so the next
primitive must reduce service inside the first active cohort.

## Static FP8 Alternate Linear Backends

Effect: capability-positive, performance-negative. Serialized static
per-tensor FP8 makes `FlashInferFP8ScaledMMLinearKernel` executable, unlike the
current dynamic-per-token layout. The entire norm/quant category is only
`6.435%` of cold-prefill GPU time, so static scale/fusion alone cannot meet the
`6.5%` R041 whole-prefill gate.

Exact Qwen shape-weighted results:

```text
static FlashInfer MM vs native CUTLASS: -8.762%
Torch row-wise MM vs native CUTLASS:     -3.954%
static CUTLASS e2e vs dynamic CUTLASS:   +1.319%
```

Carry-forward: retain native online FP8/CUTLASS. Do not serialize a static
checkpoint, enable FlashInfer for dense FP8 linear layers, or patch
`torch._scaled_mm`. Reopen only if the target image changes its backend/kernel
implementation and a same-shape H200 microbenchmark beats CUTLASS by at least
`9.467%`.

## Conditional Global Prefill Budget - Closed After R042

Effect: one cohort has positive mean direction, but no robust executable
opportunity. Across three processes per budget, wave 2 gains
`+0.006770` mean score but has robust q25 `-0.008325` and only `6/9` positive
process pairs.

```text
request oracle gain:          +0.002348  (not executable)
global-wave oracle gain:      +0.001128
global-wave robust gain:      +0.000000
process noise floor:           0.004843
```

Carry-forward: fixed/resolved 2048 remains the correct policy. Do not add an
arrival-burst selector or train Tree-Gated/XGBoost for 2048 versus 4096. A
model cannot improve beyond a robust action oracle of zero.

## Native Fused RMSNorm Plus Dynamic FP8 - Closed After R043

Effect: shape-dependent and negative for serving. Native fused beats two eager
raw ops by about 31% at M1/M20, but loses `175.056%` at M1680 and makes the
weighted prefill sequence `156.311%` slower. The real serving decode baseline
is already one Inductor Triton reduction at `2.832us/call`, faster than native
fused M20 at `3.822us`.

Carry-forward: keep generated Triton reductions. Do not infer a candidate from
an eager two-op control, and do not force `fuse_norm_quant`, manually insert
the native op or open activation fusion from this result.

## H200 Attention And GDN Backend Forcing - Closed After R044

Effect: no distinct built-in optimization remains. Full attention and
GDN/Mamba are material (`11.092%/9.945%` of cold prefill), but target H200 auto
already chooses FlashAttention-3 and FlashInfer respectively. Forcing those is
a no-op; forcing Triton selects a weaker path, and CuteDSL GDN is
Blackwell-only.

Carry-forward: keep automatic backend selection. Category share is only an
Amdahl ceiling, not proof that another implementation is faster. Reopen only
for a concrete custom kernel with a positive comparison against the exact
compiled target and H200 portability.

## Next Round Recommendation

R036 changes the selection rule before the existing queue:

- compare local and H200 at component/distribution level; local total ERS
  disagreed with Portal direction for three measured changes;
- do not use TBT median `<=20ms` as proof of TPOT saturation. Four score-below-
  50 panels meet that median but mathematically retain TPOT score loss;
- preserve I006 seq16. Its optimistic ceiling without increasing `16/120`
  TTFT support is only `56.666667`;
- keep R037 closed after its `14`-primitive exact-image audit. No path exceeded
  the `0.5`-point GPU gate; DS layout is real but low-ceiling, and MRV2 cannot
  preserve the current align cache without unbounded engineering;
- require a separate causal TTFT-support primitive for score 70+, since that
  target needs at least `48/120` requests inside the TTFT ceiling.

Follow Deep Cycle 2 in `docs/planning/post_r023_deep_research_master_plan.md`:

1. freeze I006 seq16 as score control and conservative rollback;
2. keep DR07 closed: cache control was `0.253%` of engine TTFT, pool use peaked
   at `25.323%`, and no eviction/allocation/preemption failure occurred;
3. keep I007 top-k4 as a separate quality-safe decode watchlist;
4. keep DR16/R034 W4A16 closed on the current Marlin runtime; exact
   converter/runtime gates passed, but full-trace ERS lost `10.974%`;
5. keep DR03b closed after R033: official quality evidence is acceptable, but
   fresh same-image performance is negative;
6. keep the completed conservative R024 freeze on I006 only: four reliable
   trace controls, three TPOT p99 guards and four penalty-proxy-1 GPQA controls;
7. retain Attempt 14's measured seq19 TTFT/SLO gain but not its complete score;
8. treat R026 as closed: graph11 is an optional target probe, graph6/13 are
   rejected, graph18 is parked, and I006 seq16 remains rollback.
9. close seq18/19 active-set interpolation after Attempt 16 scored `47.04` and
   park graph11 Attempt 15;
10. close fixed `2048 -> 4096` prefill-budget tuning after exact-I006 R028 lost
    `1.43%` ERS and TTFT score in every wave; retain resolved/default `2048`;
11. require the next branch to introduce a new measurable primitive that
    preserves the `<=500ms` TTFT cohort and TPOT score `1.0`; create no Portal
    artifact from R028.
12. keep R029 closed: do not implement or train a larger Tree-Gated selector
    for `2048/4096`; its robust oracle is too small and depth-2 CV is negative.
13. apply R030 metric semantics to every cache/prefill experiment; do not infer
    unique cache work from raw repeated lookup counters or GPU compute from a
    request wall-phase timer.
14. keep R031 closed: fixed `4096` loses after bidirectional drift control, and
    no dynamic scheduler or Attempt 17 is justified. Audit final-chunk/service
    isolation offline with leave-one-wave-out scoring before writing code.
15. keep R032 closed: its half-recovery stress is unstable across processes,
    waves and the very-fast cohort; do not open R032b or cite the ideal splice
    as measured speedup.
16. keep R033 closed: do not rerun or package unchanged DR03b; a future
    frontend branch must introduce a distinct causal primitive.
17. keep R035 closed: exact log-prefix accounting proves all five warnings are
    pre-readiness warmup compilations and timed JIT is zero in every R034 cell.
    Do not repeat R025 aggressive warmup or open a GPU shape sweep from the
    old complete-log counter.
18. keep R036 closed after its offline audit. Use R037 only to identify a
    distinct exact-image H200 decode-tail capability; do not run another RTX
    bracket for an already closed scalar flag.
19. keep R037 closed. Do not spend a standalone run on DS conv-state layout,
    remove prefix caching for MRV2, or bypass the MRV2 align validator. The
    next branch must have a causal mechanism for raising TTFT support beyond
    the current first active wave while retaining TPOT p99 `<=18ms` locally.
20. R038 opens only a gap-aware scheduler prototype. Preserve 16 decode slots,
    use at most eight serial staged/parked states above a `500ms` queue age,
    and require measured queue reduction `>=75%` with interference `<=10%`.
    Do not package or submit until a 20-request probe and same-image A/B/A pass.
21. R039's pure policy and static boundary audit pass, but there is no runtime
    evidence. Keep it behind the native I006 fallback; do not write a vLLM
    override, create a Compose or infer score/latency from the static tests.
22. R040 proves parked hybrid-state continuity but rejects both active policies.
    First-token swap loses `2.529%` ERS proxy, leaves support at `5/20`, and
    raises TPOT p95 to `26.491ms`. Do not run full120 or package it. Audit the
    first-active-cohort TTFT staircase before another scheduler implementation.
23. R041 attributes that staircase to queue serialization, but closes the
    executable FP8 backend alternatives. Static FlashInfer and Torch row-wise
    lose `8.762%` and `3.954%` MM throughput respectively; static CUTLASS gains
    only `1.319%` standalone quant-plus-MM. Keep native online FP8/CUTLASS and
    require an arrival-feature, out-of-sample gate before any conditional
    budget policy reopens.
24. R042 closes that conditional reopening. Wave 2 is mean-positive but has
    negative robust q25; global-wave oracle gain is `+0.001128` and robust
    gain is zero. Keep fixed 2048 and require a distinct compute-saving
    primitive before any GPU work.
25. R043 closes native fused RMSNorm/dynamic-FP8. M1680 regresses `175.056%`,
    and real compiled decode Triton reductions are about 35% faster than the
    apparent small-shape fused candidate. Compare future kernels against the
    compiled serving path, never against an eager decomposition alone.
26. R044 closes forced attention/GDN backends. H200 auto already resolves
    FlashAttention-3 and FlashInfer, while `0/8` alternatives pass all gates.
    Keep `auto`; require a real target-beating custom implementation before
    reopening GPU work.

## Part 2 Output-Only FP8 Tied Head

Effect: performance-positive, quality-blocked. Keeping the BF16 tied embedding
for input lookup while using a separate FP8 copy for output projection improved
the matched append-only full score `56.729 -> 58.857`. TPOT p50 improved
`2.584 -> 2.417ms` and TTFT p50 improved `110.524 -> 105.888ms`.

Tradeoff: no-repair tensor FP8 changed the first token on `16/96` deterministic
requests and matched no full output sequence. Row-wise scaling plus top-k32
repair was the best quality variant (`92/96` first-token and `29/96` full
matches) but remained unsafe and slower than the performance upper bound.

Carry-forward: preserve the implementation as a research seed, not a package.
Do not sweep repair top-k again; `32/128/256/512` was non-monotonic. Reopen
only for a materially better shortlist or low-rank/residual correction, and
require paired GPQA before a Portal candidate.

## Part 2 Selective Final RMSNorm

Effect: exact but not repeatably score-positive. vLLM `0.22.1` normalizes all
scheduled LFM2 hidden rows before selecting the few rows that need logits.
Deferring the unchanged BF16 RMSNorm for prefills of at least 2048 rows saves a
weighted `21.917%` of that operation and preserves every sampled value and the
complete output SHA.

Tradeoff: the absolute weighted saving is only `0.149ms`. The two full
candidate runs scored `55.8737` and `52.7145` around a middle R002k control at
`53.5857`; the second repeat raised peak active requests to 11 and TPOT p95 to
`4.568ms`. Mean gain exists, but repeatability and TPOT guards fail.

Carry-forward: do not package it alone. The graph-safe phase branch and native
fallback are reusable if a future prefill kernel creates a materially larger
combined saving; re-run a complete bracket after any such integration.

## Part 2 BF16 ShortConv Projection - Closed After R002u

Effect: the native family is material (`19.7198%` of batch-8 decode), but a
replacement GEMM has no cache-realistic margin. The opening one-weight loop
reported up to `20.50%` family gain; the required 320 MiB ten-layer cycle cut
best M8 gain to `2.256%` versus its immediate control and below the production
anchor, for `-0.516%` projected complete-step gain.

Carry-forward: keep native BF16 projections. Require full distinct-weight
working-set timing for every model-layer kernel. A future ShortConv branch
must fuse projection with a consumer or otherwise remove traffic; do not
continue standalone tile tuning.

## Part 2 ShortConv Input Projection FP8 - Closed After R002v

Effect: online per-tensor FP8 for the ten ShortConv `in_proj` matrices lowers
packaged TPOT p50 from `2.960ms` to `2.747/2.697ms`. Two complete candidate
runs score `52.6084/53.2367` versus control `49.5424`, a repeated mean gain of
`+3.3802` points with zero graded failures.

Tradeoff: generated tokens are not bit-exact. The fixed GPQA proxy has zero
aggregate accuracy drop (`21/100 -> 21/100`) but exchanges one correct answer
for one newly correct answer. CUDA allocator fragmentation also caused a
long-prefill OOM until expandable segments were enabled.

Carry-forward: retain `expandable_segments:True` as a reliability fix when
long-prefill fragmentation is observed, but do not retain the FP8 projection as
a score technique. Attempt 05 scored `60.19` and the byte-identical repeat
`59.34`; mean `59.765` is `0.080` below R002k's `59.845`, while TTFT means are
worse and TBT remains `4ms`. The target-transfer gate is closed. Reopen
ShortConv only with projection-plus-consumer fusion or a target-specific
compiled kernel whose working-set measurement demonstrates a distinct gain.

## Part 2 ShortConv Decode Epilogue Fusion - Closed After R002w

Effect: the exact width-three BF16 primitive is bit-exact and fast. On runtime
state stride `(8192,1,2048)`, ten M8 epilogues improve
`0.285780 -> 0.091605ms`; a repeated live 512-token decode panel improves
median throughput `3331.44 -> 3426.68 token/s` (`+2.86%`).

Trade-off: projection and `out_proj` staging remain outside the kernel, while
full-trace queueing/prefill noise is much larger than the removed work. The
candidate/control/candidate ERS bracket is
`50.5693/50.6229/50.1385`; candidate mean loses `0.2689` points despite zero
failed graded requests.

Carry-forward: do not deploy or submit this epilogue alone. Preserve its
sequential accumulator, runtime-stride support and native mixed/prefill
fallback as reusable code. Reopen only inside a larger projection-consumer
fusion with a repeated complete-workload gain; an isolated kernel or steady
decode throughput win is not sufficient.

## Production CUDA Graph Attribution - Required After R002x

Effect: R002w removes about 30 child kernels per pure-decode step. Exact
scheduler-weighted child-kernel time improves `2.500%`, while complete GPU
mean improves `3.501%` with `1.045%` control drift.

Trade-off: the result is not robust across active batch. M4 complete gain is
`0.851%`, exactly its control drift, and M8 regresses `1.687%`. The direct M8
ShortConv saving is `0.039981ms`, not the eager estimate `0.194175ms`; eager
timing exaggerated the opportunity by `4.86x`. Full replay remains negative.

Carry-forward: use eager kernels only for numerical and implementation
screening. Promotion requires preallocated CUDA Events around the complete
production graph step, Kineto child attribution, distinct model-layer working
sets, exact scheduler occupancy and a clean A/B/A drift bracket. Do not reopen
standalone ShortConv fusion. Profile the cached-turn prefill/first-token phase
next because it is a distinct score-limiting boundary.

## Cached-Turn Full-Attention Attribution - R002y

Effect: prefix caching is effective but cannot make full attention constant-
time. Cache reuse reaches `83.16%` and new prefill stays near `4.2k` tokens,
while complete first-token GPU service rises `60.801 -> 114.685ms`. Attention
explains `92.64%` of this growth and contributes `30.029ms` weighted.

Trade-off: the measured RTX 5080 kernel is FlashInfer because SM120 cannot use
FlashAttention with FP8 KV. H200 SM90 auto-selects packaged FA3, so local
kernel replacement timing is not target evidence. Approximate/windowed
attention would change model semantics; TRT-LLM attention requires SM100.

Carry-forward: preserve prefix caching, FP8 KV and R002k decode. Qualify only a
prefill-specific FA3 mechanism on H200, requiring backend proof and unchanged
decode/TBT. Separately retain the aligned second-step ceiling (`2.296ms`
weighted), but do not implement it without an exact Mamba-state continuity
proof.

## Tiered Full-Attention Horizons - R003d

Effect: a fixed map `2:4096,5:4096,8:8192,10:8192,12:8192,14:8192` improved
the local two-run score mean from `56.5450` to `61.8876`. TTFT p50/p95 fell
`23.296/69.797ms` and TPOT p95 fell `0.547ms`; all graded requests completed.
The same map preserved every control-pass case in a 14-case long-context probe
and gained one case at 7,109 tokens in both repeats.

Trade-off: this is a semantic attention-horizon change, not a semantics-
preserving kernel optimization. It also increases the available KV pool at the
same memory fraction. The 100-question GPQA proxy is short and returned
`23/100` versus a noisy `21..22/100` control range, so it cannot certify long
history. More aggressive `6144` and `front3` maps lost retrieval cases.

Carry-forward: keep `safe8192` as a local research reference only. Do not
submit or package it until BTC request construction, rules interpretation and
long-context quality are resolved. The next useful technique is a request-local
prefix-plus-recent retention mask that preserves selected old positions in
every affected layer without blindly truncating the middle of the context.

## Native Triton Prefix-Range Retention - R003e Rejected

Effect: explicit Triton plus a 4096 recent window is fast without prefix
metadata (`62.364 -> 75.223ms` from turn 0 to turn 5). This validates the
backend/window control.

Trade-off: enabling any native `mm_prefix_range` of 64, 128 or 256 tokens
raises turn-5 GPU time to about `187ms`, roughly `112ms` slower than the
no-prefix Triton control and more than `72ms` slower than unbounded. Prefix
size is not the useful tuning variable on this execution path. The native
predicate also requires both query and key inside the range, so it does not
retain prefix keys for later queries.

Carry-forward: do not use native disjoint prefix ranges for the current
candidate and do not sweep more sizes. Preserve the mask semantics and source
audit for a future split-range online-softmax kernel that skips the discarded
middle span. The custom target must use causal `key_in_prefix`, not native
mm-prefix semantics, and must pass profile and long-context retrieval gates.

## Compact Split-Range Triton Attention - R003f Research Pareto

Effect: exact causal `prefix OR recent` tile selection removes the native
full-span scan. P64/W4096 reaches `75.290ms` at turn 5 versus `75.223ms`
recent-only and `114.735ms` unbounded. The paired retrieval probe improves
from `5/14` to `7/14` without losing a control pass.

Trade-off: two valid full replays vary from `63.1183` to `60.3961`; mean
`61.7572` is slightly below safe8192 and below native W4096. GPQA proxy is
stable at `20/100`, below controls at `21..22/100`, even though its prompts
do not activate the window. The copied Triton kernel also lacks H200 SM90
qualification.

Carry-forward: keep the compact virtual-tile mapping, overlap de-duplication
and segmented online-softmax reduction as reusable kernel technique. Do not
package this implementation or sweep prefix sizes. Reopen only after the
full-overlap path is numerically equivalent to native and the exact CUDA 13.x/
SM90 image passes compilation, correctness and replay gates.

## Sparse ShortConv Boundary Checkpoint - R002aa Target Probe

Effect: the LFM2 cached-turn alignment step can be removed without retaining
all historical ShortConv block states. Store only the final crossed full-block
state and the final partial-tail state, then reuse that boundary in the same
model process. The no-pad implementation preserves about `1.59M` local cache
tokens, changes cached turns from two model steps to one and wins both clean
full-replay pairs by `+0.2348/+1.5078` ERS. Their mean TTFT improvement is
`6.4606ms`; two fixed GPQA runs score `22/100` versus R002k `21/100`.

Trade-off: micro timing is sign-inconsistent, the clean mean gain of `+0.8713`
is smaller than observed process variance, and TPOT rises by `0.0395ms` on
average. Shape padding looked positive in micro timing but lost `1.1367` ERS
and raised TPOT p95 in full replay. The overlay patches vLLM internals and has
not yet executed on the H200 SM90 target.

Carry-forward: retain no-pad only as an immutable H200 target probe. Keep cold
requests, allocation failures and incomplete metadata fail-closed; never
include the optional R002y profiler in the submission image. Keep R002k as the
rollback and incumbent until target evidence attributes a gain. Complete
replay outranks micro timing for every future checkpoint or padding variant.
