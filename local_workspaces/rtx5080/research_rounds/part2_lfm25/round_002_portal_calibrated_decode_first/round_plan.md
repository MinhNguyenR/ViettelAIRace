# Round 002 Plan

Status: Phases A-M passed or closed with explicit evidence. The native b16384
artifact remains the native rollback, 002k is the H200-proven incumbent, and
002v/002w are closed, non-promoted probes.

## Phase A - Workload Driver

Status: complete and qualified with `330/330` graded successes in two clean
full processes.

- Read only public metadata from `trace_grading_public.jsonl`.
- Start each conversation at its declared timestamp.
- Send turns sequentially after response completion plus `3000ms` think time.
- Generate exactly the declared input and output token counts.
- Exclude 90 primer requests and score exactly 330 graded requests.
- Record per-request TTFT, TPOT, ERS, active count and turn index.

## Phase B - Prefill/Decode Trade

Status: complete. `8192`, `4096` and `2048` scored `66.335`, `66.421` and
`66.306`; the `0.115`-point range is not material.

Run the same process in this order:

```text
A: seq8, budget8192 control
B: seq8, budget4096
C: seq8, budget2048
A: control repeat
```

Completed clean comparisons:

```text
A control repeat: 66.335 ERS
B clean:          66.421 ERS
C clean:          66.306 ERS
```

The `+0.086` difference is treated as noise. A separate B process at `52.492`
is invalid because a game was concurrently using the GPU.

Promotion requires higher mean ERS, zero failures and no process-order-only
win. A TPOT gain may spend TTFT according to the measured score, not a fixed
latency preference.

## Phase C - Active Set

Status: complete with no promotion. `seq4` averaged `66.304` across two runs,
essentially equal to the `seq8` control `66.335`, while retaining TTFT ceiling
violations. `seq5` and `seq3` were rejected.

Around the winning prefill budget, compare seq4/seq6/seq8. Prefer the smallest
active set that lowers TPOT without moving a material request cohort toward the
400ms TTFT ceiling.

## Phase D - Precision

Status: complete. Generic FP8 rejected. FP8 E4M3 attention KV repeated its
performance gain and passed a paired 100-question local drift proxy with zero
accuracy drop; hidden quality risk remains.

Test separately:

1. generic `--quantization=fp8` versus selected `fp8_per_tensor`;
2. FP8 E4M3 attention KV;
3. their combination only if each factor is independently positive.

Online accuracy fields are ignored. Every lossy candidate remains marked
quality-risk until local paired GPQA and eventual BTC post-online evaluation.

## Phase E - Runtime Qualification

Status: complete. FP8 KV plus async scheduling repeated above 67 local ERS
and is the incumbent. Session-prefix transfer was negative, `xxhash` is
unavailable in the immutable image, disabling prefix caching regressed to
`62.816`, throughput mode failed its second full repeat at `66.490`, seq10
failed its full confirmation at `64.789`, and compile-size warmup did not
remove lazy helper JIT. The float16 Mamba-state and FP8 E5M2 screens also
regressed; BF16 state and E4M3 KV remain fixed.

Only test runtime knobs that are valid in the fixed vLLM image and that can
improve the repeated incumbent without moving TPOT toward the 10ms ceiling.
Every candidate receives a cold server and exact 420-request driver. The next
branch tests local VRAM utilization near the BTC target; it must not be
interpreted as evidence that extra cache helps unless the full trace repeats.

## Phase F - Package

Status: complete.

Package Attempt 02 from the promoted b16384 configuration only after exact
image/entrypoint validation, Compose normalization, size limit and immutable
checksum. The local equivalent has already passed repeated 3 CPU/8 GiB full
traces; target `.95` memory remains inherited from the official H200 sample.

Final package gates: Compose parse passed, `25/25` tests passed, file size is
`1131` bytes and SHA-256 is
`4c0f167f73e6a60a4846c2870ef405be8963b2084d411da79d6e7d0d348cd0b0`.

## Phase G - Non-lossy Fused Greedy Head

Status: complete, packaged, quality-qualified and H200-validated.

The exact BF16 tied output head is fused into a Triton tiled dot-product and
hierarchical argmax. The standalone gate matched all tokens and selected
batch-aware tiles. The vLLM overlay accepts only empty built-in processor state
and falls back to native logits for every unsupported request. Twenty live
hidden-state audit batches matched `100%` of native argmax IDs. The full local
replay scored `58.3989` versus native control `56.7287`, with zero graded
failures; see subround 002k for artifacts and trade-offs.

The exact-base image adds only two overlay files, passes strict-greedy and
native-logprob smokes, and retains `21/100` GPQA with zero correctness flips.
Its public digest is frozen in Part 2 Attempt 04.

Portal Attempt 04 scored ERS `60.75`, TTFT p50/p95 `44/80ms`, median TBT
`4ms` and six failures. It beats native-image Attempt 02 by `+2.58` points and
becomes the H200 incumbent.

## Phase H - Selective Final RMSNorm

Status: complete; closed without packaging.

Moving final LFM2 RMSNorm after sampled-row selection is bit-exact and saves
`21.917%` of the isolated large-prefill operation, but only `0.149ms`
absolute. The phase-safe 2048-row implementation preserves output SHA and
native fallback. Its two full candidate runs score `55.8737` and `52.7145`
around a middle R002k control at `53.5857`. Mean delta is positive, but the
second run and TPOT guard fail. Keep the source primitive only for a future
larger prefill bundle; create no submission artifact from it.

## Phase I - Online Block-FP8 Target Transfer

Status: complete; closed without packaging.

The online 128x128-weight/1x128-activation representation passed source,
shape, startup and lifecycle-safety checks. Exact source exposes an H200 SM90
FlashInfer/DeepGEMM path, but local SM120 uses the block CUTLASS fallback. That
fallback lost `4.2785%` in the batch-8 decode profile. More importantly, the
candidate scored `18/100` twice versus R002k `21/100`, with stable losses on
four reference-correct cases. A diagnostic replay then scored `49.3582`
versus `58.3989` while still completing `330/330`.

The quality gate blocks target promotion independently of local kernel
portability. Keep `fp8_per_tensor`; create no R002s image, Compose, or Portal
attempt. Reopen only after a concrete calibration or numerical-preservation
change, not by forcing another backend name.

## Phase J - Small-M FP8 Kernel Feasibility

Status: complete; closed before integration.

The current numerical representation was held fixed. Source audit, exact
primitive probes and full M=1/2/4/8 shape matrices show that Torch's raw-GEMM
edge is only `0.026708ms/step`. Direct BF16 scaling loses at least `35.51%`
end to end. The custom Triton epilogue is bit-exact but costs `68.53x` more
than the maximum epilogue budget and leaves the complete M=8 path `36.34%`
slower than CUTLASS.

Do not create an overlay, quality run, replay, image, Compose or Portal
attempt. Reopen only if the runtime gains a single native primitive that
consumes per-token activation scales and a scalar weight scale while emitting
BF16, without a separately launched epilogue.

## Phase K - BF16 ShortConv Kernel Screen

Status: complete and closed before integration.

Keep all ShortConv projections BF16 and test whether a small-M primitive can
beat the native alternating 2048->6144 and 2048->2048 projection kernels.
The numerical gate passed, but a 320 MiB cyclic-weight confirmation removed
the single-matrix L2 gain. Best M8 Triton remained slower than the production
anchor and projected `-0.516%` complete-step gain. Do not reuse the rejected
FP8 copies or open a standalone BF16 GEMM overlay.

## Phase L - ShortConv in_proj FP8 Target Qualification

Status: complete; packaged, target-tested and closed without promotion.

Isolate the ten ShortConv input projections from R002n, require repeated
host-overlay and packaged gains, keep output projections BF16, preserve the
R002k fused greedy path and run the paired quality proxy. A fragmented
long-prefill OOM is resolved with expandable allocator segments. Two packaged
candidates beat packaged control by `+3.0660/+3.6943` points with zero graded
failures and zero aggregate quality drop. The immutable H200 probe scored
`60.19` and its exact-digest repeat `59.34`, versus R002k's pair mean `59.845`.
Keep the digest archived without overwriting R002k; no ShortConv precision
promotion follows.

## Phase M - ShortConv Decode Epilogue Fusion

Status: complete and closed without promotion.

R002w fuses the pure-decode BF16 ShortConv state/epilogue path while retaining
native prefill and mixed execution. Exact runtime-layout M8 timing projects an
`8.811%` complete-step gain, and a 512-token live repeat gains `2.86%` median
throughput. Numerical, source, memory and FULL_AND_PIECEWISE graph gates pass.

The required full replay rejects it: candidate/control/candidate is
`50.5693/50.6229/50.1385`, all with zero graded failures. Mean candidate delta
is `-0.2689`, so no quality run, public image, Compose or H200 probe follows.
Keep the kernel only as a possible component of a larger projection-consumer
fusion; keep R002k immutable.

## Phase N - Production Graph-Step Attribution

Status: complete and closed without promotion.

R002x profiles the full `execute_model + sample_tokens` graph boundary at
M1/M2/M4/M8 in a clean control/candidate/control bracket. A separate complete
replay supplies exact scheduler occupancy. Pure-decode eligibility is
`99.401%`, but direct M8 saving is only `0.039981ms/step`; M4 matches drift and
M8 regresses `1.687%` at the complete GPU boundary.

The registered opening gate fails. Do not implement a larger ShortConv fusion
from this branch and do not create a quality run, image, Compose or Portal
attempt. The next branch must profile a distinct score-limiting production
phase rather than tuning this decode epilogue again.

## Phase O - Cached-Turn Prefill Attribution

Status: complete; attribution passes, implementation remains target-gated.

R002y profiles the complete first-token GPU boundary across all six public
append-only turn shapes. Cache progression is valid, weighted A/B drift is
`1.418%`, and attention contributes `30.029ms`/`34.69%` while explaining
`92.64%` of turn-depth growth.

Do not write a local FlashInfer replacement from this evidence. H200 uses FA3
with FP8 KV while RTX 5080 uses FlashInfer, so local kernel speed does not
transfer. Keep R002k decode immutable. The next branch may qualify an H200 FA3
prefill-only mechanism or close on source capability; no image or Portal probe
opens until phase isolation and target execution are proven.
