# Round 003 Plan

Status: the registered `8192` candidate is fully measured and passes the local
performance/quality gates. The round is closed as a research result, with
promotion blocked only by long-context semantics and rules review.

## Phase A - Source And Rules Contract

Status: complete.

Confirmed the exact LFM2.5 topology, generic per-layer sliding-window API,
sliding-window prefix-hit algorithm and hybrid fixed-point cache coordinator.
The change is trace-independent but is not numerically equivalent to the
original model beyond the selected window. The official optimization section
allows serving/runtime techniques, but does not explicitly state whether a
model-wide attention-horizon change is acceptable; keep this as a rules-risk
item.

## Phase B - Composite Overlay

Status: complete.

One import hook preserves R002k patchers and replaces only the module-local
`Attention` constructor used by `Lfm2Attention`. It injects window `8192`,
verifies six full-attention layers and fails startup on partial patching. The
ten ShortConv layers are explicitly checked as untouched. Disabled mode is a
control kill switch.

Static acceptance passed:

1. R002k hooks execute once in control and candidate.
2. Six attention modules report window `8192`.
3. Ten ShortConv modules remain native.
4. Python compile, shell syntax and standalone overlay tests pass.
5. Image-level import smoke passes.

## Phase C - Quality-Relevance Screen

Status: complete with a scope limitation.

The 100 GPQA proxy prompts are all below the candidate window. The paired
candidate/control runs show no candidate-specific quality regression; the
candidate scores `22/100` and the control repeat also scores `22/100`.
This is sufficient for a short-prompt regression screen only, not proof of
long-context semantic equivalence.

## Phase D - Cache And Service-Time Gate

Status: complete.

The six-turn panel retains the same prefix/cache progression and new suffix
size. Turn 0 regression stays below `5%`; turn 5 saves `33.29ms` in the
two-candidate average, above the `15ms` gate. Kineto attributes the gain to
the attention family rather than a dropped ShortConv or output-head path.

## Phase E - Full Score And Boundary Checks

Status: complete.

The full append-only bracket is control/candidate/control/candidate with
`420/420` completion in every run. Candidate mean is `61.1708` versus control
mean `56.5450`; candidate TPOT p95 is lower, and candidate minimum score is
above control maximum. A bounded independent-4k check is neutral/slightly
negative (`67.9362` versus `68.3485`), proving the mechanism is conditional on
long retained history.

## Phase F - Decision And Checkpoint

Status: complete as research, not as a submission.

Update the shared maps, decision log, experiment log, technique ledger and
failed-action log. Keep the overlay outside the Portal queue until the prompt
semantics and rules interpretation are resolved. Do not claim the local score
as an H200 score; the candidate's target backend and memory capacity still
need a clean target qualification.

## Follow-Up 003a - Window Sweet-Spot Screen

Status: complete.

This is a separate follow-up, not a retroactive change to the registered
result. It tested `4096`, `6144`, `8192`, `12288` and `16384` on the
long-context six-turn panel and a calibrated paired retrieval probe. It
recorded:

- complete GPU/attention cost by turn;
- cache capacity and active-request peak;
- full score component deltas where affordable;
- GPQA parsed accuracy and exact changed-answer count;
- whether the least aggressive window preserves most of the gain.

Outcome: `4096` is the repeated score maximum but loses two control-pass
retrieval cases; `8192` is the balanced branch with no paired losses;
`16384` fails the performance gate. Do not open a Portal image until one
window passes a stronger long-context quality decision and a target-backend/
rules review.

## Follow-Up 003b - Selective Global Placement

Status: complete and rejected before full replay.

The four one/two-global-layer placements all passed the isolated latency gate,
but all lost the same two control-pass retrieval cases. This falsifies the
cheap "one global repair layer" hypothesis. Close scalar placement tuning and
move to a retention topology that preserves selected old positions in every
windowed full-attention layer.

## Follow-Up 003c - Sparse Windowed Layer Placement

Status: complete and retained as research-only evidence.

The six placements tested in R003c all passed profile latency. early2
(windowed layers 2 and 5) alone preserved every control-pass retrieval case in
two repeats and completed two full 420-request replays with zero failures.
Its mean score was 58.0922 versus 56.5450 for the unbounded control. The
quality proxy remained within the noisy 21..22/100 control range.

Decision: keep early2 as a balanced placement reference, but do not promote it
to Portal. Close further global-complement permutations. Any next round must
use a sparse retention mask or another mechanism that preserves selected old
positions in every affected layer and must repeat the long-context retrieval
gate.

## Follow-Up 003h - FlashInfer Cascade Prefix/Recent Probe

Status: complete and rejected.

The primitive FlashInfer cascade path passed its dtype and numerical smoke
checks, but the integrated overlay used a sparse `prefix OR recent` key set.
It omitted the middle range, so it changed model semantics. A full local pair
also failed the promotion comparison: candidate `60.6272` versus control
`64.7741`, with candidate TPOT p50 `2.669ms` versus `2.307ms`.

Decision: keep this as evidence about the FlashInfer API only. Do not package
the overlay. Reopen cascade work only for exact shared-prefix attention, not
for another sparse prefix/window sweep.

## Follow-Up 003d - Tiered Attention Horizons

Status: complete and retained as research-only evidence.

The tiered map screen confirmed `safe8192` as the only gate-passing candidate.
It improved the two-run local score mean by `+5.3426` points over the repeated
unbounded control, with lower TTFT and TPOT tails and zero graded failures.
`mid6144` and `front3` were rejected for paired retrieval regressions.

The result does not authorize packaging: the attention horizon is a semantic
change, the short quality proxy cannot test long history, and the larger KV
pool contributes to the measured gain. Close scalar horizon tuning. The next
branch must investigate request-local prefix-plus-recent retention or a
target-qualified attention path with an explicit long-context quality gate.

## Follow-Up 003e - Prefix Plus Recent Triton Retention

Status: complete and rejected at the profile gate.

The native Triton prefix-range contract, FP8 KV path, layer topology and CUDA
Graph compatibility were verified. A no-prefix Triton/window control passed
the performance gate, but each non-empty prefix size (`64/128/256`) increased
turn-5 GPU time to about `187ms`, more than `72ms` worse than the unbounded
control. The retrieval and replay phases were skipped by dependency. Do not
continue scalar prefix tuning on this native path.

Post-profile source audit also invalidated the original semantic assumption:
native mm-prefix ranges do not remain visible to later queries. R003f must use
a causal key-prefix predicate and prune the middle span before K/V loads.

## Follow-Up 003f - Split-Range Online-Softmax Kernel

Status: complete and retained as research-only evidence.

The source/opportunity gate, numerical primitive, engine integration, profile,
retrieval, full replay and short-quality phases are complete. Compact prefix-
plus-recent tile pruning reaches the recent-only performance floor and gains
two retrieval cases without control-pass loss. Restoring native segmented 3D
decode was necessary; the forced-2D prototype was rejected.

Two valid P64 replays average `61.7572`, slightly below safe8192 and materially
below native W4096. GPQA proxy repeats two points below the control repeat, and
the target H200 SM90 backend is untested. Close the branch without image,
Compose or Portal promotion. Reopen only for full-overlap numerical identity
and target-backend qualification, not another prefix-size sweep.

## Follow-Up 003i - Exact Shared-Prefix Cascade

Status: complete and rejected.

The source audit, synthetic activation, FP8 compatibility, public applicability
and native FlashAttention A/B phases are complete. Forced FlashInfer had zero
builds on the 420-request proxy and did not beat its matched control. Native
global cascade regressed at both batch 8 and batch 20. Preserve the source and
SM-count analysis, but do not modify the R002k submission configuration.
