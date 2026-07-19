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
