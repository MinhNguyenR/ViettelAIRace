# R002z Round Plan

Status: complete. All source gates are resolved.

## Phase A - Target Backend

Status: complete.

Verify backend priority, FP8-KV support, packaged FA versions and the exact
automatic backend selected by SM90 versus local SM120.

## Phase B - Phase Isolation

Status: complete; no built-in mechanism found.

Trace every candidate control to a runtime call site. Reject controls that are
declarations only, affect decode globally or require a different architecture.

## Phase C - H200 Scalar Controls

Status: complete; opening gate failed.

Audit block size, CUDA Graph split controls, AOT scheduling, long-prefill
`num_splits`, batch invariance and TRT-LLM attention. Open a GPU candidate only
if the control changes long FA3 prefill while preserving R002k decode.

## Phase D - Decision

Status: complete.

Automatic FA3 already owns the H200 path and no measurable prefill-only scalar
remains. Close before implementation and move to an algorithmic intervention.
