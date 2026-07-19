# Subround 003c Plan

Status: complete; early2 is retained as a research branch and no Portal
artifact is opened.

## Phase A - Contract Reuse

Reuse the R003b fail-fast selective routing overlay. Verify that the global
complement and windowed subset exactly cover the six full-attention layers and
that all ten ShortConv layers remain native.

## Phase B - Sparse Placement Screen

Run `late2`, `late3`, `spread3` and `early3`. Because `early3` loses only one
control-pass case while the other three lose two, refine the boundary with
`early2` and `first1`. For each candidate collect:

- six-turn complete GPU boundary times;
- computed/scheduled token progression;
- startup KV capacity and active-request peak;
- calibrated retrieval outputs and lost/gained control-pass cases;
- startup and request failure status.

## Phase C - Full Replay Gate

Completed. early2 had zero lost control-pass cases in both retrieval repeats,
zero request failures and a profile saving of 17.724 ms at turn 5. Two
420-request replays were compared against the registered unbounded control.

## Phase D - Decision

early2 beats the control outside the repeated process envelope and does not
show a candidate-specific GPQA regression. Keep it as a local research
candidate, not a submission: the quality evidence is a small noisy proxy and
the model semantics still change beyond the 4096 horizon. Close scalar
layer-placement permutations. Any continuation must preserve selected old
positions in every affected layer, or stop this branch.
