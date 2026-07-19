# Subround 003b Plan

Status: complete; no candidate passed the quality gate.

## Phase A - Source And Overlay

Status: complete.

The exact vLLM constructor passes `layer_idx` to `Lfm2Attention` and builds the
attention module with the prefix `...layers.<idx>.self_attn.attn`. The overlay
routes only real full-attention modules, preserves the R002k hooks and asserts
the observed global, windowed and ShortConv sets at startup.

Static checks passed:

1. Selective global routing tests for all four candidate forms.
2. Malformed/non-attention prefixes fail closed.
3. Python compilation and shell syntax checks.
4. Startup markers show the exact requested topology.

## Phase B - Placement Screen

Status: complete.

Measured `global14`, `global2`, `global8` and `global2_14` with one isolated
six-turn profile and one calibrated 14-case retrieval process each. Every
profile passed the latency opening gate and every request completed.

## Phase C - Full Replay

Status: closed without a run.

All placements failed the retrieval gate: each lost two of the four cases that
the unbounded control answered correctly. A full replay would not distinguish
the candidates on the registered quality criterion, so no 420-request process
was launched.

## Phase D - Decision And Checkpoint

Status: complete as research, not as a serving candidate.

The result rejects the hypothesis that one or two globally causal layers are
enough to make a 4096 window quality-safe. It does not reject bounded attention
as a performance mechanism; it isolates the need for a semantics-preserving
retention rule. R002k remains the submission incumbent.
