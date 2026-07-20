# R003h Round Plan

Status: complete; serving candidate rejected.

## Phase A - Primitive Contract

Inspect the exact FlashInfer 0.6.11.post2 APIs shipped in the image and
confirm the page-table, causal-level and dtype contracts.

## Phase B - Numerical Probe

Construct page-aligned prefix and recent ranges for a decode batch. Compare
the two-level cascade output with a dense reference for BF16 and FP8 KV.

## Phase C - Performance Probe

Warm the planned wrappers and measure native full attention, native recent
window attention, and prefix-plus-recent cascade using CUDA events. Keep
planning time separate from steady-state run time.

## Phase D - Integration Decision

Only if the primitive gates pass, inspect a decode-only vLLM metadata overlay.
Reject integration if arbitrary partial pages or prefill semantics cannot be
represented without changing model outputs.

The overlay was exercised on the full local replay. It is explicitly a sparse
`prefix OR recent` attention policy, so it is not an exact full-attention
implementation when a middle range is omitted.

## Phase E - Record

Write the exact result, failed cases and next decision. Do not build or submit
an image from this probe alone.

Decision: close R003h as research-only. Open a separate shared-prefix cascade
probe for the exact case supported by vLLM's native metadata contract.
