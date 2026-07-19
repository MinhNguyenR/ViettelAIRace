# Subround 003d Failed Actions

Record invalid maps, startup/backend failures, contaminated processes and
quality-gate failures without overwriting earlier evidence.

## 2026-07-20 - Generated Wrapper Argument Marker

The first safe8192 profile invocation exited before container startup because
the generated wrapper included a literal plus sign before the delegated
arguments. The parent runner read the label as MODE and rejected it. All three
affected wrappers were corrected and shell-syntax checked. This is harness
noise and contains no model measurement.

## 2026-07-20 - Quality Boundary

The safe8192 quality run completed all 100 requests and returned `23/100`
with 11 parse failures. This is not a server failure; the same short proxy
produces 21-22/100 and 10-11 parse failures in the controls. It is recorded as
a scope limitation because every proxy prompt is shorter than the attention
horizon and cannot test the semantic change that matters.
