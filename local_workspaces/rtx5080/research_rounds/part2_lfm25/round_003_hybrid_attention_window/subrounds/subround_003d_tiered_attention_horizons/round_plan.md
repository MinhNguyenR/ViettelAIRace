# Subround 003d Plan

Status: complete local research; promotion blocked by semantic/rules boundary.

## Phase A - Backend Feasibility

Read the exact vLLM 0.22.1 Attention, KV specification and backend paths in
the R002k image. Record why native attention sinks are not the first executable
candidate on this local/target envelope.

## Phase B - Tiered Overlay

Implement a fail-fast map from each full-attention layer to one positive
horizon. Require exact coverage of layers 2, 5, 8, 10, 12 and 14, reject
duplicates or malformed prefixes, and verify ten untouched ShortConv layers.

## Phase C - Profile And Retrieval Screen

Status: complete. All three profiles passed the latency gate. `safe8192` lost
zero control-pass retrieval cases in two repeats; `mid6144` lost one and
`front3` lost two. Only `safe8192` advanced.

## Phase D - Full Replay And Decision

Status: complete. Two safe8192 replays averaged `61.8876` points versus
`56.5450` for two controls; TTFT p50/p95 and TPOT p95 all improved, with zero
graded failures. The 100-case short GPQA screen was `23/100` versus a noisy
control range of `21..22/100`.

Promote only as research evidence. Do not create a Portal image or Compose
from this semantic-changing family without a separate rules decision and
long-context quality evidence. Next work should target prefix-plus-recent
retention or target-specific attention rather than another scalar sweep.
