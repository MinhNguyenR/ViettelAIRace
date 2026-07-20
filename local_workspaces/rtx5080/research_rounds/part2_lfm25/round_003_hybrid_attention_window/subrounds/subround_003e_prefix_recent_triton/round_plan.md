# Subround 003e Plan

Status: complete; native non-empty prefix ranges are rejected before
retrieval.

## Phase A - Native Path Audit

- Confirm the exact vLLM 0.22.1 Triton `mm_prefix_range` contract.
- Confirm the kernel mask order and inclusive range convention.
- Confirm FP8 E4M3 KV and CUDA Graph compatibility on the local image.

## Phase B - Overlay And Metadata Proof

- Add a fail-fast per-layer Triton wrapper.
- Add a runner hook that injects only fixed `(0, P-1)` ranges and records the
  number of metadata objects receiving a non-empty tensor.
- Add pure unit tests for map parsing and range generation.

## Phase C - Profile Matrix

- Run `p64w4096`, `p128w4096` and `p256w4096` under 3 CPU cores, 8 GiB RAM,
  RTX 5080 and GPU utilization `0.80`.
- If all three regress similarly, run `p0w4096` as a backend-only diagnostic;
  it is not a semantic candidate.
- Reject startup/backend fallbacks before interpreting latency.

## Phase D - Retrieval And Full Replay

- Compare retrieval results with the R003d safe8192 control and unbounded
  control. Stop on lost control-pass cases.
- Run two 420-request replays only for the strongest surviving candidate.
- Run the 100-case GPQA proxy and record its short-context limitation.

## Decision Rule

Prefer a candidate only when it retains the R003d latency benefit, proves the
prefix tensor is active, and does not worsen the paired long-context quality
boundary. If Triton overhead erases the gain, close this path and preserve the
source audit; do not silently fall back to FlashInfer and call it a result.

## Outcome

Phases A-C completed. `p0w4096` proved that Triton/window itself remains fast,
but every non-empty prefix candidate regressed turn-5 GPU time by more than
`72ms` versus the unbounded control. Phase D was stopped by dependency: no
semantic candidate passed the profile gate. No full replay, GPQA, image,
Compose or Portal artifact was created.
