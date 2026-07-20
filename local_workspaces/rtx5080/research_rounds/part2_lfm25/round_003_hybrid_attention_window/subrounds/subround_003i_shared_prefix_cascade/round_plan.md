# R003i Round Plan

Status: complete and rejected.

## Phase A - Source Contract

- Confirm the native two-level metadata uses all shared-prefix pages plus all
  per-request suffix pages.
- Confirm the scheduler's common-prefix threshold and batch-size heuristic.
- Record why the shipped FlashInfer builder disables the path.

Result: complete. Both backends construct all shared-prefix and private-suffix
metadata. FlashInfer disables admission unconditionally. FlashAttention uses a
`prefix >= 256`, `batch >= 8` and runtime-SM performance model. Global cascade
is disabled by default in the server arguments.

## Phase B - Diagnostic Activation

- Patch only `use_cascade_attention` through an import overlay.
- Run a fixed-prefix, eight-request diagnostic with BF16 KV.
- Log every decision without changing request data.

Result: complete. The instrumentation verifies all six attention layers and
leaves all ten ShortConv layers unchanged.

## Phase C - Exactness And Performance

- Compare cascade off/on output token IDs and per-request latency.
- Repeat at prefix lengths 1024, 4096 and 8192 where memory permits.
- Test FP8 KV only after BF16 passes and record any wrapper exception.

Result: complete for operational safety, failed for promotion. BF16 and FP8 KV
both run, but forced cascade changes greedy outputs and loses at batch 8.

## Phase D - Public-Trace Applicability

- Run the same observer on the public append-only driver.
- Measure common-prefix length, eligible batch size and activation count.
- Do not force a synthetic prefix into the public score replay.

Result: complete. The 420-request proxy replay had zero positive decisions and
zero cascade builds when FlashInfer used the upstream heuristic.

## Phase D2 - Native FlashAttention Target Bracket

- Preserve auto-selected `FLASH_ATTN`; do not force FlashInfer.
- Compare global cascade on/off on exact 8K shared-prefix waves at batch 8 and
  batch 20.
- Run one full FP8 public replay and count native positive decisions/builds.
- Model the upstream decision at the RTX 5080 SM count and the H200 `1g.18gb`
  SM fraction; keep the result explicitly target-dependent.

Result: complete. Global cascade loses materially at both batch 8 and batch
20. The target SM model predicts more frequent activation on H200 MIG, but no
measurement supports a target gain. Increased activation risk is not a
promotion signal.

## Phase E - Decision

Closed. Keep R002k unchanged. Do not package forced FlashInfer, do not add
`--no-disable-cascade-attn`, and do not create a Portal attempt from R003i.
