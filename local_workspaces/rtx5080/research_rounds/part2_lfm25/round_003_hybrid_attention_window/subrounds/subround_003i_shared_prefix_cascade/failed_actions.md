# R003i Failed Or Excluded Actions

## Assuming The Overlay Was Active

The first probe changed FlashInfer's builder, but vLLM auto-selected
`FLASH_ATTN`. The server started successfully while the intended code path was
inert. Every future backend experiment must assert the selected backend in the
startup log.

## Leaving Global Cascade Disabled

Patching a backend predicate was insufficient because vLLM 0.22.1 globally
disables cascade by default. `--no-disable-cascade-attn` is required for a real
activation test.

## Spending The Observer Budget During Graph Capture

The initial log counter counted zero-prefix CUDA Graph builds and exhausted its
80-row budget before live traffic. Counters now advance only for nonzero common
prefixes, while decision logs retain a small zero-prefix startup sample.

## Off-By-One Prime Prefix

One direct-wave run primed a prefix whose token count did not match the actual
shared prompt prefix. It is excluded from all tables. The matched replays use
the exact fixture/prompt construction and record the observed prefix length.

## Forcing Cascade At Batch 8

`R003I_FORCE=1` activates a path that the upstream performance model rejects on
the RTX 5080. It raises TPOT and lowers score at both 4K and 8K prefixes. Do not
ship a constant threshold override.

## Forcing FlashInfer For The Full Public Replay

The 420-request FP8 replay had zero cascade activations but scored `51.1766`,
below its matched same-backend cascade-off control at `52.6311`. Because both
runs had zero builds, this test does not isolate cascade runtime; it proves the
branch has no applicable prefix cohort and no score gain. Forced FlashInfer is
closed.

## Treating Exact Metadata As Bitwise Identity

Cascade includes every key/value token, but its two-level online-softmax merge
changes floating-point reduction order. Greedy output hashes diverged on the
synthetic fixed-prefix controls. Exact attention semantics do not waive the
quality gate.

## Inferring The Selected Backend From The Force Variable

The first native-FP8 harness run expected `FLASH_ATTN` because
`R003I_FORCE_FLASHINFER=0`, but RTX 5080 auto-selection chose `FLASHINFER` for
FP8 KV. A later forced-backend control had no global "Using ... backend" line
because the backend was installed per layer. Both runs stopped at startup and
are excluded. The harness now records the selected backend from either the
global selector log or the verified per-layer topology.

## Enabling Native FlashAttention Cascade Globally

The native heuristic selected one mixed-prefill cascade at batch 8 and 32
prefill/decode builds at batch 20. Both A/B pairs regressed materially in TTFT,
TPOT and score. Do not add `--no-disable-cascade-attn` to R002k based on the
H200 SM-count hypothesis alone.
