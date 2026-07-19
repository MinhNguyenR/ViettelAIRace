# R003d Source Audit - Retention Feasibility

## Exact Runtime Findings

- The image contains vLLM 0.22.1 and its LFM2 implementation constructs the
  generic Attention layer directly for six full-attention layers.
- Per-layer sliding windows are a native constructor argument and produce a
  SlidingWindowSpec, so a static horizon map is executable without replacing
  the attention kernel or cache manager.
- vLLM attention sinks are per-head sink logits, not a generic policy that
  retains arbitrary old token positions.
- The separate StaticSinkAttention path reserves global static sink blocks and
  requires precomputed sink key/value tensors. It does not represent a
  request-specific first-N-token retention policy for unrelated conversations.
- The local FlashInfer path rejects sinks unless TRTLLM attention is available
  on SM100. The RTX 5080 and H200 target are not that path. Enabling sinks
  would therefore change the backend or fail before it measured the intended
  retention mechanism.

## Decision

Do not implement a fake sink by passing a learned bias tensor or by copying
request KV through CPU memory. Those paths test a backend change or static
cross-request state, not the requested prefix-retention semantics.

Use tiered native windows as the next executable interpolation. Reopen a true
sink/prefix mask only with a request-local KV block-table design that supports
both prefix caching and the hybrid Mamba state lifecycle on the exact target
backend.
