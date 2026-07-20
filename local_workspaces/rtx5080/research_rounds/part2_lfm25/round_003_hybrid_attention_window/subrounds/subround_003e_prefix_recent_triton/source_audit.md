# R003e Source Audit - Prefix Plus Recent Feasibility

## Confirmed Native Contract

- vLLM 0.22.1 exposes `TritonAttentionBackend` with FP8 KV support and
  `supports_mm_prefix() == True`.
- `TritonAttentionMetadata` carries `mm_prefix_range_tensor` with shape
  `(num_requests, max_ranges, 2)`.
- The Triton mask computes `(causal AND sliding-window) OR mm_prefix_range`.
- Ranges are inclusive in the kernel, so `(0, P-1)` retains the first `P`
  request-local positions.
- FlashInfer remains unsuitable for this feature in the image: its sink path
  requires TRT-LLM on SM100, while this branch uses Triton explicitly.

## Expected Semantics

For each affected full-attention layer and request, the candidate attends to:

```text
all keys in [0, P-1] UNION keys in the recent causal window of width W
```

This is request-local and content-independent. It does not preserve arbitrary
middle tokens and does not alter Mamba recurrent state.

## Risks To Validate

- Triton attention is a backend change from the R002k FlashInfer path and may
  erase the saved attention work.
- Marking the runner prefix-aware causes a small metadata tensor construction;
  the first log must prove a non-empty tensor reaches Triton metadata.
- CUDA Graph capture and prefix caching must both remain valid.
- The kernel mask is semantic, so any positive latency result is still
  research-only until long-context quality and rules are resolved.
