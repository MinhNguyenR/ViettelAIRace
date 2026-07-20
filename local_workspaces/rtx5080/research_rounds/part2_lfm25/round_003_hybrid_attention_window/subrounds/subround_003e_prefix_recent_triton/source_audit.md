# R003e Source Audit - Prefix Plus Recent Feasibility

## Confirmed Native Contract

- vLLM 0.22.1 exposes `TritonAttentionBackend` with FP8 KV support and
  `supports_mm_prefix() == True`.
- `TritonAttentionMetadata` carries `mm_prefix_range_tensor` with shape
  `(num_requests, max_ranges, 2)`.
- The Triton mask computes `(causal AND sliding-window) OR
  (query_in_range AND key_in_range)`. The range is a bidirectional document
  region, not a persistent attention sink.
- Ranges are inclusive in the kernel, so `(0, P-1)` retains the first `P`
  request-local positions.
- FlashInfer remains unsuitable for this feature in the image: its sink path
  requires TRT-LLM on SM100, while this branch uses Triton explicitly.

## Actual Semantics

For a query inside `(start, end)`, native `mm_prefix_range` makes keys in the
same range bidirectionally visible. A query after `end` receives no access to
that range. Therefore `(0, P-1)` does not preserve the first `P` keys for
later queries and cannot repair a sliding-window history loss.

The exact tile-bound helper also does both of the following whenever any
mm-prefix tensor is active:

```text
max_seq_prefix_len = max(causal_prefix, full_seq_len)
sliding-window tile pruning executes only when USE_MM_PREFIX is false
```

Consequently every query block loops over the full sequence and masks after
K/V loads. This exact source behavior explains the R003e latency regression.

## Risks To Validate

- Triton attention is a backend change from the R002k FlashInfer path.
- Marking the runner prefix-aware causes a small metadata tensor construction;
  the first log must prove a non-empty tensor reaches Triton metadata.
- CUDA Graph capture and prefix caching must both remain valid.
- The kernel mask is semantic, so any positive latency result is still
  research-only until long-context quality and rules are resolved.

Post-profile conclusion: the native contract fails both required properties.
It does not retain prefix keys for later queries, and it disables the pruning
that made the recent-only window fast.
