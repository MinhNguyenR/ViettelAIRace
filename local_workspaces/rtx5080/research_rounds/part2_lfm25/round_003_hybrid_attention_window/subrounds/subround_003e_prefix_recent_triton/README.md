# Subround 003e - Prefix Plus Recent Triton Retention

Status: complete and rejected at the profile gate.

## Hypothesis

R003d proved that reducing long-history attention work is valuable locally, but
plain sliding windows discard every old token outside the horizon. This branch
tested whether vLLM 0.22.1's Triton `mm_prefix_range` could provide a shortcut:

```text
(causal AND recent-window) OR native mm-prefix range
```

Post-profile source audit proved that the native range requires both query and
key to be inside the range, so it does not retain prefix keys for later
queries. The experiment is retained as a rejected API screen, not as an
implementation of the intended mask.

## Runtime Delta

- Six LFM2 full-attention layers are constructed with explicit
  `TritonAttentionBackend`, `per_layer_sliding_window=W` and the existing FP8
  KV dtype.
- `GPUModelRunner` is marked as prefix-aware and receives a deterministic
  `(0, P-1)` range for every active request. The range is applied through the
  native Triton metadata tensor, not by adding a Python attention mask.
- Ten ShortConv layers, Mamba state, fused BF16 greedy head, scheduler,
  prefix cache and memory envelope remain unchanged.
- The prefix is trace-independent and does not read request content.

## Candidate Matrix

The first screen should use `W=4096` and vary only the retained prefix:

| Label | Recent window | Prefix tokens | Purpose |
|---|---:|---:|---|
| p0w4096 | 4096 | 0 | Triton/window backend control; no prefix mask |
| p64w4096 | 4096 | 64 | minimum semantic repair |
| p128w4096 | 4096 | 128 | balanced repair |
| p256w4096 | 4096 | 256 | stronger repair, more work |

The first executable smoke candidate is `p128w4096`. The `p0w4096` control is
run only when all non-empty prefix sizes share the same regression, so backend
overhead can be separated from prefix-mask overhead.

## Gates

1. Startup proves all six full-attention modules use Triton and all ten
   ShortConv modules remain untouched.
2. Startup/first request proves a non-empty `mm_prefix_range_tensor` reaches
   Triton metadata; otherwise the run is invalid, not a negative result.
3. Profile turn-0 regression is at most 15 percent and turn-5 saving is at
   least 10 ms versus the unbounded control.
4. The 14-case retrieval screen loses no more control-pass cases than the
   matching R003d `safe8192` reference; a zero-loss result is preferred.
5. Only a gate-passing candidate opens two full replays and the short GPQA
   screen. Any result remains research-only until long-context quality and
   target-H200 backend qualification are addressed.

## Boundary

Triton is a different attention backend from the R002k FlashInfer path. A
negative result may be backend overhead rather than a failure of prefix-plus-
recent retention. A positive result must therefore report both mask delivery
and backend change, and must not be described as a portable H200 improvement.

## Measured Result

The canonical profile matrix is recorded in `results.md` and
`analysis/profile_screen.json`. Triton/window without prefix metadata reached
`62.364/75.223ms` at turns 0/5 versus the unbounded control at
`60.782/114.735ms`. Non-empty prefixes of 64, 128 and 256 tokens instead
reached turn-5 times of `187.100/187.862/187.059ms`.

Prefix size is therefore not the limiting variable. Enabling the native
disjoint prefix-range path adds about `112ms` at turn 5 versus the no-prefix
Triton control. All semantic candidates fail the latency gate, so no
retrieval, full replay, quality or packaging phase opens.

Exact source additionally proves that this API has the wrong retention
semantics and disables sliding-window tile pruning. A custom causal sink-plus-
recent kernel is required; native `mm_prefix_range` must not be reused.
