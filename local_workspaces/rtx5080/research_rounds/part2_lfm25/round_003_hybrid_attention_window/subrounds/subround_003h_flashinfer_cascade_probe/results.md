# R003h Results

Status: complete; rejected for promotion.

## Environment

- Image: `localhost/part2-lfm25-r002k:fused-bf16-greedy`
- PyTorch `2.11.0+cu130`, CUDA runtime `13.0`, vLLM `0.22.1`
- Local GPU: RTX 5080; process envelope `3 CPU / 8 GB RAM`
- Runtime policy: FP8 per-tensor weights, FP8 E4M3 KV, BF16 Mamba state,
  `max-num-seqs=8`, `max-num-batched-tokens=16384`, async scheduling
- Attention overlay: FlashInfer on six full-attention layers, window `4096`,
  cascade enabled only for decode batches that satisfy the page checks

## Primitive Probe

The two-level wrapper was numerically finite and close to a dense reference
whose key set was intentionally the same `prefix OR recent` set. At batch 20
and sequence length 32768, BF16 cascade was `0.04289 ms` versus `0.39649 ms`
for the selected full reference (`9.24x`), with maximum absolute error
`0.000613`. FP8 was `0.04125 ms` versus `0.21231 ms` (`5.15x`), with maximum
absolute error `0.000516`. At batch 8 the speedups were `3.88x` (BF16) and
`1.93x` (FP8). At short context the merge/planning overhead dominated.

These numbers validate the FlashInfer primitive, not the original model's
full-attention semantics: the omitted middle range is not represented.

## Serving Smoke And Quality

The decode-only overlay activated successfully in the six-panel smoke run:

```text
t0  prompt=4000   gpu=62.3156 ms  ttft=68.6679 ms
t1  prompt=8199   gpu=72.3066 ms  ttft=81.4555 ms
t5  prompt=24994  gpu=71.5603 ms  ttft=90.9509 ms
```

The 100-case GPQA proxy scored `22/100`, with no cascade activation because
the prompts were shorter than the activation threshold. This is a no-regression
check for the inactive path only; it does not validate long-context sparse
attention.

## Full Replay

The matched runs used the same image, trace, request driver, window and memory
settings. Both completed all `420` requests and all `330` graded requests:

| run | score points | TTFT p50/p95 (ms) | TPOT p50/p95 (ms) | graded failures |
| --- | ---: | ---: | ---: | ---: |
| cascade candidate | `60.6272` | `90.42 / 205.35` | `2.669 / 3.538` | `0` |
| cascade off control | `64.7741` | `84.20 / 232.63` | `2.307 / 3.330` | `0` |

The candidate improved the TTFT p95 in this pair but was slower at the median,
slower on TPOT, and lower on the aggregate local score. The difference is not
large enough to claim a stable gain, especially because the overlay emitted
only one activation marker and the full replay has normal run-to-run
scheduling variance.

The candidate and control output-sequence hashes also differed. That is
consistent with the sparse middle omission and is an additional quality risk;
the separate runs are not a deterministic equivalence proof.

## Decision

Reject R003h for the submission path. Keep the primitive and measurements as
evidence that FlashInfer can merge page-level attention states, but do not use
the sparse prefix-plus-recent overlay in the final image. The next experiment
should target vLLM's exact shared-prefix cascade path, where all non-shared
tokens remain visible.
