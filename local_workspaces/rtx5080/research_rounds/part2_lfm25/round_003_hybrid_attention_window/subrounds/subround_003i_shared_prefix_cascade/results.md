# R003i Results

Status: complete and rejected.

## Source And Topology

- Exact image: `localhost/part2-lfm25-r002k:fused-bf16-greedy`, vLLM 0.22.1.
- Model topology: six full-attention layers and ten ShortConv layers.
- LFM2.5 has 32 query heads and 8 KV heads.
- FlashInfer's builder contains exact two-level metadata but returns `False`
  from `use_cascade_attention` in the shipped source.
- FlashAttention uses the upstream cascade heuristic, but the API server's
  global cascade setting is disabled unless `--no-disable-cascade-attn` is
  supplied.

## Synthetic Shared-Prefix Replay

The 48-request burst is diagnostic and all rows are tagged warmup by the copied
trace, so latency below is computed by `summarize_all_requests.py` rather than
the graded score summary.

| KV / path | Positive decisions | Builds | TTFT p50/p95 ms | TPOT p50/p95 ms |
| --- | ---: | ---: | ---: | ---: |
| BF16 forced cascade | 76/80 | 80 | 99.50 / 761.95 | 3.26 / 4.68 |
| BF16 cascade off | 0 | 0 | 95.02 / 765.89 | 2.34 / 4.64 |
| FP8 forced cascade | 76/80 | 80 | 100.36 / 769.44 | 3.39 / 4.51 |

Forced cascade is operational with both KV dtypes, but it adds about `0.92ms`
to median TPOT in the matched BF16 burst and does not improve TTFT p50.

## Direct Exact-Prefix Waves

| Prefix / batch | Mode | Builds | Score | TTFT p50 ms | TPOT p50 ms | TTFT <=400 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4K / 8 | forced | 31 | 28.3622 | 581.20 | 3.18 | 0/8 |
| 4K / 8 | off | 0 | 35.2872 | 461.33 | 2.11 | 0/8 |
| 8K / 8 | forced | 31 | 31.2152 | 440.58 | 2.83 | 0/8 |
| 8K / 8 | off | 0 | 40.2467 | 432.87 | 1.87 | 0/8 |
| 8K / 20 | forced | 32 | 44.5656 | 296.44 | 1.84 | 20/20 |
| 8K / 20 | off | 0 | 44.2298 | 414.83 | 1.39 | 1/20 |
| 8K / 20 | upstream heuristic | 32 | 42.6031 | 288.98 | 2.10 | 20/20 |
| 8K / 8 | upstream heuristic | 1 | 40.3390 | 448.46 | 1.87 | 0/8 |

There is a real crossover. At batch 8 forced cascade loses; at batch 20 it
trades roughly `0.45ms` median TPOT for roughly `118ms` median TTFT and produces
a small `+0.336` point diagnostic gain over its control. The upstream heuristic
correctly avoids sustained batch-8 decode cascade and selects batch 20.

Output token hashes differ between forced cascade and control even with
temperature zero. The metadata retains every token, so this is reduction-order
numerics amplified by greedy decoding, not sparse token omission. It still
invalidates any claim of bitwise output identity and requires GPQA before a
candidate with real activation can be promoted.

## Public Proxy Applicability

The eight-conversation observer saw long per-conversation prefixes only at
batch 1 and recorded `0/10` positive decisions. The complete FP8 replay then
recorded `0/80` positive decisions and zero cascade builds:

| Run | Success | Score | TTFT p50/p95 ms | TPOT p50/p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Forced FlashInfer, upstream heuristic | 330/330 | 51.1766 | 132.97 / 420.01 | 2.74 / 4.13 |
| Matched FlashInfer, cascade off | 330/330 | 52.6311 | 128.36 / 370.36 | 2.70 / 3.91 |

Both runs recorded zero cascade builds. Their output hashes and peak HTTP
inflight counts differ, so the `-1.4545` point gap includes run/scheduling
variance and cannot be attributed entirely to the predicate observer. The
valid conclusion is narrower: the candidate creates no workload benefit and
does not beat its same-backend control. This branch is rejected.

## Native FlashAttention A/B

The native path keeps BF16 KV so the RTX 5080 selects `FLASH_ATTN`. Candidate
and control differ only in the global cascade argument.

| Prefix / batch | Global cascade | Positive/builds | Score | TTFT p50 ms | TPOT p50 ms | TTFT <=400 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8K / 8 | on | 1/1 | 35.6212 | 442.59 | 2.35 | 0/8 |
| 8K / 8 | off | 0/0 | 41.1370 | 285.66 | 2.27 | 8/8 |
| 8K / 20 | on | 32/32 | 19.8181 | 522.99 | 4.94 | 0/20 |
| 8K / 20 | off | 0/0 | 38.2097 | 315.58 | 2.38 | 20/20 |

At batch 8 the one mixed-prefill cascade adds roughly `157ms` TTFT. At batch
20 the heuristic also selects decode cascade, adding roughly `207ms` TTFT and
`2.56ms` median TPOT. The native heuristic therefore mispredicts this SM120
workload at both activation boundaries.

## Target-SM Hypothesis

The native FlashAttention decision depends on runtime `num_sms`. For a decode
batch, the upstream model compares cascade waves against FlashDecoding waves.
With 32 query heads, 8 KV heads and an 8K prefix, the reproduced model gives:

| Effective SMs | Batch 8 | Batch 20 |
| ---: | ---: | ---: |
| 16 | cascade | cascade |
| 18 | cascade | cascade |
| 19 | cascade | cascade |
| 84 | FlashDecoding | cascade |
| 132 | FlashDecoding | cascade |

NVIDIA documents H200 `1g.18gb` as one seventh of the full GPU's SM fraction.
Therefore local batch-8 non-activation cannot be directly transferred to BTC.
However, the measured local activations are strongly negative, and the H200
SM fraction predicts more activation. This is a target uncertainty, not a
reason to submit. Native global cascade is rejected without an H200-specific
A/B mechanism.

## Decision

- Keep R002k's cascade default unchanged.
- Do not force FlashInfer solely for shared-prefix cascade.
- Do not ship `--no-disable-cascade-attn`.
- Preserve the heuristic model and exact-cascade evidence for a future engine
  version or direct H200 lab, but close this score branch now.
