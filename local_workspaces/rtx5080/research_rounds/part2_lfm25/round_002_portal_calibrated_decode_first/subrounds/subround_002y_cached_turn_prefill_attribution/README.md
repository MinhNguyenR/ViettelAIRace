# Subround 002y - Cached-Turn Prefill Attribution

Status: complete. Attribution passed; no serving candidate was promoted.

## Question

R002x closes standalone ShortConv decode fusion. The active append-only proxy
still has TTFT p50 increasing from about `68ms` at turn 0 to `141ms` at turn
5 while TPOT remains near `2.6-2.9ms`. R002y determines which production GPU
phase grows across the six cached turns before another kernel is written.

## Frozen Serving Contract

- Image: R002k fused BF16 greedy head.
- Model: local fixed `LiquidAI/LFM2.5-1.2B-Instruct` artifact.
- Precision: online FP8 per-tensor weights, FP8 E4M3 attention KV and BF16
  aligned Mamba state.
- Runtime: seq8, batched tokens 16384, prefix caching, chunked prefill, async
  scheduling and FULL_AND_PIECEWISE CUDA Graphs.
- Local envelope: RTX 5080, exactly 3 server CPU cores and 8 GiB server RAM.
- Profile-only GPU memory fraction: matched `.80` in every process.

No scheduler, model, precision, output head or prompt-policy change is allowed.

## Workload Panels

Create one deterministic append-only conversation per repeat. Each turn adds
the public trace's approximately 4k-token segment plus a deterministic
assistant history. The server generates only the first token because the
measurement boundary ends there; the next prompt still carries the declared
assistant-history length so total prompt shapes match the six-turn proxy.
The 15 public primer conversations run unmeasured before any panel is armed,
covering all 90 official warmup request shapes.

For turn 0-5, record:

- prompt length, initial computed/cache tokens and newly scheduled tokens;
- complete GPU duration from first prefill execution through first sampling;
- individual prefill-chunk GPU and CPU durations;
- client first-token latency;
- Kineto child-kernel duration, launch count and family attribution.

## Evidence Boundary

This is a controlled service-time attribution, not a full ERS candidate. It
removes queue interference deliberately and cannot claim the hidden prompt's
exact cache-hit shape. A later optimization must return to the complete
420-request replay and Portal-calibrated score gate.

## Opening Gate

A follow-up implementation may open only if:

1. all six turns complete with the expected prompt/cache progression;
2. weighted complete-GPU drift between control processes is at most `10%`;
3. one mutable kernel family contributes at least `2ms` weighted per first
   token and at least `15%` of weighted child-kernel time;
4. the family is present in at least four of six turns and has an executable
   vLLM/H200-compatible intervention;
5. the intervention can leave the R002k decode path unchanged.

Otherwise close the branch without an overlay, image, Compose, quality run or
Portal attempt.

## Answer

The final A/Kineto/B bracket is stable at `1.418%` weighted complete-GPU
drift. Cache reuse rises from `0%` at turn 0 to `83.16%` at turn 5 while new
prefill remains near `4.2k` tokens. Complete first-token GPU time still rises
`60.801 -> 114.685ms`.

Attention explains `92.64%` of that growth: it rises
`4.825 -> 54.742ms`, while FP8 and BF16 GEMMs remain approximately flat. The
architecture-level bottleneck is therefore the six full-attention layers
scanning the retained KV context, not failed prefix caching or Mamba state
recomputation.

The local path is FlashInfer because SM120 cannot use FlashAttention with FP8
KV. Exact source proves the H200 target instead auto-selects packaged FA3,
which supports FP8 KV on SM90. This backend mismatch blocks direct promotion
of a local kernel result. R002y creates no overlay or submission and opens only
an H200-specific FA3 qualification branch. Full measurements are in
`results.md`; backend evidence is in `source_audit.md`.
