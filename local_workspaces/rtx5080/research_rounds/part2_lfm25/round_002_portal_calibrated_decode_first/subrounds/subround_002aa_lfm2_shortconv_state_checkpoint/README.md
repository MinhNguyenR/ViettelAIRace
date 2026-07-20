# Subround 002aa - LFM2 ShortConv State Checkpoint

Status: active; implementation and contaminated directional screen complete.
Idle-GPU qualification, quality and full replay remain pending. The incumbent
R002k serving path remains frozen.

## Question

R002y found that cached turns 1-5 execute a large `4192-4208` token model
step followed by a second full-model step containing only `2-14` tokens. The
second step costs `2.662-2.923ms`, or `2.296ms` per public turn when weighted
over all six turn indices.

The split is not caused by the `16384` scheduler budget. vLLM deliberately
stops at the last full cache block when `mamba-cache-mode=align`, because the
LFM2 ShortConv wrapper stores only the state at the end of a forward. R002aa
tests whether the existing block-checkpoint path can preserve both the cache
boundary and final running state in one model execution.

## Frozen Contract

- Base image: `localhost/part2-lfm25-r002k:fused-bf16-greedy`.
- Model, FP8 per-tensor weights, FP8 E4M3 attention KV and BF16 ShortConv
  state remain unchanged.
- Runtime remains seq8, batched tokens 16384, async scheduling, chunked
  prefill, prefix caching and FULL_AND_PIECEWISE CUDA Graphs.
- Local server remains limited to 3 CPU cores and 8 GiB RAM.
- No Portal artifact is created before exact output, cache progression,
  capacity and full replay gates pass.

## Phases

1. Wire the already packaged causal-conv APC path into LFM2 and qualify
   `mamba-cache-mode=all` as a correctness/performance proof.
2. Compare output behavior, model-step count and complete first-token GPU time
   against matched `align` processes.
3. Measure the state-memory cost. `all` stores a ShortConv checkpoint for every
   cache block and is not assumed promotable.
4. Only if phase 1 removes the extra step without changing outputs, implement
   a sparse dual checkpoint that stores the crossed block boundary and final
   running state, not every historical block.
5. Run paired quality and complete 420-request replay before promotion.

## Opening Gates

The sparse implementation opened because all-mode:

- starts without fallback and executes the APC ShortConv path;
- changes cached turns 1-5 from two model steps to one;
- preserves the expected prefix-cache progression;
- improves weighted complete-GPU first-token time by at least `1.5ms`.

Exact 16-token sequence identity is retained as a diagnostic, not a hard gate.
The FP8 incumbent itself reproduced only `5/12` complete sequences and `11/12`
first tokens across two identical control processes. Sparse measured `3/12`
and `10/12`; the packaged all-mode path also measured `3/12` and `10/12`.
Cold-request GPQA and full replay are therefore the promotion gates.

## Current Checkpoint

- Sparse dual checkpoint keeps align-like capacity: `1,592,154` tokens at
  local memory fraction `.78`, versus about `530k` for all-mode block16.
- Cold requests remain on the native align path; sparse checkpointing is
  enabled only when a request already has a nonzero cached prefix.
- Cached turns become one model step while cache progression remains
  `4000/8192/12384/16592/20784`.
- In-process screens indicate a real gain, but all three alignment screens ran
  while a game consumed `23-39%` background GPU. Their latency values are
  directional only and cannot select no-pad versus pad32.
- No submission image, Compose file or Portal artifact exists for R002aa.

The remaining gate requires an idle-GPU in-process repeat, no cold-quality
drop, no failures, no TPOT regression above `3%`, and an ERS gain larger than
matched-run noise.

The local runner now samples GPU utilization and memory five times before
startup. By default it rejects a run above `10%` utilization or `2048 MiB`
used memory, so contaminated measurements cannot silently enter the result
set. Threshold overrides must be explicit environment settings.
