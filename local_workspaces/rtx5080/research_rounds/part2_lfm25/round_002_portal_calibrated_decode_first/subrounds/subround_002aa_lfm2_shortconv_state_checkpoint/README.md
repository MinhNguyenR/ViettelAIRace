# Subround 002aa - LFM2 ShortConv State Checkpoint

Status: locally qualified and packaged as immutable Part 2 Attempt 06;
H200/Portal validation remains pending. The incumbent R002k serving path
remains frozen and is still the rollback.

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
5. Run paired quality and complete 420-request replay before creating an
   immutable local package.

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

## Qualified Checkpoint

- Sparse dual checkpoint keeps align-like capacity: `1,592,154` tokens at
  local memory fraction `.78`, versus about `530k` for all-mode block16.
- Cold requests remain on the native align path; sparse checkpointing is
  enabled only when a request already has a nonzero cached prefix.
- Cached turns become one model step while cache progression remains
  `4000/8192/12384/16592/20784`.
- Idle in-process A/B rejected pad16. Pad32 won both micro repeats but lost
  `1.1367` ERS points on the complete trace, so the package uses no padding.
- Two clean no-pad full-replay pairs are both positive: `+0.2348` and
  `+1.5078` points, mean `+0.8713`. Mean paired TTFT changes by `-6.4606ms`;
  mean TPOT changes by only `+0.0395ms`. Every run completed `420/420`.
- A third pair is excluded because the candidate preflight sampled `13%` GPU
  utilization; its negative `-2.4908` points are retained as contamination
  evidence, not hidden.
- Two no-pad GPQA runs score `22/100` against R002k `21/100`, with zero
  reference-correct answers lost and zero request errors.
- Local image `localhost/part2-lfm25-r002aa:shortconv-sparse-nopad` runs the
  complete trace from baked code with `330/330` graded successes. Its local
  digest is `sha256:9e5586a637dfedec185a0665bb10284d00284be13cea0ce09f9238dfc7739765`.

The package is a target probe, not the new incumbent. Its functional replay
ran under `41-46%` desktop GPU load, so that run proves packaging only and its
score is excluded. The qualified image was pushed at registry digest
`sha256:9e5586a637dfedec185a0665bb10284d00284be13cea0ce09f9238dfc7739765`
and pinned in
`portal_submissions/part2_attempt_06_lfm25_r002aa_shortconv_sparse_nopad_seq8_b16384.yml`.
The Compose is byte-for-byte identical to Attempt 04 except for the image
digest; an H200 result has not been submitted or attributed yet.

The local runner samples GPU utilization and memory five times before startup.
Analysis accepts performance evidence only when every sample is at most `10%`
and memory use is at most `2048 MiB`; a separate `20%` hard limit and one-spike
runtime default keep harmless compositor activity from blocking functional
work while preserving the stricter post-run classification.
