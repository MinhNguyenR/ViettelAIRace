# Subround 002z - H200 FA3 Exactness Boundary

Status: complete by exact-image source audit. No serving candidate opened.

## Question

R002y proves that cached-turn first-token cost grows in the six full-attention
layers, but the RTX 5080 measures FlashInfer while BTC H200 is expected to use
FA3. R002z asks whether the immutable vLLM `0.22.1` image exposes a bounded,
prefill-only FA3 control that can be qualified without changing R002k decode.

## Answer

The target path is already the optimized automatic path:

- SM90 selects packaged FA3 for FP8 E4M3 KV;
- FA3 AOT scheduling is enabled automatically;
- long prefill uses FA3's automatic `num_splits=0` heuristic;
- default block size `16` already matches the supported FA3 granularity;
- the public split-count control applies only to full CUDA Graph shapes, while
  R002k captures only up to 16 tokens and the measured prefill is about 4k;
- the declared prefill/decode-attention split option has no runtime call site;
- TRT-LLM attention requires SM100 and is not an H200 route.

Forcing a backend, batch-invariant mode or a global attention change would also
change decode and violate the frozen incumbent contract. A custom FA3
`num_splits` patch cannot be measured on SM120 and would replace an H200-native
heuristic without target evidence.

R002z therefore closes source-only. It creates no overlay, GPU run, quality
run, image, Compose or Portal attempt. The next score-focused branch changes
the algorithmic amount of attention work using vLLM's per-layer sliding-window
primitive and must pass explicit cache, quality and rules gates.
