# Subround 003d - Tiered Attention Horizons

Status: complete local research; `safe8192` is retained as a conditional
winner and is not a submission artifact.

## Question

Can the R003c quality-safe early placement retain its 4096 horizon while the
other four full-attention layers use a longer finite horizon? This should
recover part of the all-layer bounded-attention score without returning to
the quality-failing all-4096 candidate.

The candidate is trace-independent. It assigns one fixed horizon to each of
the six full-attention layer indices and leaves all ten ShortConv layers,
precision, cache policy, scheduler, CUDA Graph policy and fused greedy head
unchanged.

## Candidate Matrix

| Label | Layers 2,5 | Layer 8 | Layers 10,12,14 | Purpose |
|---|---:|---:|---:|---|
| safe8192 | 4096 | 8192 | 8192 | safest interpolation from early2 to all-8192 |
| mid6144 | 4096 | 6144 | 6144 | stronger latency, moderate horizon |
| front3 | 4096 | 4096 | 8192 | test one extra early bounded layer |

## Gates

1. Static overlay tests and exact six-layer topology verification pass.
2. Turn-0 profile regression is no more than 5 percent and turn-5 saving is
   at least 15 ms versus the registered unbounded profile.
3. All 14 retrieval requests succeed and no control-pass case is lost.
4. Only the strongest candidate passing gates 1-3 opens two full replays.
5. A candidate remains research-only until short quality, long-context
   semantics and target-H200 rules are all resolved.

## Reproduction

- scripts/run_profile.sh LABEL WINDOW_MAP PORT
- scripts/run_retrieval.sh LABEL WINDOW_MAP PORT
- scripts/run_full_replay.sh LABEL WINDOW_MAP PORT
- scripts/run_quality.sh LABEL WINDOW_MAP PORT

The exact map is written into every run directory as window_map.txt. Raw run
directories remain ignored; compact comparison JSON files are committed.

## Measured Result

The fixed local envelope was 3 CPU cores, 8 GiB RAM, RTX 5080 and the exact
R002k fused-BF16-greedy image. Profile results versus the registered unbounded
control were:

| Candidate | Turn 0 ms | Turn 5 ms | Turn 5 saving | Gate |
|---|---:|---:|---:|---|
| safe8192 | 59.407 | 73.928 | 40.807 ms | pass |
| mid6144 | 58.821 | 71.625 | 43.110 ms | pass |
| front3 | 59.552 | 72.716 | 42.018 ms | pass |

The paired retrieval gate separated them. `safe8192` completed 14/14 requests,
lost zero control-pass cases and gained one 7,109-token case in both repeats.
`mid6144` lost one control-pass case and `front3` lost two, so neither was
opened for full replay.

`safe8192` then completed two independent 420-request replays:

| Metric | Control mean (2) | safe8192 mean (2) | Delta |
|---|---:|---:|---:|
| Score points | 56.5450 | 61.8876 | +5.3426 |
| TTFT p50 | 114.262 ms | 90.966 ms | -23.296 ms |
| TTFT p95 | 315.560 ms | 245.763 ms | -69.797 ms |
| TPOT p50 | 2.571 ms | 2.409 ms | -0.163 ms |
| TPOT p95 | 4.159 ms | 3.612 ms | -0.547 ms |
| Graded failures | 0 | 0 | unchanged |
| TTFT <=400 ms | 323.5/330 | 329.5/330 | +6 |

Individual candidate scores were `61.8737` and `61.9014`; their range was only
`0.0277` points, versus `1.1918` for the two controls. The exact aggregate is
in `analysis/full_replay_comparison.json`.

The 100-question short GPQA proxy produced `23/100`, with 11 parse failures.
Controls observed `21/100` and `22/100` with 11 and 10 parse failures. This is
not a long-context accuracy proof because the proxy prompts are shorter than
the 8192-token horizon.

## Interpretation And Boundary

The result is a composite resource/compute effect: bounded attention stops the
long-history attention growth and the same memory fraction allocates a larger
KV pool. It is not evidence that a new attention kernel is faster on H200.
The independent 4k workload and the retrieval losses of more aggressive maps
show that the semantic boundary is real.

Keep `safe8192` as the balanced local research reference. Do not merge it into
R002k, build a Portal image or create a Compose candidate until BTC's
long-context request construction and the acceptability of a changed
attention horizon are resolved. The next technical branch should investigate
request-local prefix-plus-recent retention or a target-qualified attention
path, not another blind scalar window sweep.
