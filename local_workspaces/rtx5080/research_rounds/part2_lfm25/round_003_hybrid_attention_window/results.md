# Round 003 Results - Hybrid Full-Attention Window

Status: complete as a local research candidate. The `8192` candidate is the
current local winner for the inferred append-only closed-loop workload, but it
is not yet a Portal artifact. It changes the model's effective attention
horizon for long contexts, so semantic and rules review remain open.

## Fixed Setup

- Image: `localhost/part2-lfm25-r002k:fused-bf16-greedy` (vLLM `0.22.1`).
- Model mount: `LiquidAI/LFM2.5-1.2B-Instruct`.
- Server envelope: 3 CPU cores, 8 GiB RAM, local RTX 5080, GPU utilization
  `0.80` for both sides of every comparison.
- Unchanged R002k policy: FP8 per-tensor weights, FP8 E4M3 attention KV,
  BF16 aligned ShortConv state, prefix caching, chunked prefill, async
  scheduling, interactivity mode, `max-num-seqs=8`, b16384 and fused BF16
  greedy head.
- Candidate delta: inject `per_layer_sliding_window=8192` only into the six
  LFM2 full-attention modules. All ten ShortConv modules remain unchanged.
- Public replay: `trace_grading_public.jsonl`, 70 conversations, 90 primer
  requests and 330 graded requests. The primary driver is
  `append_only_static_assistant`, which resends the growing conversation
  history with deterministic assistant token IDs.

## Mechanism Verification

The startup assertion reports exactly:

```text
full-attention layers: [2, 5, 8, 10, 12, 14]
ShortConv layers:      [0, 1, 3, 4, 6, 7, 9, 11, 13, 15]
window:                8192
```

The R002k patchers and `FULL_AND_PIECEWISE` CUDA Graph path are active in both
control and candidate. The six-turn profile keeps the same cache progression:

```text
(computed, scheduled) =
(0, 4000), (4000, 4199), (8192, 4206), (12384, 4213),
(16592, 4204), (20784, 4210)
```

The candidate's attention cost plateaus after the window rather than growing
with retained history. Its startup KV pool is larger (`2,177,934` tokens
versus `1,649,210` in the control at the same memory fraction), so the total
serving effect is a combination of bounded attention work and extra cache
capacity. This is an intended resource trade-off, not a pure kernel-only
comparison.

## Isolated Profile Gate

Two candidate runs versus one control on graded conversation `15` pass the
registered opening gate. Candidate mean complete GPU time by turn is:

```text
turn 0: 62.552ms   control 60.782ms   +2.91%
turn 1: 75.881ms   control 73.763ms   +2.87%
turn 2: 84.643ms   control 83.607ms   +1.24%
turn 3: 84.542ms   control 93.569ms   -9.65%
turn 4: 82.963ms   control 103.034ms  -19.48%
turn 5: 81.448ms   control 114.735ms  -28.53%
```

The earlier Kineto run shows the same causal shape: attention grows from about
`4.8ms` to `54.7ms` in the unbounded control but remains near `19ms` with the
window. Cache progression is unchanged. See
`analysis/profile_comparison.json` and `analysis/candidate-kineto-c15-summary/`.

## Full Append-Only Replay

All four runs completed `420/420` requests with `90` warmup, `330` graded and
zero graded failures.

| run | score points | TTFT p50/p95 ms | TPOT p50/p95 ms | graded <=400ms |
|---|---:|---:|---:|---:|
| control A | 55.9491 | 117.31 / 325.15 | 2.600 / 4.131 | 324/330 |
| candidate A | 60.8540 | 96.17 / 271.78 | 2.421 / 3.646 | 330/330 |
| control B | 57.1409 | 111.21 / 305.97 | 2.543 / 4.187 | 323/330 |
| candidate B | 61.4875 | 94.52 / 268.01 | 2.394 / 3.677 | 329/330 |

Two-run means:

```text
candidate score:       61.1708
control score:         56.5450
absolute gain:          +4.6258 points (+8.18%)
TTFT p50:               95.35ms vs 114.26ms (-16.55%)
TTFT p95:              269.89ms vs 315.56ms (-14.47%)
TPOT p50:                2.408ms vs 2.571ms (-6.37%)
TPOT p95:                3.661ms vs 4.159ms (-11.97%)
requests <=400ms:      329.5 vs 323.5 on average
```

The candidate minimum (`60.8540`) is above the control maximum (`57.1409`),
and the candidate TPOT tail is not worse. The full machine-readable analysis
is `analysis/full_replay_comparison.json`.

Score gain by turn, candidate minus control, is:

```text
turn 0: -0.124 points
turn 1: +1.328 points
turn 2: +3.082 points
turn 3: +6.274 points
turn 4: +7.821 points
turn 5: +9.373 points
```

This monotonic depth relationship is stronger evidence than the aggregate
score alone. It matches the attention-growth diagnosis and does not resemble
a decode-only improvement.

## Independent 4k Boundary Check

To test the literal public-trace input bound, the same server was run on the
first 30 conversations in `independent` mode. This gives 90 graded requests,
with every prompt approximately 4k tokens and no accumulated history:

```text
control:   68.3485 points, TTFT p50/p95 71.87/132.96ms
candidate: 67.9362 points, TTFT p50/p95 68.97/127.77ms
delta:     -0.4123 points
```

Therefore the window is not a general 4k-prefill optimization. It is valuable
only if BTC's inferred closed-loop protocol resends enough prior conversation
history for the full prompt to exceed 8192 tokens. The official statement and
machine-readable contract describe this protocol as inferred, not explicitly
specified; this is the main deployment uncertainty.

## Quality Gate Evidence

The exact 100-case GPQA proxy has input lengths `104..847` tokens; neither an
8192 nor a 4096 window is active for those prompts plus the local eight-token
probe. Paired runs produced:

```text
control first:  21/100
candidate:       22/100
control repeat:  22/100
candidate-specific parsed/hash flips: none versus the control repeat
```

This passes the local short-prompt quality screen, but it cannot prove that
discarding attention beyond 8192 preserves long-conversation answer quality.
The official post-online GPQA gate remains mandatory.

## Decision

- Keep the `8192` overlay as the best local research candidate for the
  append-only closed-loop hypothesis.
- Do not merge it into R002k or create a Docker Hub image/Compose yet. The
  candidate is semantically non-equivalent for long contexts and the BTC
  request-body semantics are not explicit.
- Do not use the independent 4k result to reject the technique; it answers a
  different workload hypothesis and correctly shows the boundary condition.
- Next research should compare `4096`, `8192` and `16384` only after this gate,
  using a small long-context panel plus the same 100-case quality proxy. The
  aim is to find the least aggressive window that retains most of the score
  gain, not to assume `8192` is universally safe.

## Reproduction Paths

- `scripts/run_prefill_process.sh`: isolated six-turn profiling.
- `scripts/run_quality_process.sh`: paired GPQA proxy.
- `scripts/run_full_replay.sh`: full or bounded replay; set
  `R003_PROMPT_MODE=independent` and `R003_CONVERSATION_LIMIT=30` for the
  4k boundary check.
- `scripts/analyze_full_replay.py`: score-component and per-turn aggregation.
