# Part 2 Trace Requalification Plan

Status: active. The Part 2 statement and metadata trace are available. The
hidden prompts and exact client request body remain unavailable, so local
benchmarking must bracket prefix-reuse behavior instead of assuming it.

## Objective

Qualify the native LFM2.5 vLLM stack for a closed-loop, six-turn conversation
workload without carrying Qwen3.5 sweet spots by assumption.

## P2-000 - Contract And Driver

Confirm:

1. model repository and revision;
2. hardware and container limits;
3. score formula and quality gate;
4. whether hidden prompts share reusable prefixes across turns;
5. how conversation state reaches the server;
6. whether warmup and graded conversations share one execution timeline.

Exit gate: an official contract and locally reproducible no-reuse and
high-reuse request drivers.

## P2-001 - Native Baseline

Run the simplest supported vLLM path with no lossy precision changes. Record
per turn:

- TTFT, TPOT, total latency and output count;
- queue, prefill and decode time;
- request-attributed cache hit and compute tokens;
- active request count and decode batch size;
- Transformer KV and Mamba state allocation/eviction;
- CPU, RAM, VRAM and startup behavior.

Run at least two processes to establish variance.

## P2-002 - Session Reuse Gate

Compare prefix cache off/on and verify the full six-turn lifecycle.

Required proof:

- turn 0 computes its prompt normally;
- turns 1-5 compute only new suffix work when history is append-only;
- both Transformer KV and hybrid Mamba state resume correctly;
- state remains resident across the 3000 ms think interval;
- completed warmup sessions do not evict live graded sessions;
- outputs remain identical under deterministic decoding.

The statement bounds each request context at about 4k tokens. Do not model the
six turns as an append-only 25k-token final request. Prefix reuse remains a
valuable hidden-workload variable, but the public metadata cannot quantify it.

## P2-003 - Precision Factorial

Test independently, then combine:

1. online FP8 linear weights;
2. FP8 E4M3 attention KV;
3. BF16 LFM2 short-convolution state;
4. the three-way combination.

Each lossy change requires the official quality gate. Preserve native BF16
recurrent compute; prior FP8/INT8 recurrent-state kernels were slower and
drifted.

## P2-004 - Concurrency And CUDA Graphs

The revised model is much smaller than Qwen3.5 and each request is only about
4k input tokens. Start with low-concurrency latency and expand only if measured
service time predicts it:

```text
max active: 4, 6, 8, 10, 12
decode graph sizes: 1, 2, 4, 6, 8, 10, 12
```

Promote by score distribution and TPOT tail, not maximum throughput alone.
The old batch-20 graph focus is no longer the default.

## P2-005 - Prefill Budget

Each new turn is approximately 4000 input tokens. Compare:

```text
2048
4096
8192 or whole-prefill
```

Measure scheduler overhead and decode interference. A 4096 budget is now a
stronger hypothesis than it was on the old 13K-27K burst trace, but it is not
preselected.

## P2-006 - Decode Path

The graded output cap is 66,000 tokens, 2.75 times Phase 1. Re-test:

1. native online FP8 methods before any custom tied-head path;
2. graph coverage for the measured common decode batches;
3. continuous batching and decode-step policy;
4. automatic H200 attention/GDN backends.

The tied head is the first custom path to revisit. GDN recurrent kernels remain
closed until target profiling shows a new opportunity.

## P2-007 - Runtime Bakeoff

Only vLLM is permitted by the revised statement. SGLang and TensorRT-LLM are
out of scope for submission.

## P2-008 - Final Integration

Integrate only independently positive techniques. Repeat:

- performance A/B/A;
- quality gate;
- 3-core/8GB resource enforcement;
- startup and healthcheck;
- full warmup plus graded timeline;
- Docker package rollback.

## Explicitly Deprioritized

- exact trace priming, stored outputs or request-identity shortcuts;
- six-wave Phase 1 schedulers;
- Tree-Gated routing from static request shape, because Part 2 lengths and
  output caps are nearly constant;
- CPU/NVMe offload and physical prefill/decode disaggregation;
- forced attention/GDN backends that duplicate or lose to H200 auto selection.

## H200 Calibration - 2026-07-18

R002k's strict non-lossy Triton fused greedy head transferred successfully:
Attempt 04 scored `60.75` versus native-image Attempt 02 at `58.17`. TTFT
p50/p95 improved `58/82 -> 44/80ms`, failures fell `8 -> 6`, and median TBT
remained `4ms`.

Carry-forward rule: preserve R002k in every performance candidate unless the
submission is an explicit attribution or native rollback. New custom work must
target a whole-request opportunity of at least `1%` or a measured failure/tail
cohort; small scalar flag sweeps remain closed.

R002m closes the remaining output-head cleanup. Direct output ownership gains
only `0.011%` weighted GPU time, the absolute no-clone floor is `0.489%`, a
transposed BF16 copy loses despite 256 MiB extra VRAM, and manual GEMV loses
about `154%` across the measured batch mix. Future custom-kernel work must move
inside a profile-proven model hotspot; do not reopen head allocation/layout
without a new target-architecture profile above the 1% whole-request gate.

The exact R002k Portal repeat scored `58.94` after the first `60.75`, producing
a two-run mean `59.845` and range `1.81`. A future candidate must not advance
from one Portal score inside that range. Require a projected effect clearly
above two points, repeated local A/B evidence, or a separately identified
failure/tail mechanism before spending a submission.

R002n-R002t exhaust the lossy ShortConv, FP8 activation, graph-fusion, backend,
block-scaling and standalone FP8-epilogue branches. The final FP8 epilogue has
only a `1.2119%` zero-overhead ceiling and loses `36.34%` end to end when
implemented separately. R002u therefore reopens only the measured BF16
ShortConv projection family (`19.499%` of local decode child-kernel time).
It must preserve checkpoint dtype and state/cache policy, beat exact native
projection shapes by enough to project above 1% complete-step gain, and pass
numerical gates before any live integration.

R002u completes that BF16-only screen. The measured family ceiling is real,
but the apparent single-weight gain was an L2-residency artifact. Cycling all
320 MiB of ten-layer projection weights leaves best M8 Triton slower than the
production anchor and projects `-0.516%` complete-step gain. Standalone
ShortConv GEMM replacement is closed; future work requires fusion with a
consumer or target-only evidence, not another tile sweep.

R002v then isolates the previously positive ShortConv `in_proj` FP8 factor.
Matched host candidates beat control twice, and the immutable package beats a
matched packaged control twice by `+3.0660/+3.6943` ERS with zero failures.
Expandable allocator segments are mandatory: the package without them OOMed
on a fragmented 512 MiB long-prefill allocation. The fixed quality proxy has
zero aggregate drop but one correct-to-wrong movement. The H200 pair is
`60.19` (`50/90ms`, `6/420` failures) and `59.34` (`61/97ms`, `5/420`), for
mean `59.765`; R002k's exact pair is `59.845`. Close R002v as a neutral target
probe, retain its allocator fix as reliability evidence, and require a fused
consumer or another distinct mechanism before reopening ShortConv precision.

## R003 Follow-Up - Long-History Attention Bound

The driver ambiguity is now measured rather than left implicit. R003 runs both
the documented independent-4k boundary and the inferred append-only
closed-loop branch. An `8192` window in the six full-attention layers is
positive only on the latter: two full-replay means are `61.1708` versus
`56.5450` control, while the bounded independent-4k screen is
`67.9362` versus `68.3485`.

This does not override P2-002's warning that BTC has not published the exact
request body. It establishes a conditional research result: if BTC resends
conversation history, attention-window work and cache capacity are promising;
if BTC sends complete 4k prompts, the technique should be removed. R003 is
therefore not a submission artifact. The next research step is the
`4096/8192/16384` window-size and long-context quality comparison under the
same two semantics.
