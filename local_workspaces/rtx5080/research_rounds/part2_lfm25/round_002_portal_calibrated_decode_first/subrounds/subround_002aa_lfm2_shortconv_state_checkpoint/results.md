# R002aa Results - Sparse ShortConv Boundary Checkpoint

Status: locally qualified no-pad target probe with immutable registry artifact.
R002k remains the Portal incumbent until R002aa is validated on H200.

## Answer

The extra cached-turn model step can be removed without all-mode's large state
memory cost. The sparse implementation stores only the final crossed
full-block ShortConv state plus the final partial-tail state. It preserves
align-like cache capacity and changes turns 1-5 from two model steps to one.

The local candidate uses sparse checkpointing with no shape padding. Pad32
looked best in micro timing but failed the complete-trace gate. Two clean
no-pad replay pairs and two quality runs pass; a baked image passes functional
replay without mounting source from the workspace.

## Structural Results

```text
align .78 capacity:              about 1.59M tokens
sparse .78 capacity:             1,592,154 tokens
all/block16 .78 capacity:        530,570 tokens
cached prefix progression:       4000/8192/12384/16592/20784
align cached model steps:        2
sparse cached model steps:       1
scheduled new prompt tokens:     4199/4206/4213/4204/4210
```

All-mode proves the packaged APC causal-conv path can write boundary states in
one forward, but its roughly threefold capacity loss rejects it as a serving
design. Sparse recovers the capacity by allocating one boundary checkpoint
per crossing request rather than one checkpoint per historical block.

## Output Diagnostic

Two conversations, six turns and 16 greedy output tokens per panel produced:

```text
comparison                 exact panels   first token   token-position match
align repeat vs align      5/12           11/12         58.8542%
sparse vs align            3/12           10/12         51.5625%
all/block16 vs align       3/12           10/12         58.8542%
```

The align repeat proves complete sequence identity is unstable in the current
FP8 stack. Sparse is slightly below that first-token noise floor, so cold GPQA
and full public replay remain mandatory. Cold turn-0 outputs matched in the
two tested conversations.

## Kernel Attribution

At cached turn 1, native align spent approximately `102.63ms` on the main
4192-token step and `4.24ms` on the seven-token tail step. Sparse used one
4199-token step. Kineto attributes only about `0.156ms` total to all ten
causal-conv kernels; checkpoint writing is not the limiting overhead.

The dominant variation comes from FP8/BF16 GEMM and normalization at different
token shapes. This opened a runner-native padding screen rather than a custom
ShortConv kernel.

## Initial Contaminated Directional Screen

All values below came from one-process alternating A/B, but the GPU already
showed `23-39%` utilization because a game was active:

```text
policy       cached GPU delta    cached TTFT delta
no pad       -2.992ms            -3.332ms
pad16        -0.712ms            -1.678ms
pad32 B      -3.744ms            -2.808ms
pad32 C      -3.894ms            -4.250ms
```

Negative is faster. These results support continuing sparse and dropping
pad16, but they do not authorize choosing no-pad or pad32.

## Idle Micro Qualification

After the game closed, two reversed-order A/B repeats produced:

```text
policy/run     cached GPU delta    cached TTFT delta
no-pad 1             +1.055ms             +0.945ms
no-pad 2             -1.704ms             -1.445ms
pad32 1              -0.474ms             -0.242ms
pad32 2              -1.889ms             -1.539ms
```

No-pad did not pass the micro repeat-sign gate. Pad32 did, but that gate only
opened full replay; it did not authorize promotion.

## Complete Replay

Pad32 scored `55.0941` against matched control `56.2308`, a `-1.1367`-point
loss. It also raised TPOT p95 from `4.1627ms` to `4.5111ms`. Reject padding.

No-pad then ran three matched pairs. The analyzer requires every preflight GPU
sample to be at most `10%` before a pair contributes to promotion evidence:

```text
pair   control   candidate   delta      max GPU preflight   classification
1      56.2308    56.4657   +0.2348          6% / 6%        clean
2      54.3962    55.9040   +1.5078          8% / 4%        clean
3      56.5328    54.0420   -2.4908          4% / 13%       contaminated
```

The two clean pairs both finish `420/420` with zero graded failures. Their
mean score delta is `+0.8713` points; mean TTFT delta is `-6.4606ms`, while
mean TPOT delta is `+0.0395ms`. The contaminated pair is retained in the
record and excluded by the registered environmental criterion.

## Quality Gate

Two no-pad runs over the fixed 100-case GPQA proxy both score `22/100`, versus
R002k `21/100`:

```text
run                 A          B
request errors      0          0
parse failures     10         12
prediction changes  4          8
reference correct -> wrong
                     0          0
reference wrong -> correct
                     1          1
```

The exact prediction guard remains false because the FP8 incumbent itself is
not sequence-stable, but both accuracy and reference-correctness guards pass.
The quality run intentionally ignored desktop latency; GPU background cannot
alter its pass/fail arithmetic in the absence of request errors.

## Packaging Gate

The first baked image included R002y's profiling hook but omitted its research
path variables. The first request raised `KeyError: R002Y_PROFILE_ARM`, killed
EngineCore and caused all 330 graded requests to fail. The serving overlay now
installs R002y only when all three profiler variables exist, and the submission
image no longer includes the profiler.

The rebuilt image has local digest
`sha256:9e5586a637dfedec185a0665bb10284d00284be13cea0ce09f9238dfc7739765`
and size `9,223,634,914` bytes. It starts with sparse mode and R002k fused
greedy enabled, leaves R002y disabled, and completes `420/420` requests with
`330/330` graded successes. That functional run sampled `41-46%` desktop GPU
before launch; its `52.5512` score is explicitly not performance evidence.

## Decision

Keep R002aa as a locally qualified target probe and retain R002k as incumbent.
The immutable digest is pinned in Part 2 Attempt 06; the next gate is its H200
execution, not another local tuning sweep. Do not submit pad32, do not include
R002y in the image, and do not use the contaminated functional score for
expected Portal performance.

## Verification

The focused CPU-only suite passes `6/6` checks: cached-only final-boundary
scheduler bypass, one-block boundary reservation/allocation with cold-request
exclusion, cold metadata fail-closed behavior, the non-sparse all-mode manager
guard, runtime mode-file switching and a package path that cannot import R002y
without profiler configuration. Python compilation, shell syntax and
ShellCheck pass. Idle preflight repeatedly rejected `20-46%` desktop activity
before Docker startup, and the baked image completed its full functional run.
A post-package idle check after the game was closed still measured
`17/19/18/3/2%` GPU utilization (`1242 MiB` used), so the strict gate rejected
another replay before Docker startup. No additional latency claim is inferred.
