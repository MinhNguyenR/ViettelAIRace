# R002aa Results - Sparse ShortConv Boundary Checkpoint

Status: active checkpoint. Functional and capacity proofs pass; final latency,
quality and replay gates are pending an idle GPU.

## Answer

The extra cached-turn model step can be removed without all-mode's large state
memory cost. The sparse implementation stores only the final crossed
full-block ShortConv state plus the final partial-tail state. It preserves
align-like cache capacity and changes turns 1-5 from two model steps to one.

This is not yet a submission candidate. The available no-pad/pad16/pad32
latency screens were collected while a game used the GPU and cannot choose the
final padding policy.

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

## Contaminated Directional Screen

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

## Next Gate

1. With the game closed, require a clean GPU background check.
2. Run two in-process A/B repeats for no-pad and pad32, using two reversed-order
   pairs per run.
3. Keep a policy only if cached GPU and TTFT deltas are positive in both runs.
4. Run cold 100-case GPQA and the complete 420-request public replay.
5. Promote only on no accuracy drop, no failures, TPOT regression below 3%
   and ERS gain beyond replay noise.

No image, Compose file or Portal submission has been created.

## Verification

The focused CPU-only suite passes `5/5` checks: cached-only final-boundary
scheduler bypass, one-block boundary reservation/allocation with cold-request
exclusion, cold metadata fail-closed behavior, the non-sparse all-mode manager
guard and runtime mode-file switching. Python compilation and shell syntax
checks also pass. The newly
strengthened cold-request native fallback and idle-GPU preflight still require
their first clean runtime exercise. A synthetic `35%` utilization / `3000 MiB`
preflight was rejected with exit code `3` before Docker startup, confirming the
contamination guard itself.
