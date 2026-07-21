# R002aa Round Plan

Status: phases A-E complete through immutable artifact creation. H200/Portal
target validation is pending.

## A. Source Audit

- Locate the scheduler split and both call sites.
- Trace align-mode state allocation and copy behavior.
- Verify causal-conv APC can write block checkpoints during one forward.
- Quantify all-mode memory before treating it as a candidate.

## B. All-Mode Proof

- Patch only the missing LFM2 constructor gate and ShortConv metadata wiring.
- Preserve R002k's fused greedy output head and R002y's profiler hooks.
- Run startup, six-turn output and timing panels for align and all.
- Reject on fallback, cache miss, runtime error or quality regression beyond
  the incumbent's measured nondeterminism.

## C. Sparse Dual Checkpoint

- Allocate one state at the crossed full-block boundary and one running state
  at the partial end of the request.
- Pass a compact state table to causal-conv instead of allocating every block.
- Keep decode and full-attention paths byte-for-byte unchanged.
- Add focused scheduler/allocation/state-index unit tests.

Implementation status: functional proof complete. Sparse capacity is
align-like and cached turns execute in one model step. Six focused
pure-Python tests pass, including the package-without-profiler contract.

## C2. One-Process Timing Qualification

- Alternate native and sparse requests inside one engine to remove startup,
  compile-cache, clock and thermal drift.
- Pre-JIT both paths before timing and reverse request order in the second
  pair.
- Screen no padding, pad16 and pad32 with identical prompt lengths.
- Accept a padding policy only after two idle-GPU repeats agree and background
  utilization is below the recorded contamination band.
- Reject the run before engine startup when the five-sample idle gate exceeds
  `10%` utilization or `2048 MiB` used memory.

Status: complete. Pad32 passed both micro repeats but failed complete replay.
No-pad was mixed in micro timing and therefore advanced only through matched
full replay. Two strictly clean pairs were positive; one pair with a `13%`
candidate sample is classified as contaminated and excluded.

## D. Promotion

- Run paired 100-question quality regression.
- Run matched complete public replays under the 3 CPU / 8 GiB envelope.
- Promote only after output, TPOT, TTFT, capacity and ERS gates all pass.
- Otherwise close R002aa and retain R002k.

Status: locally complete. No-pad passes paired accuracy, failure, capacity and
clean full-replay gates. The baked package also passes a 420-request functional
replay after removing the research profiler from the serving image.

## E. Immutable Target Artifact

- Push only the repaired no-pad image; never publish the profiler-dependent
  failure image.
- Pin the registry digest rather than the mutable tag.
- Keep every Attempt 04 flag unchanged so the target comparison changes only
  the serving image.
- Validate Compose syntax, one-variable diff and the 1 MiB upload limit.

Status: complete. Registry digest
`sha256:9e5586a637dfedec185a0665bb10284d00284be13cea0ce09f9238dfc7739765`
is pinned in Part 2 Attempt 06. R002k remains incumbent until that artifact is
probed on H200.
