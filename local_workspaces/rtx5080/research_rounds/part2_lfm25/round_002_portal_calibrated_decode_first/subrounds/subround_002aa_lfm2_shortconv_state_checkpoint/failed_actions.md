# R002aa Failed Actions

## Contaminated alignment screen

- Runs: `inprocess-nopad`, `inprocess-pad16`, `inprocess-pad32b` and
  `inprocess-pad32c` on 2026-07-20.
- Background state before launch: `23-39%` GPU utilization and roughly
  `126-174W`; the user confirmed a game had been running during the screen.
- Observation: in-process sparse still reduced cached-turn GPU mean by
  `2.992ms` no-pad, `0.712ms` pad16 and `3.744-3.894ms` pad32.
- Classification: useful directional evidence that one-step works and pad16
  is weak, but invalid for final policy selection or score projection.
- Resolution: stop GPU work and repeat no-pad/pad32 twice only after the game
  is closed and a clean background check passes.
- Prevention: the runner records five samples, hard-rejects above `20%` or
  `2048 MiB`, and allows at most one `10-20%` compositor spike for functional
  work. Performance analysis applies a stricter post-run `<=10%` rule.

## Packaged profiler dependency crash

- Run: `packaging/outputs/full_replay/functional-candidate-...-112850`.
- Symptom: startup and health passed, then the first request killed EngineCore;
  all 330 graded requests failed.
- Root cause: the initial package copied and installed R002y's research
  profiler without setting `R002Y_PROFILE_ARM`, `R002Y_PROFILE_FINALIZE` and
  `R002Y_PROFILE_OUTPUT_ROOT`. The first model execution raised
  `KeyError: R002Y_PROFILE_ARM`.
- Resolution: `_patch_runner` installs R002y only when all three variables are
  present; the serving Dockerfile no longer copies the profiler or adds its
  directory to `PYTHONPATH`.
- Verification: a focused no-profiler unit test passes and the rebuilt image
  completes `420/420`, with `330/330` graded successes.

## Pad32 micro-to-replay reversal

- Two clean in-process repeats showed cached GPU deltas of `-0.474ms` and
  `-1.889ms`, so pad32 passed the micro opening gate.
- Complete replay scored `55.0941` versus matched control `56.2308`; TPOT p95
  regressed by about `8.37%`.
- Classification: shape-local micro gain does not transfer through scheduler
  occupancy and decode contention. Pad32 is rejected and not packaged.

## Desktop-compositor replay contamination

- The third no-pad candidate sampled `13%` GPU in preflight and later lost
  `2.4908` points to its control. It is excluded by the strict analysis gate.
- Two later strict retries were rejected before Docker startup at `20-31%`,
  and a packaged strict retry was rejected at `41-46%`.
- `nvidia-smi pmon` attributed active graphics work primarily to `gnome-shell`;
  no competing compute process was present.
- Functional/quality runs may explicitly bypass the gate, but their latency
  and ERS must never enter performance promotion evidence.
- A later five-sample check after the game was closed still measured
  `17/19/18/3/2%` GPU utilization with `1242 MiB` used. The strict `10%`
  criterion rejected another packaged replay before Docker startup; no timing
  result was recorded from this attempt.

## Cross-process timing comparison

- Symptom: identical controls varied by more than the expected feature gain;
  one sparse/control pair suggested a 10-18ms gain while a later pair showed
  less than 2ms.
- Cause: separate model startup, GPU clock/thermal drift and desktop load.
- Resolution: replace this gate with alternating native/sparse panels inside
  one engine, pre-JIT both paths and reverse order in the second pair.

## Exact sequence identity as a hard gate

- Two identical align processes matched only `5/12` complete 16-token panels
  and `11/12` first tokens.
- Sparse matched `3/12` complete panels and `10/12` first tokens; all-mode had
  the same panel/first-token counts.
- Classification: exact greedy sequence identity cannot distinguish APC drift
  from incumbent FP8 near-tie nondeterminism. Keep it as a diagnostic and use
  cold GPQA plus full replay as promotion gates.

## Local memory admission at `.80`

- The first sparse proof requested slightly more memory than was free while
  desktop processes were active and failed before engine startup.
- Re-running at `.78` succeeded with `1,592,154` cache tokens.
- Classification: environmental admission failure, not a sparse allocation
  failure.

## All-mode parity manager guard

- Run: `parityall-all-20260720-180517`.
- Symptom: the first warmup request terminated EngineCore with
  `AttributeError: 'MambaManager' object has no attribute
  '_allocated_block_reqs'`.
- Cause: the sparse allocation wrapper read an align-only manager field before
  checking whether sparse mode was enabled. Timing and sparse proof runs were
  unaffected because their align manager owns that field.
- Resolution: return directly to the native allocator before touching sparse
  state whenever sparse mode is disabled or the manager is not in align mode.
- Classification: harness defect; this run contains no all-mode parity result.

Observed implementation failures:

- `proof-all-20260720-172030` reached Ready but was stopped before panels.
  `MambaModelConfig` changed the requested `all` mode back to `align` before
  model construction because LFM2 does not advertise the generic capability.
  The run exercised no candidate kernel and is not benchmark evidence. R002aa
  now overrides that capability only for architecture `Lfm2ForCausalLM`.

Rejected before implementation:

- Disabling `_mamba_block_aligned_split` alone. It loses the intermediate
  full-block ShortConv state and can silently corrupt later prefix hits.
- Treating all-mode as a submission candidate before measuring cache capacity.
  Its per-block BF16 state cost is materially larger than align mode.
