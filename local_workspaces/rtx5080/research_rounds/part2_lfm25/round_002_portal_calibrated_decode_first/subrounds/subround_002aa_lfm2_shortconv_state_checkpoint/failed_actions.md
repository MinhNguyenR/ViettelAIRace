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
- Prevention: the runner now rejects startup by default if any of five
  preflight samples exceeds `10%` GPU utilization or `2048 MiB` used memory.

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
