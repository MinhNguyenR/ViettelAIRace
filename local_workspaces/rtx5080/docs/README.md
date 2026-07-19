# RTX 5080 Lab Docs Map

Read this file first for local RTX 5080 lab notes.

## Folder Meaning

| Folder | Meaning |
|---|---|
| `baselines/` | Local baseline notes. These are not official BTC H200 baselines. |
| `experiments/` | Completed local experiment notes. |
| `workflow/` | Local measurement workflow and scope rules. |

## Files

- `roadmap.md` - final synthesis roadmap; keep locked until local evidence and
  H200 calibration are read together.
- `round_workflow_checklist.md` - required read-before-round checklist and after-round update rules.
- `model.md` - fixed local `/model` mount and downloaded model notes.
- `script_policy.md` - where common scripts and round-specific scripts should live.
- `round_baselines.md` - quick baseline tracker across rounds.
- `config_registry.md` - current local best config, best technique/framework signal, fallbacks and carry-forward policy.
- `model_contract_audit.md` - local model artifact audit; confirms Qwen3.5 hybrid linear-attention/full-attention behavior and warns against pure Transformer KV assumptions.
- `metric_tuning_playbook.md` - metric-to-technique playbook for TTFT, TPOT, throughput, VRAM and stability trade-offs.
- `technique_ledger.md` - shared technique notes across rounds; use this instead of one technical note per round.
- `failed_actions_log.md` - separate log of failed/rejected actions so we do not repeat them blindly.
- `../research_rounds/part2_lfm25/round_003_hybrid_attention_window/README.md` -
  completed `8192` per-layer full-attention window study; append-only replay
  gains `+4.6258` local points over matched controls, while independent 4k
  traffic is neutral/slightly negative. Research-only pending semantic/rules
  review.
- `../research_rounds/part2_lfm25/round_003_hybrid_attention_window/subrounds/subround_003a_window_sweetspot/README.md`
  - completed attention-window frontier. `4096` is the repeated local
  score-max branch at `64.5504` but fails paired long-context retrieval;
  `8192` is the balanced branch. Neither changes the R002k submission
  incumbent.
- `../research_rounds/score_100_program/scoreboard.md` - compact R000-R039
  integration results; I006 seq16/default2048 remains the Portal champion and
  rollback at `51.59`.
- `../research_rounds/score_100_program/champions.json` - machine-readable
  decode, TTFT and integrated champion registry.
- `../research_rounds/score_100_program/round_010_readiness_warmup_integration/README.md`
  - R010 answer, package architecture, A/B/A, relay isolation and frozen
  Attempt 11 digest.
- `../research_rounds/score_100_program/round_034_gptq_w4a16_marlin/README.md`
  - completed GPTQ W4A16/G128 artifact, quality and same-image performance
  qualification; quality-safe but rejected after `-10.974%` relative ERS.
- `../research_rounds/score_100_program/round_035_timed_jit_accounting_audit/README.md`
  - exact prebenchmark/timed log-boundary audit; corrects R034 to five
  pre-readiness JIT warnings and zero timed warnings without a GPU run.
- `../research_rounds/score_100_program/round_036_h200_transfer_residual_audit/README.md`
  - no-GPU local/Portal component audit; identifies two H200 decode-dominant
  comparisons and the `56.666667` score ceiling at current TTFT support.
- `../research_rounds/score_100_program/round_037_exact_i006_h200_decode_capability_audit/README.md`
  - exact I006/vLLM 0.22.1 capability inventory; classifies `14` paths, closes
  DS conv-state layout below ceiling and rejects easy MRV2 variants without a
  GPU run.
- `../research_rounds/score_100_program/round_038_first_token_staging_opportunity/README.md`
  - six-panel causal opportunity model for a one-model serial first-token
  staging lane; opens only a gap-aware scheduler design under strict
  queue/interference/TPOT gates.
- `../research_rounds/score_100_program/round_039_gap_aware_scheduler_prototype/README.md`
  - policy plus exact-image shadow scheduler probe; `13/13` tests and
  native/shadow/native 20-request mechanism runs pass, but no mutating stage
  candidate is promoted.
- `../../../docs/techniques/phase_aware_kv_pipeline.md` - architecture note for
  the proposed single-model TTFT/TPOT phase split using internal KVHandle
  handoff.
- `../research_rounds/ready_rounds/round_008_engine_native_ers_scheduler/README.md` - active engine-native two-policy round and Phase A result.
- `../research_rounds/ready_rounds/round_008_engine_native_ers_scheduler/scheduler_source_map.md` - pinned SGLang 0.5.14 scheduler/cache source contract and telemetry hooks.
- `../research_rounds/ready_rounds/round_008_engine_native_ers_scheduler/subround_008d_conservative_active_rescue/README.md` - rejected global-front active probe, causal score decomposition and locality-preserving follow-up constraint.
- `../research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006j_native_tree_gated_v2/README.md` - scheduler-outcome dataset, C++ Tree-Gated v2 latency audit and bounded rejuvenation replay.
- `../research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006k_cache_local_tree_vs_heuristic/README.md` - offline Tree/heuristic comparison and rejected exact-cache-node live shadow path.
- `../research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006l_parent_cache_nonposition_replay/README.md` - offline ancestry reconstruction and rejected static no-position cache ranking.
- `baselines/baseline_20260707_111358.md` - first local RTX 5080 baseline.
- `experiments/experiment_001_sweetspot.md` - local sweet spot search result.
- `workflow/measurement_workflow.md` - how to measure and summarize local runs.
- `workflow/rules_and_scope.md` - what local lab results can and cannot prove.

## Research Rounds

Research rounds live outside this docs folder:

```text
local_workspaces/rtx5080/research_rounds/
```

Each round has its own folder for code, configs, outputs and a result note.

Always update:

```text
local_workspaces/rtx5080/research_rounds/warmup_rounds/round_answers.md
local_workspaces/rtx5080/research_rounds/ready_rounds/round_answers.md
local_workspaces/rtx5080/docs/technique_ledger.md
local_workspaces/rtx5080/docs/failed_actions_log.md
```
