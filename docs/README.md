# Docs Map

Read this file first. It is the map for all project notes.

## Active Part 2 Requalification - Read Before Everything Else

BTC has published the revised Part 2 statement. The active model is
`LiquidAI/LFM2.5-1.2B-Instruct`; the public workload is 70 closed-loop,
six-turn conversations with 90 primer requests and 330 graded requests. The
old Qwen3.5 contract and all Score 100 rounds remain historical evidence, not
numeric baselines for the active optimization contract.

Read in this order:

1. `docs/official/de_bai.md` - active raw BTC statement.
2. `competition_reset/rapid_requalification_kit/contracts/part2_lfm25_official.json`
   - machine-readable active contract and provenance assumptions.
3. `docs/planning/part2_trace_requalification_plan.md` - ordered native,
   precision, concurrency, prefill and decode research plan.
4. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_001_one_shot_requalification/README.md`
   - active first-submission research round.
5. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_001_one_shot_requalification/results.md`
   - measured BF16/FP8/scheduler/quality matrix and one-shot decision.
6. `portal_submissions/part2_attempt_01_lfm25_fp8pt_interactive_seq8.yml`
   - graded first Part 2 Portal artifact: ERS `56.53`, TTFT p50/p95
     `73/116ms`, TBT median `4ms`, zero failures.
7. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/README.md`
   - active Portal-calibrated decode-first research round and evidence rules.
8. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/results.md`
   - complete FP8-KV, async, runtime, VRAM and budget results; b16384 is
     promoted at two-run mean `68.310` local ERS.
9. `portal_submissions/part2_attempt_02_lfm25_fp8pt_kvfp8_async_interactive_seq8_b16384.yml`
   - graded native-image rollback at ERS `58.17`, TTFT p50/p95 `58/82ms`, TBT
     median `4ms`; prefix caching and Mamba `align` remain required.
10. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002a_turn_semantics_bracket/results.md`
    - fixed-size versus append-only causal bracket; fixed-size full A/B favors
      disabling prefix caching `72.068 -> 68.458`, while append-only requires it.
11. `portal_submissions/part2_attempt_03_lfm25_fp8pt_kvfp8_async_no_prefix_seq8_b16384.yml`
    - graded negative hedge at ERS `47.91`, TTFT p50/p95 `98/201ms`; no-prefix
      and Mamba `none` are rejected for the hidden append-only driver.
12. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002b_fixed4k_no_prefix_tuning/README.md`
    - completed no-prefix fixed-4k tuning round; budget, chunking, seq6 and
      throughput produced no repeated gain, so Attempt 03 remains unchanged.
13. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002c_ngram_gpu_spec_decode/README.md`
    - completed negative built-in speculative-decode study; only `1.15%` of
      drafts were accepted and FlashInfer lost its full graph path.
14. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002d_fp8_tied_lm_head/README.md`
    - output-only tied-head FP8 implementation and fixed-size qualification.
15. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002e_prefill_admission/results.md`
    - completed scheduler-admission negative screen.
16. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002f_append_only_head_integration/results.md`
    - matched full-trace output-head result: `58.857` versus `56.729`, blocked
      from packaging by token drift.
17. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002g_hadamard_fp8_head/README.md`
    - pure Hadamard output-head transform; closed for weak full-sequence
      agreement.
18. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002h_signed_hadamard_fp8_head/README.md`
    - deterministic signed-Hadamard variant; closed without promotion.
19. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002i_residual_fp8_head/README.md`
    - low-rank residual repair; closed after weak exact-sequence agreement.
20. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002j_candidate_coverage/README.md`
    - tensor top-k512/1024 coverage bracket; no quality-safe promotion.
21. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002k_fused_bf16_greedy_head/README.md`
    - non-lossy Triton fused greedy head; full local ERS `58.3989` versus
      native `56.7287`, zero graded failures; clean image startup, native
      fallback and 100-case GPQA qualification all pass.
22. `portal_submissions/part2_attempt_04_lfm25_r002k_fused_greedy_seq8_b16384.yml`
    - exact R002k digest graded twice at ERS `60.75/58.94` (mean `59.845`,
      range `1.81`); best TTFT p50/p95 is `44/80ms`, TBT median remains `4ms`.
23. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002l_selective_final_rmsnorm/README.md`
    - completed exact prefill-only final RMSNorm deferral. Numerical and source
      gates pass, but a full B/A/B bracket fails repeatability and TPOT guards;
      closed without changing R002k Attempt 04.
24. `competition_reset/rapid_requalification_kit/README.md` - executable
   contract diff, trace audit, scorer and technique matrix.
25. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002m_decode_head_v2/README.md`
   - completed exact output-head V2 audit. Direct ownership is neutral,
     transposed BF16 loses with 256 MiB extra VRAM, and manual GEMV repeats
     weight traffic; no candidate changes R002k.
26. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002n_decode_hotspot_profile/README.md`
   - completed Kineto decode-hotspot profiling and ShortConv FP8 ablation.
     `in_proj` is retained as a research candidate after positive paired local
     replay evidence; `out_proj` and all-projection FP8 are closed, and no
     submission artifact exists.
27. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002o_phase_aware_short_conv/results.md`
   - completed one-model, phase-aware ShortConv projection study. Decode-only
     FP8 passes startup, graph, profile and local quality gates, but two exact
     full replays score `53.2023/52.9304` versus control `55.4200`; the branch
     is closed without an image or submission artifact.
28. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002p_fp8_activation_scale_audit/results.md`
   - completed FP8 activation-scale audit. Static scaling removes 44 dynamic
     quant producers and gains `1.2296%` batch-8 child-kernel time, but both
     prefill-only and decode-aware scale maps fail repeated quality. No replay,
     image or submission artifact was promoted.
29. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002q_graph_native_fp8_fusion_audit/results.md`
   - completed exact-image graph-native FP8 fusion audit. Norm fusion matches
     five patterns but gains only `0.597%` complete decode time; activation
     matches zero and attention changes graph mode with zero matches. No
     quality, replay, image or submission artifact was promoted.
30. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002r_fp8_linear_backend_screen/results.md`
   - completed exact-image FP8 backend screen. Auto already selects CUTLASS;
     FlashInfer/Marlin cannot initialize this layout and Torch is `0.0396%`
     slower in a low-drift A/B/A profile. No quality, replay, image or
     submission artifact was promoted.
31. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002s_online_block_fp8_target_transfer/results.md`
   - completed online block-FP8 target-transfer audit. H200 SM90 has a real
     FlashInfer/DeepGEMM path, but the local fallback loses `4.2785%`, quality
     falls `21 -> 18/100`, and replay ERS falls `58.3989 -> 49.3582`; closed
     without a package or submission artifact.
32. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002t_small_m_fp8_kernel_feasibility/results.md`
   - completed exact small-M FP8 primitive and custom Triton epilogue screen.
     Both paths are bit-exact, but direct BF16 scaling loses at least `35.51%`
     and the custom M=8 path loses `36.34%` end to end; no integration or
     submission artifact was opened.
33. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002u_bf16_shortconv_kernel_screen/README.md`
   - completed exact BF16 kernel screen for ten 2048->6144 and ten 2048->2048
     ShortConv projections. A single-weight loop was L2-biased; the 320 MiB
     cyclic confirmation projects `-0.516%` complete-step gain and closes the
     branch before integration.
34. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002v_shortconv_in_fp8_target_qualification/README.md`
   - completed isolated ShortConv `in_proj` FP8 qualification. Two packaged
     candidates beat matched control by `+3.0660/+3.6943` points; expandable
     allocator segments resolve the long-prefill OOM, GPQA aggregate drop is
     zero; the H200 pair is `60.19/59.34` with mean `59.765`, so R002k remains
     the incumbent and the precision overlay is not promoted.
35. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002w_shortconv_decode_epilogue_fusion/README.md`
   - completed bit-exact decode-only ShortConv epilogue/state fusion. The
     kernel gains `2.86%` median steady decode throughput, but candidate mean
     full-replay delta is `-0.2689` ERS; closed without a Portal artifact.
36. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002x_production_graph_step_attribution/results.md`
   - complete production graph-step attribution. Exact telemetry shows
     `98.744%` pure-decode steps, but direct M8 saving is only `0.039981ms` and
     M8 complete GPU time regresses `1.687%`; standalone ShortConv fusion is
     closed.
37. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002y_cached_turn_prefill_attribution/results.md`
    - complete cached-turn first-token attribution. Prefix cache progression is
      valid, but attention explains `92.64%` of turn-depth GPU growth. Local
      SM120 uses FlashInfer while H200 SM90 uses FA3, so no local kernel or
      submission is promoted.
38. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_003_hybrid_attention_window/README.md`
    - completed the bounded full-attention window study. An `8192` window on
      the six attention layers raises append-only full-replay local ERS from
      a two-run control mean `56.5450` to `61.1708`, with lower TTFT and TPOT;
      independent 4k traffic is neutral/slightly negative, and long-context
      semantics/rules review still block Portal promotion.
39. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_003_hybrid_attention_window/subrounds/subround_003a_window_sweetspot/README.md`
    - completed the `4096..16384` score/quality frontier. Window `4096`
      repeats at mean `64.5504` but loses two paired retrieval cases; `8192`
      remains the balanced branch with no paired losses. R002k remains the
      submission incumbent and no bounded-window artifact is published.
40. `local_workspaces/rtx5080/research_rounds/part2_lfm25/round_003_hybrid_attention_window/subrounds/subround_003e_prefix_recent_triton/results.md`
    - completed native Triton prefix-plus-recent screen. The no-prefix window
      path is fast, but every non-empty prefix range adds about `112ms` at
      turn 5 versus that control. Exact source also proves the API does not
      retain prefix keys for later queries, so the branch closes before replay.

Do not reuse Qwen-specific numeric sweet spots or artifacts in a Part 2
submission without a fresh LFM2.5 measurement.

## Reading Order

1. `docs/official/de_bai.md` - raw official problem statement pasted from BTC.
2. `docs/official/phase1_official_problem_notes.md` - interpreted official rules, scoring, environment and submission workflow.
3. `docs/official/submission_contract_and_flag_policy.md` - what the official compose must preserve and which flags can be tuned.
4. `docs/baselines/phase1_official_baseline.md` - BTC baseline image, command and flag set.
5. `docs/methodology/optimization_targets.md` - what must decrease, what must increase and which levers to try.
6. `docs/methodology/benchmark_methodology.md` - how to replay trace and record comparable metrics.
7. `docs/planning/phase1_roadmap.md` - roadmap distilled from completed work.
   `docs/planning/score_100_architecture_plan.md` - staged Score 100
   architecture; R000-R039 and H200 Attempts 10/11/12/13/14/16 have measured
   answers.
   R021a promoted the packaged FP8 tied head into local I007; R021b keeps
   seq16. R018 hybrid snapshots are rejected by their compute/VRAM bound.
   R022a confirms local I006 auto already resolves the only executable FP8
   linear backend, CUTLASS; no forced backend proceeds to full trace.
   R022b proves exact120+seq20 locally. R022c qualifies that preprocessing
   probe at ERS `0.999870`, then repeats exact local ERS `1.000000` with zero
   GPQA flips. R022d packages the single user-authorized risk trial; Attempt
   13 scored `49.45`, so the exact-trace branch is closed and I006 seq16
   remains the H200 champion at `51.59`.
   R022e proved fixed-checkpoint MTP capability; R022f rejected it on ERS,
   TTFT, TPOT and throughput, so R022g/R022h are closed.
   R023a then proved `2,240` duplicate in-flight tokens at block560. R023b
   rejected admission-only singleflight because the corrected hold stalled
   hybrid-cache publication and increased compute. R023c/R023d found no
   retention miss or exact runtime-render hit. R023e deadline scheduling and
   R023f bounded burst admission both lost their A/B/A controls. R023g proved
   one-storage FP8 tied weights save about `484-485MiB` statically but lose
   runtime cache/workspace margin and miss the TPOT guard. R023h found no
   remaining profile-backed kernel above its opening threshold. R023 is
   closed; I006 seq16 remains unchanged.
   `docs/planning/score_100_research_roadmap.md` - complete active A-Z queue
   from R000 evidence through R039 static scheduler research. Read it before
   opening any optimization round.
   `docs/planning/post_attempt13_research_plan.md` - completed post-Attempt13
   execution overlay: all R023 generic reuse, scheduling, storage and hotspot
   candidates are resolved; it records why none entered integration.
   `docs/planning/post_r023_deep_research_master_plan.md` - completed Deep
   Cycle 2 roadmap. It ran DR00-DR21 before conservative R024/R025 and covers
   opportunity modeling, wave-local scheduling, CPU/frontend work, sparse
   hybrid state, Triton/CUDA kernels, precision, framework capability,
   integration and H200 transfer, with explicit green/yellow/red compliance
   zones.
   `local_workspaces/rtx5080/research_rounds/score_100_program/scoreboard.md` -
   compact R000-R039 result table, Portal anchors and current champions.
   `local_workspaces/rtx5080/research_rounds/score_100_program/NEXT_GPU_RUNS.md`
   - exact reproduction history and current queue through R039. Deep
   Cycle 2 DR00/DR01/DR02/DR03 are resolved, DR03b is a local integration
   watchlist, DR02 closed new kernel work, and DR07 closed cache/allocator,
   sparse-state and native-scheduler work. DR16 completed its static precision
   screen; R034 later completed GPTQ W4A16 conversion/quality/runtime and
   rejected Marlin after a `10.974%` ERS loss. DR20 integration/removal ablation
   independently confirmed DR03b's performance gain but rejected it on the
   pre-registered quality-attribution repeat. Deep Cycle 2 is complete without
   a promoted primitive. R024 passed the conservative I006 seq16 freeze. R025
   then selected I007 seq19 and froze Attempt 14. R026 completed the graph
   follow-up; graph11 is only a weak H200 probe and larger additions lost.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_024_final_quality_reliability_package_gate/README.md`
   - canonical R024 answer, immutable I006 artifact, aggregate gates and
   retained TPOT/process-variance warnings.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_025_score_sprint/README.md`
   - completed one-shot score sprint. It records the seq19 win, rejected
   aggressive warmup, GPQA gate and immutable Attempt 14 package.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_026_seq19_transition_graph/README.md`
   - completed seq19 graph research. R026a parked graph18; R026b found graph11
   practically neutral after reverse-order confirmation, rejected graph6+11
   and skipped graph13. Attempt 14 remains immutable.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_027_seq18_h200_interpolation/README.md`
   - completed active-set interpolation. Seq18 passed all local probe gates,
   but Portal Attempt 16 scored `47.04`; decode improved while TTFT p95 rose to
   `8799ms`. The branch is closed and I006 seq16 remains champion.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_028_i006_prefill_budget4096/README.md`
   - completed exact-I006 `2048/4096/2048` A/B/A. Fixed `4096` improved the
   raw TTFT tail and TPOT but lost `1.43%` ERS because squared TTFT score fell
   in all six waves; default/resolved `2048` remains selected.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_029_score_shape_policy_opportunity/README.md`
   - completed legal-feature oracle and leave-one-wave-out audit. Robust oracle
   gain was only `0.351%` and depth-2 CV was negative, so dynamic budget
   switching closed without GPU.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_030_prefix_metric_semantics_audit/README.md`
   - corrected R028 aggregate metric semantics from exact I006 source. Raw
   lookup hits include repeated attempts; true request cache attribution fell
   only `0.143%`. R028/R029 decisions remain closed.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_031_i006_prefill_lifecycle_attribution/README.md`
   - six-run bidirectional first-token lifecycle attribution. Fixed `4096`
   lowers queue time but increases scheduled-to-first-token service and loses
   five of six wave scores; no scheduler or Attempt 17 is promoted.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_032_final_chunk_isolation_opportunity/README.md`
   - offline `3 x 3` process opportunity bound. Half recovery is unstable in
   process, wave and very-fast cohorts, so final-chunk isolation closes before
   GPU or implementation.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_033_dr03b_score_first_requalification/README.md`
   - score-first DR03b requalification. Historical Accuracy Gate evidence
   passes, but fresh same-image off/on/off loses `0.257%` ERS and fails the
   wave/request distribution gates; quality confirmation and Attempt 17 are
   skipped by dependency.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_034_gptq_w4a16_marlin/README.md`
   - completed isolated GPTQ W4A16/G128 conversion, artifact and three-process
   GPQA gate. Native Marlin startup passes, but full-trace ERS loses `10.974%`,
   TPOT p99 reaches `23.213ms`, and all six waves lose; no package or Attempt
   17 is created.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_036_h200_transfer_residual_audit/README.md`
   - completed no-GPU local/Portal transfer audit. It proves two H200
   decode-dominant comparisons, shows median TBT can hide an unsaturated mean
   TPOT component, and bounds I006 seq16 at `56.666667` without more than its
   current `16/120` TTFT support.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_037_exact_i006_h200_decode_capability_audit/README.md`
   - completed exact-image vLLM 0.22.1 source/capability audit. Fourteen
   primitives were classified, none passed the `0.5`-point opening gate.
   `VLLM_SSM_CONV_STATE_LAYOUT=DS` is real but causal convolution is only
   `1.296%` of pure decode and `1.196%` of cold prefill; MRV2 remains
   incompatible with retained Mamba `align` prefix caching.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_038_first_token_staging_opportunity/README.md`
   - causal timing-only opportunity model for one-model first-token staging.
   A guarded gap-aware point gains `+0.016036` mean ERS across `6/6` process
   panels and `36/36` waves while keeping modeled TPOT p95/p99 at
   `18.183/19.032ms`. It opens a design contract only, not a GPU run or
   submission artifact.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_039_gap_aware_scheduler_prototype/README.md`
   - gap-aware lifecycle policy plus exact-image shadow `AsyncScheduler` probe.
   Thirteen deterministic boundary/behavior tests pass and native/shadow/native
   20-request runs are clean; no mutating stage action or score-bearing serving
   candidate was promoted.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_040_parked_state_scheduler/README.md`
   - completed one-model hybrid-state parking feasibility. State continuity
   passes, but direct hold and first-token swap lose ERS/TPOT and are closed
   without full trace or submission.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_041_first_wave_service_slope_audit/README.md`
   - queue/service attribution plus static FlashInfer and Torch row-wise FP8
   backend screens. Both alternatives are slower than native CUTLASS; no
   checkpoint, overlay, full trace or submission was promoted.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_042_global_budget_cohort_audit/README.md`
   - six-process conditional 2048/4096 audit. Wave 2 is mean-positive but not
   robust; the executable global-wave oracle is only `+0.001128` and robust
   gain is zero, so fixed 2048 remains mandatory.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_043_fused_reduction_envelope/README.md`
   - exact-image native fused RMSNorm/dynamic-FP8 microbenchmark. It regresses
   large prefill shapes and loses to the actual compiled Triton decode
   reduction, so no matcher or kernel overlay is promoted.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_044_remaining_hotspot_capability_audit/README.md`
   - source-level target-H200 capability audit for the remaining attention and
   GDN/Mamba prefill hotspots. Automatic resolution already selects
   FlashAttention-3 and FlashInfer, and `0/8` alternatives pass all gates.
   `local_workspaces/rtx5080/research_rounds/score_100_program/round_010_readiness_warmup_integration/README.md`
   - I006 package architecture, A/B/A evidence, relay isolation and graded
   Attempt 11 artifact.
8. `docs/techniques/candidate_techniques.md` - algorithm and technique reference index.
   `docs/techniques/phase_aware_kv_pipeline.md` - proposed single-model
   TTFT/TPOT phase split with internal KVHandle handoff.
9. `docs/methodology/technical_conclusions.md` - current technical conclusions from RTX 5080 experiments.
10. `docs/logs/status_checkpoint.md` - current project state, verified items, limitations and next task.
11. `docs/logs/decision_log.md` - accepted decisions and reasons.
12. `docs/logs/experiment_log.md` - experiment history and results.
13. `docs/logs/open_questions.md` - unresolved BTC/environment questions.
14. `docs/local/rtx5080_local_lab.md` - local RTX 5080 workspace boundary and rules.
15. `local_workspaces/rtx5080/docs/model_contract_audit.md` - local model artifact audit; confirms Qwen3.5 hybrid linear-attention/full-attention behavior and flags pure Transformer KV assumptions as unverified.
    `local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006k_cache_local_tree_vs_heuristic/README.md` - final Tree-vs-heuristic exact-cache-cohort result and rejection rationale.
    `local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006l_parent_cache_nonposition_replay/README.md` - offline ancestry/no-position replay and branch closure.
16. `docs/methodology/toolchain.md` - local tooling, commands and serving baseline notes.
17. `docs/official/rules_and_anti_cheating.md` - hard constraints and anti-cheating interpretation.
    `docs/official/exact_trace_preload_review.md` - recorded exact-trace cache
    proposal, technical distinction, gray-area assessment and submission hold.
18. `docs/official/submission_checklist.md` - checklist before portal submission.
19. `portal_submissions/README.md` - current Portal upload target and immutable-attempt rules.
20. `portal_submissions/submission_plan.md` - five-attempt plan, result template and decision tree.

## Folder Meaning

| Folder | Meaning |
|---|---|
| `docs/official/` | Official statement, rules, submission contract and BTC-facing checklist. |
| `docs/baselines/` | Baseline configurations and baseline interpretation. |
| `docs/methodology/` | Benchmark method, optimization targets, toolchain and technical conclusions. |
| `docs/planning/` | Forward-looking plans, roadmaps and candidate sequence. |
| `docs/techniques/` | Algorithm and technique references before they become roadmap work. |
| `docs/logs/` | Decisions, experiments, open questions and current status. |
| `docs/local/` | Local-only workspace notes, especially RTX 5080. |

## Current Important Notes

- Active state: Part 2 R002k Attempt 04 is the H200 incumbent at ERS `60.75`,
  with exact-repeat mean `59.845`, TTFT p50/p95 `44/80ms`, median TBT `4ms`
  and six failures on the first panel. R002v Attempt 05 and its exact repeat
  scored `60.19/59.34` (mean `59.765`) and were not promoted.
- Active target hardware is 1 MiG H200 with 18GB VRAM, 3 CPU cores and 8GB RAM.
- Active model is `LiquidAI/LFM2.5-1.2B-Instruct` under vLLM `0.22.1`.
- Active public trace is `input_part2/trace_grading_public.jsonl`: 70
  conversations, six turns, 90 primer requests and 330 graded requests.
- Active H200 artifact is
  `portal_submissions/part2_attempt_04_lfm25_r002k_fused_greedy_seq8_b16384.yml`;
  retain Attempt 02 as the native-image rollback and Attempt 03 only as
  rejected cache-policy evidence.
- R002k's fused BF16 greedy head transferred from local ERS `58.3989` to
  Portal ERS `60.75`. Its package is clean, fallback-safe and locally GPQA-
  qualified; final BTC quality remains a post-round selected-submission gate.
- R002w decode-only ShortConv epilogue fusion is complete. Its exact kernel
  and graph gates pass, but repeated full replay loses `0.2689` mean ERS;
  the local image is research-only and no submission exists.
- R002x explains the mismatch: eager timing overstated the usable M8 saving by
  `4.86x`; M4 is at the noise boundary and M8 reverses. No larger ShortConv
  fusion, image, Compose or Portal attempt is authorized.
- R002y confirms that cached-turn TTFT growth is dominated by full attention,
  not failed cache reuse: complete GPU rises `60.801 -> 114.685ms` while the
  cached fraction reaches `83.16%`. Attention explains `92.64%` of the growth.
  The measured local backend is FlashInfer; H200 auto-selects packaged FA3, so
  only a target-specific qualification branch is open.
- R003's `8192` per-layer attention window is the current local research
  candidate. It wins the inferred append-only replay by `+4.6258` points over
  drift-matched controls and improves the turn-5 score by `+9.373` points, but
  has no benefit on independent 4k prompts. It changes long-context model
  semantics and is not a submission artifact until the request semantics,
  long-context quality and rules interpretation are closed.
- The Qwen3.5 trace, submissions and Portal attempts described below are
  historical Phase 1 evidence only.
- Current Portal upload queue is `portal_submissions/`; Attempt 04 failed on
  Portal-injected `--trust-remote-code`, and Attempt 06 failed on the next
  injected `--disable-uvicorn-access-log`. Attempt 05 official vLLM was graded
  `50.13` with `120/120`, penalty `1`, TTFT p50 `24025ms` and TBT median `13ms`;
  it is now the operational baseline. Attempt 07 passed local/public checks but
  timed out waiting for the Portal Pod to become Ready. Attempt 08 is the ready
  direct-vLLM `seq8 + FP8 E4M3` candidate and scored `45.29`: FP8 accuracy drop
  was `0`, but TBT median reached `21ms` and seq8 is rejected. Attempt 09
  `seq6 + FP8` scored `49.98`, with TTFT p50/p95 `21496/24498ms`, TBT `18ms`,
  accuracy drop `2` and penalty `1`; it is reliable but does not beat Attempt
  05. Explicit batched tokens `16384` was also rejected. Attempt 10 I004 scored
  `44.09` with TTFT `4522/5422ms`, TBT `21ms`, accuracy drop `0` and `22/120`
  passed SLO. Attempt 11 I006 scored `46.03` with TTFT `4306/5334ms`, TBT
  `20ms`, accuracy drop `2` and `20/120` passed SLO. Warmup transferred
  positively. Attempt 12 then graded exact public I007 seq16 at `51.37`, ERC
  `0.133333`, TTFT p50/p95 `4148/7130ms`, TBT `15ms`, accuracy drop `0`, no
  failures and penalty `1`. The I006 seq16 rerun scores `51.59`, with ERC
  `0.133333`, `16/120` passed SLO, TTFT p50/p95 `4783/6800ms`, TBT `16ms`,
  accuracy drop `0`, no failures and penalty `1`; it is now the score and
  complete-panel champion.
- Attempt 13 R022d was graded `49.45` at
  `portal_submissions/attempt_13_vllm_r022d_exact_trace_score100.yml`, SHA-256
  `d5c9395beb5713b5e38c1278f5431a8b81bd3f4431bd22e08b4b87f75de28dd4`,
  using public image digest
  `sha256:228c4fce4bdbcf973ebed3ded1d9c3413359e05320b3d6f0c80228d0cd97c229`.
  It is the completed single user-authorized exact-trace risk trial. Local
  self-contained qualification reached ERS `1.000000`, but Portal returned
  TTFT p50/p95 `4066/5367ms`, TBT `21ms`, `20/120` passed SLO and score
  `49.45`. The local fast path did not transfer materially; future work uses
  only generic post-arrival reuse. BTC rule interpretation remains unconfirmed.
- BTC baseline image is `vllm/vllm-openai:v0.22.1`.
- `--max-model-len=262144` is the BTC baseline, not a marked fixed field.
  Candidate `32768` passed exact-token/full-trace, quality and startup gates
  and is approved inside Attempt 10, with Attempt 09 context `262144` retained
  as rollback.
- RTX 5080 results are directional only and are not official H200 MiG baseline scores.
- Local `Qwen/Qwen3.5-2B` artifact is hybrid: 18 `linear_attention` layers and 6 `full_attention` layers. Pure Transformer KV-cache conclusions need revalidation unless BTC hash proves a different artifact.
- Clean packaged-core champion I004 has ERS `0.586594`, TTFT p50/p95
  `1289/3587ms`, TPOT p95/p99 `18.597/20.326ms`, and unchanged local GPQA
  predictions after BF16 Mamba SSM cache conversion. Packaged local champion
  I006 adds synthetic shape warmup behind readiness and reaches ERS `0.619885`
  with zero timed JIT. Attempt 11 proved Portal startup, `120/120`, penalty `1`
  and a modest H200 gain over Attempt 10, but not a score win.
- R007 is complete negative: its lifecycle 4096/2048 scheduler passed runtime
  and reliability gates but scored ERS `0.580253` versus control midpoint
  `0.582919`. I004 remains unchanged.
- R008 is complete local negative: forced full-attention `TRITON_ATTN` lost
  paired TTFT/TPOT `120/120`, while requested GDN FlashInfer fell back on RTX
  5080. Keep I004 automatic backends.
- R009 is complete positive: legal synthetic shape warmup reached ERS
  `0.617657` versus control midpoint `0.581198`, with zero new timed JIT.
  This formed runtime-only I005.
- R010 is complete locally: packaged I006 reached ERS `0.619885` versus
  packaged-control midpoint `0.589324`; relay isolation found no measurable
  regression, public readiness was `102s`, and Attempt 11 completed on Portal.
- R011 is complete: seq18/graph18 is a conditional local Pareto fallback at
  ERS `0.593090`, TPOT p99 `18.457ms` and `814.132 tok/s`; seq17 produced no
  material gain. Concurrency-only descent is closed. R012 hybrid hotspot
  profiling is complete.
- R012-R018 are resolved. R013 FP8 tied LM-head plus BF16 top-k4 repair is the
  only retained kernel; R014 norm fusion, R015 GDN decode/state
  quantization and R016 alternate prefill paths are rejected. R017 mixed state
  is skipped because its per-layer dependency is slower. R018's ideal block-16
  state split saves only `5.89%` prompt compute and at most `4.18%` of the
  Portal TTFT p50 anchor; dense snapshots exceed VRAM. R019/R020 remain closed
  for this representation.
- R021a is complete positive: same-image packaged top-k4 reached ERS
  `0.626978` versus control midpoint `0.623107`, TPOT p95 `14.492ms`, output
  throughput `819.557 tok/s`, and repeated GPQA `35/100` versus control
  `34/100` with penalty proxy `1`. This is local I007.
- R021b is complete: seq17's same-session ERS edge over seq16 was only
  `0.000747` and raised TPOT p99 by `0.863ms`; session drift was much larger.
  Keep seq16. I007 is frozen as
  `portal_submissions/attempt_12_vllm_i007_fp8_head_seq16.yml` at public digest
  `sha256:d6d626e8...b7611`; Compose/anonymous-manifest gates pass. Attempt 12
  validates it on H200 at `51.37`, TBT `15ms` and accuracy drop `0`, but I006
  seq16 scores `51.59`. R021c is complete negative: its head-off seq17 bracket
  loses ERS `0.530%` and adds no fast-TTFT request. The complete `51.59` panel
  now directs the next legal branch toward TTFT distribution while preserving
  local TPOT p99 `<=18ms`.
- R022d is complete and target-negative: it embeds only the public trace, stores no
  generated output, passed image/Compose/public-manifest checks, completed
  `120/120` locally at ERS `1.000000`, and repeated GPQA `34/100` with zero
  reference flips. This is performance/package evidence, not BTC rule
  approval. Attempt 13 then scored `49.45` on H200; exact-trace derivatives
  are closed and I006 seq16 remains champion at `51.59`.
- R023 is complete with no promoted candidate. Native retention already served
  every later shared prefix; online exact rendering had `0/120` hits;
  deadline-aware scheduling lost ERS and fast-count versus its control
  midpoint; bounded `5ms` cohort reordering lost ERS and throughput; and
  one-storage FP8 tied weights missed the TPOT/ERS gates despite saving about
  `484-485MiB` statically. The closing hotspot audit found no untested path
  above the `8%` decode or `10%` end-to-end benefit thresholds. No I008-I011
  or Attempt 14 was created.
- Deep Cycle 2 DR02 is complete. Fresh I006 seq16 CUPTI traces show pure
  decode at `94.977%` GPU busy with idle-gap p99 `3.902us`; no new kernel
  branch opens. DR07 then attributed the `96.597%` cache-hit branch: cache
  control was only `0.253%` of summed engine TTFT, physical pool use peaked at
  `25.323%`, and eviction/allocation/preemption counts were zero. No memory or
  scheduler implementation opened. DR16 capability-first precision screening
  parked GPTQ W4A16 until R034 completed its converter/runtime proof and
  rejected it on full-trace performance. DR20 then measured DR03b at
  `+0.7436%` ERS and TPOT p99 `17.556ms`, but its repeat quality panel violated
  the zero stable-change guard. DR03b is excluded. R024 then passed its
  conservative freeze with four reliable trace controls, three of four TPOT
  p99 guards, four complete GPQA controls, startup at most `107s`, exact
  3-CPU/8GB contracts and the Portal score-`51.59` artifact. I006 remains the
  conservative rollback. R025 subsequently promoted I007 seq19 locally at ERS
  `0.610256`, rejected aggressive warmup and froze Attempt 14. Portal scored it
  `46.86`: SLO `19/120` and TTFT `4295/6586ms` validate a TTFT-tail benefit,
  while TBT `19ms` explains the decode-repair priority. R026b did not prove a
  local repair: graph11 was nearly neutral and graph6+11 regressed. R027 then
  measured seq18 at TPOT p95/p99 `15.556/17.899ms`, one fewer fast request and
  `0.400%` lower ERS than seq19 midpoint. Attempt 16 scored `47.04` on H200:
  TBT improved to `18ms`, but TTFT p95 regressed to `8799ms`. Reject active-set
  interpolation and park Attempt 15. R028 then rejected fixed prefill budget
  `4096` on exact I006: ERS `0.601842` versus midpoint `0.610570`, despite six
  more requests below the TTFT ceiling. No Attempt 17 was created; I006 seq16
  remains champion at `51.59` with resolved/default budget `2048`.
- R029 closes the dynamic derivative: robust request-level oracle gain is only
  `0.351%`, the depth-2 held-out result is `-0.000224` ERS, and all six opening
  gates fail. No Tree-Gated scheduler, GPU run, image or Compose was created.
- R030 corrects the R028 metric interpretation: lookup hit rate rose
  `0.800pp`, actual request cache attribution fell only `0.143%`, and request
  prefill time is a wall phase rather than GPU compute. No decision, champion
  or submission artifact changes.
- R031 isolates that wall phase. Across six exact-I006 GPU cells, `4096`
  lowers queue time `138.716ms` but raises scheduled-to-first-token service
  `122.927ms` and loses ERS `1.80%`; the strict six-wave gate fails, so no
  implementation or submission artifact is created.
- R032 closes its final-chunk derivative. At `50%` net service recovery only
  `6/9` process pairs win; waves 1/5 and the `<=500ms` cohort lose. No R032b,
  GPU run or submission artifact is created.
- R033 corrects DR20's quality decision rule to the official BTC penalty, then
  rejects DR03b on fresh same-image performance: ERS `0.612869` versus
  `0.614450`, W/T/L `32/35/53`, only three positive waves. No fresh GPQA run,
  image, Compose or Attempt 17 is created.
- R034 converts and validates exact GPTQ W4A16/G128, passes native Marlin
  startup and GPQA `37/37/35` with penalty proxy `1`, then rejects it on a
  same-image full-trace bracket: ERS `0.542724` versus midpoint `0.609622`,
  TPOT p99 `23.213ms`, W/T/L `0/37/83`, all waves negative. No image, Compose
  or Attempt 17 is created.
- R035 audits the R034 log boundary. All three prebenchmark snapshots are
  exact prefixes with five warmup JIT warnings, while every timed suffix has
  zero. The analyzer is corrected; no GPU/artifact opens and R034 is unchanged.
- R036 audits local-to-H200 transfer. Two Portal comparisons are decode-
  dominant, four low-median panels retain TPOT score loss, and local total ERS
  disagrees with target direction three times. I006 seq16 cannot exceed
  `56.666667` without raising `16/120` TTFT support.
- R037 audits the exact I006 image and closes with `14` classified
  capabilities, zero gate-qualified GPU candidates and manifest SHA
  `85b11280...c72d`. DS convolution-state layout is executable but below the
  score ceiling; MRV2 cannot preserve the current align cache without an
  unbounded state-manager implementation. No GPU, image, Compose or Attempt
  17 was created.
- R038 audits logical first-token staging with a causal serial-lane model. The
  opportunity survives only at `>=75%` queue removal and `<=10%` decode
  interference; it opens gap-aware scheduler design work but changes no
  champion, image, Compose or Portal artifact.
- Part 2 R003b closes selective-global attention placement. Keeping one or two
  full-attention layers global preserves the `4096` latency gain but loses the
  same two paired long-context retrieval cases in every placement. No full
  replay, image or Compose follows; see the subround results under
  `local_workspaces/rtx5080/research_rounds/part2_lfm25/`.
- Part 2 R003f proves compact causal prefix-plus-recent Triton pruning can
  reach the recent-only profile floor and improve paired retrieval. P64 full
  replay mean `61.7572` remains below safe8192/native W4096, and GPQA proxy
  `20/100` is below controls `21..22/100`. Keep it research-only; no image,
  Compose or Portal artifact follows.

## Where To Write New Notes

- New rule or submission constraint: `docs/official/`.
- New baseline interpretation: `docs/baselines/`.
- New benchmark process or tuning target: `docs/methodology/`.
- New roadmap, plan or candidate backlog: `docs/planning/`.
- New algorithm or technique reference: `docs/techniques/`.
- New experiment result: `docs/logs/experiment_log.md`.
- New accepted/rejected decision: `docs/logs/decision_log.md`.
- New local RTX 5080 observation: `docs/local/` or `local_workspaces/rtx5080/docs/`.
