# Experiment Log

Use this file for manual notes. Raw outputs should go under `experiments/000X-*`, `metrics/` and `reports/`.

## Template

```text
Experiment:
Date:
Commit:
Image:
Compose:
Hardware:
Runtime flags:
Change:

Results:
- ERS:
- Score if accuracy gate passes:
- TTFT p50/p95/p99:
- TPOT p50/p95:
- Latency p50/p95/p99:
- Success/errors/timeouts:
- Output tokens:
- VRAM peak:
- GPU utilization:
- Accuracy / GPQA:

Conclusion:
Next decision:
```

## 2026-07-06 - Phase 1 Trace Integrated

Change:

- Added official public trace and baseline compose under `input/`.
- Replaced generic prep notes with Phase 1-specific docs.
- Local harness now supports BTC trace body format and ERS scoring.

Conclusion:

- Ready to run real baseline once model weights and server are available locally or inside the submission environment.

## 2026-07-07 - RTX 5080 Local Baseline

Experiment:

- Local baseline in `local_workspaces/rtx5080/`.
- Model: `Qwen/Qwen3.5-2B`
- Runtime: `vllm/vllm-openai:v0.22.1`
- Hardware: RTX 5080 local desktop, not BTC H200 MiG.
- Trace: `input/trace-round1.jsonl`
- Concurrency: 120
- `max_model_len`: 32768
- `gpu_memory_utilization`: 0.65
- Prefix caching: enabled

Results:

- Success/errors/zero-token: 120 / 0 / 0
- ERS: 0.025640811640791
- Score if accuracy gate passes: 2.5640811640791
- TTFT p50/p95/p99: 42262.359 / 93561.53205 / 100049.86899999999 ms
- TPOT p50/p95/p99: 41.6454925 / 47.302845500000004 / 48.422043800000004 ms
- Latency p50/p95/p99: 50599.387 / 101781.2695 / 103169.69295 ms
- Wall tokens/s: 186.37962246368292

Conclusion:

- Local baseline is valid for RTX 5080 trend tracking only.
- Main bottleneck is TTFT from prefill, queueing and KV pressure.
- Every request missed the official TTFT ceiling; TPOT was close to the official ceiling.
- See `local_workspaces/rtx5080/docs/baseline_20260707_111358.md`.

## 2026-07-07 - Experiment 001 RTX 5080 Sweet Spot

Experiment:

- Swept `GPU_MEMORY_UTILIZATION` and then capped `MAX_NUM_SEQS`.
- Fixed model/runtime/trace: `Qwen/Qwen3.5-2B`, `vllm/vllm-openai:v0.22.1`, `input/trace-round1.jsonl`.

Completed candidates:

- `0.65 / max_num_seqs=120`: ERS 0.025641 initial, 0.016112 rerun
- `0.75 / max_num_seqs=120`: ERS 0.018118
- `0.70 / max_num_seqs=120`: ERS 0.015397
- `0.60 / max_num_seqs=120`: ERS 0.016209
- `0.65 / max_num_seqs=16`: ERS 0.056109

Conclusion:

- Best completed RTX 5080 local baseline is `GPU_MEMORY_UTILIZATION=0.65`, `MAX_NUM_SEQS=16`.
- Main lesson: more KV cache alone increased concurrent running requests but hurt TPOT too much.
- `MAX_NUM_SEQS=8` was not run because Docker/GPU execution hit the Codex usage limit.
- See `local_workspaces/rtx5080/docs/experiment_001_sweetspot.md`.

## 2026-07-08 - GPQA Diamond Local Quality Set

Experiment:

- Prepared a local private proxy for the Accuracy Gate.
- Source: GPQA Diamond public dataset.
- Selection: deterministic 100-question subset from 198 Diamond records.
- Selection seed: `20260708`.
- Output path: `input/quality/private/gpqa_diamond_100.jsonl`.

Results:

- Questions: 100
- Domain distribution: Biology 6, Chemistry 44, Physics 50
- Raw examples are git-ignored under `input/quality/private/`.

Conclusion:

- This is now the local main quality proxy for comparing serving configs.
- It is not the BTC hidden/fixed 100-question set, so results are directional.
- Next quality experiment should run `KV_CACHE_DTYPE=auto` vs `KV_CACHE_DTYPE=fp8_e4m3` on this exact file.

## 2026-07-08 - Round 007b GPQA KV Cache Accuracy

Experiment:

- Subround: `local_workspaces/rtx5080/research_rounds/warmup_rounds/round_007_kv_cache_dtype/subrounds/round_007b_gpqa_diamond_accuracy/`
- Dataset: `input/quality/private/gpqa_diamond_100.jsonl`
- Compared `KV_CACHE_DTYPE=auto` against `KV_CACHE_DTYPE=fp8_e4m3`.
- Same local serving baseline: vLLM `v0.22.1`, `MAX_MODEL_LEN=32768`, `GPU_MEMORY_UTILIZATION=0.72`, `MAX_NUM_SEQS=4`, prefix cache on, chunked prefill on.

Results:

- Auto KV accuracy: `0.38` (`38/100`)
- FP8 E4M3 KV accuracy: `0.34` (`34/100`)
- Relative drop: `0.04`
- OK/errors: `100/0` for both
- Parse failures: `0` for both
- Changed prediction: `9`
- Auto correct, FP8 wrong: `4`
- Auto wrong, FP8 correct: `0`
- Official-style proxy with baseline `0.4`: auto delta `0.02`, FP8 delta `0.06`, both penalty `1.0`

Conclusion:

- FP8 E4M3 KV remains lossy but did not fail the local GPQA proxy.
- Promote FP8 E4M3 KV from performance-only to provisional candidate.
- Keep auto KV as rollback and rerun the proxy after major or lossy changes.

## 2026-07-08 - Round 007c Full Score Replay

Experiment:

- Subround: `local_workspaces/rtx5080/research_rounds/warmup_rounds/round_007_kv_cache_dtype/subrounds/round_007c_full_score_replay/`
- Run: `local_workspaces/rtx5080/runs/20260708-121041`
- Candidate: FP8 E4M3 KV cache with Round 007 baseline knobs.
- Score proxy used Round 007b GPQA accuracy proxy `0.34`.

Results:

- Success/errors/zero-token: `120 / 0 / 0`
- ERS: `0.5`
- Accuracy delta proxy: `0.06`
- Accuracy penalty proxy: `1.0`
- Final score proxy: `50.0`
- Target score: `95.0`
- TTFT p50/p95/p99: `16440.635 / 28315.94975 / 30294.24758 ms`
- TPOT p50/p95/p99: `8.0753675 / 9.3355985 / 9.36906665 ms`
- Wall tokens/s: `418.35204530920987`

Conclusion:

- Do not submit this candidate under the `>=95` target.
- TPOT is already excellent; the score is capped by TTFT being far above the official 1500 ms ceiling.
- Next optimization must attack TTFT/queueing, not just decode speed.

## 2026-07-08 - Round 007d TTFT Scheduler Sweep

Experiment:

- Subround: `local_workspaces/rtx5080/research_rounds/warmup_rounds/round_007_kv_cache_dtype/subrounds/round_007d_ttft_scheduler_sweep/`
- Fixed resource limits: `CPU_LIMIT=3`, `MEM_LIMIT=8g`
- Fixed base: FP8 E4M3 KV, `GPU_MEMORY_UTILIZATION=0.72`, prefix cache on, chunked prefill on.
- Ran a small early-stop sweep: `MAX_NUM_SEQS=4`, `8`, `12`.

Results:

- `seqs_04`: ERS `0.5`, score proxy `50.0`, TTFT p50/p95 `18810.9795 / 31400.71595 ms`, TPOT p50/p95 `8.37152 / 9.98833775 ms`
- `seqs_08`: ERS `0.442994`, score proxy `44.299415`, TTFT p50/p95 `9813.8335 / 25580.64285 ms`, TPOT p50/p95 `12.2161125 / 38.07663175 ms`
- `seqs_12`: ERS `0.449426`, score proxy `44.942615`, TTFT p50/p95 `6280.0605 / 14946.32745 ms`, TPOT p50/p95 `15.6488725 / 29.919245 ms`

Conclusion:

- Higher scheduler concurrency does reduce queue TTFT, but not enough to reach the official 1500 ms ceiling.
- TPOT tail worsens enough that ERS/score drop below the `MAX_NUM_SEQS=4` control.
- Stop the rest of the sweep for now; plain concurrency increase is not the TTFT solution.

## 2026-07-08 - Round 008 Brain 1 Shadow Dataset

Experiment:

- Round: `local_workspaces/rtx5080/research_rounds/warmup_rounds/round_008_brain1_shadow_dataset/`
- Built a request-level dataset from `input/trace-round1.jsonl` and all available local RTX 5080 run metrics.
- No GPU serving run was needed.

Results:

- Dataset CSV: `outputs/shadow_dataset.csv`
- Dataset JSONL: `outputs/shadow_dataset.jsonl`
- Balanced RAM/VRAM CSV: `outputs/shadow_dataset_balanced_ram_vram.csv`
- Balanced RAM/VRAM JSONL: `outputs/shadow_dataset_balanced_ram_vram.jsonl`
- Rows: `7080`
- Runs: `59`
- Requests: `120`
- Feature columns: `77`
- TTFT over ceiling rows: `7080 / 7080`
- TPOT over ceiling rows: `1014 / 7080`
- Score proxy bad rows: `1756 / 7080`
- RAM/VRAM target tiers: RAM `3540`, VRAM `3540`

Conclusion:

- Round 008 completed the Brain 1 data foundation.
- The dataset was enriched after reviewing the Tree-Gated spec: it now contains `sem_gravity`, `temp_decay`, `info_density`, `neuro_interest`, `surviv_index` and balanced RAM/VRAM-only labels.
- SSD/offload labels were removed from the active dataset because the BTC environment does not guarantee a useful SSD path.
- There are no local TTFT-pass examples, so Round 009 should learn relative risk/ranking, bad TPOT/score classes or RAM/VRAM placement behavior rather than a TTFT-pass classifier.
- Round 010 can use this dataset for admission/scheduler simulation.

## 2026-07-08 - Round 009 Brain 1 Shadow Ranker

Experiment:

- Round: `local_workspaces/rtx5080/research_rounds/warmup_rounds/round_009_brain1_xgboost_shadow/`
- Dataset: Round 008 `shadow_dataset_balanced_ram_vram.csv`
- Model: pure-NumPy boosted-stump fallback because `xgboost`, `sklearn` and `pandas` are not installed locally.
- Split methods: `request_id`, `run_id`
- Feature sets: `trace_only`, `trace_pressure`, `trace_pressure_config`
- Targets: `ram_vram_target`, `score_proxy_bad`, `tpot_over_ceiling`, `ttft_over_ceiling`

Results:

- Usable models: `18`
- Skipped models: `6` TTFT-over-ceiling targets, because every local row exceeds the official TTFT ceiling.
- Primary production-safe model: `request_id.trace_pressure.ram_vram_target`
- Primary ROC AUC / balanced accuracy / F1: `0.973690 / 0.932974 / 0.931900`
- Primary top features: `timestamp_ms`, `vram_usage_proxy`, `surviv_index`
- Bad-score trace_pressure ROC AUC: `0.871762`
- TPOT-over-ceiling trace_pressure ROC AUC: `0.905334`
- Best diagnostic config-set RAM/VRAM ROC AUC: `0.989514`

Conclusion:

- Brain 1 can learn the Round 008 RAM/VRAM heuristic target and can rank TPOT/score risk when online pressure features are available.
- `trace_pressure_config` is diagnostic only because it can lean on fixed config fields such as `dtype` and `max_num_seqs`.
- Use `request_id + trace_pressure` probabilities for Round 010 offline simulation before changing the serving path.
- This is not proof that real KV migration improves ERS; it is a shadow model for controlled simulation.

## 2026-07-08 - Round 009b Brain 1 Hyperparameter Sweep

Experiment:

- Subround: `local_workspaces/rtx5080/research_rounds/warmup_rounds/round_009_brain1_xgboost_shadow/subrounds/round_009b_hyperparameter_sweep/`
- Dataset: Round 008 `shadow_dataset_balanced_ram_vram.csv`
- Feature set: `trace_pressure`
- Broad sweep: `48` configs on `request_id` split and `ram_vram_target`
- Deep shortlist: `60` evals across `request_id`, `run_id`, `score_proxy_bad` and `tpot_over_ceiling`

Results:

- Sweet spot: `n60.lr0.200.thr8.leaf24`
- Request primary ROC AUC / balanced accuracy / F1: `0.972375 / 0.932974 / 0.931900`
- Run primary ROC AUC / balanced accuracy / F1: `0.946675 / 0.909237 / 0.900405`
- Risk AUC mean: `0.854707`
- Top features: `message_count`, `vram_usage_proxy`, `surviv_index`, `entropy_score`
- Model bundle: `models/sweetspot.request_id.trace_pressure.ram_vram_target.json`
- Performance-heavy fallback: `n60.lr0.200.thr16.leaf24`

Conclusion:

- Round 009 default was close, but 009b gives a smaller and more regularized model for Round 010.
- Use sweet spot first in simulation; compare the fallback only if Round 010 needs a performance-heavy variant.
- Still shadow-only; no serving behavior changed.

## 2026-07-08 - Round 010 Brain 1 Admission Simulation

Experiment:

- Round: `local_workspaces/rtx5080/research_rounds/warmup_rounds/round_010_brain1_admission_simulation/`
- Model: Round 009b sweet spot `sweetspot.request_id.trace_pressure.ram_vram_target.json`
- Simulation type: offline proxy only, no vLLM serving changes.
- Policies tested: `p_vram` thresholds and top-fraction policies.
- Hybrid replay pairs: safe `seqs_04`, balanced `seqs_08`, aggressive `seqs_12`, and FP8 full-score replay.

Results:

- Best existing baseline: `safe_seq4`, run `20260708-123026`
- Baseline ERS / score proxy: `0.500000 / 50.000000`
- Baseline TTFT p50/p95: `18810.979500 / 31400.715950 ms`
- Best policy quality: threshold `p_vram >= 0.15`
- Policy balanced accuracy: `0.976190`
- Policy VRAM precision/recall/F1: `0.975000 / 1.000000 / 0.987342`
- Best hybrid proxy: top `40%` p_vram uses `aggressive_seq12`, remainder uses `fp8_full_score`
- Hybrid ERS / score proxy: `0.500000 / 50.000000`
- Hybrid TTFT p50/p95: `8016.931000 / 28315.949750 ms`
- Hybrid TPOT p50/p95: `9.168675 / 15.872813 ms`
- Best observed per-request oracle ERS: `0.500000`
- Min-TTFT oracle ERS: `0.450347`

Conclusion:

- Brain 1 probabilities identify the RAM/VRAM target and request priority well.
- The best proxy policy reduces TTFT p50 materially while keeping ERS equal to the best baseline, but TTFT remains far above the official `1500 ms` ceiling.
- No observed-run proxy shows ERS headroom beyond `0.5`; real progress still requires a TTFT breakthrough.
- Do not implement or submit Brain 1 admission from this round.

## 2026-07-08 - Round 012 Warmup/JIT And Full Usable VRAM

Experiment:

- Round: `local_workspaces/rtx5080/research_rounds/warmup_rounds/round_012_warmup_jit/`
- Runtime: `vllm/vllm-openai:v0.22.1`
- Fixed constraints: RTX 5080 local, Docker `--cpus 3`, Docker `--memory 8g`, model mounted at `/model`.
- Base serving config: `MAX_NUM_SEQS=4`, `MAX_MODEL_LEN=32768`, prefix caching on, chunked prefill on, `KV_CACHE_DTYPE=fp8_e4m3`.
- Tested cold start, synthetic Torch warmup, endpoint synthetic warmup, and high local VRAM settings.

Results:

- `GPU_MEMORY_UTILIZATION=0.95` failed locally: vLLM saw `12.91/15.44 GiB` free and required `14.67 GiB`; desktop/app processes were already using about `2.1GB` VRAM.
- `0.83` near-max usable VRAM completed several runs but was flaky.
- Cold `0.83`: run `local_workspaces/rtx5080/runs/20260708-233429`, ERS `0.5`, TTFT p50/p95 `17673.4095 / 30627.24865 ms`, TPOT p50/p95 `8.4574325 / 9.740635 ms`.
- Torch warmup `10s` at `0.83`: run `local_workspaces/rtx5080/runs/20260708-234003`, ERS `0.5`, TTFT p50/p95 `17421.6925 / 31319.2015 ms`, TPOT p50/p95 `8.71467 / 9.482708 ms`.
- Endpoint short warmup `0.82`: run `local_workspaces/rtx5080/runs/20260708-234652`, ERS `0.5006135105048746`, score proxy `50.06135105048746`, TTFT p50/p95 `17849.3315 / 31575.4577 ms`, TPOT p50/p95 `8.6846975 / 10.096045 ms`.
- Endpoint warmup synthetic request TTFTs: `938.286 ms`, `23.262 ms`, `22.321 ms`.

Conclusion:

- Endpoint warmup is a real but small TTFT/JIT-shape win and should stay on the technique watchlist.
- Pure Torch warmup is not strong enough to promote from Round 012.
- `0.83` is an experimental high-VRAM setting for the active desktop; `0.82` is the stable high-VRAM fallback.
- Round 012 best score proxy is still only `50.061351`, so it is not a submission candidate.
- Next work should be a Round 012b endpoint-shape warmup subround with medium/long/concurrent synthetic warmup prompts.

## 2026-07-09 - Round 012a High VRAM Sweep

Experiment:

- Subround: `local_workspaces/rtx5080/research_rounds/warmup_rounds/round_012_warmup_jit/subrounds/round_012a_high_vram_sweep/`
- Fixed constraints: RTX 5080 local, Docker `--cpus 3`, Docker `--memory 8g`.
- Fixed serving config: `MAX_NUM_SEQS=4`, `MAX_MODEL_LEN=32768`, prefix caching on, chunked prefill on, `KV_CACHE_DTYPE=fp8_e4m3`.
- Variable: `GPU_MEMORY_UTILIZATION=0.82, 0.84, 0.86, 0.88, 0.90`.
- VRAM before run: about `14594 MiB` free.

Results:

- All candidates completed `120/120` requests.
- `0.82`: ERS `0.500000`, TTFT p50/p95 `16462.7585 / 28407.40815 ms`, TPOT p50/p95 `8.09667 / 9.375602 ms`, KV `6.13 GiB / 883850 tokens / 26.97x`.
- `0.90`: ERS `0.500000`, TTFT p50/p95 `16504.6455 / 28455.75355 ms`, TPOT p50/p95 `8.0987875 / 9.40156925 ms`, KV `7.36 GiB / 1062745 tokens / 32.43x`.
- Capacity delta `0.82 -> 0.90`: `+178895` KV tokens, about `+20.24%`.

Conclusion:

- Higher VRAM allocation is proven useful for KV capacity.
- Higher VRAM allocation is not proven useful for this exact trace score while `MAX_NUM_SEQS=4`.
- Keep `0.90` as a capacity/H200 watchlist, not as local default.
- To exploit H200 MiG `18GB`, pair high VRAM with scheduler/concurrency/prefill changes.

## 2026-07-10 - Ready 008a Engine-Native Scheduler Telemetry

Experiment:

- Pinned SGLang `0.5.14` image digest `5027e95b...`.
- Added an opt-in, read-only import overlay around native Scheduler,
  SchedulePolicy and MambaRadixCache lifecycle methods.
- A/B replayed the exact current SGLang config with 3 CPU cores, 8GB RAM and
  concurrency 120.

Results:

- Native control: ERS `0.512927`, success `120/120`, TTFT<=1.5s `23/120`,
  TPOT p95 `19.392 ms`.
- Instrumented: ERS `0.511176`, success `120/120`, TTFT<=1.5s `24/120`,
  TPOT p95 `17.918 ms`.
- Same-session ERS delta: `-0.001751`, inside the `0.005` overhead budget.
- Telemetry: 7147 valid timed events, no malformed line, and complete
  `120 queued / 120 first-token / 120 finished` coverage.
- Native batch shape: 237 prefill, 2800 decode, no mixed batch.
- First-prefill prefix match ratio mean `0.7767`.
- Cache activity: 343 unfinished commits, 51 eviction calls, 114461 full-KV
  tokens evicted, zero Mamba states evicted and zero request retractions.

Conclusion:

- Accept telemetry as a measurement layer only.
- Native LPM prefix reuse is valuable; LPM calculation is not the main measured
  prefill scheduling cost.
- Next experiment is a conditional high-prefix/deadline rescue budget inside
  one SGLang engine, with TPOT guard, not a global scheduler switch.

## 2026-07-10 - Portal Attempt 01 Result And Attempt 02 Selection

Portal result:

- Artifact: `portal_submissions/attempt_01_vllm_official_control.yml`
- Portal record time: `10/07/2026 17:52`
- Status: `Đã chấm`
- Final score: `14.98000`
- ERS display / ERC: `14.98 / 0.7`
- Passed SLO: `84/120`
- TTFT p50/p95: `737/10066ms`
- TBT median: `59ms`
- Failed / warmup: `0 / 0`
- Accuracy drop / penalty: `0 / 1`
- Raw logs: not provided

Attempt 02 selection:

- Source: Ready 006e `hybrid_fair_lpm_res64_steps1`
- Local run: `local_workspaces/rtx5080/runs/20260709-234511`
- Local ERS / success: `0.524547 / 120/120`
- Local TTFT p50/p95: `3896.429 / 35006.865 ms`
- Local TPOT p50/p95: `12.508 / 17.036 ms`
- Runtime: native SGLang LPM/Radix, prefill and decode CUDA graphs, Mamba
  FlashInfer, BF16 SSM, auto KV cache
- Submitted memory setting: `mem-fraction-static=0.86`
- Excluded: proxy, Tree-Gated, two-policy logic, telemetry and active adaptive
  scheduler changes

Conclusion:

- Submit a simple TPOT-safe native candidate rather than the absolute TPOT
  floor. TPOT below `20 ms` is already fully scored, so remaining budget should
  reduce TTFT.
- Keep official context `262144`; local `32768` metrics remain directional.

Submission state correction:

- Attempt 02 was submitted with `mem-fraction-static=0.86`, SHA-256
  `018a88480f1c7aa8b30fe99b1523c53b52ad05cb16d064d06b2bdd1287dd7a24`.
- The H200-sized `0.88` variant is now isolated as the unsubmitted Attempt 03
  comparison candidate, SHA-256
  `a4f021125f906c3cdce52814c84a0911ff82db0a886d96fbf41e44e5e11e0250`.

## 2026-07-10 - Subround 006h Attempt 02 KV FP8 Gate

Experiment:

- Compared Attempt 02 auto KV with SGLang FP8 E4M3 and E5M2.
- Fixed context `262144`, max running `10`, LPM/Radix, per-phase CUDA graphs,
  Mamba FlashInfer, BF16 SSM, CPU `3`, RAM `8GB`.
- Local memory fraction was `0.72` because desktop processes left about
  `12.6GB` VRAM free; both candidates used the same value.
- Ran each dtype twice on the deterministic local 100-question GPQA proxy.

Quality results:

- Auto accuracy A/B: `0.37 / 0.38`; repeat changes `2/100`.
- E4M3 accuracy A/B: `0.36 / 0.33`; repeat changes `9/100`; quality gate fail.
- E5M2 accuracy A/B: `0.35 / 0.36`; repeat changes `10/100`; accuracy sub-gate
  passed but deterministic stability failed.
- Both FP8 variants retained official-style penalty proxy `1.0` locally.

E5M2 trace results:

- Auto/E5M2 ERS: `0.495794 / 0.487801`, delta `-0.007994`.
- Auto/E5M2 TTFT p50: `14136.840 / 5326.200 ms`.
- TTFT<=1500 remained `16/120` for both.
- Auto/E5M2 TPOT p95: `25.765 / 27.858 ms`.
- KV capacity increased `241871 -> 494760` tokens.

Conclusion:

- Reject unscaled FP8 KV for Portal. Capacity doubled, but repeatability and
  ERS worsened.
- SGLang warned that E4M3 used scale `1.0`; exact-image source inspection found
  no Qwen3.5 KV-scale loader, so calibrated scales require a custom image.

## 2026-07-10 - Subround 006i Brain1 Tree-Gated GPQA Gate

Experiment:

- Compared direct and Brain1 Tree-Gated paths through one shared Attempt 02
  SGLang engine and cache pool.
- Fixed engine random seed `42`, auto KV, context `262144`, local memory `0.72`,
  max running `10`, CPU `3`, RAM `8GB`.
- Tree-Gated used the Ready 003 proxy, Round 009b sweet-spot model, backend cap
  `8`, dispatch window `25ms`, release-on-complete and no priority injection.
- Ran 100 GPQA questions twice sequentially and twice with 16 workers for both
  direct and Tree paths: eight complete passes, 800 responses total.

Results:

- Direct sequential A/B: `0.37 / 0.38`; Tree: `0.38 / 0.38`.
- Direct concurrent A/B: `0.38 / 0.38`; Tree: `0.38 / 0.37`.
- Worst direct-minus-Tree drop: `0.01`; Tree penalty proxy minimum `1.0`.
- Tree repeat changes: sequential `0`, concurrent `3`; both inside the direct
  variance budget.
- Proxy logged `400/400` Tree requests with no errors and queued `198/200`
  concurrent requests.
- Queue wait p50/p95: `25.244 / 132.308 ms`.
- Brain1 priority was constant `0.879565` for all GPQA requests.

Conclusion:

- Accept Tree-Gated proxy/admission as locally GPQA-safe.
- Do not claim ranked Tree policy quality coverage because GPQA occupied one
  learned priority leaf.
- Do not submit based on quality alone; its queue overhead still requires a
  positive production-trace ERS A/B.

## 2026-07-10 - Subround 006j Native Tree-Gated V2

Experiment:

- Rebuilt Tree-Gated from 11 native SGLang scheduler telemetry timelines and
  grouped six same-session cohorts into train/validation/test.
- Produced `54,729` full policy-decision rows and `11,574` capped/weighted
  training rows for `1,320` timeline/request pairs.
- Replaced the circular heuristic RAM/VRAM target with observed positive TTFT
  score. Excluded queue age/deadline progress from model features and kept them
  only in the explicit rejuvenation controller.
- Trained 64 boosted stumps in single-threaded C++, exported a compact model,
  exposed a C ABI scorer and replayed fixed-capacity top-K placement.

Results:

- Grouped test/actionable-window ROC AUC: `0.910670 / 0.900176`.
- Actionable score standard deviation: `0.137673`.
- Pure C++ / Python-to-C++ / whole-queue top-K p99:
  `0.088 / 0.454 / 23.867us`.
- Observed SGLang policy calculation p99: `5409.762us`.
- Native/Python prediction and selected-set parity passed.
- Native training used one core for `0.54s`, peaked at `6,900KB` RSS and used
  no swap, below the 3 CPU/8GB target constraint.
- Model gain used only native position, remaining prefill, scheduler pressure
  and normalized position.

Rejuvenation:

- Gentle locality policy rescued `161` placements with positive coverage
  unchanged at `1.0` and selected age p50 about `517ms` lower.
- Precision and prefix ratio regressed slightly; unselected age p95 rose about
  `17.5ms`, so this is near-neutral shadow evidence rather than a win.
- Strong soft-aging and forced-deadline promotion materially damaged precision
  and locality and were rejected.

Conclusion:

- Native classification latency is acceptable and not the current bottleneck.
- Keep Tree-Gated v2 and gentle rejuvenation shadow-only. No serving baseline
  or Portal artifact changed.
- Compare the four-feature model against an explicit heuristic, then measure
  live feature extraction and causal ERS inside cache-local LPM cohorts.
- No real RAM/VRAM migration or context restoration was measured.

## 2026-07-11 - Subround 006k Cache-Local Tree Versus Heuristic

Experiment:

- Compared Native Tree-Gated v2 against an explicit heuristic using the same
  four effective features.
- Selected policy/weights on grouped validation only; test remained audit-only.
- Audited pinned SGLang `0.5.14` source and used identical non-root
  `req.last_node.id` as an opaque cache-local cohort boundary.
- Ran telemetry-only control and read-only dual-scorer shadow with one model,
  CPU `3`, RAM `8GB`, `mem-fraction-static=0.82`, network disabled and all
  native queue/cache behavior unchanged.

Offline results:

- Validation AUC Tree/heuristic: `0.953414 / 0.974930`.
- Actionable test AUC: `0.900176 / 0.916530`.
- Top-K selected-set agreement: `0.864407`.
- The direct heuristic was selected.

Live results:

- Control/shadow ERS: `0.524547 / 0.518136`, delta `-0.006411`.
- Success: `120/120` each; TTFT<=1500 changed `25 -> 20`.
- TTFT p50/p95 improved slightly in shadow, as did TPOT and throughput, so the
  one-run ERS delta is dominated by request ordering around the 1500ms cliff
  and cannot be attributed wholly to observer overhead.
- Observed `263` exact-node cohorts across `60` opaque node tags.
- Tree and heuristic would-reorder counts were both `0`.
- Evaluator p99 was `333.063us`; wrapper max observed call was `378.180us`.
- Policy errors, active order changes, rescues and mutations were all zero.

Conclusion:

- Reject active exact-node Tree/heuristic scheduling. Both rankers reproduce
  the native LPM cohort head and therefore have no causal action support.
- Do not rerun solely to lower observer cost and do not change Compose/Portal.
- Keep the dataset, native scorer and source-audited hook as research tools.

## 2026-07-11 - Subround 006l Parent-Cache Non-Position Replay

Experiment:

- Reused 006k timed telemetry offline; no GPU/server run.
- Joined `120/120` arrivals/outcomes across `178` native policy decisions.
- Reconstructed `146` cache nodes with zero conflicting parent relation.
- Tested direct parent, fixed-hop ancestor and depth-normalized ancestor
  boundaries, requiring a structurally broader current node/branch.
- Swept ten transparent policies using service cost, prefix ratios, bounded
  projected score and cache depth. Native position, IDs, trace index, text and
  future outcome were forbidden policy features.

Topology results:

- Direct-parent same-key groups: `276`; distinct current nodes/branches `0/0`.
- Fixed-hop groups at hops `2/3/4/5`: `248/217/49/11`; distinct nodes `0`.
- Depth-normalized groups at depths `3/4/5`: `165/164/160`; distinct nodes `0`.
- No parent/ancestor candidate existed beyond the exact-node boundary.

Exact-node diagnostic:

- Best no-position policy was `service_only`.
- Validation cohorts/changes: `216/126`; unique changed selections targeted
  only four requests.
- Support wins/losses: `0/6`; native/candidate positive capture `1.0/0.0`.
- Positive precision delta `-0.027778`; observed TTFT-component delta
  `-0.002374`; matched-ratio delta `+0.000377`.

Conclusion:

- Reject ancestry expansion and static no-position ranking.
- Do not run a live 006l shadow/active policy or change submission artifacts.
- Future work must use temporal/causal state or optimize engine resource
  budgets instead of reranking the same cache snapshot.

## 2026-07-11 - Final Exact-Entrypoint SGLang/VLLM Validation

Experiment:

- Built a custom SGLang image that preserves the exact external BTC command
  `python3 -m vllm.entrypoints.openai.api_server` and validates/translates only
  the protected baseline arguments.
- Registered a five-request synthetic mixed text warmup inside SGLang before
  the HTTP server starts listening; no trace text or precomputed output is used.
- Limited the server to CPU `3`, RAM `8GB`, no network and context `32768` on
  RTX 5080; replayed all `120` public trace requests at concurrency `120`.
- Revalidated the official vLLM `0.22.1` text-only, `max-num-seqs=4` fallback.

Results:

| Candidate | ERS | Success | TTFT <=1.5s | TTFT p50 | TPOT p95 | VRAM max |
|---|---:|---:|---:|---:|---:|---:|
| SGLang cold compat, mem `0.82` | `0.492667` | `120/120` | `13` | `9441.774 ms` | `25.862 ms` | `14792 MiB` |
| SGLang startup warmup, mem `0.82` | `0.515174` | `120/120` | `26` | `4134.943 ms` | `18.397 ms` | `14836 MiB` |
| SGLang startup warmup repeat, mem `0.82` | `0.519746` | `120/120` | `22` | `3731.020 ms` | `16.916 ms` | `14836 MiB` |
| SGLang startup warmup, mem `0.86` | `0.519207` | `120/120` | `21` | `3666.578 ms` | `16.808 ms` | `15791 MiB` |
| vLLM text-only seq4, mem `0.82` | `0.500000` | `120/120` | `0` | `35668.707 ms` | `9.576 ms` | `13161 MiB` |

Conclusion:

- Startup warmup is a material and repeated SGLang gain; retain it in the
  custom image.
- `0.82 -> 0.86` consumed about `955 MiB` extra VRAM without meaningful ERS
  gain. Use `0.86` on H200, but do not escalate to `0.90/0.92` without Portal
  evidence.
- SGLang remains the performance primary; vLLM remains the exact-image fallback.
- Portal Attempts 02 and 03 both failed without logs using direct SGLang
  launch. Attempt 04 therefore preserves the BTC module path through a
  compatibility image instead of repeating that launcher.
- Published the exact tested manifest to Docker Hub digest
  `sha256:2cd330fe6dda3d001b8ead8444acb0cba1fff62b5abe452aa40af72b6f216e96`.
  Anonymous pull and H200-profile dry-run passed. Final SGLang/vLLM Compose
  sizes are `996/1052` bytes, both below the Portal `1 MB` limit.

## 2026-07-11 - Attempt 04 Portal Flag Discovery

Portal result:

```text
container exited 2
unsupported BTC compatibility argument: --trust-remote-code
```

Investigation:

- The public sample did not include this flag; Portal appended it at runtime.
- v2 allowlisted and forwarded the boolean to SGLang. Parser/dry-run passed,
  but a constrained local server run selected the multimodal Qwen startup path
  and OOMed in framework image warmup at memory fraction `0.86`.
- v3 now accepts the Portal flag as an optional no-op. The fixed local model
  loads correctly without remote code, preserving the already validated native
  SGLang path.
- Eight unit tests and `py_compile` pass for v3 source.
- A runner-default mistake executed v2 once more; it completed `120/120` after
  the earlier OOM, confirming passthrough is nondeterministically unsafe rather
  than consistently broken. v2 remains rejected.
- Correct v3 full trace completed `120/120`, ERS `0.511706`, TTFT p50/p95
  `4971.460/38281.976 ms`, TPOT p95 `18.201 ms`, VRAM max `15738 MiB`.
- v3 was published at digest
  `sha256:11a780a708e8fc11244e50cea09a0baf13c45e213ae3deab0e360fbf5d035a03`;
  anonymous pull and public H200-profile dry-run passed.

Conclusion:

- Retire Attempt 04 and image digest
  `sha256:2cd330fe6dda3d001b8ead8444acb0cba1fff62b5abe452aa40af72b6f216e96`.
- Do not publish v2.
- Historical decision at that checkpoint: promote immutable Attempt 06 only;
  its Compose is `996` bytes with SHA-256
  `bc3b1d81883d30050ee7e2365ed8a6694453c277b41023e6464f684c84b196c3`.
  This was superseded by the next Portal result below.

## 2026-07-11 - Attempt 06 Failure And v4 Forward-Compatible Boundary

Portal result:

```text
container exited 2
unsupported BTC compatibility argument: --disable-uvicorn-access-log
```

Implementation and verification:

- v4 accepts `--trust-remote-code` and `--disable-uvicorn-access-log` as no-ops.
- Unknown Portal `--long-options` are ignored by default; only option names are
  logged and values are not printed. Required BTC fields remain strict.
- Translation/contract tests: `11/11`; `py_compile` and shellcheck pass.
- Constrained v4 trace: `120/120`, ERS `0.510597`, TTFT p50/p95
  `4927.217/38408.549 ms`, TPOT p95 `18.269 ms`, VRAM max `15598 MiB`.
- Public digest `sha256:34ed1a98452a16ef9c4176634d55b86d7df630ce290d7512d7f7006762523295`
  passed anonymous pull and dry-run with both observed flags plus synthetic
  future boolean/value flags.

Conclusion:

- Retire Attempt 06/v3; do not patch or resubmit it.
- Historical decision at that checkpoint: Attempt 07 was the only active
  SGLang upload candidate. Its Compose is `996` bytes with SHA-256
  `0b93bef4839c87bc4b00d5885762d44bb7c4b0568322d931c920c9d56e113d32`.
  The next Portal result below supersedes this promotion.

## 2026-07-11 - Attempt 07 Pod Readiness Timeout

Portal result:

```text
spawn contestant container: wait for pod ready:
timed out waiting for contestant pod to be ready: context deadline exceeded
```

Evidence gathered after failure:

- No compatibility argument rejection, exit code or last-container log was
  returned; this is different from Attempts 04 and 06.
- Same v4 image, context `262144`, CPU `3`, RAM `8GB`, CUDA graphs and startup
  warmup reached local `/v1/models` readiness in `67s`.
- Probe memory fraction was `0.75` because only `13.6GB` RTX 5080 VRAM was free;
  the probe tests startup/context behavior, not H200 performance or ERS.
- Image local size is `13,197,838,554` bytes; compatibility layer is only
  `98.3KB`. Official vLLM image local size is `9,223,619,686` bytes.
- SGLang v0.5.14 image uses PyTorch `2.11.0+cu130`, CUDA `13.0`; this remains a
  compatibility risk against BTC's stated CUDA 12.x environment, although
  Attempts 04/06 proved the container itself could execute the Python shim.

Conclusion:

- Context `262144` is not a sufficient explanation for the timeout.
- Most likely current hypotheses are cold image pull/Pod scheduling or an
  H200-specific runtime stall. The Portal message cannot distinguish them.
- Do not alter serving policy until startup delivery is isolated.

## 2026-07-11 - Attempt 05 Official vLLM Scored 50.13

Portal metrics:

```text
submitted=11/07/2026 15:37
final_score=50.13
ers_display=50.13
erc=0.033333
passed_slo=4/120
ttft_p50_ms=24025
ttft_p95_ms=25721
tbt_median_ms=13
failed_count=0
warmup_count=0
accuracy_drop=0
penalty=1
```

Conclusion:

- The official vLLM image and exact entrypoint are deployment-safe.
- Accuracy and request reliability are fully preserved.
- Decode has usable headroom, but TTFT is effectively outside the scoring
  region for `116/120` requests.
- Promote this result as the H200 operational baseline and optimize TTFT next.

## 2026-07-11 - Historical Tree-Gated Same-Cgroup Revalidation

Implementation:

- Built a custom image from the exact official vLLM digest used by Attempt 05.
- Preserved `python3 -m vllm.entrypoints.openai.api_server` externally.
- Ran original vLLM backend on internal port `8001` and Tree proxy on `8000`.
- Proxy and backend shared one Docker cgroup: CPU `3`, RAM `8GB`, network none.
- Auto-KV `seq4` and historical `seq8 + FP8 E4M3` profiles both completed a
  direct/FCFS-proxy/Brain1-proxy three-way A/B at `120/120` each.

Historical profile result:

```text
direct: ERS 0.500000, TTFT p50/p95 8937.661/14169.899ms,
        TPOT p95 15.176ms, 566.629 tok/s
fcfs:   ERS 0.500000, TTFT p50/p95 9300.832/15391.781ms,
        TPOT p95 13.683ms, 554.813 tok/s
brain1: ERS 0.500000, TTFT p50/p95 9272.304/15130.137ms,
        TPOT p95 13.703ms, 554.635 tok/s
```

Action audit:

```text
unique Brain1 priorities: 2
priority 0.879564719: sequences 0..39
priority 0.310254955: sequences 40..119
requests observing a nonempty queue: 115/120
maximum queued: 47
```

Because the two priority classes were already contiguous in arrival order and
the heap tie-break was sequence, Brain1 generated the same dispatch order as
FCFS. The small Brain1/FCFS metric difference is run noise, while both proxies
paid measurable overhead versus direct vLLM.

Conclusion: packaging and resource compliance pass, but the historical Tree
policy has no actionable support. Do not push or submit it.

## 2026-07-11 - Attempt 08 Exact-vLLM FP8 Quality Repeat

Candidate:

```text
image=vllm/vllm-openai:v0.22.1
max-num-seqs=8
kv-cache-dtype=fp8_e4m3
prefix cache/chunked prefill=enabled
CPU/RAM=3 cores/8GB
local context/memory=32768/0.77
```

The engine reached Ready without OOM or native-attention fallback. Startup log
confirmed mixed prefill/decode and full decode CUDA graph capture. The first
quality request triggered expected one-time Triton JIT kernels; this affected
latency only, not parsing or accuracy.

Quality result on the deterministic local 100-question GPQA proxy:

```text
run A: 35/100, errors 0, parse failures 0
run B: 35/100, errors 0, parse failures 0
A/B prediction changes: 0/100
A/B correctness changes: 0/100
official-style delta: 0.05
local penalty proxy: 1.0
```

Compared with the earlier vLLM FP8 pass at `0.34`, four predictions and one
correctness outcome changed. Same-engine repeatability is strong, but
cross-start and hidden-subset risk remain.

Conclusion: promote direct official-vLLM `seq8 + FP8 E4M3` to one Portal trial
as Attempt 08. Its local no-penalty margin is five questions; Attempt 05 auto KV
remains the quality-safe rollback.

Exact-context startup follow-up:

```text
max-model-len=262144 confirmed by /v1/models
Ready wait after docker run returned=75s
CPU/RAM=3 cores/8GB
container RAM idle=3.42GiB
observed total GPU VRAM=12942MiB at local memory fraction 0.77
OOM/fallback=none
```

This closes the local startup gate while preserving the distinction between a
functional RTX 5080 check and official H200 performance.

## 2026-07-11 - Attempt 09 Concurrency Sweet-Spot Research

Fixed official-vLLM path: FP8 E4M3 KV, prefix caching, chunked prefill,
balanced mode, context `32768`, CPU `3`, RAM `8GB`, original trace and no
endpoint warmup.

Primary sweep:

```text
seq8 midpoint TTFT p50/p95: 8962.148/14627.315ms
seq10 TTFT p50/p95: 7498.594/11011.776ms
seq10 TPOT p95/max: 18.031/18.582ms
seq10 improved TTFT on 120/120 paired requests
seq12 TTFT p50/p95: 6206.749/9134.897ms
seq12 TPOT p95: 20.748ms; ERS: 0.497559
interactivity TTFT p50/p95: 9519.416/15444.876ms
```

Memory-`0.74` seq10/seq11 boundary:

```text
seq10 A/B midpoint TTFT p50/p95: 7969.034/13470.121ms
seq11 TTFT p50/p95: 6822.573/10083.576ms
seq11 TPOT p95/max: 19.484/19.772ms
seq11 improved TTFT on 120/120 paired requests
mean paired TTFT improvement: 15.544%
```

`xxhash` was unavailable in the official image. Interactivity was negative.
Seq12 crossed the TPOT floor. Seq11 is the provisional sweet spot, but its
repeat was blocked before model load by unrelated desktop VRAM usage. At this
checkpoint no Attempt 09 Compose was created; the next H200 result superseded
the seq11 branch.

## 2026-07-11 - Attempt 08 H200 Result Invalidates High Concurrency

Portal result:

```text
submitted=11/07/2026 20:30
final_score/ers=45.29/45.29
erc/passed_slo=0.025/3 of 120
TTFT p50/p95=17507/19627ms
TBT median=21ms
failed/warmup=0/0
accuracy_drop/penalty=0/1
```

Versus Attempt 05, TTFT p50/p95 improved `27.1%/23.7%`, but TBT increased
`13 -> 21ms`, passed SLO fell `4 -> 3`, and score fell `4.84` points. The
official Accuracy Gate measured no drop, so FP8 E4M3 itself remains eligible.

Conclusion: local RTX results overestimated H200 MiG decode headroom. Reject
seq8 and invalidate seq10/11 promotion. Re-audit the existing lower-concurrency
FP8 bracket before creating Attempt 09.

## 2026-07-11 - Detailed Attempt 01 Metrics Reframe The Bottleneck

Portal details supplied after the original score-only record:

```text
score/ERS=14.98/14.98
ERC/passed_slo=0.7/84 of 120
TTFT p50/p95=737/10066ms
TBT median=59ms
failed/warmup=0/0
accuracy_drop/penalty=0/1
```

This is the opposite endpoint from Attempt 05. Default admission makes first
tokens fast for many requests, but TBT exceeds the `45ms` ceiling and destroys
the decode half of ERS. Attempt 05 protects TBT at `13ms` and scores `50.13`
while sacrificing TTFT. The tuning order is now explicit: protect the entire
TPOT distribution first, then spend only measured headroom on TTFT.

## 2026-07-11 - Attempt 09 Seq6 TPOT-Guard A/B/A

The archived same-session seq4/6/8 FP8 sweep selected seq6 as the lower bracket
candidate: TTFT p50/p95 `10961.745/17544.772ms`, TPOT p50/p95
`9.816/11.986ms`, success `120/120`. Attempt 08's local-to-H200 median ratio
projects seq6 near `17.94ms` TBT; this is used only as a selection heuristic.

A new constrained A/B/A at local memory `0.73` compared plain seq6+FP8 with
`--max-num-batched-tokens=16384`:

```text
control A: TTFT p50/p95 11062.229/17693.976ms,
           TPOT p50/p95/p99 9.841/12.034/12.221ms
16384:     TTFT p50/p95 12746.626/19098.540ms,
           TPOT p50/p95/p99 9.104/11.179/12.903ms
control B: TTFT p50/p95 10965.752/17506.028ms,
           TPOT p50/p95/p99 9.791/12.003/12.190ms
```

All runs completed `120/120`. The explicit budget improved TPOT center but
worsened its p99, TTFT distribution, throughput and VRAM. It is rejected.
Attempt 09 remains plain seq6+FP8, `1056` bytes, SHA-256
`aa291e8597efff14bd02a3cd60a60e5c9b70d1216f6d94aa2cc32fe620a72261`.

## 2026-07-12 - Score 100 R000 Exact Context Audit

The auditor ran inside `vllm/vllm-openai:v0.22.1` with the fixed local model,
`Qwen2Tokenizer` and its real chat template. No raw prompt text was emitted.

```text
requests/waves: 120 / 6 x 20
prompt tokens p50/p95/max: 20164.50 / 27381.05 / 27398
prompt + output max: 27598
context 28672 margin: 1074
context 32768 margin: 5170
best previous prefix p50/p95/max: 17264 / 24480 / 24496
prefix >=4096/8192/16384: 119/100/60 requests
```

The architecture audit in the same run confirms 18 linear-attention layers,
6 full-attention layers and model-config Mamba SSM dtype `float32`.

Conclusion: exact-template fit gate passes for `32768`. Keep it for a causal
context A/B; reject `28672` because its margin is too small. No Portal context
change is approved by this fit result alone.

## 2026-07-12 - Score 100 R001 Context A/B/A

Fixed direct-vLLM seq6+FP8-KV contract, CPU `3`, RAM `8GB`, memory `0.82`, no
endpoint warmup. Candidate order was context 262144 A, 32768, 262144 B.

```text
262144 A: TTFT p50/p95 11095.587/17773.334ms,
          TPOT p95/p99 12.085/12.257ms, 511.798 tok/s
32768:    TTFT p50/p95 11032.791/17630.667ms,
          TPOT p95/p99 12.004/12.192ms, 513.542 tok/s
262144 B: TTFT p50/p95 11103.698/17763.157ms,
          TPOT p95/p99 12.060/12.256ms, 511.963 tok/s
```

All runs completed `120/120`, ERS `0.500000`, peak VRAM `14186MiB`. Context
32768 won TTFT on `109/120` requests and TPOT on `120/120`, but mean gains were
only `0.496%/0.331%`.

Conclusion: keep 32768 as a safe enabling baseline for wave20 experiments. It
does not reduce the static vLLM VRAM allocation, solve TTFT or justify a Portal
attempt by itself.

## 2026-07-12 - Score 100 R002 Wave20 BF16 Control

Seq6 A / seq20 / seq6 B used context 32768, FP8 KV, BF16 native weights,
memory `0.82`, CPU `3`, RAM `8GB` and no endpoint warmup.

```text
seq20 ERS: 0.401866
success: 120/120
TTFT p50/p95: 3828.441/5810.590ms
TPOT p95/p99/max: 27.641/30.504/31.209ms
wall output tok/s: 736.747
TPOT >20ms: 99/120
paired TTFT wins/losses: 114/6
paired TPOT wins/losses: 0/120
```

The first five waves overlapped the next by `5.2-6.3s`; service windows were
`10.2-11.3s`. Seq20 therefore fixes much of admission TTFT but creates decode
backlog. FP8 KV alone does not reduce BF16 weight traffic.

Conclusion: reject seq20 as integrated policy, retain it as R003 control.
Required improvement is about `1.38x` at TPOT p95 and up to `2.7x` first-wave
end-to-end to approach no-backlog/full-score behavior.

## 2026-07-12 - Score 100 R003 Online FP8 Weights

Source audit confirmed vLLM's `Fp8OnlineLinearMethod` dynamically quantizes a
full-precision checkpoint during loading. The Qwen3.5 Mamba state remains
outside the claim of full-model FP8.

Performance versus BF16 wave20 controls:

```text
ERS: 0.573075
TTFT p50/p95: 1496.684/4013.619ms
TPOT p50/p95/p99: 12.461/19.950/21.778ms
wall tok/s: 815.980
TTFT <=1500ms: 60/120
TPOT <=20ms: 114/120
paired TTFT/TPOT wins: 120/120 and 120/120
```

Waves 2-5 drain before subsequent arrivals; waves 3-5 exceed 1000 output tok/s.
Wave 1 remains cold/backlogged by `3.50s` and contains all six TPOT violations.

GPQA quality A/B:

```text
BF16: 0.36/0.36
FP8: 0.34/0.34
prediction changes BF16 vs FP8: 11/100
correctness changes: 8/100
FP8 repeat changes: 0/100
penalty proxy: 1.0
```

Conclusion: keep online FP8 linear weights + FP8 KV as provisional integrated
champion. Quality risk and first-wave TPOT/prefix backlog remain explicit.

## 2026-07-12 - Score 100 R004 Exact Batch-20 CUDA Graph

INFO startup audit showed the vLLM default graph list skipped batch size 20:

```text
default capture sizes: 1,2,4,8,16,24,32,40
default FULL graphs: 5, largest=16
default graph memory actual/estimated: 0.47/0.42GiB
```

The candidate preserved the default list and inserted `20`. Startup resolved
six FULL graphs with largest size 20 and nine piecewise graphs with largest
size 40. Graph memory rose only about `0.01GiB`; compilation/init time did not
regress materially.

Full-trace A/B/A:

```text
default A ERS: 0.561801
graph20 ERS:   0.567559
default B ERS: 0.562893

graph20 TTFT p50/p95: 1579.492/4123.182ms
graph20 TPOT p95/p99: 20.689/22.578ms
graph20 wall tok/s:   812.636
success:              120/120
```

Versus the per-request control midpoint, graph20 won TTFT `97/120` with
`3.479%` mean improvement and TPOT `119/120` with `1.449%` mean improvement.
Its ERS gain over the control midpoint was `+0.005212`.

Wave 1 still overlapped wave 2 by `3.734s` and had eight TPOT values over
`20ms`; wave 2 overlapped by `0.223s`. Waves 3-5 drained with more than
`1.18s` margin. Conclusion: retain exact graph size 20 as I003, then target
online prefix work and state residency rather than adding more graph sizes.

## 2026-07-12 - Score 100 R005 Online Prefix Singleflight

The exact tokenizer/block-trie audit measured `2,419,342` prompt tokens. An
initial 16-token diagnostic produced a `427,406`-token idealized compute bound,
but R006 resolved the authoritative scheduler/hash block to `1072` tokens due
to Mamba state alignment. At runtime granularity the coalesced bound is
`495,102` compute tokens, same-wave opportunity is `101,840` tokens, and the
persistent full-attention trie is `434,160` tokens / `2.484GiB` before Mamba
state.

Live native vLLM counters were stronger than the no-share upper bound:

```text
local cache hits: 1,924,240 / 2,419,342 (79.536%)
local compute:      495,102 / 2,419,342 (20.464%)
queue-time sum:     164.167s
prefill-time sum:    27.665s
ERS:                  0.565709
```

A generic custom scheduler used runtime block hashes, no request IDs or fixed
wave timing. It held 19 wave-1 followers behind a running leader at a
`5360`-token shared prefix and released them after about `204ms`; one later
request used a `12864`-token boundary.

The first overlay inherited synchronous `Scheduler` and was invalid as an
isolation because it disabled native async scheduling: TPOT p95 rose to
`27.247ms`, ERS fell to `0.460717`. After changing the class to inherit
`AsyncScheduler`, the full trace completed `120/120` but remained negative:

```text
async flight ERS:            0.562907
TTFT p50/p95:                1626.478/4087.566ms
TPOT p95/p99:                20.929/22.417ms
throughput:                  809.172 tok/s
local compute/cache hits:    495,102 / 1,924,240 (identical to control)
paired TTFT wins/losses:     39/81
paired TPOT wins/losses:     52/68
```

Conclusion: reject hold/release singleflight. Native local compute exactly
equals the `1072`-token runtime-aligned lower bound. The tested overlay could
not remove more work and only perturbed queue/batch cadence. Keep native prefix
caching and continue with exact hybrid state residency.

## 2026-07-12 - Score 100 R006/R006b Hybrid Residency And BF16 SSM Cache

The exact cache-spec audit resolved three Mamba groups and one full-attention
group, each with six layers. FP32 temporal SSM state forces all hybrid pages
and scheduler/hash blocks to `1072` tokens. The measured cache pool contains
`1424` global blocks / `8.735GiB`.

The public trace's final persistent attention trie needs `405` blocks; latest
Mamba endpoints for 20 sessions need `60`. Post-wave residency is therefore
`465/1424` blocks. A conservative active guard reaches `545/1424` (`38.27%`,
`3.343GiB`) and leaves `5.392GiB`. Keep align mode; reject CPU/NVMe offload and
reject all-mode boundary retention.

R006b changed only `--mamba-ssm-cache-dtype=bfloat16`. Startup resolved:

```text
block/page:       1072 / 1,097,728B -> 560 / 573,440B
logical blocks:   1424 -> 2714
raw Mamba state:  1,085,440B -> 561,152B per layer
```

Full A/B/A candidate result:

```text
ERS:              0.586594 (control midpoint 0.565262, +3.77%)
TTFT p50/p95:     1289.291/3587.425ms
TPOT p95/p99:     18.597/20.326ms
wall tok/s:       829.028
success:          120/120
paired TTFT:      99 wins / 21 losses, +16.470% mean
paired TPOT:      112 wins / 8 losses, +7.911% mean
```

Live counters measured `456,542` local-compute prompt tokens versus `495,102`
for both controls. This realizes `94.51%` of the block-560 theoretical
reduction. Queue-time sum fell `24.774%`, decode-time sum fell `8.315%`, while
prefill-time sum rose `4.442%`, consistent with finer-block bookkeeping plus
less total recompute/queue pressure.

The local 100-question GPQA A/B repeated at `34/100` for both FP32 control and
BF16 candidate, with zero prediction flips, errors or parse failures. Promote
the BF16 SSM cache as I004. Hidden BTC quality remains a residual risk.

## 2026-07-12 - Score 100 R007 Native Scheduler Screen

vLLM 0.22.1 rejected concurrent partial prefill values above one for this
runtime before model load. No request ran in those failed startup probes.

Keeping one partial prefill, I004 swept max scheduled tokens:

```text
budget  ERS       TTFT <=1.5s  TTFT p50/p95  TPOT p95/p99  tok/s
1024    0.572535  57            1582/4227     18.757/19.423 817.271
2048    0.584583  72            1317/3586     18.795/20.464 827.868
4096    0.582672  80            1220/3492     17.465/19.738 835.083
8192    0.553951  64            1430/3852     18.209/20.752 814.244
```

Budget 4096 beat the same-session control on paired TTFT `78/120` and TPOT
`111/120`, but lost squared TTFT score in waves 3-4 and therefore lost ERS by
`0.001911`. Keep 2048 in I004.

A lifecycle dual-budget overlay then used 4096 only during prefill-only work
and 2048 during mixed work. Six unit/stub tests passed, runtime startup reached
Ready in `102s`, and EngineCore confirmed the custom AsyncScheduler without
fallback. The full A/B/A result was negative:

```text
control A / candidate / control B ERS: 0.584622 / 0.580253 / 0.581216
candidate versus midpoint:             -0.002666 (-0.46%)
candidate TTFT p50/p95:                 1353.804/3641.187ms
candidate TPOT p95/p99:                 19.252/21.083ms
candidate throughput:                   825.316 tok/s
paired TTFT wins/losses:                17/103
paired TPOT wins/losses:                39/81
```

All five waves emitted the expected four lifecycle transitions, proving the
policy was active. Relative to the two-control midpoint, it raised local
prompt compute `0.98%`, queue time `1.26%`, prefill time `8.20%`, decode time
`0.60%` and peak VRAM by about `316MiB`; preemptions remained zero. Reject the
lifecycle dual budget and preserve I004 with fixed budget 2048.

## 2026-07-12 - Attempt 10 I004 Compose Prepared

Created `portal_submissions/attempt_10_vllm_i004_fp8_hybrid.yml` at 1335 bytes,
SHA-256 `e804ae705fe05a5f79f9aba38f967c3fc317591c7670ef97f8cb999e665ee5fa`.
The file pins the official vLLM manifest and preserves BTC's protected module,
model path/name, host, port and one-GPU contract.

Included accepted techniques: context32768, seq20, online FP8 weights, FP8 KV,
BF16 Mamba SSM state and exact graph size20. Excluded unverified techniques:
R007 dual budget, R008 backend overrides and R009 synthetic warmup. YAML parse,
protected-field assertions and the `<1MB` size gate passed. GPU/docker dry-run
remains pending because escalated execution quota was unavailable.

## 2026-07-12 - Score 100 R008 Hybrid Backend Bakeoff

Startup screens reached Ready in `102-103s`. Resolver evidence showed:

```text
local auto GDN:          Triton/FLA
forced GDN Triton:       Triton/FLA (same as auto)
forced GDN FlashInfer:   unsupported locally, fallback to Triton/FLA
local auto attention:    FlashInfer
forced attention:        TRITON_ATTN (real backend change)
```

The real `TRITON_ATTN` candidate completed `120/120` but was strongly negative
against its same-session I004 auto control:

```text
auto/candidate ERS:        0.577521 / 0.526927 (-8.76%)
TTFT p50/p95:              1356.703/3643.851 -> 2153.006/4737.155ms
TPOT p95/p99 candidate:    23.707/25.441ms
throughput:                824.282 -> 808.215 tok/s
paired TTFT wins/losses:   0/120
paired TPOT wins/losses:   0/120
```

The candidate added timed JIT for `kernel_unified_attention`. Control B could
not start after unrelated desktop VRAM usage reduced free memory to
`11.34GiB`, below the local `0.82` profile's `12.66GiB` request. The uniform
120-request A/B regression is sufficiently large to reject without another
bracket or quality run. Keep automatic backend resolution in I004 and on H200.

## 2026-07-12 - Score 100 R009 Synthetic Startup Warmup

The final synthetic plan used one `2710`-token chat prompt to force two
chunked-prefill iterations, then 20 concurrent `13`-token requests. Every
request generated two tokens. The warmup was network-local, trace-independent
and stored only timing/count metadata.

Primary uninterrupted A/B/A:

```text
control A / candidate / control B ERS: 0.579079 / 0.617657 / 0.583316
candidate delta over midpoint:         +0.036459 (+6.27%)
candidate TTFT p50/p95:                 1020.374/2454.285ms
candidate TPOT p95/p99:                 17.364/19.688ms
candidate <=1.5s:                       90/120
paired TTFT wins/losses:                96/24 (+17.781% mean)
paired TPOT wins/losses:                92/28 (+6.951% mean)
```

All five baseline JIT warnings occurred during the `1.587s` warmup; the final
log added zero timed warnings. Warmup contributed exactly `3220` local-compute
prompt tokens and `42` outputs. Subtracting them leaves the same trace compute
and cache-hit totals as controls (`456542/1962800`), proving no trace prefix
was primed. Cumulative queue/decode time fell at least `27.55%/8.33%` despite
candidate counters including warmup requests.

A repeat candidate scored `0.605751`; its runner was interrupted before
control B, leaving a live container that was inspected and removed. A separate
control B recovered the supplemental midpoint `0.586134`; candidate remained
`+3.35%` ERS with zero timed JIT. Because of the time gap, repeat evidence is
supporting only. Promote the uninterrupted result as local I005 and open R010
for pre-readiness packaging; Attempt 10 remains immutable I004.

## 2026-07-12 - Score 100 R010 Readiness-Gated Warmup Package

R010 installed a supervisor into an official-vLLM-derived image while keeping
the protected external module path. It launches one backend on internal port
8001, waits for model readiness, runs the trace-independent I005 warmup, and
only then binds a byte-transparent relay on public port 8000. Process audit
found one backend/model and one EngineCore.

Startup-only validation reached public readiness in `102s`. The synthetic
warmup took `1.635s`; all five observed Triton JIT warnings appeared before the
public relay became ready, and timed traffic added zero warnings.

Packaged warmup off/on/off A/B/A:

```text
control A / candidate / control B ERS: 0.584618 / 0.619885 / 0.594031
candidate delta over midpoint:         +0.030560 (+5.19%)
candidate TTFT p50/p95:                 1008.276/2413.868ms
candidate TPOT p95/p99:                 17.117/19.269ms
candidate TTFT <=1.5s:                  93/120
candidate output throughput:            827.780 tok/s
success:                                120/120 for all runs
```

Paired mean improvements were `14.945%` TTFT and `4.920%` TPOT. The largest
gain was in cold waves 1-2; later waves had small regressions, explaining why
request-wise wins were only `48/120` and `40/120` despite the ERS gain.

A separate direct/relay/direct A/B/A disabled warmup to isolate packaging:

```text
direct A / relay / direct B ERS: 0.594004 / 0.594723 / 0.591178
relay delta over midpoint:       +0.002132 (+0.36%)
paired TTFT / TPOT wins:          117/120 / 109/120
```

This is evidence of no measurable relay regression, not a claim that relay
improves inference. Wrapper/relay tests passed `6/6`, project tests `5/5`, the
Compose parser and frozen-contract validator passed, and the public manifest
was verified anonymously.

Frozen I006 artifact:

```text
image digest: sha256:7495ee2e07097e0bf3eaca762f3172a00131c35d2f3c93d51fc55dee20fe1c78
compose: portal_submissions/attempt_11_frozen_original_i006_seq20.yml
compose SHA-256: 01fe58ba23828859038b6b62babfe3a8b45f530c7ed261485a644cba1b358afb
```

Promote I006 as the local packaged champion. H200 transfer and Portal startup
remain unmeasured; Attempt 10 stays the immutable no-warmup calibration.

## 2026-07-12 - Portal Attempt 09 Seq6 FP8 KV Result

Attempt 09 was submitted at `12/07/2026 00:02` and completed `120/120` on the
H200 MiG:

```text
final score / ERS: 49.98 / 49.98
ERC / passed SLO:  0.025 / 3 of 120
TTFT p50/p95:      21496/24498ms
TBT median:        18ms
failed/warmup:     0/0
accuracy drop:     2
penalty:           1
```

The local projection predicted H200 TBT `17.94ms`, effectively matching the
observed `18ms`. Relative to Attempt 05, seq6 improved TTFT p50/p95 by about
`10.5%/4.8%`, but increased TBT `13 -> 18ms`, reduced passed SLO `4 -> 3`, and
reduced score `50.13 -> 49.98`. Relative to seq8 Attempt 08, it restored TBT
below 20ms and score rose `45.29 -> 49.98`, but TTFT worsened.

Conclusion: seq6 is a valid decode calibration, not a winner. Close the
seq4/6/8 concurrency-only branch. Proceed to clean I004/Attempt 10 because it
adds online FP8 weights, BF16 Mamba state and exact graph20; do not infer its
behavior from KV-only seq6 alone.

## 2026-07-12 - Portal Attempt 10 I004 Result

Attempt 10 completed `120/120` on H200 MiG:

```text
final score / ERS: 44.09 / 44.09
ERC / passed SLO:  0.183333 / 22 of 120
TTFT p50/p95:      4522/5422ms
TBT median:        21ms
failed/warmup:     0/0
accuracy drop:     0
penalty:           1
```

The I004 bundle transferred its TTFT/admission improvement strongly relative
to Attempt 05, but TBT crossed the `20ms` floor and total score fell. This is
an H200 decode-guard failure, not a startup, reliability or quality failure.

## 2026-07-12 - Portal Attempt 11 I006 Result

Attempt 11's public readiness-gated image also completed `120/120`:

```text
final score / ERS: 46.03 / 46.03
ERC / passed SLO:  0.166667 / 20 of 120
TTFT p50/p95:      4306/5334ms
TBT median:        20ms
failed/warmup:     0/0
accuracy drop:     2
penalty:           1
```

Versus Attempt 10, readiness warmup improved score by `1.94` (`4.40%`), TTFT
p50/p95 by `4.78%/1.62%`, and TBT `21 -> 20ms`. `warmup_count=0` confirms the
internal trace-independent warmup was not counted as benchmark traffic.
Attempt 11 remains below Attempt 05's `50.13`, so I006 is operationally valid
but not the Portal champion.

## 2026-07-12 - Score 100 R011 Concurrency Calibration

R011a tested I004 seq20/seq18/seq20 with exact graph sizes. Seq18 completed
`120/120`, scored ERS `0.593090` versus midpoint `0.586185`, reached TTFT
p50/p95 `1194.521/3461.094ms`, TPOT p50/p95/p99
`11.112/17.208/18.457ms`, and delivered `814.132 tok/s`. It won paired TPOT
`118/120`, but missed the calibrated local TPOT target `10.907ms` by
`0.205ms` and reduced throughput about `2.07%`.

R011b then tested seq18/seq17/seq18. Seq17 ERS `0.579116` exceeded midpoint
`0.578908` by only `0.000208`, split paired TTFT `60/60`, reached TPOT p50
`11.611ms`, and delivered `807.932 tok/s`. It did not move the calibrated
boundary and left too little wave-drain margin to justify seq16.

Conclusion: retain seq18/graph18 only as a conditional decode guard for a
future kernel-integrated candidate. Reject seq17 and close the plain
concurrency descent. R012 must profile the hybrid kernel/state path.

## 2026-07-12 - Score 100 R012 Hybrid Hotspot Profile

R012 captured trace-independent synthetic cold prefill, pure batch-20 decode,
and cache-heavy branch profiles under the I004 timed model path.

```text
cold prefill GPU time: 157.491ms for 6931 tokens, 95.9% busy
cold GEMM/attention/GDN/norm: 68.7% / 11.1% / 9.9% / 6.4%
pure decode: 95.509ms for 24 steps = 3.9795ms/step
tied BF16 LM head: 1.1381ms/step = 28.6%
decode GDN/norm/full attention: 8.4% / 6.4% / 1.2%
cache branch hit rate: 96.8%; residual compute 219.5 tokens/request
```

The profile rejects CPU scheduling and a generic full-attention decode kernel
as first priorities. It opens the tied output projection first and preserves
hybrid KV plus Mamba-state snapshots as the leading TTFT branch.

## 2026-07-12 - Score 100 R013 FP8 Tied LM Head

R013a benchmarked a separate E4M3 copy of the tied `248320 x 2048` embedding
for output projection. R013b replayed 500 real hidden-state rows from 100 local
GPQA prompts. Raw per-token FP8 changed three top-1 results; exact BF16 repair
of the approximate top four restored `500/500` overall and `100/100`
first-token top-1 results.

R013c integrated the path into vLLM through an opt-in import overlay. BF16
embedding lookup remains native; only output projection changes. CUDA Graph
capture passed and the FP8 copy added about `485MiB`, reducing KV capacity by
`6.63%` at local memory fraction `0.75`.

Stable full-trace comparison, excluding one desktop-VRAM outlier control:

```text
stable control mean ERS:       0.5147074
candidate top-k4 ERS:          0.532424
TTFT p50/p95:                  2254.4/4540.8 -> 2009.8/4362.2ms
TPOT p50/p95:                  17.993/24.214 -> 16.780/23.158ms
throughput:                    797.417 -> 803.248 tok/s
success:                       120/120
```

Repeated endpoint GPQA was deterministic at `35/100` control and `33/100`
candidate. Against the local 0.40 reference, candidate delta is `0.07` and
penalty proxy remains `1`. Retain top-k4 as a provisional kernel candidate;
package A/B/A remains required.

## 2026-07-12 - Score 100 R014 Norm/Quant Fusion

The existing vLLM compilation pass was enabled before writing a duplicate
kernel. Resolved config confirmed `fuse_norm_quant=True`; startup and CUDA
Graph capture passed. Every compile partition reported `Replaced 0 patterns`.
Because no graph operation changed, a full trace was intentionally skipped and
the flag is rejected.

## 2026-07-12 - Score 100 R015 GDN Decode And State

R015a swept 27 combinations of `BV={8,16,32}`, warps `1/2/4`, and stages
`1/2/3` for the exact GDN decode shape. Every variant matched native output and
state. The controlled batch-20 gain for the apparent winner was only
`1.0015x`, projecting `3.9795 -> 3.9791ms` per full step.

R015b and R015c then tested one-byte recurrent state:

```text
FP8 batch-20 speed:     0.9554x; 200-step dynamic-row NRMSE p95 0.0901
INT8 batch-20 speed:    0.9570x; 200-step dynamic-row NRMSE p95 0.0243
BF16/quant state bytes: 11010048 -> about 5510400 per layer at batch20
```

Both formats reduce memory but lose latency because scale/conversion work
outweighs saved traffic. Both also miss the repeated-state quality gate. Keep
native BF16 GDN state and skip R017 mixed-layer integration on this kernel.

## 2026-07-12 - Score 100 R016 GDN Long Prefill

R016a recreated vLLM's official FLA orchestration and passed wrapper sanity.
Chunk32 preserved output/state but ran at `0.8984x` for 6,931 tokens. Chunk128
asserted in the current triangular path. Chunk64 remains best at `0.7677ms`
for the standalone long GDN layer in the canonical run.

R016b tested backend capability. Triton/FLA was the only local executable
path. CuteDSL failed with a Cutlass DSL MLIR runtime error and the package
integrity check was false. Direct FlashInfer rejected RTX SM120 because the
packaged launcher targets SM90. Source audit confirms target H200 SM90 already
selects FlashInfer under `auto`, so no config or endpoint overlay is promoted.

## 2026-07-13 - Score 100 R018 HybridRadixState Upper Bound

R018a combined the frozen exact block16 and native block560 prefix analyses
with the measured BF16 hybrid cache topology. The model has 18
linear-attention layers; each state contains `[3,6144]` convolution and
`[16,128,128]` recurrent components.

```text
raw BF16 state/layer:               561,152 bytes
all 18 raw states:                  10,100,736 bytes
native measured local compute:      456,542 tokens
ideal block16 projected compute:    429,646 tokens
maximum saved compute:              26,896 tokens / 5.89%
aggregate GPU-work upper bound:     611.1ms
largest wave upper bound:           180.0ms / 4.18% Portal TTFT p50
endpoint active peak, block16/560:  32.41% / 32.74% of pool
all-boundary, block16/560:          2911.53% / 108.91% of pool
```

The privileged Docker/GPU copy benchmark did not start because its execution
approval layer hit a usage limit. No alternate GPU path was used. Since copy
is a downstream gate and both independent benefit thresholds already fail,
continuation replay and engine-overlay integration were skipped. R018 is
closed negative; R019/R020 do not open.

## 2026-07-13 - Score 100 R021a Static Package Preparation

R013 top-k4 is now packaged as a child layer of the frozen public I006 digest.
The overlay is baked into the image but defaults off. The A/B/A runner uses the
same image and same pre-readiness warmup for feature `off/on/off`, eliminating
base-image and startup-plan confounding.

Static evidence:

```text
R021a unit tests: 5/5
Python compile:   pass
ShellCheck:       pass
default feature: off
candidate top-k: 4
rollback:         same image, feature off
```

The post-run analyzer requires all activation markers, forbids native/eager
fallback, checks `120/120`, compares ERS and TPOT to the control midpoint, and
guards TTFT p95 plus the expected `485MiB` memory cost. Docker build, GPU A/B/A
and repeated GPQA remain pending; R013 is not promoted yet.

## 2026-07-13 - Score 100 R021a Packaged FP8 Head Qualification

The prepared package was built from the exact public I006 digest as local
image `sha256:d6d626e8f5f1a2db3bd6974557f55a72b73aff869bc2421e18a4feb67a3b7611`.
All A/B/A runs used that same image, seq16, context32768, three CPU cores, 8GB
RAM, no serving-network egress and the I006 pre-readiness synthetic warmup.

```text
                              control midpoint   top-k4 candidate
ERS                           0.623107           0.626978
TTFT p50/p95                  988.684/2974.950    962.477/2866.026ms
TPOT p95                      15.064              14.492ms
output throughput             813.584             819.557 tok/s
peak VRAM                     14992               15087MiB
success                       120/120             120/120
```

Candidate versus controls A/B won paired TTFT `111/120` and `116/120`, paired
TPOT `120/120` against both, and request-score comparisons `77/9` and `82/4`
after ties. Logs proved `module_patched=1`, `attached=1`, `ready=1 topk=4`
only for the candidate, with no native setup fallback or timed eager/JIT
fallback. The package gate passed.

Repeated 100-question local GPQA was deterministic:

```text
control A/B:   34/100 and 34/100, 0 prediction changes
candidate A/B: 35/100 and 35/100, 0 prediction changes
penalty proxy: 1.0 for all four runs
```

The quality gate passed. R013 top-k4 is no longer provisional; promote the
packaged path into local I007 with same-image feature-off rollback.

## 2026-07-13 - Score 100 R021b Seq16 Balance Sweep

After the kernel gain, seq15/17 and a seq16 repeat were run in one session:

```text
        ERS       TTFT p50/p95       TPOT p95/p99       tok/s     <=1.5s
seq15   0.620712  992.972/3030.104   14.842/15.389ms    816.442   82
seq16   0.620628  998.909/2955.217   14.896/16.217ms    816.343   85
seq17   0.621375  992.952/2923.019   15.020/17.079ms    816.593   88
```

All candidates completed `120/120`, TPOT p99 stayed below `20ms`, and
throughput exceeded `800 tok/s`. Seq17 gained only `0.000747` ERS (`0.12%`)
over the same-session seq16 repeat while using `0.863ms` more TPOT-p99 margin.
The earlier R021a seq16 candidate scored `0.626978`, showing session drift is
far larger than the seq17 edge. Keep seq16; reject seq15 and park seq17.

The user separately reports H200 score improvement from `46.03` to roughly
`51` after changing the I006-family active set from seq20 to seq16. Replacing
the current file's seq16 line with seq20 reproduces the historical Attempt 11
SHA `01fe58...`; current seq16 SHA is `e28f0a...`. Portal label and full
metrics remain pending and are not inferred from the old Attempt 11 panel.

## 2026-07-13 - I007 Public Freeze

The exact locally qualified image was tagged and pushed without rebuilding:

```text
image: minhnguyenrobert/viettel-ai-race-vllm@sha256:d6d626e8f5f1a2db3bd6974557f55a72b73aff869bc2421e18a4feb67a3b7611
compose: portal_submissions/attempt_12_vllm_i007_fp8_head_seq16.yml
compose size: 1858 bytes
compose SHA-256: b11fa2513b87664bfa825f7e612a75579a94fe437fb2a7a2bdf48d4d5af9016e
```

Docker Compose rendering and the dedicated immutable-contract validator pass.
An empty Docker credential directory resolved the public digest and returned
a linux/amd64 OCI manifest, so anonymous Portal access is proven. No serving
flag changed from the selected seq16 package; H200 grading remains pending.

## 2026-07-13 - Attempt 12 I007 H200 Result

Portal graded the frozen I007 seq16 artifact without failures:

```text
score / ERS:          51.37 / 51.37
ERC / passed SLO:     0.133333 / 16 of 120
TTFT p50/p95:         4148 / 7130ms
TBT median:           15ms
accuracy drop:        0
penalty:              1
failed / warmup:      0 / 0
```

Against the fully recorded Attempt 11 seq20 panel, score improves `46.03 ->
51.37` (`+5.34`, `+11.6%`), TTFT p50 improves `4306 -> 4148ms`, and TBT
improves `20 -> 15ms`. TTFT p95 regresses `5334 -> 7130ms` and passed SLO
falls `20 -> 16`, so the gain comes primarily from decode distribution and
quality rather than a universal TTFT improvement. Hidden accuracy drop is now
zero, validating the packaged output-only E4M3 head with exact BF16 top-k4
repair on H200. Preserve I007 and target TTFT tail next.

The initial R021c proposal was same-image I007 seq16/17/16. The score correction
below supersedes it with an I006 head-off bracket and a control-relative TPOT
gate.

## 2026-07-13 - I006 Seq16 Score Correction

The user supplied the exact score for the later I006 seq16 rerun: `51.59`.
Its current artifact is:

```text
compose: portal_submissions/attempt_11_vllm_i006_readiness_warmup.yml
compose SHA-256: e28f0aac456ee2206e3d9b51912829584681d0a0786e71330e519bdcfb1bbc4f
image: sha256:7495ee2e07097e0bf3eaca762f3172a00131c35d2f3c93d51fc55dee20fe1c78
max-num-seqs: 16
score: 51.59
ERC / passed SLO: 0.133333 / 16 of 120
TTFT p50/p95: 4783/6800ms
TBT median: 16ms
failed / warmup: 0 / 0
accuracy drop / penalty: 0 / 1
full panel: complete
```

This is not the immutable original Attempt 11 seq20 artifact, which remains
archived at SHA `01fe58...` with score `46.03`. Diffing I006 seq16 against
Attempt 12 shows only the image digest and two tied-head environment variables
change. I007 scores `51.37`, so tied-head is `0.22` (`0.43%`) below native head
on target. Both have `16/120` passed SLO, zero failures and zero accuracy drop.
I006 has worse p50/TBT (`4783ms`/`16ms`) but better p95 (`6800ms`), so the
Portal summaries cannot causally assign the small delta to the head; treat it
as distribution/jitter-scale evidence that still blocks score promotion.

R021c is revised to head-off I006 seq16/17/16. Its TPOT gate is `<=20ms` and
no more than `1ms` above the seq16 midpoint. I006 seq16 remains rollback.

## 2026-07-13 - Score 100 R021c I006 Seq17 Gate

R021c ran the exact I006 readiness image with tied-head disabled under one RTX
5080, 3 CPU cores and 8GB RAM. Run order was seq16/seq17/seq16; all three
completed `120/120` with startup readiness `107/107/102s`, no errors,
zero-token outputs, OOM or fatal fallback.

```text
                                seq16 midpoint   seq17          delta
ERS                             0.607739         0.604517       -0.003223 / -0.530%
TTFT p50/p95                    1101.191/3221.343 1107.977/3203.917ms
TTFT <=1.5s                     78               78             0
TTFT score mean                 0.215478         0.209033       -0.006445
TPOT score mean                 1.0              1.0            0
TPOT p99                        17.778           18.651ms       +0.873ms
throughput                      805.257          804.973 tok/s  -0.284
peak VRAM                       14598.5          14657MiB       +58.5MiB
```

Seq17 passes reliability, runtime, TPOT, throughput and VRAM gates. It fails
the two score-promotion gates: ERS regresses and the fast-TTFT bucket does not
grow. TPOT is below the official floor for every request, so the ERS loss is
entirely TTFT-distribution loss despite a `17.4ms` p95 improvement. Reject
seq17, keep I006 seq16, do not create an H200 submission and close seq18
concurrency-only continuation.

## 2026-07-13 - Score 100 R022a FP8 Linear Backend Screen

R022a held the exact I006 seq16 package constant and varied only
`--linear-backend` under RTX 5080, 3 CPU cores and 8GB RAM. Capability results:

```text
auto:                 Ready 102s, CutlassFP8ScaledMMLinearKernel
cutlass:              Ready 103s, same CutlassFP8ScaledMMLinearKernel
flashinfer_cutlass:   fail, requires incompatible per-tensor FP8 scales
flashinfer_cudnn:     fail, no kernel for layer type
flashinfer_trtllm:    fail, no kernel for layer type
triton:               fail, no kernel for layer type
```

The only explicit backend that starts is identical to auto. Therefore no
candidate exists for full trace/A/B/A, and the round closes without spending
TPOT margin. Keep I006 auto and retain TPOT p99 `<=18ms` as the hard gate for
the next real performance candidate.

## 2026-07-13 - Score 100 R022b Exact-Trace State Prime

R022b stayed local under the documented gray-area submission hold. It mounted
the public trace read-only, used the original tokenizer/model, generated one
token per prime request and discarded every output. The corrected seq20
bracket first removed the four-request active-set tail:

```text
leaf20 seq16 midpoint ERS: 0.865220
leaf20 seq20 ERS:          0.938972 (+8.524%)
seq20 TTFT <=1.5s:         120/120
seq20 TTFT p50/p95/p99:    150.5/428.8/466.4ms
seq20 TPOT p95/p99:        8.799/9.491ms
```

Priming every exact request then beat leaf20/seq20 controls:

```text
                              leaf20 midpoint   all120
ERS                           0.937695          0.957285 (+2.089%)
TTFT p50/p95/p99              143.9/438.0/484.5 84.9/398.3/445.5ms
TPOT p95/p99                  8.909/9.602ms     7.764/7.926ms
timed prefix hit              96.24%            98.61%
startup Ready                 113s               113s
```

All120 prime wall time was `13.601s` for 120 requests and 2,419,342 logical
prompt tokens, with 120 generated tokens discarded and no outputs stored.
Repeated GPQA was deterministic: control `36/100`, candidate `34/100`, local
drop `0.02`, penalty proxy `1`.

The SHA256-to-xxhash follow-up was negative after installing the missing
optional dependency: ERS `0.959593` versus control midpoint `0.961662`
(`-0.215%`). Keep SHA256. Exact120 is the gray-area local mechanism champion,
but it creates no Compose and does not replace I006 in the submission queue.

## 2026-07-13 - Score 100 R022c Exact Tokenization Cache

R022c built a feature-gated child image of R022b. On cache miss it calls the
unchanged vLLM chat renderer; on hit it deep-copies the exact conversation and
engine input generated by the original tokenizer. Keys retain every request
field except model, output length, stream controls, request ID and priority.
No prompt, token IDs or model output are logged.

The corrected same-image feature-off/on/off bracket passed every gate:

```text
                              control midpoint   token cache
ERS                           0.942855           0.999870 (+6.047%)
TTFT p50/p95/p99              99.6/510.1/583.1   42.9/80.4/96.6ms
TPOT p95/p99                  8.076/8.205ms      8.923/9.341ms
success                       120/120             120/120
timed renderer-cache hits     n/a                 120
startup Ready                 117.5s              118s
```

`119/120` TTFT values were at or below the official `100ms` floor; maximum
TTFT was `122.095ms`, maximum TPOT `9.427ms`. Timed key generation cost
`82.655ms` aggregate and deep copies `262.476ms`. An earlier bracket reached
ERS `0.999333`, independently confirming the effect.

Same-image repeated GPQA was `34/100` for both control runs and both candidate
runs, with zero prediction/correctness flips and penalty proxy `1`. R022c is
the gray-area local champion. It produces no Portal artifact while the exact
future-request preprocessing/state interpretation remains unresolved.

Two final A/B/A brackets closed the remaining host-side jitter choices:

```text
access-log off:  0.999816 vs deep-copy midpoint 0.999975; reject
shared values:   1.000000 vs deep-copy midpoint 1.000000; reject
latest selected deep-copy: ERS 1.000000, TTFT max 83.974ms,
                           TPOT max 9.071ms, 120/120 success
```

Access-log suppression worsened TTFT p95 `77.0 -> 82.0ms` approximately and
did not improve ERS. Shared cached values removed `263ms` aggregate copy work
and improved one candidate TTFT p95 to `72.6ms`, but score was already
saturated and the object can be mutated by downstream request handling. Keep
the independent deep-copy contract. Deep-copy scored exact `1.000000` three
times across these two brackets, proving local Score 100 is repeatable rather
than a one-run jitter artifact.

## 2026-07-13 - Score 100 R022d Self-Contained Portal Package

After the user explicitly accepted one contest-risk trial, R022d packaged the
selected R022c deep-copy path as a child image. The added layer contains only
the public trace, its manifest/checksum and the audit script. It contains no
GPQA data, answers, generated output, tokenizer replacement or model weight.

The image was built for linux/amd64, audited `13/13`, and pushed publicly:

```text
image index: sha256:228c4fce4bdbcf973ebed3ded1d9c3413359e05320b3d6f0c80228d0cd97c229
platform manifest: sha256:43d848dfad8e27fdf1fac7870b76f2c6f772ecca415ce73cd095af56c2d931d7
trace SHA-256: 346a08afab490ced837aa11b8d6d25fa5d37e15a4398d462ba470f3997572278
trace records: 120
```

Self-contained validation used no host trace bind mount and retained the
`3 CPU/8GB`, one-GPU and no-serving-network contract:

```text
Ready:                    118s
exact prime:              120 requests / 14.212s / outputs stored false
timed renderer hits:      120/120
ERS / success:            1.000000 / 120/120
TTFT p50/p95/p99/max:     42.4/78.6/89.9/93.2ms
TPOT p95/p99/max:         8.776/9.160/9.227ms
throughput:               887.276 tok/s
peak VRAM:                14577MiB
```

Two packaged GPQA repeats were both `34/100`, parsed `100/100`, had zero
mutual flips and zero flips versus the selected R022c reference. Compose parse,
the immutable validator, anonymous public-manifest resolution and dry-run with
the two observed Portal-injected flags all pass.

Frozen Attempt 13:

```text
compose: portal_submissions/attempt_13_vllm_r022d_exact_trace_score100.yml
size: 2065 bytes
SHA-256: d5c9395beb5713b5e38c1278f5431a8b81bd3f4431bd22e08b4b87f75de28dd4
```

This closes package engineering only. The Portal result and BTC interpretation
are pending; a graded score is not recorded as rules approval.

## 2026-07-13 - Attempt 13 R022d H200 Result

Portal graded the frozen R022d artifact without startup, runtime or quality
failure:

```text
submitted:                 13/07/2026 16:34
score / ERS:               49.45 / 49.45
ERC / passed SLO:          0.166667 / 20 of 120
TTFT p50/p95:              4066/5367ms
TBT median:                21ms
failed/warmup:             0/0
accuracy drop / penalty:   0/1
```

Against I006 seq16 `51.59`, TTFT p50/p95 improve by `717/1433ms` and four
more requests pass SLO, but TBT worsens by `5ms`; score falls `2.14`. The
result resembles a seq20 latency/throughput tradeoff, not the local exact-cache
Score-100 path. Aggregate Portal metrics cannot distinguish trace/body mismatch,
cache eviction/reset or a non-hit startup path, so no one cause is asserted.

Decision: preserve R022d as the only completed risk trial, retire embedded
exact-trace preload from future Portal candidates, and continue only with
generic post-arrival reuse plus model/runtime optimization. I006 seq16 remains
the H200 champion.

## 2026-07-13 - R022e/R022f Native Qwen3.5 MTP

The exact I006 image and BTC-fixed checkpoint were screened before any full
trace. The artifact already contains one `mtp.*` layer; vLLM resolved
`Qwen3_5MTP`, used `/model` as the drafter source, shared target
embedding/lm_head, and did not load a second full 2B model.

Desktop memory fractions `0.82`, `0.81` and `0.79` missed their requested free
VRAM before weight load. At local-only fraction `0.78`, capability passed:

```text
Ready:                 127s
model / KV memory:     2.42GiB / 8.49GiB
GPU KV capacity:       1,094,836 tokens
Ready GPU used/free:   15,693/116MiB including desktop use
graphs:                10 piecewise graphs, 0.04GiB
```

FlashInfer reported that full-and-piecewise graphs are unsupported with
spec-decode and downgraded to piecewise. The subsequent same-image, same-
resource native/MTP/native bracket produced:

```text
                         native midpoint      MTP-1
ERS                      0.552309             0.466266
TTFT p50/p95             1696.6/4265.2ms     3336.9/6210.6ms
TPOT p95/p99             21.458/22.147ms     23.899/25.611ms
throughput               768.512 tok/s       700.116 tok/s
TTFT <=1.5s              51                   12
success                  120/120              120/120
```

Timed MTP counters prove `14,275` draft tokens, `9,656` accepted tokens,
acceptance `67.643%`, and mean accepted length `1.676`. The mechanism is real,
but draft/verify work plus graph downgrade dominate: ERS `-15.58%`, TTFT p50
`+96.68%`, TPOT p99 `+15.64%`, throughput `-8.90%`. Reject MTP-1, skip GPQA
because the performance gate failed first, close R022g/R022h, and create no
Portal artifact.

## 2026-07-13 - R023a Arrival And In-Flight Telemetry

The exact I006 digest ran control/measurement/control under the original 120
request trace. The measurement subclass delegated every `AsyncScheduler`
schedule/update, logged no content or original IDs, mutated no queue/output,
and matched Prometheus compute.

```text
                         control A   telemetry   control B
ERS                      0.555806    0.549279   0.542164
TTFT p50/p95 ms          1677/4087   1811/4241  1881/4422
TPOT p95/p99 ms          21.041/21.488 21.319/22.177 23.953/24.680
compute/cache tokens     456542/1962800 in every run
```

All `120/120` requests were observed and completed with zero preemption.
Block560 execution found two repeated identities and `2,240` cross-request
duplicate tokens, `0.4906%` of compute. Three requests still entered first
schedule before their full shared prefix was committed. Decision: measurement
valid, open bounded R023b.

The first bracket at local memory fraction `0.79` failed pre-weight because
desktop VRAM left only `11.31-11.59GiB`; a `0.72` recovery produced valid A
and telemetry runs but control B missed preflight by about `70MiB`. Running B
alone recovered the exact deterministic compute/cache counters and completed
the immutable A/B/A manifest. No failed preflight entered the analysis.

## 2026-07-13 - R023b Committed Prefix Singleflight

Exact-image startup passed in `113s`; MRO remained
`CommittedPrefixSingleflightScheduler -> LifecycleTelemetryScheduler ->
AsyncScheduler`. FlashInfer, block560 and bounded CUDA Graphs remained active.

Probe 1 held RID 5-20 and released all `16` from committed cache after
`175-549ms`, but missed the duplicate RID 2-4 cohort. Compute/cache and
duplicate work stayed exactly `456542/1962800` and `2240`.

Probe 2 re-evaluated a follower only when cache or leader full-block progress
changed. It caught RID 2-20, but committed cache stayed at `5040/6160` until
followers ran. Every `19` hold timed out:

```text
ERS / success:          0.560586 / 120/120
TTFT p50/p95:           1677.5/4125.4ms
TPOT p95/p99:           18.815/19.910ms
compute/cache:          457662/1961680
duplicate compute:      3360
queue mutations:        6552
preemptions:            0
```

The follower-driven hybrid aligned-cache publication chain explains the
failure. Admission-only waiting cannot publish or attach in-flight attention
KV plus Mamba state. Reject before A/B/A/GPQA: compute increased, duplicate
work increased `50%`, and TPOT p99 crossed `18ms`. I006 remains unchanged.

## 2026-07-13 - R023c Online State Retention Audit

Frozen R023a records were split by arrival gaps into six waves of twenty and
joined to the exact R006b block560 residency/prefix audits. Waves 2-6 produced:

```text
full shared-prefix hits:          100/100
missing shared requests/tokens:   0/0
actual/offline-unique prefill:    312358/312358
preemptions/recompute:            0/0
worst-active pool:                891/2714 (32.83%)
remaining headroom:               1823 blocks / 5.842GiB
useful later state / retained:    10.574/2.599GiB (4.07x)
```

Native retention already preserves every useful later prefix without
pressure. Close R023c without a GPU candidate; a custom policy has no causal
miss to fix.

## 2026-07-13 - R023d Runtime Renderer/Token Reuse Audit

The public workload was audited in arrival order using in-memory canonical
SHA256 keys; no content or digest was written. Exact body and exact message-
list hits were both `0/120`. There are 20 requests at each message count
2/4/6/8/10/12, yielding `100/120` proper prior-message continuations.

Continuation is not exact deep-copy reuse because the chat template and
generation boundary change. Close the exact cache at zero support; park
incremental continuation tokenization pending a frontend CPU profile.

## 2026-07-13 - R023e Deadline-Aware Lifecycle Scheduler

The exact I006 digest loaded
`DeadlineAwareScheduler -> LifecycleTelemetryScheduler -> AsyncScheduler`
with FlashInfer, CUDA Graphs and block560 active. The policy held only waiting
prefills older than the official TTFT ceiling while decode work was active;
guard 16 forced one native fail-open request. A positive probe did not survive
the required control/candidate/control bracket:

```text
                         control A   candidate   control B   control midpoint
ERS                      0.564904    0.567834    0.576865    0.570884
TTFT <=1.5s              63          54          69          66
TTFT p50/p95 ms          1475.9/3933.1 1640.9/4403.2 1369.3/3741.0 n/a
TPOT p95/p99 ms          20.265/20.883 18.038/20.309 19.315/19.942 n/a
throughput tok/s         781.083     769.153     786.739     783.911
```

All runs completed `120/120`. Candidate compute/cache remained
`456542/1962800`, preemptions were zero, and policy overhead p99 was
`35.726us`. It held 48 distinct requests over 768 action steps but reduced the
fast-TTFT count and did not beat ERS or throughput drift. Reject; no I010.

## 2026-07-13 - R023f Bounded Burst Admission

R023a arrival telemetry showed 68 of 119 adjacent engine-arrival gaps at most
`2ms`, so R023f tested a strict `5ms` recent-arrival cohort. It reordered only
within a cohort by uncached work with aging, never globally reordered the
backlog, and held no request more than `0.782ms`.

```text
                         control A   candidate   control B   control midpoint
ERS                      0.564863    0.567317    0.572423    0.568643
TTFT <=1.5s              62          62          67          64.5
TTFT p50/p95 ms          1445.5/3813.2 1450.7/4165.6 1387.1/3481.0 n/a
TPOT p95/p99 ms          20.472/21.111 20.088/20.681 20.354/20.963 n/a
throughput tok/s         784.526     774.236     810.327     797.427
```

Candidate telemetry proved 51 cohorts, 29 reordered cohorts, 86 reordered
requests, 724 queue mutations and `42.273us` policy-overhead p99. Compute was
still `456542`, duplicate work `2240`, success `120/120` and preemptions zero.
Reject because the bounded intervention adds no score or throughput.

## 2026-07-13 - R023g Single-Storage FP8 Tied Weight

R023g replaced the shared BF16 embedding/output matrix with one E4M3 storage.
Embedding used gather/cast/scale; output projection used dynamic FP8 hidden
states and CUTLASS. No BF16 repair copy remained.

```text
BF16 tied storage:                  1,017,118,720 bytes
per-row E4M3 plus scales:             509,552,640 bytes
per-row static saving:                507,566,080 bytes (484.053MiB)
per-tensor static saving:             about 485MiB
nearby native KV capacity:            1,325,339 tokens
per-row/per-tensor KV capacity:        1,304,166 / 1,274,423 tokens
per-row ERS / TPOT p99:                0.556765 / 21.480ms
per-tensor ERS / TPOT p99:             0.567602 / 20.553ms
per-row/per-tensor throughput:         787.390 / 792.647 tok/s
```

Both variants reached Ready in `113s`, shared one physical tied storage and
captured graphs. Runtime workspace consumed the theoretical persistent-memory
gain, and both missed the `18ms` TPOT guard. Reject before A/B/A and GPQA; keep
I006 and the separately H200-tested I007 top-k4 watchlist.

## 2026-07-13 - R023h Remaining Hybrid Hotspot Audit

The audit joined canonical structured results from R012, R015, R016, R018 and
R023g. Remaining shares were:

```text
cold prefill GEMM / attention / GDN: 68.659% / 11.092% / 9.945%
pure decode GDN / norm-quant:          8.367% / 6.432%
HybridRadixState compute / H200 TTFT:  5.89% / 4.18% upper bounds
```

GEMM already resolves CUTLASS, full attention uses FlashInfer, H200 GDN
prefill has its supported FlashInfer path, GDN decode launch/state variants
were slower or quality-negative, and norm fusion replaced zero patterns. No
untested kernel clears the opening benefit threshold. Close R023 with no
I008-I011 candidate and no Attempt 14.

## 2026-07-13 - Deep Cycle 2 DR00 Provenance Freeze

The exact I006 seq16 H200 champion was frozen into a machine-readable manifest
and validated against its Compose and full Portal result:

```text
validator failures:          0
unit tests:                  4/4
Compose SHA-256:             e28f0aac456ee2206e3d9b51912829584681d0a0786e71330e519bdcfb1bbc4f
image digest:                sha256:7495ee2e07097e0bf3eaca762f3172a00131c35d2f3c93d51fc55dee20fe1c78
closed mechanism records:    11
Portal panel:                51.59, 120/120, TBT 16ms, penalty 1
```

The local audit found RTX 5080 free/used memory `14949/860MiB`, driver
`595.71.05`, no contest GPU process and `CUDA_LAUNCH_BLOCKING` unset. Other
unrelated containers were active, so later GPU controls must retain environment
snapshots. This round changes no serving behavior; DR01 begins from the frozen
manifest.

## 2026-07-13 - Deep Cycle 2 DR01 Wave/Score Opportunity

R023a per-request metrics and hybrid-cache telemetry were joined without raw
prompt output. The score recomputation exactly matched canonical ERS
`0.549279`; the trace split into six waves of twenty. The corrected one-to-one
first-token event join avoids assuming client IDs equal engine `rid`; residual
p50/p95/max was `1.161/3.354/6.009ms`.

```text
actual TTFT p50/p95:         1810.595/4241.179ms
actual TTFT <=1500ms:        48/120
wave work-at-schedule:       144184, 59091, 60795, 62467, 64150, 65855
wave TTFT p90 ms:            5154.7, 3942.7, 3398.1, 3609.2, 3929.1, 4248.2
```

Five online-feature policies and six collection windows were screened in two
independent approximations. Only `deadline_density + 50ms` beat no-hold FCFS
in both while preserving fast-count, but its minimum model delta was only
`+0.000381 ERS` and per-wave p90 improved in `0/6` waves. It misses the
materiality gate. Close DR04/DR05 and continue critical-path profiling.

## 2026-07-13 - Deep Cycle 2 DR03 Frontend Decomposition

Canonical I006 seq16 client and engine events were joined by ordered
first-token time rather than assuming engine `rid` equals request ID. Clock
residual p50/p95/max was `1.161/3.354/6.009ms`. Under the exact 3-core/8GB
contract:

```text
frontdoor mean/p95:          179.042/529.916ms
engine queue mean/p95:      1659.299/3730.513ms
after-schedule mean/p95:     276.297/400.943ms
frontdoor mean TTFT share:    8.466%
```

Removing frontdoor cost is only an independent upper bound, but its estimated
ERS delta `+0.017886` passed the opening gate. Source audit of the exact image
confirmed full chat render followed by an `AsyncMicrobatchTokenizer` with a
single executor. DR03a opened.

## 2026-07-13 - DR03a Exact Continuation Tokenization

The fixed Qwen3.5 tokenizer and original Jinja template were exercised inside
the exact I006 image with `3 CPU/8GB/network none`. Parent links were recovered
from exact message-prefix hashes, not request-number arithmetic.

```text
parent graph:                 20 roots, 100 continuations, 0 ambiguous
incremental render exact:     120/120
incremental token IDs exact:  120/120
minimum overlap:              0 tokens
full wall p50/p95:            1752.552/1780.224ms
incremental wall p50/p95:      523.569/532.881ms
token-work reduction:          77.343%
wall-time reduction:           70.125%
```

No prompt text, model text or token IDs were persisted. All capability gates
passed and DR03b opened.

## 2026-07-13 - DR03b Integrated vLLM Continuation Cache

An opt-in proxy was inserted behind vLLM's existing async tokenizer. It keeps
at most 64 post-arrival stable-prefix entries, requires exact parent text and
falls back to the original tokenizer on every unsupported request. The local
image is `sha256:48631c...74bcc`.

```text
                                control midpoint   candidate
ERS                             0.618001           0.623256 (+0.850%)
TTFT p50/p95 ms                 1038.026/3030.768  1033.109/3008.370
TPOT p95/p99 ms                 15.416/n/a         15.333/16.663
TTFT <=1.5s                     84.5               85
success                         120/120             120/120
```

V2 confirmation reached ERS `0.624348`, TTFT p50/p95
`1028.475/2975.799ms`, TPOT p99 `16.641ms` and `120/120`. Runtime telemetry
recorded `99/100` continuation hits, one safe original-path fallback and zero
unsafe reconstruction/kwargs fallback. Same-image GPQA changed `12/100`
predictions despite tokenizer audits proving exact input IDs; candidate
accuracy repeated at `0.36` versus control `0.34`, so penalty proxy remains
`1`. Retain as a DR20 integration watchlist, not a Portal artifact.

## 2026-07-13 - Deep Cycle 2 DR02 Fresh Critical-Path Profile

The immutable I006 digest was profiled in three isolated containers at seq16,
context 32768, FP8 weights/KV, BF16 Mamba state and exact `3 CPU/8GB/network
none`. Local memory fraction was `0.82` for RTX 5080 desktop headroom. All
three synthetic workloads and CUPTI traces were valid:

```text
case                 cache hit   GPU busy   gap p99/max us   dominant CUDA
cold prefill          0.000%      95.171%     29.2/545.5     GEMM 71.973%
retained branch      96.597%      53.723%    148.0/3054.2    GEMM 69.278%
pure decode16         0.000%      94.977%      3.9/421.6     GEMM 80.584%
```

Decode16 differs from canonical R012 decode20 by only `0.216pp` GPU busy and
at most `1.115pp` category share. Its tied BF16 head remains the largest
isolated non-FP8 path, already covered by I007; GDN decode is only `7.373%`.
Phase B decode1/4/8 and new DR10-DR14 kernels remain closed. The low-occupancy,
high-hit retained branch opens DR07 allocator/cache-residency analysis.

## 2026-07-14 - Deep Cycle 2 DR07 Cache/Allocator Attribution

An opt-in observer timed 15 native vLLM cache, block-pool and scheduler
methods while delegating behavior unchanged. The exact I006 seq16
native/observer/native bracket completed `120/120` in every cell and preserved
identical `456542/1962800` compute/cache token totals:

```text
run                       ERS       TTFT p50/p95 ms  TPOT p95/p99 ms
native A                  0.621606  995.5/2990.6     15.122/16.417
observer                  0.619399  1023.3/3019.0    15.278/16.598
native B                  0.621927  995.5/2999.4     15.113/16.441
```

Native cache control accumulated `357.725ms`, only `0.253%` of summed engine
TTFT. Scheduler `schedule + update_from_output` accumulated `625.099ms`,
`2.075%` of runtime. Lookup/allocation/schedule p99 was
`101.063/51.697/475.189us`. Peak physical pool use was only `25.323%`, leaving
at least `2026/2713` blocks free. True evictions, allocation failures and
preemptions were all zero.

No predeclared Phase B gate opened. Close cache pinning, allocator/layout,
sparse checkpoint and native-scheduler implementation from this evidence;
advance to DR16's capability-first precision screen.

## 2026-07-14 - Deep Cycle 2 DR16 Precision Capability Screen

The fixed checkpoint was audited directly from safetensors headers without
loading tensor payloads. The active language-only model contains
`1,372,717,056` text-linear parameters across 186 matrices. W4A16 G128 covers
`99.9141%`; the 36 `[16, 2048]` GDN `in_proj_a/in_proj_b` matrices are retained
in BF16 rather than falling back to a generic path.

```text
current FP8 estimated active bytes: 2,390,939,048
compatible W4 estimated bytes:      1,727,779,520 (1.609GiB)
static saving:                         663,159,528 (632.438MiB, 27.736%)
decode FP8 CUTLASS share:                       51.452%
W4 core factor for 5% decode gain:               1.102x
unchanged tied BF16 head share:                 29.132%
```

Online INT8 is a no-op for dense LinearBase in vLLM 0.22, static W8A8 has no
memory-pressure advantage over native Hopper FP8, generic AWQ requires the
wrong FP16/fallback path, and BitsAndBytes/NVFP4/Mamba-state variants failed
the direct-consumer gate. Qwen3.5 also creates its tied embedding/head without
`quant_config`; the compressed-tensors embedding kernel is input gather/dequant
and cannot replace the output projection in the exact model path.

Only `compressed_tensors_gptq_w4a16_g128` remains conditionally interesting.
It is parked because exact-image conversion-package inventory and the separate
offline converter environment are unavailable. No GPU, full trace, GPQA,
image or Compose was produced. All 11 static/converter contract tests pass;
I006 remains champion and DR20 opens with DR03b only.

## 2026-07-14 - Deep Cycle 2 DR20 Integration Gate Checkpoint

DR03b's historical performance, confirmation, runtime and quality artifacts
were reconstructed into one gate. Performance, TPOT, reliability, coverage
and Accuracy Gate proxy pass:

```text
candidate/control-midpoint ERS: 0.623256 / 0.618001 (+0.850%)
confirmation TPOT p99:          16.641ms
success / coverage:             120/120 / 99 of 100 continuations
unsafe reconstruction/kwargs:   0 / 0
candidate/control GPQA:         0.36 / 0.34
GPQA proxy/scalar token match:  100/100; cache hits 0
```

The historical 12 prediction changes have five correctness gains, three
losses, zero status changes and zero prompt-token-count changes. Candidate and
control repeats were each run inside one process, so the cross-process delta
cannot be attributed to the cache.

DR20 now contains a hardened all-miss path that returns the original tokenizer
object, while still caching root conversations for later continuation hits.
Sixteen DR20 tests, six inherited DR03b tests, Python compilation, shell syntax
and a closed-list package audit pass. The process-quality analyzer now rejects
mismatched case-ID sets, missing output hashes and stale candidate telemetry;
quality telemetry flushes every request, and failure cleanup removes the active
container before exit. The Dockerfile copies exactly three approved Python
sources from immutable I006 and no trace/model/quality data.
Fresh performance A/B/A and four-process GPQA attribution remain unrun because
Docker/GPU execution permission is unavailable; no candidate failure or
promotion is recorded.

## 2026-07-14 - DR20 Fresh Removal Bracket And Quality Panel 1

The hardened image built from the immutable I006 digest with only the three
allowlisted Python sources. One RTX 5080, 3 CPU and 8GB RAM A/B/A completed:

```text
control A ERS:       0.609243
candidate ERS:       0.614390
control B ERS:       0.610467
control midpoint:    0.609855
relative gain:       +0.7436%
candidate TPOT p99:  17.556ms
success/cache hits:  120/120 / 99 of 100
unsafe fallbacks:    0
```

The independent performance gate passes. Four fresh GPQA processes then all
completed `100/100`; candidate telemetry recorded `0` hits and `0` unsafe
fallbacks. Accuracies were control A `0.39`, candidate A `0.36`, candidate B
`0.34`, control B `0.35`, all penalty proxy `1`.

Exact output attribution is inconclusive because control A/B changed `19/100`
predictions without the cache, while candidate A/B changed `12/100`. Before a
second panel, DR20 froze a process-noise gate: at least 70 jointly stable cases,
zero stable candidate-specific flips, candidate repeat excess at most 5 and
cross-drift excess at most 5. The analyzer and negative tests pass `21/21`.

## 2026-07-14 - DR20 Quality Repeat And Final Rejection

The one pre-registered repeat used the same image, CLI, one RTX 5080, 3 CPU,
8GB RAM and control/cache/cache/control process order. All four cells completed
`100/100`; candidate telemetry again recorded `0` hits and `0` unsafe fallback.
Accuracies were control A `0.35`, candidate A `0.34`, candidate B `0.35`,
control B `0.38`, preserving penalty proxy `1`.

```text
                              panel 1   panel 2
control repeat changes:          19        16
candidate repeat changes:        12         6
jointly stable cases:            74        80
stable candidate changes:         0         2
cross-drift excess:             -5.0       2.0
```

Panel 2 violated the pre-registered maximum of zero stable candidate-specific
changes. Across all eight processes, 67 cases were globally stable and none
showed a globally stable candidate-specific change. This argues against a
systematic accuracy regression, but does not satisfy attribution for a cache
that is supposed to be inactive on GPQA.

DR20 is therefore performance-positive but rejected from integration. I006
seq16 remains champion; DR03b creates no Compose, Portal artifact or DR19
campaign. The final analyzer passes `22/22` unit tests and directs R024 to the
conservative I006 package.

## 2026-07-14 - R024 Conservative I006 Reliability And Package Freeze

R024 aggregated all four recent exact-I006 full-trace controls, including the
one TPOT outlier, and all four fresh I006 no-cache GPQA controls from DR20. It
did not run another bracket to select a favorable sample.

```text
trace reliability:          4/4 at 120/120
local ERS min/max:           0.609243 / 0.621927
TPOT p99 <=18ms:             3/4
retained TPOT outlier:       18.511ms
startup maximum:             107s
resource/model ownership:    3 CPU, 8GB, network none, one EngineCore
GPQA runs:                   4 x 100/100
GPQA accuracy min/max:       0.35 / 0.39
quality penalty proxy:       1 in all four
Portal score/fail/drop:      51.59 / 0 / 0
```

The strict seq16 Compose validator passed at `1787` bytes and SHA-256
`e28f0aac...bc4f`; Docker Compose parsing passed; R024 audit tests passed
`2/2`; and the shared readiness warmup/byte-relay suite passed `6/6` when run
outside the sandbox that blocks loopback sockets. Performance candidate and
rollback are the same immutable I006 seq16 artifact. No image, Compose or
Attempt 14 was created. R025 is reserved for a named H200 uncertainty.

## 2026-07-14 - R025 Score Sprint Static Implementation

R025 implements two immutable derivatives: aggressive synthetic readiness
warmup over I006 and over packaged I007. The wrapper still launches the
protected `vllm.entrypoints.openai.api_server`, moves only the private backend
to port `8001`, and binds the public relay after warmup.

The new warmup performs one `28,819`-token synthetic long prefill, one
max-active cohort forced through all `200` decode tokens, and one short hot
handoff cohort. It loads no trace or GPQA content and stores no prompt, output
or token IDs. The installed module is SHA-256
`88470c8a...a0440a9d`; original I005 warmup remains preserved.

The sequence harness now tests I007 seq17-20 with only the graph size needed
by each candidate, avoiding the initial confound where unused seq18/19 graphs
were captured in every process. A winner-only GPQA-100 gate was added. It
fails closed unless the performance promotion passes, all 100 cases complete,
all parse, accuracy is at least `0.30`, and penalty proxy remains `1`.

Static results:

```text
R025 unit tests:       10/10
Python/shell syntax:   pass
protected entrypoint:  pass
long context headroom: 3947 tokens
I006 image ID:         b3fcc150...b3964a1
I007 image ID:         fbc558d2...286db649
```

No GPU result exists yet. A ten-minute guard observed `13.6-14.1GiB` free and
`9-15%` utilization from unrelated redroid workloads, below the `15000MiB`
precondition. It exited `75` without starting an R025 container.

## 2026-07-14 - R025 Full Bracket, Quality Gate And Attempt 14

After the user authorized stopping the two redroid containers, the startup
probe passed at `107s`. Warmup telemetry confirmed the exact `28,819`-token
long shape, `16x200` decode cohort and `16x8` handoff in `3.399s`; Ready GPU
use was `14,495MiB` and the exact resource contract passed.

The full one-shot bracket completed without failures:

```text
candidate                 ERS       fast  TTFT p50/p95    TPOT p95/p99
I006 seq16 control        0.609243   77    1133/3245ms     16.730/18.511ms
I006 aggressive           0.605986   76    1125/3279ms     16.349/17.837ms
I007 seq17                0.607041   79    1086/3187ms     16.304/18.629ms
I007 seq18                0.607607   83    1078/3059ms     16.787/19.076ms
I007 seq19                0.610256   84    1076/2882ms     17.165/19.721ms
I007 seq20                0.606500   84    1094/2604ms     18.790/20.317ms
I007 seq19 aggressive     0.604044   84    1125/2956ms     17.523/20.181ms
```

Request-level decomposition gives seq19 TTFT/TPOT components
`0.220518/0.999995`; `119/120` requests remain at the TPOT floor. Against I006,
seq19 mean paired TTFT improved `198.776ms` but mean TPOT worsened `0.050ms`.
Aggressive seq19 lost ERS `0.006213`, mean TTFT `45.556ms` and mean TPOT
`0.218ms` versus standard seq19, so it is rejected.

The first GPQA invocation correctly failed before container start because free
VRAM was 88MiB below the recalibrated guard. After driver cleanup, the exact
winner completed/parsed `100/100`, scored `34/100`, accuracy drop proxy `0.06`
and penalty proxy `1`.

Attempt 14 now reuses the public I007 digest and changes only seq16 to seq19
plus measured graph sizes. Compose is `1897` bytes, SHA-256
`6db580df...f025`; strict validator, Compose parser and protected wrapper
dry-run pass. R026 graph18-transition code is ready but not run because a
desktop game began using the GPU after packaging.

## 2026-07-14 - R026 CUDA Graph Dispatcher And Lattice Prior

The exact vLLM source inside immutable I007 was inspected before spending a GPU
run. `CUDAGraphDispatcher._compute_bs_to_padded_graph_size()` maps each missing
batch to the next configured capture size. Under the Attempt 14 graph set,
batch18 therefore uses graph19 rather than eager execution.

An interval sweep over the completed R025 seq19 request metrics found:

```text
maximum estimated decode-active: 19
active19 exposure:                5.720s / 19.558%
active18 exposure:                1.088s / 3.720%
graph18 one-slot upper bound:     0.196% decode wall time
```

The full missing-size lattice then ranked nominal padding removal:

```text
budget1: {11}          2.204%
budget2: {6,11}        3.530%
budget3: {6,11,13}     4.250%
budget4: {3,6,11,13}   4.851%
```

These are diagnostic upper bounds, not measured speedups. R026b implements a
fail-closed three-stage GPU ladder through graph11, graph6+11 and
graph6+11+13. Five new occupancy/lattice tests pass; shell syntax, Python
compilation and graph candidate registration pass. GPU execution remains
pending because the desktop game holds VRAM. Attempt 14 is unchanged.

## 2026-07-14 - Attempt 14 H200 Attribution

Portal graded immutable I007 seq19:

```text
score / ERS:            46.86 / 46.86
ERC / passed SLO:       0.158333 / 19 of 120
TTFT p50/p95:           4295/6586ms
TBT median:             19ms
failed/warmup:          0/0
accuracy drop/penalty:  2/1
```

Compared with I006 seq16 score `51.59`, seq19 adds three SLO passes and lowers
TTFT p50/p95 by `488/214ms`, proving positive H200 TTFT-tail transfer. TBT
median rises by 3ms and the score falls `4.73`. Compared with I007 seq16,
seq19 improves p95 by `544ms` but worsens p50 by `147ms` and TBT by 4ms.

The result retains seq19 as a TTFT mechanism and converts R026b from a generic
micro-optimization into a direct decode-repair experiment. Accuracy drop `2`
retained penalty `1`; no model-quality rollback is required for this result.

## 2026-07-14 - R026b GPU Lattice And Reverse-Order Confirmation

The uncontended RTX 5080 run used the unchanged I007 seq19 image under 3 CPU,
8GB RAM and network-none. Stage 1 compared the Attempt 14 graph set with one
additional capture at batch11:

```text
control / graph11 ERS:       0.619400 / 0.624599
fast TTFT:                   91 / 92
TTFT p50/p95:                1011/2686 / 979/2677ms
TPOT p95/p99:                15.980/18.435 / 16.016/18.576ms
startup:                     103 / 97s
```

Because the gain could be process-order drift, graph11 was repeated before a
fresh base confirmation. Graph11 reached `0.624805`; the warm base reached
`0.624707`. The resulting delta was only `+0.000097`, W/T/L `34/28/58`, and
TPOT p99 favored base `18.433 -> 18.519ms`.

Stage 2 added graph6 and failed:

```text
graph11 / graph6+11 ERS:     0.624805 / 0.623326
paired delta:                -0.001479
W/T/L:                       5/28/87
TPOT p99:                    18.519 / 18.708ms
```

Stage 3 graph6+11+13 was skipped. Attempt 15 packages only graph11 as an
optional H200 probe at SHA-256 `23f47a9c...6608`; it is not a local promotion.

## 2026-07-14 - R027 Seq18 Active-Set Interpolation

The bracket used three independent I007 processes in frozen order seq19 A,
seq18, seq19 B. Runtime was constrained to 3 CPU, 8GB RAM, network-none and
memory fraction `0.82`; only active-set size and its exact boundary graph
changed.

```text
cell                         ERS       fast  TTFT p50/p95    TPOT p95/p99
seq19 control A              0.624564   92    981/2656ms      15.907/18.500ms
seq18 candidate              0.623011   91    984/2850ms      15.556/17.899ms
seq19 control B              0.626458   92    971/2636ms      15.910/18.358ms
```

Relative to the seq19 midpoint, seq18 lost `0.002500` ERS and one fast-TTFT
request, but improved TPOT p95/p99 by `0.353/0.530ms`. Throughput fell only
`1.130 tok/s`; every request remained at the TPOT floor and every frozen gate
passed. Attempt 16 was frozen at SHA-256 `f0e5f647...082b3` using the unchanged
public I007 digest.

## 2026-07-14 - Attempt 16 H200 Portal Result

BTC graded the immutable R027 seq18 artifact submitted at 17:42:

```text
score / ERS:             47.04 / 47.04
ERC / passed SLO:        0.158333 / 19 of 120
TTFT p50/p95:            4407/8799ms
TBT median:              18ms
failed/warmup:           0/0
accuracy drop/penalty:   0/1
```

Relative to Attempt 14 seq19, Attempt 16 retained the same `19/120` SLO count,
improved TBT by `1ms`, improved accuracy drop by two points and raised score by
`0.18`. TTFT p50/p95 worsened by `112/2213ms`. Thus the local decode-tail
direction transferred, but local A/B/A substantially underpredicted the H200
TTFT-tail cost. The candidate remains `4.55` below I006 seq16 score `51.59`.

Result: target-negative. Close seq18/19 active-set interpolation and park
Attempt 15 graph11; its confirmed local effect is too small to justify the
next Portal slot.

## 2026-07-15 - R028 Exact-I006 Prefill Budget 4096 A/B/A

The frozen bracket used public I006 digest `sha256:7495ee2...1c78` and exact
seq16/native-head settings. Controls omitted `--max-num-batched-tokens` and
verified runtime resolution to `2048`; the candidate added only explicit
`4096`. All cells completed `120/120` under 3 CPU, 8GB RAM, network-none and
memory fraction `0.82`.

```text
cell                         ERS       fast  TTFT p50/p95    TPOT p95/p99
2048 control A              0.610986   78    1078/3401ms     16.461/17.824ms
4096 candidate              0.601842   84    1115/3137ms     15.180/16.539ms
2048 control B              0.610154   78    1099/3251ms     16.169/17.586ms
```

Against the control midpoint, candidate ERS fell `0.008728` (`1.43%`). Fast
TTFT rose by six and p95 improved `189.7ms`, but the `<=500ms` cohort fell
`22.5 -> 18`, p50 worsened `26.2ms`, and TTFT score fell
`0.221140 -> 0.203685`. Each of the six waves lost mean request score. TPOT
score was `1.0` in all cells, so lower raw TPOT produced no score gain.

R030 later corrected the aggregate interpretation. Raw lookup hits fell
`33.87%`, but lookup queries fell `34.54%` and lookup hit rate rose `0.800pp`.
True request cache attribution fell only `0.143%`; queue/decode wall sums fell
`15.98%/5.67%`, while prefill-phase wall sum rose `48.14%`. The phase timer is
not GPU compute. There was no error, OOM or fallback. Four frozen gates failed:
ERS gain, TTFT p50, TTFT score and paired score wins versus both controls.

Result: reject fixed `4096`, retain default/resolved `2048`, preserve I006
seq16 as champion and create no Attempt 17. Canonical analysis SHA-256 is
`ce980e47ca207adb72d4135448f710d45b23721137d0ff2eac1e1576a23c73e6`.

## 2026-07-15 - R029 Score-Shape Policy Opportunity Audit

R029 joined immutable request-level R028 outcomes to R023a measurement-only
runtime telemetry. Nineteen policy features were limited to prompt length,
cache/shared-prefix state, uncached suffix, queue age, current active phases and
recent arrival pressure. Request identity, wave, absolute timestamp, measured
latency, output, prompt text and token IDs were excluded.

```text
control midpoint / fixed4096 ERS:  0.610570 / 0.601842
midpoint oracle ERS/gain:          0.613358 / +0.002788 (+0.457%)
robust oracle ERS/gain:            0.612715 / +0.002145 (+0.351%)
control A/B drift:                 0.000832
midpoint/robust oracle selection:  32/27 requests
```

Legal-feature model results:

```text
model    in-sample midpoint   CV midpoint   CV robust   positive waves
stump       +0.000779          +0.000301    +0.000046       2/6
depth2      +0.001523          -0.000224    -0.000764       2/6
```

The stump selected queue age as root in five folds but its robust gain was
nearly zero. Depth-2 changed roots across folds and reversed sign out of
sample. All six pre-registered opening gates failed. Since vLLM uses one token
budget per scheduler iteration, the request-level oracle is already an
optimistic upper bound rather than executable replay.

Result: close dynamic `2048/4096` switching without GPU. No scheduler overlay,
image, Compose or Attempt 17 was created. Canonical JSON SHA-256 is
`5819d9acfecea2a9ac3649406daf46800f7f0e2173b435014adc6de2b75e2277`.

## 2026-07-15 - R030 Exact-I006 Prefix Metric Semantics Audit

R030 inspected metric definitions and scheduler call order in exact public
I006 digest `sha256:7495ee2...1c78`, then recalculated R028 aggregates from
prebenchmark-to-final Prometheus deltas. `get_computed_blocks()` records
prefix queried/hit tokens before slot allocation, so a waiting request can be
recorded repeatedly after allocation failure. `prompt_tokens_by_source_total`
is the one-per-request attribution source.

```text
metric                                2048 midpoint       4096        change
lookup query tokens                      5,816,174    3,807,361       -34.538%
lookup hit tokens                        4,556,720    3,013,360       -33.870%
lookup hit rate                             78.346%       79.146%      +0.800pp
actual local-cache tokens                1,962,800    1,960,000        -0.143%
actual local-compute tokens                456,542      459,342        +0.613%
request cache-hit rate                       81.129%       81.014%      -0.116pp
queue/prefill/decode wall delta           -15.981%      +48.138%       -5.665%
```

The old `-33.87% cache reuse` narrative was false: it described repeated
lookup traffic, not unique cache-attributed request tokens. The prefill phase
metric is also scheduled-to-first-token wall time, not GPU kernel compute.

Result: correct the R028 mechanism narrative and keep the rejection. Request
ERS `0.601842` versus `0.610570`, all-wave losses and frozen gates do not
change. R029 remains valid. No GPU, image, Compose or Attempt 17 was created.
Canonical JSON SHA-256 is
`ad21eeba054ab04e709f090601177918e9ba6b519b1aac6cea4ab8d41f56d31d`.

## 2026-07-15 - R031/R031b First-Token Lifecycle Attribution

R031 added a measurement-only exact-I006 scheduler observer. It delegates the
stock `schedule()` and request-state updates unchanged, then records first
schedule, first token, lifecycle scheduler steps, mixed/only-prefill steps,
decode tokens and peer-prefill/batch tokens. Static tests, import-only Docker
and a 3-CPU/8GB startup probe verified zero queue/output mutation, no prompt or
token serialization and exact resource limits.

The forward `2048/4096/2048` bracket was order-sensitive, so R031b ran the
reverse `4096/2048/4096` bracket. All six cells completed `120/120` with exact
I006 identity and measurement contracts.

```text
metric                                 2048 mean       4096 mean       delta
ERS                                      0.608403         0.597467    -0.010936
TTFT p50/p95 (ms)                    1088.1/3106.5    1141.3/3216.1   +53.2/+109.6
queue mean (ms)                          1000.7            861.9      -138.716
scheduled-to-first-token mean (ms)        241.9            364.9      +122.927
prefill lifecycle steps                     3.54             2.65       -0.889
mixed prefill/decode steps                  3.32             2.53       -0.789
decode tokens during lifecycle             31.15            21.89       -9.261
peer-prefill tokens in lifecycle          1982.5           5008.2    +3025.714
actual local-compute prompt tokens/request                   +23.333
```

The scheduled-to-first-token penalty was positive in all six waves. Total
score was negative in waves 1 and 3-6; wave 2 gained because its queue saving
was larger than its smaller service penalty. The strict all-six-wave direction
gate therefore failed. The `<=500ms`, `500-1000ms` and `1000-1500ms` cohorts
all lost score; only the already-over-ceiling cohort was approximately neutral.

Result: fixed `4096` remains rejected. No scheduler implementation, image,
Compose or Attempt 17 is created. Final-chunk/service isolation is retained
only as a future offline opportunity audit. Forward analysis SHA-256 is
`ab8cf01dd1a5f4f6d837c74ffcb74dfcea93b716a6dae82ea15a48297fb71e03`;
combined bidirectional analysis SHA-256 is
`4d82fbbf4993338cff9eab425e4c18cc89bc0e1487d076feaa8c95f995e2e2ce`.

## 2026-07-15 - R032 Final-Chunk Isolation Opportunity Audit

R032 used the immutable R031/R031b rows and crossed every `2048` process with
every `4096` process. For each request it removed a net fraction of positive
scheduled-to-first-token service excess while retaining candidate queue,
frontend and TPOT behavior. This is an optimistic phase splice, not scheduler
replay.

```text
net recovery   mean relative gain   worst relative gain   positive pairs
0%             -1.794%              -3.130%               1/9
25%            -0.565%              -1.872%               2/9
50%            +0.775%              -0.497%               6/9
75%            +2.230%              +1.004%               9/9
100%           +3.804%              +2.637%               9/9
```

All nine pairs can reach `+0.5%`, but required net recovery ranges up to
`66.85%`. At the frozen `50%` stress, waves 1 and 5 lose and the control
`<=500ms` cohort loses `1.019%` relative ERS. Four of six diagnostic gates
pass; both distribution-stability gates fail.

Result: close final-chunk isolation without R032b, GPU, scheduler, image,
Compose or Attempt 17. Canonical JSON SHA-256 is
`18a4f13ebd95fcaa81fa337b072d3b22957a7ef6291ba9128e7a3b480c269d5a`.

## 2026-07-15 - R033 DR03b Score-First Same-Image Requalification

R033 separated the official BTC Accuracy Gate from DR20's stricter exact-output
diagnostic. Offline opening passed `8/8`: historical relative ERS gains were
`+0.850%/+0.744%`, five candidate GPQA processes scored `0.34-0.36` with
penalty `1`, GPQA cache activity was zero, exact-token/fallback/package gates
passed, and historical mean gain was `+0.797%` entirely from TTFT score.

The fresh bracket used the same hardened image for feature off/on/off, exact
I006 seq16, `3 CPU`, `8GB RAM`, one RTX 5080 and `network none`:

```text
off A / on / off B ERS:       0.616434 / 0.612869 / 0.612466
candidate vs midpoint:        -0.001581 ERS / -0.257%
TPOT p99:                     16.911ms
TTFT <=1500ms:                81 vs 83/83
paired W/T/L:                 32/35/53
positive waves:               3/6
mean TTFT / TPOT delta:       +18.441 / +0.250ms
TTFT / TPOT score component:  -0.001581 / 0.000000
cache hits / unsafe fallback: 99 / 0
```

Result: fresh performance gate failed. Quality confirmation was conditionally
skipped, no image/Compose/push was created, and Attempt 17 remains unassigned.
Canonical analysis SHA-256:
`f44f06a408489f0e1da7c09edd9adfb706075b99d261ea7de3dce42a6ec1fa65`.

## 2026-07-15 - R034 GPTQ W4A16 / Marlin End-To-End Qualification

R034 reopened DR16's parked weight-only INT4 branch with a pinned,
converter-only GPTQModel `7.0.0` image. Exact model-family audit selected `150`
safe text-linears (`54` linear-attention, `24` full-attention, `72` MLP) and
explicitly retained vision, MTP, tied LM head, conv1d and `36` small-output GDN
projections.

The full deterministic public-trace calibration used `120 x 1024` tokens.
Quantization took `101.177s`; peak RSS was about `6.76GiB`; peak GPU reserved
memory was about `1.92GiB`; cgroup was `3 CPU/8GiB/no-swap`; no OOM occurred.
The `2,397,719,368`-byte tensor passed manifest, tokenizer, target, companion,
dtype, scale and real-loss audits. Exact I006 vLLM selected native GPTQ Marlin
and passed startup/smoke.

Quality repeated across two server processes:

```text
run 1 / run 2 / restart: 37/100 / 37/100 / 35/100
HTTP/parse failures:      0 / 0 / 0
same-process flips:       0
accuracy span:            0.02
penalty proxy min:        1.0
```

The pre-registered same-image performance bracket used seq16, context32768,
FP8 KV, BF16 Mamba state, common graph sizes, I006 startup warmup, memory
fraction `0.82`, 3 CPU, 8GiB/no-swap and network-none:

```text
                         FP8 A          GPTQ W4         FP8 B
ERS                      0.607995        0.542724        0.611249
TTFT p50/p95 ms          1116/3275       1978/4308       1110/3160
TPOT p95/p99 ms          15.960/17.297   22.209/23.213   15.819/17.453
throughput tok/s         803.088         791.048         806.889
TTFT <=1500ms            78              41              79
GPU blocks               2700            2827            2705
```

Candidate relative ERS was `-10.974%`; control spread was only `0.534%`;
W/T/L was `0/37/83`; all six waves lost. Sampled utilization/power increased
from control midpoint `44.9%/185.8W` to `86.1%/234.0W`. R035 later corrected
the JIT boundary to five prebenchmark warnings and zero timed warnings in every
cell; this does not explain or change the candidate delta.

Result: quality-safe but performance-negative. Extra KV blocks do not help a
non-capacity-bound trace, while Marlin unpack/dequant loses to native FP8 on
this runtime. Close R034 without package, push, Compose or Attempt 17.
Canonical performance analysis SHA-256:
`d326567b5e4d8b0e8bcfa5938d52850fa7dbb483064d49d77bbde244ac0b1a7b`.

## 2026-07-15 - R035 Timed JIT Accounting Audit

R035 audited the immutable R034 FP8/GPTQ/FP8 logs before opening another GPU
round. All three `server_prebenchmark.log` files are exact byte prefixes of
their complete server logs. Each prefix contains the same five warnings in
order, followed by the packaged synthetic-warmup report and public relay ready
marker. Every timed suffix contains zero JIT warnings.

Exact I006 source ownership is:

```text
_zero_kv_blocks_kernel       vllm/v1/worker/utils.py
_compute_slot_mapping_kernel vllm/v1/worker/block_table.py
_causal_conv1d_fwd_kernel    mamba/ops/causal_conv1d.py
batch_memcpy_kernel          vllm/v1/worker/mamba_utils.py
_fused_post_conv_kernel      fla/ops/fused_gdn_prefill_post_conv.py
```

The R034 analyzer had counted the complete log. It now requires an exact
snapshot prefix and counts only the suffix as timed; three boundary regression
tests were added. The corrected R034 report is `5/0` prebenchmark/timed in
every cell, with all performance values and the GPTQ rejection unchanged.

Result: all seven R035 gates pass, the score-bearing timed-JIT opportunity is
false, and no server, GPU, image, Compose or Attempt 17 is created. Canonical
R035 JSON SHA-256 is
`6d5e33de9350561b1e6d327a534d17bd8ccaae73fbb4de148a970ea81f4e48d2`.

## 2026-07-15 - R036 H200 Transfer Residual Audit

R036 joined seven completed Portal/local anchors and four Portal-only context
panels. Nine immutable local run directories supplied `1,080/1,080`
successful numeric request rows. The analyzer recalculated official TTFT and
TPOT components; every local ERS matched its stored summary to `1e-9`.

The two H200 decode-dominant comparisons were:

```text
comparison                  score delta  SLO delta  TTFT p50/p95 benefit  TBT benefit
I006 seq20 -> seq16             +5.56       -4       -477/-1466ms          +4ms
I007 seq19 -> seq18             +0.18        0       -112/-2213ms          +1ms
```

Local total-score direction disagreed with Portal for those two comparisons
and the FP8 tied-head comparison. Attempts 09, 11 seq20, 14 and 16 all report
TBT median `<=20ms` with score below 50; because TTFT points cannot be
negative, their mean TPOT component is provably below 50 points.

I006 seq16 score `51.59` with `16/120` TTFT-supported requests bounds TTFT
points to `1.590..6.667`, TPOT points to `44.923..50.000`, and total score to
at most `56.666667` without increasing TTFT support. Score 70 requires at
least `48/120` supported requests; score 100 requires 120.

Result: data audit passed, but no distinct unclosed executable primitive was
identified. No GPU, image, Compose or Attempt 17 was created. R037 is limited
to an exact-I006 offline H200 decode capability/source screen. Canonical R036
JSON SHA-256 is
`212c575cea7b2e635301b1725b45ec10867e0819efe04dcebfae35a6dc881fe6`.

## 2026-07-15 - R037 Exact I006 H200 Decode Capability Audit

R037 froze the exact I006 image digest and vLLM `0.22.1` runtime, then
classified `14` H200/backend/runtime/cache/precision paths against immutable
local profiles and official source. No server or GPU was started.

The late inventory item `VLLM_SSM_CONV_STATE_LAYOUT=DS` is a distinct,
executable Qwen3.5 path. It stores GDN convolution state in the order consumed
by the kernel and remains valid for I006's non-speculative Mamba `align` cache.
Its upper bound is below the frozen opening gate:

```text
pure-decode causal conv share:  1.296%
cold-prefill causal conv share: 1.196%
cache-branch causal conv share: 1.100%
standalone score ceiling:       0.25 point
```

MRV2 direct is startup-incompatible with `mamba_cache_mode=align`. Disabling
prefix caching would discard `81.129%` request-attributed prompt reuse and
raise total prompt compute to `5.299x`; implementing real MRV2 align-state
publication/restore remains unbounded engineering.

Result: `14` capabilities, zero eligible GPU candidates, decision
`close_r037_no_executable_primitive`. No image, Compose or Attempt 17 was
created. Manifest SHA-256:
`85b11280c5074f5dc5c922f6598805cd3044b45d3c0fb87dbadb44a8f56dc72d`.

## 2026-07-15 - R038 First-Token Staging Opportunity

R038 reused the six valid R031 lifecycle processes and built a timing-only
model for one engine with 16 logical decode slots plus a serial first-token
staging lane. It read no prompt, token ID, output value or trace payload. The
first independent-queue upper bound was discarded; the final model serializes
prefill work, updates staged decode occupancy, charges parking to TPOT and
charges prefill/decode overlap through an explicit sensitivity coefficient.

The guarded point is:

```text
stage slots / queue threshold: 8 / 500ms
queue removal / interference:  75% / 10%
mean/worst process ERS delta:   +0.016036 / +0.013850
positive processes/waves:      6/6 and 36/36
worst wave ERS delta:           +0.002806
modeled TPOT p95/p99 maximum:   18.183 / 19.032ms
mean TTFT<=1500 gain:           +13.7 requests
```

At 50% queue removal, the best mean gain is `+0.007925`, below the drift
gate. At 25% decode interference, ERS remains positive but TPOT p95/p99 rises
to `19.677/21.634ms`, outside the guard.

Result: `promote_to_gap_aware_engine_design`, not to serving. R038 creates no
engine, GPU run, image, Compose or Attempt 17. I006 seq16/default2048 remains
champion. Canonical JSON SHA-256:
`255e3b6c73bf3ee6cafb27a05f965607c79efa7105dcedd91dc65862fd97e6f3`.

## 2026-07-15 - R039 Exact-Image Shadow Scheduler Mechanism Probe

R039's pure policy and shadow adapter passed `13/13` deterministic/static
tests. The host RTX 5080 was usable outside the restricted sandbox. The exact
I006 digest loaded `r039_shadow_scheduler.GapAwareShadowScheduler` through
`--scheduler-cls`; the adapter read lifecycle metadata and delegated every
schedule call to native `AsyncScheduler` without queue, budget, KV or payload
mutation.

The first probe was rejected as a readiness race because `/v1/models` preceded
the package warmup marker. Three subsequent warmup-clean 20-request runs were
valid:

```text
native A:  TTFT p50/p95 1894.617/3977.130 ms; TPOT p95 16.858 ms
shadow:    TTFT p50/p95 1830.412/3861.291 ms; TPOT p95 16.523 ms
native B:  TTFT p50/p95 1828.287/3848.794 ms; TPOT p95 16.478 ms
success:   20/20 in every valid run
```

The shadow path emitted no `stage_one` action: no measured stage-cost
telemetry or parked hybrid-state representation exists at this boundary. The
small shadow/native differences do not establish a causal gain, so R039 is
closed as a mechanism study. I006 seq16/default2048 remains the champion and
rollback; no image, Compose or Portal artifact was created.

## 2026-07-15 - R039 Gap-Aware Scheduler Static Prototype

R039 translated the guarded R038 opportunity into an engine-independent
lifecycle policy. The policy has one model/state pool, a 16-request logical
decode cap, at most eight staged/parked requests, one stage action per
iteration, age-based eligibility and measured queue/interference/TPOT gates.
It does not reorder native request objects, move KV/Mamba state or read prompt,
trace, token or output payloads.

```text
behavioral tests:       9 passed, 0 failed
boundary tests:         2 passed, 0 failed
AST violations:         0
compile / diff check:   passed / passed
runtime score:          not measured
```

The static result is a design/protocol checkpoint, not a latency result. The
local NVIDIA driver remains unavailable, so exact-image startup, the
20-request mechanism probe and same-image A/B/A are pending. No image,
Compose, champion or Attempt 17 changed. Evidence is in
`round_039_gap_aware_scheduler_prototype/outputs/static_validation.latest.*`.

## 2026-07-15 - R040 Parked-State Scheduler And First-Token Swap

R040 source-audited vLLM 0.22.1 and then exercised a real one-model hybrid
state parking path. R040a safely parked and resumed two post-first-token
requests without freeing KV blocks or copying Mamba state through CPU. It
completed `20/20` outputs but direct hold lost ERS proxy and raised TPOT p95
from `16.478` to `22.425ms`.

R040b shortened displacement to one staged request's first-token interval.
At a resource-matched local GPU reservation of `0.72`, the event chain emitted
`5/2/2/2` displace/bind/swap/resume actions, with three safe rollbacks and no
fallback. The candidate and two native controls all returned `20/20` requests
with exactly 200 tokens and zero preemption, error or OOM.

```text
candidate / native-midpoint ERS: 0.454380 / 0.466172 (-2.529%)
candidate TTFT/TPOT components:  0.046069 / 0.862692
native-midpoint components:      0.043405 / 0.888938
TTFT <=1500ms:                   5/20 in every run
candidate/native TPOT p95:       26.491 / 24.434ms
```

Paired TTFT improved `119.274ms` on average, but the two staged requests still
landed at `4323/4583ms`; no new request entered the TTFT score ceiling. R040 is
closed without full trace, package or Attempt 17. The retained result is state
feasibility only; both measured parking policies are score-negative.

## 2026-07-15 - R041 First-Wave Service Slope And FP8 Backend Audit

R041 validated seven native first-wave processes and three exact R031 phase
decompositions. The active seq16 TTFT staircase is queue serialization rather
than increasing request service: at GPU reservation `0.82`, request-index
correlation is at least `0.999852` and slope is `158.037ms/request`; the mean
queue span is `2510.046ms`, while service span is `-117.523ms`.

The official-score model requires at least `6.5%` whole-prefill reduction, or
`9.467%` reduction inside the measured `68.659%` GEMM category, to gain at
least `0.005` ERS and one supported request in every process.

Exact-image source/runtime audit proved serialized static FP8 can use
FlashInfer while current online dynamic-per-token FP8 cannot. Two matrix
screens then reconstructed all 570 material GEMMs in R012's cold prefill:

```text
static FlashInfer MM vs CUTLASS:       -8.762%
static FlashInfer quant+MM vs current: -10.270%
static CUTLASS quant+MM vs current:     +1.319%
Torch row-wise MM vs CUTLASS:           -3.954%
Torch row-wise quant+MM vs current:      -0.867%
```

All numerical gates passed, so the result is performance-negative rather than
incompatible. No checkpoint conversion, kernel overlay, full120, image,
Compose or Attempt 17 was created. I006 seq16/default2048 remains champion.

## 2026-07-15 - R042 Global Budget Cohort Audit

R042 combined the R031 forward and reverse brackets into three complete
processes per prefill budget. It reconstructed official request score, used
only arrival-time prompt/cache/inflight features, and treated request ID, wave
ID and absolute timestamps as analysis joins rather than policy inputs.

```text
2048/4096 ERS mean:            0.608403 / 0.597467
fixed 4096 delta:             -0.010936
wave 2 forward/reverse:       +0.013964 / +0.001507
wave 2 pairwise support:       6/9
wave 2 mean/robust q25:       +0.006770 / -0.008325
request oracle gain:          +0.002348
global-wave oracle gain:      +0.001128
global-wave robust gain:      +0.000000
```

The request oracle is diagnostic because vLLM has one token budget per
scheduler iteration. The global-wave oracle is the executable upper bound and
misses both the `0.005` gain gate and the `0.004843` process-noise floor. A
robust generic rule selects no 4096 wave in full data or held-out folds. R042
is closed without GPU, scheduler code, image, Compose or Attempt 17.

## 2026-07-15 - R043 Native Fused Reduction Envelope

R043 source-audited the exact vLLM image and found the callable
`rms_norm_dynamic_per_token_quant` CUDA op. Unlike R014, it benchmarked the
op directly instead of relying on a graph matcher that replaced zero nodes.
Seven rotated trials used preallocated buffers, 3 CPU, 8GB RAM and no network.

```text
M1/M20 reduction vs two raw ops:  +31.524% / +30.790%
M211 reduction:                    +6.789%
M1680 reduction:                 -175.056%
weighted prefill reduction:      -156.311%
whole-prefill projection:         -10.058%
max dequantized NRMSE:              1.622e-2
```

The small-shape raw-op comparison does not open decode integration. The real
R012 serving trace already uses one Inductor Triton RMSNorm/quant reduction at
`2.832us/call`; native fused `M=20` is `3.822us`, about 35% slower. Both
prefill and decode branches are closed without a server, full120, image,
Compose or Attempt 17.

## 2026-07-15 - R044 Remaining Hotspot Capability Audit

R044 ranked the remaining R012 cold-prefill categories and resolved the exact
vLLM target path from source rather than inferring it from RTX 5080 behavior.
Full attention consumes `17.469ms` (`11.092%`) and hybrid GDN/Mamba consumes
`15.663ms` (`9.945%`) of the profiled prefill.

On H200 SM90 with FP8 KV, automatic full-attention selection is
FlashAttention-3. Qwen GDN prefill automatically selects FlashInfer; CuteDSL
GDN is available only on Blackwell in this image. R008 also already rejected
forced Triton attention with an `8.76%` ERS loss.

The deterministic analyzer screened eight forced/custom paths for category
share, target executability, distinction from target auto, prior rejection,
possible gain evidence and a same-path benchmark. Zero passed all gates.
R044 therefore created no GPU run, server, full120, image, Compose or Attempt
17 and retained automatic backend selection.

## 2026-07-16 - Leaderboard Reset Rapid Requalification Kit

BTC's replacement statement was not yet available, so no model, trace, Docker
or GPU benchmark was run. A separate contract-driven intake kit was created
instead of continuing R045.

The initial pending audit compares the frozen legacy contract with an
intentionally unknown replacement contract:

```text
contract status:                  pending
validation errors:                0
unresolved required fields:      47
total/critical changes:       61 / 55
critical changes to unknown:     54
techniques blocked/prohibited: 13 / 2
benchmark authorization:         false
```

The four explicit blockers are incomplete official contract, missing new
trace, disabled/unknown scoring and unvalidated official statement file/hash.
Eighteen unit tests cover validation, diffing, TPOT denominator semantics,
piecewise quality penalty, aggregate-only trace handling, statement/trace hash
matching, matrix states and both blocked/authorized end-to-end paths.

## 2026-07-16 - Part 2 Public Conversation Trace Audit

The new public metadata file was parsed without prompt emission:

```text
records:                         420
conversations:                    70
turns per conversation:            6
warmup requests:                  90
graded requests:                 330
input tokens per turn p50:      3999
output cap per request:          200
think time:                     3000ms
conversation start span:      302866ms
```

Compared with the old exact tokenizer profile:

```text
graded request multiplier:       2.75x
graded output multiplier:        2.75x
incremental input fraction:      0.545x
append-only replay fraction:     1.977x
decode-heaviness increase:       5.04x
reuse-avoidable prompt replay:  72.42%
```

A closed-loop service-time model predicts p95/max active requests of `9/12`
at 4 seconds and `11/14` at 5 seconds. The result is an offline workload
qualification, not a score or official driver proof.

Artifacts:

- `src/llm_race_lab/requalification/conversation_trace.py`
- `scripts/analyze_part2_trace.py`
- `competition_reset/rapid_requalification_kit/outputs/part2_public_trace/`
- `docs/planning/part2_trace_requalification_plan.md`

## 2026-07-17 - Part 2 Round 002 Portal-Calibrated Decode First

The exact vLLM 0.22.1 LFM2.5 stack was tested under `3 CPU / 8 GiB RAM` with
the public closed-loop lifecycle, 90 primer and 330 graded requests. FP8 E4M3
attention KV plus BF16 Mamba state and async scheduling became the runtime
incumbent after repeated full traces and a paired 100-question quality proxy.

Rejected branches included generic FP8, E5M2 KV, FP16 Mamba state, seq10,
balanced/throughput modes, disabled prefix cache, xxhash, DBO, forced linear
backends, custom graph capture, compile size 4096 and local memory `.83/.86`.

The final prefill-budget bracket observed large screen drift, so b16384 was
promoted only after cold full scores `67.879/68.740` with `660/660` graded
successes. The ready compose adds only b16384, FP8 E4M3 KV and async scheduling
to the graded Attempt 01 contract.

## 2026-07-17 - Part 2 Subround 002c GPU N-Gram Speculation

The exact image accepted `ngram_gpu` with async scheduling and loaded no draft
model. The runtime did, however, reduce FlashInfer graph mode from
full-and-piecewise to piecewise. A bounded 96-request diagnostic generated
`2910` draft steps and `5820` draft tokens but accepted only `67` tokens
(`1.15%`); first-position acceptance was `40/2910` (`1.37%`).

Because every eligible step already produced both requested draft tokens,
lookup coverage was complete enough and broader n-grams could not solve the
failure. The registered early-stop skipped k3, n1-3, the 90-graded screen and
the full trace. No submission artifact changed.

## 2026-07-18 - Part 2 Subrounds 002d-002f Tied-Head And Admission

An opt-in import overlay preserved the tied BF16 input embedding and created a
separate FP8 output-head copy. Tensor, block-128 and row-wise scaling plus
exact shortlist repair were source-qualified without eager fallback. On the
fixed-4k proxy, tensor top-k32 repeated about `+1.45` points over the two-run
control mean, but exact token matching remained weak.

The admission follow-up rejected `--no-scheduler-reserve-full-isl`
(`71.622` versus `71.862`) and found concurrent partial prefill unsupported in
vLLM 0.22.1. The append-only integration then produced:

```text
matched native full:             56.729
tensor FP8 no-repair full:       58.857
score delta:                     +2.128
native -> candidate TPOT p50:    2.584 -> 2.417ms
native -> candidate TTFT p50:  110.524 -> 105.888ms
graded failures:                    0 -> 0
```

The deterministic quality smoke rejected promotion. Tensor top-k0 matched
`0/96` full sequences, while the best row-wise point (top-k32) matched only
`29/96`. Larger row-wise repair sizes were non-monotonic and worse. The branch
is retained as a measured decode ceiling with no Portal artifact.

## 2026-07-18 - Part 2 Portal Attempts 02 And 03

Attempt 02 scored `58.17` with TTFT `58/82ms` and TBT median `4ms`; Attempt 03
scored `47.91` with TTFT `98/201ms` and the same TBT median. This target result
resolves the hidden workload toward append-only/session reuse and rejects
no-prefix/Mamba-none. Both runs reported unexpected primer accounting and
nonzero failures, which remain outside local reproduction.

## 2026-07-18 - Part 2 Subround 002k Fused BF16 Greedy Head

The exact vLLM `0.22.1` image was instrumented with an opt-in Triton output-head
overlay. It computes BF16 tied-head dot products and hierarchical argmax without
materializing the `[batch, 65536]` logits tensor. Empty built-in `MinP`,
`MinTokens` and `LogitBias` state is handled by the fused path; all active or
unknown processor cases retain native logits.

Results:

- standalone M1/M2/M4/M8 speedups: `1.0098x/1.0239x/1.0280x/1.0411x`, all
  compared IDs matched;
- 20 live native-delegate audit batches, including batch M8: `100%` argmax
  agreement;
- full local replay: native ERS `56.7287`, fused ERS `58.3989`, delta
  `+1.6702`, zero failures;
- fused TTFT p50/p95 `106.503/336.066ms`, TPOT p50/p95 `2.477/3.649ms`.

The custom kernel is locally promoted for packaging/quality qualification, but
no Portal artifact changed in this experiment.

## 2026-07-18 - Part 2 R002k Packaging And Quality Qualification

The 002k overlay was baked into the exact `vllm/vllm-openai:v0.22.1` base.
The resulting local image ID is
`sha256:10f819c0e765bb579b89caccac4de5f17984c2fd5342335130832dc7ad93f6f5`;
layer history and runtime inspection show only `fused_greedy_head.py` and
`sitecustomize.py` were added.

Independent startup completed in about 40 seconds. The image captured
PIECEWISE 16 and FULL 8 CUDA graphs, exposed `1,621,489` KV tokens at local
memory reservation `.80`, and required no host overlay mount. The packaged
strict-greedy smoke completed `6/6` graded requests through the fused path.
A separate six-request logprob smoke completed through the native runner.

The private 100-case GPQA proxy completed `100/100` requests and retained
`21/100` accuracy. Compared with the incumbent process, four parsed answers
moved, one additional output was unparsed and zero correctness decisions
changed. The same-hidden-state 20-batch native-delegate audit remains the
stronger no-loss correctness evidence.

The first quality invocation failed because the restricted host namespace
could not reach a Docker host-network endpoint. Its all-connection-error
output is retained and excluded; the successful evaluator ran inside the
container without changing image layers.

The qualified image was then published to
`minhnguyenrobert/viettel-ai-race-vllm` and Docker Hub returned the exact local
digest. The immutable Part 2 Attempt 04 Compose keeps every graded Attempt 02
flag and changes only the image reference. Docker Compose parsing passes and
the 1,216-byte file is below the 1 MiB Portal limit.

## 2026-07-18 - Part 2 Subround 002l Selective Final RMSNorm

The exact source path showed that LFM2 final RMSNorm precedes logits-row
selection. An exact BF16 microbenchmark compared native fused add-RMSNorm plus
selection against residual addition, selection and sampled-row RMSNorm.
Values were bit-exact; the deferred operation gained a weighted `21.917%` for
large prefill but saved only `0.149ms` absolute. Decode shapes regressed by
about `40.556%`, so integration used a graph-safe 2048-row threshold.

Live serving confirmed `4000 rows -> deferred` and `1 row -> native` phase
transitions. Twenty native/fused argmax audit batches matched, a 96-request
screen passed, all full runs produced the same output SHA, and a six-request
logprob smoke logged native runner fallback.

The full B/A/B bracket scored `55.8737` candidate, `53.5857` R002k control and
`52.7145` candidate repeat. Candidate mean is `+0.7084` points and all six
turn means are positive, but repeat 2 loses, reaches peak active 11 and raises
TPOT p95 to `4.568ms`. The analyzer passes correctness and mean-score gates but
fails repeatability and TPOT guards. No image, Compose or Portal attempt was
created.

## 2026-07-18 - Part 2 Attempt 04 H200 Result

The immutable image digest
`sha256:10f819c0e765bb579b89caccac4de5f17984c2fd5342335130832dc7ad93f6f5`
was submitted at 14:51 and scored `60.75` ERS on the H200 evaluator. Portal
reported TTFT p50/p95 `44/80ms`, median TBT `4ms`, six failures, total count
420 and `warmup_count=0`.

Against Attempt 02's native image under the same retained serving flags, this
is `+2.58` ERS points, `-14/-2ms` TTFT p50/p95 and two fewer failures with
unchanged median TBT. This validates R002k's non-lossy fused greedy head on the
target architecture. The Compose predates R002l and that selective-RMSNorm
code is not in the submitted image.

The displayed accuracy drop zero is an online placeholder. Independent local
evidence remains `21/100` GPQA with zero correctness flips plus exact live
native-argmax audits; final BTC GPQA is post-round on selected submissions.

## 2026-07-18 - Part 2 Subround 002m Exact Decode-Head V2

R002m compared the full R002k production path against direct-owned output,
reusable-output lower bounds, a persistent transposed BF16 head and three
manual GEMV tiles. The final exact-image run used nine timing samples with 100
iterations each for M1/M2/M4/M8; all variants matched `240/240` native IDs.

Direct ownership was effectively neutral at `+0.011%` weighted GPU and
`+0.430%` wall time. The impossible-to-use-without-ownership-hooks no-clone
floor was only `+0.489%`. Transpose lost `1.299%` GPU time and added 256 MiB.
Manual GEMV lost about `154%` because each batch row streamed the full weight
again. Every candidate failed the micro gate, so live qualification and
packaging were skipped.

## 2026-07-18 - Part 2 Attempt 04 Exact-Digest Repeat

The byte-identical R002k Compose and image were resubmitted at 16:19. The
repeat scored ERS `58.94`, TTFT p50/p95 `55/79ms`, TBT median `4ms`, seven
failures, total 420, warmup count zero, displayed accuracy drop zero and both
penalty factors one. The first exact run scored `60.75`, `44/80ms`, `4ms` and
six failures.

The two-run mean is `59.845` and same-artifact ERS range is `1.81`; TTFT p50
moved 11ms while p95 moved only 1ms. This is direct process/evaluator variance
evidence. The repeat remains `+0.77` above the single native Attempt 02 score,
so R002k stays incumbent, but future effects inside the A/A range require
repeat evidence rather than a code-level explanation.

## 2026-07-18 - Part 2 Subround 002n Decode Hotspot and ShortConv FP8

Kineto profiling of the exact R002k runtime captured 64 clean decode-only
steps at active batches 1, 2, 4 and 8. Existing FP8 GEMMs consumed `58.269%`
of child-kernel time, the R002k output head `16.050%`, and ShortConv BF16
projection kernels `19.499%`. Source inspection showed that vLLM's LFM2
ShortConv constructors do not pass the active FP8 quantization config to their
`in_proj` and `out_proj` modules.

An opt-in overlay reused vLLM's existing static-FP8/CUTLASS method, with
independent `in`, `out` and `all` modes. At batch 8, child-kernel time was
`141.598ms` for control, `131.718ms` for in-only, `142.546ms` for out-only and
`130.119ms` for both. Out-only was closed immediately.

The all-projection branch scored `18/100` twice on the local GPQA proxy versus
the `21/100` reference and scored `58.2706` on the exact 420-request replay,
slightly below the R002k anchor `58.3989`; it was not packaged. In-only scored
`22/100` twice, with no reference-correct-to-candidate-wrong transition. Its
two exact replays scored `58.9590` and `59.4062`, with TPOT p50/p95
`2.421/3.756ms` and `2.405/4.006ms`. The immediate control bracket scored
`57.2978`, but the size of the control movement demonstrates substantial local
process variance. Each candidate had `329/330` requests under the 400ms TTFT
ceiling and matched only `268/330` first tokens against the paired control.

A `.95` local startup probe failed before healthcheck because only
`13.89/15.44 GiB` was free while vLLM requested `14.67 GiB`; `.95` remains a
clean-MIG-only validation step. R002n retains in-only as research-only and
leaves R002k as the submission artifact.

The attempted reversed-order in-only repeat was invalid: after six requests,
the piecewise CUDA-Graph path requested `512MiB` with only `409.69MiB` free on
the display-contaminated RTX 5080 and the engine exited. Its `0.0075` ERS is
excluded from all comparisons; no control was run after the failed candidate.

## 2026-07-18 - Part 2 Subround 002o Phase-Aware ShortConv

R002o created one-model dual projection paths for all ten ShortConv layers:
native BF16 when an iteration contains prefill and online FP8 only for clean
decode iterations. The shared-load design aliased checkpoint parameters before
vLLM quantization and produced independent BF16/FP8 storage afterward. Startup,
real phase transitions and PIECEWISE-16/FULL-8 graph capture all passed.

The clean batch-8 A/B/A profile measured non-output-head child-kernel time at
`120.464ms` control A1, `111.825ms` candidate and `119.293ms` control A2 over 64
steps. Against the `119.879ms` control mean, the candidate improved `6.72%`.
The selective `0,1` layer mask improved only `0.845%` and stopped before full
replay. The all-layer quality panel retained `21/100` twice, with zero
correctness changes relative to R002k.

Two all-layer full replays scored `53.2023` and `52.9304`; the immediate R002k
control scored `55.4200`. Candidate TTFT/TPOT p50/p95 were
`127.170/379.093ms` and `2.665/3.828ms` in run 1, then
`121.644/495.384ms` and `2.553/8.072ms` in run 2. All 990 graded observations
succeeded, but candidate mean was `2.3537` ERS points below control and its
TTFT/TPOT tails were worse. No image, Compose or Portal artifact was created.

## 2026-07-19 - Part 2 Subround 002p FP8 Activation Scales

Clean R002n traces showed 44 FP8 scaled-MM calls and 44 fused activation-quant
producers per decode step. Dynamic activation quantization accounted for
`4.6195%` mean child-kernel time. After correcting the LFM2 MLP width to
`8192`, two exact-shape microbenchmarks put oracle static activation scaling at
`+6.5263%/+6.5963%` quant-plus-MM gain.

The live implementation selected static CUTLASS kernels for exactly 44 model
linears and captured both required CUDA Graph modes. Shape thresholds through
`1024` and a scheduler-state phase hook remained on the dynamic graph after
capture. A force-static diagnostic removed all 44 dynamic producers and
improved batch-8 child-kernel time from a `2.1928129ms` control mean to
`2.1658492ms`, or `1.2296%`, with `0.4006%` control drift.

Quality rejected the branch. The original scale map scored `20/100` and
`18/100`; prompt-plus-decode recalibration scored `19/100` and `18/100`, versus
R002k at `21/100`. Every request completed, so this is numerical regression.
The quality stop rule skipped full replay, image, Compose and Portal work.

## 2026-07-19 - Part 2 Subround 002q Graph-Native FP8 Fusion

The exact vLLM 0.22.1 image was audited with GPU-backed help and source
introspection. Nested `PassConfig` JSON was accepted. Startup probes under the
3-CPU/8GB envelope all returned `8/8` successful requests and retained the
R002k head. `fuse_norm_quant` replaced five LFM2.5 graph patterns while
retaining `FULL_AND_PIECEWISE`; `fuse_act_quant` matched zero patterns;
norm+act added no activation matches; and `fuse_attn_quant` matched zero
attention/MLA patterns while changing graph mode to FULL.

The fixed batch-8, 64-step Kineto A/B/A bracket measured `2.2050282`,
`2.1898797` and `2.2010363 ms/step`, respectively. Control drift was `0.1812%`
and complete gain was `0.5970%`. Dynamic quant producers fell `44 -> 22`, but
the displaced work appeared in the `other` family; FP8 GEMM remained 44 calls
per step. The pre-registered 1% gate failed, so no quality/replay/image/
Compose work was run and R002k remains incumbent.

## 2026-07-19 - Part 2 Subround 002r FP8 Linear Backend

The exact vLLM 0.22.1 registry was audited before GPU testing. Auto selected
CUTLASS. Explicit CUTLASS and Torch retained `FULL_AND_PIECEWISE`, the R002k
head and `8/8` successful requests. Torch resolved to
`ChannelWiseTorchFP8ScaledMMLinearKernel` because the online activation scale
is dynamic per-token. FlashInfer failed that scale-layout constraint and also
lacks SM90 support; Marlin failed its native-FP8 guard. Triton and DeepGEMM
contain no matching per-tensor-online class.

The clean batch-8, 64-step CUTLASS/Torch/CUTLASS bracket measured
`2.2056096`, `2.2047690` and `2.2021816 ms/step`. Control drift was `0.1555%`
and Torch gain was `-0.0396%`. Torch raw GEMMs were slightly faster locally,
but separate FP32 scaling/dequantization and conversion consumed the saving.
The 1% gate failed, so no quality, replay, image, Compose or Portal artifact
was created.

## 2026-07-19 - Part 2 Subround 002s Online Block FP8

Exact-image source inspection confirmed that `fp8_per_block` performs online
E4M3 128x128 weight quantization and dynamic 1x128 activation quantization.
All 44 LFM2.5 linear calls are shape-compatible. Local SM120 auto selected
`CutlassFp8BlockScaledMMKernel`; exact H200 SM90 source exposes FlashInfer for
decode-like M<32 and DeepGEMM for larger M. Auto, explicit CUTLASS, explicit
Triton and the per-tensor control all retained the R002k head, both CUDA Graph
modes and `8/8` startup responses.

The clean per-tensor/block/per-tensor batch-8 bracket measured `2.2207708`,
`2.3176776` and `2.2243989ms/step`. Control drift was `0.1632%`; local block
CUTLASS lost `4.2785%`. Repeated quality scored `18/100` twice versus R002k
`21/100`, with four reference-correct losses and one new correct answer in
each comparison.

A diagnostic 420-observation replay completed `330/330` graded requests but
scored `49.3582` versus R002k `58.3989`. Candidate TTFT p50/p95 was
`133.855/423.597ms`, TPOT p50/p95 `2.749/5.126ms`, and peak active 11. Paired
analysis found only 24 TTFT wins and 41 TPOT wins across 330 requests. R002s
is closed without image, Compose or Portal work.

## 2026-07-19 - Part 2 Subround 002t Small-M FP8 Kernel

Exact vLLM 0.22.1 source established the fixed contract: online per-tensor
E4M3 weights, dynamic per-token activation scales and BF16 output. The R002r
control/candidate/control traces put CUTLASS and Torch raw GEMMs at
`1.1297934` and `1.1030849ms/step`, respectively. The apparent saving is only
`0.0267084ms`, with a zero-overhead ceiling of `1.2119%`; clearing the 1% gate
requires removing `83.07%` of Torch's measured extra overhead.

The direct BF16 `_scaled_mm` probe showed that scalar weight scales support
only M=1. A `1.242 MiB` persistent expansion to row-wise scales enabled all
registered M values and remained bit-exact, but exact-shape timing lost
`35.51-132.98%` end to end versus CUTLASS.

A custom Triton epilogue then multiplied the FP32 raw GEMM by the existing row
and scalar scales and wrote BF16. Blocks 256/512/1024 were tested over all 16
M/shape cells with nine randomized 100-iteration samples. Every output and
row argmax matched. The best M=8 epilogue still cost `0.320015ms` per 44
calls, `68.53x` its live +1% budget, and quantize+GEMM+epilogue measured
`1.167806ms` versus CUTLASS `0.856557ms` (`-36.34%`). R002t is closed before
server integration or packaging.

## 2026-07-19 - Part 2 Subround 002u BF16 ShortConv Kernel

Two clean batch-8 traces attribute ten BF16 in-projections and ten BF16
out-projections at `0.302585` and `0.132019ms/step`. Their total
`0.434604ms` is `19.7198%` of the `2.203896ms` complete step, so a `5.0710%`
family gain would clear the 1% integration gate.

The first exact-shape matrix loop selected a Triton 64x64/four-warp/three-stage
kernel and reported `11.31-20.50%` family gains. That result was not promoted:
each cell repeated one 8-24 MiB matrix and made weight reads artificially
L2-hot.

The confirmation allocated ten distinct in/out pairs totaling exactly 320
MiB and alternated all 20 calls per timed iteration. At M8, cyclic native was
`0.456264ms`, only `4.98%` above the live anchor. Best Triton was
`0.445971ms`, a same-process `2.256%` gain but still slower than the
`0.434604ms` production anchor, projecting `-0.516%` complete-step gain.
Worst Triton NRMSE was `0.00024876`, minimum exact fraction `0.997233`, and all
outputs were finite. R002u closes on performance before integration.

## 2026-07-19 - Part 2 Subround 002v ShortConv in_proj FP8

R002v copied the retained R002n implementation into an isolated overlay and
verified ten `Fp8PerTensorOnlineLinearMethod` input projections, ten native
BF16 output projections and `FULL_AND_PIECEWISE` CUDA Graph capture. Local
`.90` could not pass the desktop free-memory guard; `.84` was selected to
match the absolute headroom of a clean 18 GiB H200 at `.95`.

The host-overlay replay scored `57.9057/55.6036` for two candidates versus
control `54.4081`. An initial package failed after six graded requests on a
fragmented 512 MiB allocation. With expandable allocator segments, immutable
packaged candidates completed `420/420` and scored `52.6084/53.2367` versus
packaged control `49.5424`. Mean gain is `+3.3802`; candidate TPOT is lower on
`284/330` and `290/330` paired requests.

Packaged GPQA remains `21/100`, equal to R002k, with one correct loss and one
new correct answer. The public image digest is
`sha256:eb9f961172f0668ec62af5ae66e1689670c9ec8d3249d41688f9cb06d399f701`.
Attempt 05 first Portal run then scored `60.19`, TTFT p50/p95 `50/90ms`, TBT
median `4ms`, `6/420` failures and accuracy drop `0`. This is `+0.345` over
the R002k exact-artifact mean `59.845`, inside its `1.81`-point A/A range. An
unchanged repeat is prepared; no technical delta is claimed yet.

## 2026-07-19 - R002v Exact-Digest Repeat Closes Target Probe

The byte-identical Attempt 05 repeat returned `59.34` ERS, TTFT p50/p95
`61/97ms`, TBT median `4ms`, `5/420` failures, `warmup_count=0` and
`accuracy_drop=0`. The first panel was `60.19`, `50/90ms`, `4ms` and `6/420`
failures. The pair mean/range is therefore `59.765/0.85`.

R002k's exact A/A pair is `60.75/58.94`, mean/range `59.845/1.81`. R002v is
`-0.080` points below that mean; its TTFT p50/p95 means are `55.5/93.5ms`
versus R002k `49.5/79.5ms`, while TBT is unchanged. The lower failure count in
the repeat is not a reliable score improvement. The local packaged ShortConv
`in_proj` FP8 gain does not transfer as an attributable H200 gain. Keep R002k
as incumbent, archive the digest, and do not submit the same artifact again.

## 2026-07-19 - Part 2 Subround 002w ShortConv Decode Epilogue

R002w implemented a pure-decode Triton fusion for all ten LFM2.5 ShortConv
layers. It keeps the native BF16 `in_proj/out_proj`, but fuses `B*x`, the
sequential width-three convolution, recurrent state shift and `C*conv` output
staging. Prefill, mixed and unsupported batches remain native.

The final distinct-weight microbenchmark reproduces runtime state stride
`(8192,1,2048)` and index `(M,1)`. Output and state are bit-exact over 16
histories at M1/M2/M4/M8. M8 native/fused is `0.285780/0.091605ms`, projecting
`8.811%` complete-step gain. FULL_AND_PIECEWISE graph capture records ten
eligible layers and ten fused dispatches. A repeated batch-8, 512-token live
panel improves median throughput by `2.86%` and mean by `1.86%`.

The public-trace candidate/control/candidate replay completes all `990` graded
observations but scores `50.5693/50.6229/50.1385`. Mean candidate delta is
`-0.2689`; candidate A trades four more TTFT-under-400 requests for worse TPOT,
while candidate B loses the TTFT tail. Two unchanged controls match only
`5/330` complete output hashes, so output-sequence mismatch is baseline FP8/
batch nondeterminism rather than kernel-specific evidence. Performance rejects
the branch before GPQA, publication or Portal work.

## 2026-07-19 - Part 2 Subround 002x Production Graph-Step Attribution

R002x instrumented vLLM's complete `execute_model + sample_tokens` boundary
with preallocated CUDA Events and captured Kineto child kernels without
per-step synchronization. Three clean processes formed an R002k/R002w/R002k
bracket. Each process completed 64 pure-decode steps at M1/M2/M4/M8 with zero
request failures and the expected graph mode.

Exact scheduler telemetry from a separate `420/420` replay counted `47,657`
pure-decode steps out of `48,263` active steps (`98.744%`) and `83,109` pure
decode rows out of `83,610` decode rows (`99.401%`). R002w's native mixed
fallback is therefore not the reason its full replay failed.

The scheduler-weighted complete GPU gain is `3.501%` with `1.045%` control
drift; child-kernel time gains `2.500%` and kernel count falls `13.163%`.
Per-batch complete GPU gains are `+3.496/+5.100/+0.851/-1.687%` at M1/M2/M4/
M8. M4 equals its drift and M8 reverses. Direct M8 ShortConv saving is only
`0.039981ms/step`, versus `0.194175ms` projected by eager timing, a `4.86x`
overestimate. The registered M8-positive gate fails and no larger fusion or
serving artifact opens.

## 2026-07-19 - Part 2 Subround 002y Cached-Turn Prefill Attribution

R002y instrumented the complete first-prefill through first-sample GPU
boundary for the six append-only turn shapes. Timing A and B used four repeats
per turn; a middle process captured Kineto kernels. Every process first ran all
90 public primer requests unarmed. Weighted control drift is `1.418%`.

Initial cache/computed tokens progress from `0` to `20,784`, cache fraction
from `0%` to `83.16%`, and newly scheduled tokens remain `4,000-4,213`.
Complete GPU time nevertheless grows `60.801 -> 114.685ms`. Attention grows
`4.825 -> 54.742ms`, explaining `92.64%` of that increase. Weighted child time
is `39.713ms` FP8 GEMM, `30.029ms` attention and `13.930ms` BF16 GEMM.

Turns 1-5 split at an aligned Mamba boundary and pay a second `2-14` token
model step costing `2.662-2.923ms`; its equal-turn ceiling is `2.296ms`.
Source audit shows the dominant local kernel is FlashInfer only because SM120
rejects FlashAttention with FP8 KV. H200 SM90 instead has packaged FA3 support
and auto-selects it. R002y closes without an implementation or package and
opens only target-specific FA3 qualification.

## 2026-07-19 - Part 2 R003 Hybrid Full-Attention Window

R003 injected a trace-independent `per_layer_sliding_window=8192` into the six
LFM2.5 full-attention modules, preserving R002k's ten ShortConv modules,
precision, cache mode, scheduler, CUDA Graph policy and fused greedy head.
Startup assertions and standalone overlay tests passed.

The isolated six-turn panel preserved cache progression and flattened the
attention tail: complete GPU savings at turns 3/4/5 were approximately
`9.0/20.1/33.3ms`. The full append-only control/candidate/control/candidate
replay completed `420/420` every time. Scores were
`55.9491/60.8540/57.1409/61.4875`; two-run means are therefore
`56.5450` control and `61.1708` candidate. Candidate TTFT p50/p95 improved by
`18.91/45.67ms`, and TPOT p95 improved by `0.498ms`.

The independent 4k boundary check on 30 conversations did not reproduce the
gain: control `68.3485`, candidate `67.9362`. This is consistent with the
window being inactive for a 4k prompt and makes the request-history semantics
the central unresolved variable. Candidate startup also has a larger KV pool,
so the end-to-end gain includes capacity/admission relief in addition to
bounded attention work. The branch is a strong local research candidate, but
it remains outside the Portal queue pending long-context quality and rules
review.

## 2026-07-19 - Part 2 R003a Attention Window Sweet Spot

The profile sweep measured windows `4096/6144/8192/12288/16384` against the
same unbounded R002k process. Turn-5 complete GPU time was
`71.141/79.483/81.448/87.440/101.264ms` versus `114.735ms` control. Every
valid run preserved the exact computed/scheduled-token progression. Window
`4096` also raised startup KV capacity to about `2.623M` tokens versus
`1.649M` unbounded.

A calibrated 14-case paired retrieval panel covered actual prompt lengths
from about 3.1k to 22.3k tokens. Control passed four cases. Windows
`4096/6144/8192/12288/16384` passed `2/3/4/4/5`; relative to control they lost
`2/1/0/0/0` cases. This is a regression probe only because control accuracy
is low.

Full replay scores were `64.5121/64.5886` at `4096`, `61.7945` once at
`6144`, `61.1708` mean at `8192`, `57.0213` once at `12288`, and `56.5450`
mean unbounded. The `4096` mean is `+8.0054` points, with TTFT p50/p95
`84.49/234.29ms` and TPOT p50/p95 `2.318/3.342ms`. The branch is the stable
local score maximum but fails the paired retrieval gate; `8192` remains the
balanced research branch. No submission artifact follows.

## 2026-07-19 - Part 2 R003b Selective Global Placement

The exact image was run with one or two full-attention layers left unbounded
and a `4096` window on the other full-attention layers. Profile turn-5 GPU
savings were `36.840/34.306/37.932/32.443ms` for global layers
`14/2/8/2,14`, respectively. All six-turn profiles and all 14 retrieval
requests completed.

The paired retrieval result was negative for every placement: candidates
answered `2/14` key cases versus `4/14` for the unbounded control, losing the
same `5084`- and `7109`-token cases. No full replay or package was opened.

Conclusion: a later global layer cannot reconstruct information discarded by
earlier windowed layers. A future candidate must preserve selected old
positions in every affected layer. R003b is a completed research rejection.

## 2026-07-20 - Part 2 R003d Tiered Attention Horizons

Experiment:

- Image: `localhost/part2-lfm25-r002k:fused-bf16-greedy`.
- Hardware envelope: RTX 5080, 3 CPU cores, 8 GiB RAM, GPU utilization `0.80`.
- Maps: `safe8192`, `mid6144` and `front3`; full-attention layers only.

Results:

- All three profiles passed the latency gate.
- `safe8192` lost zero control-pass retrieval cases in two repeats and gained
  one case at 7,109 actual prompt tokens. `mid6144` lost one; `front3` lost two.
- Two safe8192 full replays scored `61.8737` and `61.9014`, mean `61.8876`,
  versus two unbounded controls at mean `56.5450`.
- Mean TTFT p50/p95 changed `114.262/315.560ms -> 90.966/245.763ms`.
- Mean TPOT p95 changed `4.159ms -> 3.612ms`; all four runs had zero graded
  failures.
- GPQA proxy: `23/100`, 11 parse failures; controls observed `21..22/100`.

Conclusion:

- `safe8192` is a stable local research winner for the inferred long-history
  workload, not a portable H200 kernel result.
- The gain combines bounded attention with a larger KV pool at the same memory
  fraction. The short GPQA proxy cannot test changed long-context semantics.
- No image, Compose or Portal artifact follows. Next work requires a
  prefix-plus-recent retention design or target-qualified attention path.
