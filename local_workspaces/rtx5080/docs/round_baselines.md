# Round Baseline Tracker

This file tracks the best known local baseline after each research round.

It is a quick comparison table. Put detailed explanations in:

```text
local_workspaces/rtx5080/research_rounds/warmup_rounds/round_answers.md
```

## Active Part 2 LFM2.5 Baselines

| Branch | Runs | Score mean | TTFT p50/p95 ms | TPOT p50/p95 ms | Decision |
|---|---:|---:|---:|---:|---|
| R002k unbounded local control | 2 | `56.5450` | `114.26 / 315.56` | `2.571 / 4.159` | Exact local submission control. |
| R002aa sparse no-pad paired delta | 2 clean pairs | `+0.8713` | paired delta `-6.46` | paired delta `+0.0395` | Locally qualified H200 probe; does not replace R002k. |
| R003a window 4096 | 2 | `64.5504` | `84.49 / 234.29` | `2.318 / 3.342` | Score-max research only; two paired retrieval losses. |
| R003 window 8192 | 2 | `61.1708` | `95.35 / 269.89` | `2.408 / 3.661` | Balanced research branch; no paired retrieval losses. |
| R002k Portal Attempt 04 | 2 | `59.845` | target pair `49.5 / 79.5` | TBT median `4` | Submission incumbent. |

All local Part 2 rows use 3 CPU cores, 8 GiB RAM and RTX 5080 GPU utilization
`0.80`. Do not compare their absolute values directly with H200 Portal runs.
No bounded-window branch replaces R002k until long-context semantics, rules and
target behavior are qualified.

## Current Best Local Baseline

```text
Round: Score 100 R021a/R021b - packaged I007 seq16
Run: local_workspaces/rtx5080/research_rounds/score_100_program/round_021_graph_workspace_memory/subround_021a_packaged_fp8_head/outputs/packaged_aba/runs/20260713-004106_vllm_direct_i007_packaged_fp8head_topk4
Config: /model, I006-derived vLLM 0.22.1 package, context 32768, GPU memory 0.82 local, max_num_seqs 16, online FP8 weights, FP8 E4M3 KV, BF16 Mamba SSM cache, output-only E4M3 tied head with exact BF16 top-k4 repair, native async scheduler/auto hybrid backends, legal synthetic shape warmup before public readiness, CPU 3, RAM 8GB
ERS: 0.6269777079497513
Success: 120/120
TTFT p50/p95: 962.4775 / 2866.02575 ms
TPOT p95/p99: 14.491644 / 15.80472535 ms
Reason: R021a beat the same-image control midpoint ERS `0.623107`, improved TTFT and TPOT, passed activation/fallback/resource gates, and repeated GPQA at 35/100 versus control 34/100 with penalty proxy 1. R021b kept seq16. Attempt 12 validated I007 quality on H200 at `51.37`, but I006 seq16 scored `51.59`. R021c then rejected head-off seq17 at ERS `0.604517` versus seq16 midpoint `0.607739`; I006 seq16 remains the target score champion.
```

## Current Portal Score Champion

```text
Artifact: portal_submissions/attempt_11_vllm_i006_readiness_warmup.yml
Compose SHA: e28f0aac456ee2206e3d9b51912829584681d0a0786e71330e519bdcfb1bbc4f
Config: I006 seq16, native output head
Score: 51.59
Full metric panel: pending
Best complete panel: Attempt 12 I007 at 51.37, TTFT 4148/7130ms, TBT 15ms, accuracy drop 0
Decision: use I006 seq16 as the target baseline and tied-head off for the next admission experiment.
```

## Quality-Safe Control

```text
Round: 005 / 007 auto-control
Run: local_workspaces/rtx5080/runs/20260707-204049 or same-session auto control local_workspaces/rtx5080/runs/20260708-105935
Config: auto KV cache, explicit prefix caching, explicit chunked prefill
Reason: Round 007a auto repeat exact-matched 8/8 sampled outputs, while FP8 E4M3 exact-matched 0/8 versus auto. Round 007b auto GPQA proxy accuracy was 0.38.
Use: rollback/control when FP8 combines with other lossy optimizations or if hidden Accuracy Gate fails.
```

## Baseline Table

| Round | Run | Config Summary | ERS | Success | TTFT p50 | TPOT p50 | Decision |
|---|---|---|---:|---:|---:|---:|---|
| 001 | `runs/20260707-152939` | `/model`, vLLM, `max_model_len=32768`, `gpu_memory_utilization=0.65`, `max_num_seqs=16`, CPU 3, RAM 8g | `0.055487` | `120/120` | `39293.15 ms` | `52.53 ms` | Current contract baseline. |
| 002 | `runs/20260707-155728` | `/model`, vLLM, `max_model_len=32768`, `gpu_memory_utilization=0.65`, `max_num_seqs=4`, CPU 3, RAM 8g | `0.499987` | `120/120` | `49884.60 ms` | `13.91 ms` | Current local best; TPOT improves enough to dominate ERS despite worse TTFT. |
| 003 | `runs/20260707-192232` | `/model`, vLLM, `max_model_len=32768`, `gpu_memory_utilization=0.72`, `max_num_seqs=4`, CPU 3, RAM 8g | `0.500000` | `120/120` | `38655.35 ms` | `10.03 ms` | Current local best by ERS tie-break; next bottleneck is TTFT. |
| 004 | `runs/20260707-195253` | `/model`, vLLM, `max_model_len=32768`, `gpu_memory_utilization=0.72`, `max_num_seqs=4`, `max_num_batched_tokens=16384`, CPU 3, RAM 8g | `0.500000` | `120/120` | `40307.05 ms` | `8.24 ms` | Not promoted; TPOT improves but TTFT and VRAM worsen versus Round 003. |
| 005 | `runs/20260707-204049` | `/model`, vLLM, `max_model_len=32768`, `gpu_memory_utilization=0.72`, `max_num_seqs=4`, `enable_prefix_caching=true`, `enable_chunked_prefill=true`, CPU 3, RAM 8g | `0.500000` | `120/120` | `37711.23 ms` | `9.84 ms` | Current local best by tail latency and stability; partial prefill not ready. |
| 005a | `runs/20260707-215032` | Round 005 baseline plus `torch_matmul` synthetic ramp `10s` before vLLM startup | `0.500000` | `120/120` | `36720.11 ms` | `9.58 ms` | Promising TTFT watchlist; not promoted until repeat and healthcheck-safe integration. |
| 005a-ext | `runs/20260707-222022` | Round 005 baseline plus heavy `torch_matmul` synthetic ramp `10s`, `6144->2048`, `sync_every=2` | `0.499995` | `120/120` | `37771.78 ms` | `9.90 ms` | Best extended-sweep stability candidate; keep as ramp watchlist, not global baseline. |
| 005b | `runs/20260707-223853` | Round 005 baseline plus `torch_matmul` ramp `2s`, no explicit batched-token cap | `0.500000` | `120/120` | `37716.12 ms` | `9.92 ms` | Best 005b overall; startup power max 325.79 W, no extra 16384 VRAM cost. |
| 005b-tpot | `runs/20260707-231704` | Heavy ramp `2s`, `6144->2048`, plus `max_num_batched_tokens=16384` | `0.500000` | `120/120` | `39524.67 ms` | `8.08 ms` | 005b-only TPOT-safe hypothesis; later 005c repeat was negative, so not a stable fallback. |
| 005c | `runs/20260707-234135` | Round 005 baseline plus light `torch_matmul` ramp `2s`, `3072->1024`, no explicit batched-token cap | `0.500000` | `120/120` | `37838.21 ms` | `10.04 ms` | Best repeated ramp candidate; not promoted globally yet. |
| 005c-tpot | `runs/20260707-234953` | Heavy ramp `2s`, `6144->2048`, plus `max_num_batched_tokens=16384` | `0.482821` | `120/120` | `39628.56 ms` | `8.73 ms` | Negative repeat; TPOT p99 worsened to 23.53 ms, so this is not a stable TPOT guard. |
| 006 | `runs/20260708-010904` | Round 005 baseline knobs with explicit `enable_prefix_caching=true`, explicit `enable_chunked_prefill=true` | `0.500000` | `120/120` | `38407.93 ms` | `10.00 ms` | Prefix cache audit; not promoted over Round 005 but makes explicit prefix caching mandatory. |
| 006-negative-default | `runs/20260708-010415` | Same knobs but no prefix-cache flag, using vLLM default behavior | `0.473541` | `120/120` | `63676.76 ms` | `16.97 ms` | Negative control; default/no-flag behavior is unsafe for this trace. |
| 006-negative-off | `runs/20260708-011310` | Same knobs with explicit `no-enable-prefix-caching` | `0.484700` | `120/120` | `60645.57 ms` | `16.42 ms` | Negative control; disabling prefix cache hurts TTFT, TPOT and throughput. |
| 007-control | `runs/20260708-105935` | Round 006 baseline with `kv_cache_dtype=auto` | `0.499899` | `120/120` | `37489.74 ms` | `9.86 ms` | Same-session control; TPOT p99 crossed 20 ms. |
| 007 | `runs/20260708-110846` | Round 006 baseline with `kv_cache_dtype=fp8_e4m3`; `calculate_kv_scales` requested but disabled by vLLM | `0.500000` | `120/120` | `17052.64 ms` | `8.24 ms` | Current provisional local candidate after Round 007b GPQA proxy. |
| 007-e5m2 | `runs/20260708-111206` | Round 006 baseline with `kv_cache_dtype=fp8_e5m2`; `calculate_kv_scales` requested but disabled by vLLM | `0.500000` | `120/120` | `17218.05 ms` | `8.29 ms` | Completed but worse tail/throughput than E4M3; keep only as alternate FP8 variant. |
| 007a-quality | `round_007a_fp8_kv_output_drift/outputs/drift_summary_20260708-112328.md` | 8 sampled public-trace prompts, auto repeat vs FP8 E4M3 output drift | `n/a` | `8/8 captures` | `n/a` | `n/a` | High quality-risk signal: auto repeat exact `8/8`, FP8 vs auto exact `0/8`, similarity avg `0.386`. |
| 007b-quality | `round_007b_gpqa_diamond_accuracy/outputs/gpqa_comparison_20260708-115628.md` | 100-question local GPQA Diamond proxy, auto KV vs FP8 E4M3 KV | `n/a` | `100/100 parsed each` | `n/a` | `n/a` | FP8 local accuracy drop was `0.04`: auto `0.38`, FP8 `0.34`; both no-penalty under official-style baseline `0.4` proxy. |
| 007c-score | `runs/20260708-121041` | Full trace replay with FP8 E4M3 KV plus Round 007b quality proxy | `0.500000` | `120/120` | `16440.64 ms` | `8.08 ms` | Score proxy `50.0`, below target `95.0`; not submission-ready because TTFT is above ceiling. |
| 007d-ttft | `runs/20260708-123026` | FP8 E4M3 KV, scheduler control `MAX_NUM_SEQS=4` | `0.500000` | `120/120` | `18810.98 ms` | `8.37 ms` | Best in 007d. `MAX_NUM_SEQS=8/12` reduced TTFT but dropped ERS to `0.442994` / `0.449426`, so raw concurrency is not promoted. |
| 012-cold83 | `runs/20260708-233429` | FP8 E4M3 KV, `gpu_memory_utilization=0.83`, `MAX_NUM_SEQS=4`, no extra warmup | `0.500000` | `120/120` | `17673.41 ms` | `8.46 ms` | Near-max usable VRAM completed, but 0.83 is flaky under active desktop VRAM overhead. |
| 012-torch10 | `runs/20260708-234003` | Same as 012-cold83 plus 10s synthetic Torch CUDA warmup before vLLM startup | `0.500000` | `120/120` | `17421.69 ms` | `8.71 ms` | Best 0.83 TTFT p50 in Round 012, but worse tail/throughput than cold 0.83; not promoted. |
| 012-endpoint82 | `runs/20260708-234652` | FP8 E4M3 KV, `gpu_memory_utilization=0.82`, 3 short synthetic streaming endpoint warmup requests before trace | `0.500614` | `120/120` | `17849.33 ms` | `8.68 ms` | New best local ERS by a small margin. Keep as endpoint-warmup watchlist, not submission-ready. |
| 012-failed095 | `round_012_warmup_jit/outputs/cold_fullvram_fp8_20260708-232644.log` | `gpu_memory_utilization=0.95` full-card attempt | `n/a` | `not_ready` | `n/a` | `n/a` | Failed because only `12.91/15.44 GiB` was free but vLLM needed `14.67 GiB`; desktop/app VRAM overhead prevents 0.95 locally. |
| 012a-vram90 | `runs/20260709-001453` | FP8 E4M3 KV, `gpu_memory_utilization=0.90`, no endpoint warmup, `MAX_NUM_SEQS=4` | `0.500000` | `120/120` | `16504.65 ms` | `8.10 ms` | Capacity watchlist: KV tokens increased to `1062745` (`+20.24%` vs 0.82), but score/latency did not beat same-session 0.82. |
| ready002-baseline-raw | `runs/20260709-123358` | Raw/no-technique vLLM control: auto KV, prefix off, chunked off, `gpu_memory_utilization=0.65`, `MAX_NUM_SEQS=16` | `0.171784` | `120/120` | `52560.59 ms` | `31.44 ms` | Raw control only. Confirms the integrated stack is necessary. |
| ready002-integrated-all | `runs/20260709-123806` | FP8 E4M3 KV, prefix on, chunked on, `gpu_memory_utilization=0.88`, `MAX_NUM_SEQS=8`, mixed endpoint warmup, Tree-Gated proxy | `0.502460` | `120/120` | `7890.54 ms` | `10.51 ms` | Current best full-stack local baseline. GPQA proxy 0.35, no penalty locally; still TTFT-capped. |
| ready003-sglang-raw | `runs/20260709-131417` | Raw SGLang control: `mem_fraction_static=0.72`, `max_running_requests=4`, `kv_cache_dtype=auto`, default FlashInfer attention, RadixAttention on | `0.499091` | `120/120` | `40827.87 ms` | `10.30 ms` | Usable SGLang control; TPOT strong, TTFT too high. GPQA proxy 0.37. |
| ready003-sglang-lpm-cudagraph | `runs/20260709-131931` | SGLang, `mem_fraction_static=0.86`, `max_running_requests=8`, `schedule_policy=lpm`, default FlashInfer, RadixAttention on, decode CUDA graph full, prefill CUDA graph tc_piecewise, mixed endpoint warmup | `0.507862` | `120/120` | `4655.24 ms` | `9.64 ms` | Superseded by Ready 003a run10. GPQA proxy 0.37, no penalty; still useful as lower-TPOT comparison. |
| ready003-sglang-fp8kv | `runs/20260709-132112` | SGLang, `mem_fraction_static=0.88`, `max_running_requests=8`, `schedule_policy=lpm`, `attention_backend=flashinfer`, `kv_cache_dtype=fp8_e4m3`, mixed endpoint warmup | `0.507400` | `120/120` | `3846.18 ms` | `9.34 ms` | Best TTFT/TPOT profile, but slightly lower ERS than auto-KV SGLang and GPQA proxy drops to 0.34. Keep as watchlist. |
| ready003a-auto-mem082-run8 | `runs/20260709-134023` | SGLang auto KV, `mem_fraction_static=0.82`, `max_running_requests=8`, LPM, per-phase CUDA graph, mixed endpoint warmup | `0.506578` | `120/120` | `4633.19 ms` | `9.63 ms` | Safer-VRAM fallback: VRAM max `14952 MiB`, lower score than current best. |
| ready003a-auto-mem084-run8 | `runs/20260709-134223` | SGLang auto KV, `mem_fraction_static=0.84`, `max_running_requests=8`, LPM, per-phase CUDA graph, mixed endpoint warmup | `0.504164` | `120/120` | `4101.13 ms` | `9.54 ms` | Lower TTFT p50 than mem082 but lower ERS; not promoted. |
| ready003a-auto-mem086-run6 | `runs/20260709-134420` | SGLang auto KV, `mem_fraction_static=0.86`, `max_running_requests=6`, LPM, per-phase CUDA graph, mixed endpoint warmup | `0.513863` | `120/120` | `4602.24 ms` | `8.22 ms` | Strong TPOT profile but lower ERS than run10; keep as lower-TPOT fallback. |
| ready003a-auto-mem086-run10 | `runs/20260709-134625` | SGLang auto KV, `mem_fraction_static=0.86`, `max_running_requests=10`, LPM, per-phase CUDA graph, mixed endpoint warmup | `0.517733` | `120/120` | `4318.31 ms` | `13.25 ms` | New current local ERS baseline. GPQA proxy 0.37, no penalty; TPOT p99 `20.79 ms` is a watchpoint. |
| ready003a-fp8-mem086-run8 | `runs/20260709-134807` | SGLang FP8 E4M3 KV, `mem_fraction_static=0.86`, `max_running_requests=8`, FlashInfer, LPM, mixed endpoint warmup | `0.513875` | `120/120` | `4522.63 ms` | `10.32 ms` | Improved versus earlier FP8 candidate but does not beat auto run10; GPQA would be required before promotion. |
| ready003a-fp8-mem088-run6 | `runs/20260709-134954` | SGLang FP8 E4M3 KV, `mem_fraction_static=0.88`, `max_running_requests=6`, FlashInfer, LPM, mixed endpoint warmup | `0.450861` | `105/120` | `3766.34 ms` | `8.06 ms` | Rejected: 15 errors/zero-token outputs, despite good latency on successful requests. |
| ready003e-prefill-broad-best | `runs/20260709-155557` | Current SGLang baseline plus `--max-prefill-tokens=8192` | `0.515851` | `120/120` | `4576.19 ms` | `13.88 ms` | Broad sweep best and beat same-session baseline `0.512680`, but required refine because result was small and could be session noise. |
| ready003e-prefill-refine-baseline | `runs/20260709-160310` | Current SGLang baseline, same-session control for refine sweep | `0.517014` | `120/120` | `4533.44 ms` | `13.99 ms` | Refine sweep winner; confirms no 003e prefill knob replaced current Ready 003a baseline. |
| ready003e-prefill-refine-watchlist | `runs/20260709-160713` | Current SGLang baseline plus `--max-prefill-tokens=6144` | `0.515615` | `120/120` | `4621.26 ms` | `13.90 ms` | Watchlist only: improved TTFT p95 and TPOT p95 versus refine baseline but lost ERS. |
| ready003f-decode-broad-best | `runs/20260709-162725` | Current SGLang baseline plus `--num-reserved-decode-tokens=200` | `0.517276` | `120/120` | `4407.22 ms` | `13.65 ms` | Broad sweep best, beating same-session baseline `0.516420`; refined because trace uses exactly 200 max output tokens. |
| ready003f-decode-refine-best | `runs/20260709-164504` | Current SGLang baseline plus `--num-reserved-decode-tokens=200 --num-continuous-decode-steps=4` | `0.517185` | `120/120` | `4499.35 ms` | `13.84 ms` | Best 003f refine candidate, beating same-session baseline `0.514151` but still below global best `0.517733`. Strong watchlist. |
| ready003f-decode-refine-reserve224 | `runs/20260709-164304` | Current SGLang baseline plus `--num-reserved-decode-tokens=224` | `0.516981` | `120/120` | `4407.98 ms` | `13.80 ms` | Slightly safer reserve headroom over public `max_tokens=200`; strong watchlist for integrated repeat. |
| ready003f-decode-final-reserve224-steps4 | `runs/20260709-173405` | Current SGLang baseline plus `--num-reserved-decode-tokens=224 --num-continuous-decode-steps=4` | `0.523150` | `120/120` | `3754.60 ms` | `12.45 ms` | First current-best run. GPQA proxy accuracy `0.37`, penalty `1.0`; still TTFT-capped. |
| ready005-tpotfloor-control | `runs/20260709-185111` | Lowest-TPOT SGLang policy replay: `MAX_RUNNING_REQUESTS=4`, `num-continuous-decode-steps=8`, `num-reserved-decode-tokens=224`, dynamic chunking, LPM | `0.508153` | `120/120` | `20444.54 ms` | `7.00 ms` | TPOT-floor control only; preserves very low TPOT but TTFT is too costly. |
| ready005-tpotfloor-ttft-overlay | `runs/20260709-192607` | TPOT-floor policy plus `--chunked-prefill-size=1024 --prefill-max-requests=4 --enable-mixed-chunk` | `0.513911` | `120/120` | `12178.54 ms` | `9.11 ms` | Best TPOT-floor-compatible TTFT overlay. Useful module candidate but below global Ready 003f baseline. |
| ready005-chunk1024-req1-tail | `runs/20260709-195859` | TPOT-floor policy plus `--chunked-prefill-size=1024 --prefill-max-requests=1 --enable-mixed-chunk` | `0.511664` | `120/120` | `12381.90 ms` | `8.98 ms` | Stable TPOT-tail variant; improves TPOT p95 versus req4 (`11.087` vs `11.549 ms`) but loses ERS, so not promoted. |
| ready005-chunk256-negative | `runs/20260709-194516` | TPOT-floor policy plus `--chunked-prefill-size=256 --prefill-max-requests=4 --enable-mixed-chunk` | `0.504052` | `120/120` | `27566.90 ms` | `11.09 ms` | Rejected: chunk 256 is too small; overhead hurts TTFT, TPOT and throughput. |
| ready005-chunk256-req1-negative | `runs/20260709-195432` | TPOT-floor policy plus `--chunked-prefill-size=256 --prefill-max-requests=1 --enable-mixed-chunk` | `0.501992` | `120/120` | `27984.35 ms` | `10.72 ms` | Rejected: one-request prefill improves TPOT p95 but worsens TTFT/ERS; chunk 256 remains bad. |
| ready006-hybrid-control-replay | `runs/20260709-214526` | Current SGLang control shape after model-contract audit; hybrid artifact confirmed with 18 linear-attention and 6 full-attention layers | `0.513978` | `120/120` | `4206.16 ms` | `13.88 ms` | Valid round-006 control. First control attempt crashed with 0/120 and is ignored as transient server failure. |
| ready006-mamba-flashinfer | `runs/20260709-213936` | Current SGLang control plus `--mamba-backend=flashinfer` | `0.514223` | `120/120` | `4377.76 ms` | `13.63 ms` | Usable hybrid backend watchlist; slightly better TPOT p95 and throughput than replay control, but not an ERS baseline. |
| ready006-ssm-bfloat16 | `runs/20260709-214138` | Current SGLang control plus `--mamba-ssm-dtype=bfloat16` | `0.519408` | `120/120` | `4172.92 ms` | `13.62 ms` | Best round-006 hybrid signal, but still below global Ready 003f and quality-risk until GPQA/output-drift passes. |
| ready006a-ssm-bfloat16-quality | `round_006_hybrid_architecture_audit/subround_006a_mamba_state_quality_gate/outputs/gpqa_comparison_20260709-215832.md` | Local 100-question GPQA proxy: default-float32 SSM control vs `--mamba-ssm-dtype=bfloat16` | `n/a` | `100/100 parsed each` | `n/a` | `n/a` | BF16 SSM matched control exactly: accuracy `0.37` vs `0.37`, changed predictions `0/100`, changed correctness `0/100`, official-style proxy penalty `1.0`. Quality risk reduced; still not promoted because ERS is below Ready 003f. |
| ready006b-ssm-bfloat16-replay | `runs/20260709-220837` vs control `runs/20260709-220632` | Same-session trace replay: current Ready 003f control vs `--mamba-ssm-dtype=bfloat16` | `0.513613` vs `0.512065` | `120/120` each | `4112.22 ms` vs `4518.25 ms` | `13.81 ms` vs `13.66 ms` | BF16 SSM won same-session ERS and TTFT p50, with slightly worse TPOT p95 and TTFT tail. Keep as watchlist; not global baseline because Ready 003f still has ERS `0.523150`. |
| ready006c-control-repeat-best | `runs/20260709-221331` | Same Ready 003f control config replayed before hybrid combo test | `0.523891` | `120/120` | `3941.15 ms` | `12.92 ms` | New best observed run for the existing Ready 003f config; config unchanged. |
| ready006c-ssm-bfloat16 | `runs/20260709-221531` | Ready 003f control plus `--mamba-ssm-dtype=bfloat16` in combo replay session | `0.522122` | `120/120` | `3776.41 ms` | `12.36 ms` | Better TPOT/throughput than control, but lower ERS due request-score distribution and lower TTFT<=1500 count. |
| ready006c-mamba-flashinfer-ssm-bfloat16 | `runs/20260709-221727` | Ready 003f control plus `--mamba-backend=flashinfer --mamba-ssm-dtype=bfloat16` | `0.518213` | `120/120` | `3762.96 ms` | `12.38 ms` | Best TPOT p95 `16.824 ms`, wall tok/s `507.813`, VRAM max `15422 MiB`; not promoted because ERS is lower than control `0.523891`. |
| ready006d-artifacts | `round_006_hybrid_architecture_audit/subround_006d_hybrid_ttft_combo_hunt/outputs/ttft_combo_artifacts_20260709.md` | User-requested artifact bundle: SGLang command, input/output token histogram, concurrency/rate, per-request TTFT, versions | `n/a` | `n/a` | `n/a` | `n/a` | Artifact pack for selected stable TTFT combo `hybrid_mamba_flashinfer_ssm_bfloat16`. |
| ready006d-run14-steps6-negative | `runs/20260709-231354` | No-warmup `MAX_RUNNING_REQUESTS=14`, dynamic chunking, continuous decode steps 6 | `0.459727` | `120/120` | `11794.26 ms` | `17.48 ms` | Rejected: did not reduce TTFT; TPOT p95 `30.631 ms`. |
| ready006d-run14-hybrid-negative | `runs/20260709-231546` | No-warmup run14+steps6 plus `--mamba-ssm-dtype=bfloat16 --mamba-backend=flashinfer` | `0.461250` | `120/120` | `4351.33 ms` | `16.39 ms` | Better than run14 alone but still worse than selected stable hybrid combo; TPOT p95 `34.649 ms`. |
| ready006d-run16-hybrid-negative | `runs/20260709-232305` | No-warmup run16+steps6 plus BF16 SSM and Mamba FlashInfer | `0.408281` | `120/120` | `4381.74 ms` | `17.75 ms` | Rejected: TPOT p95 `40.979 ms`, near official ceiling, and TTFT p50 did not beat selected combo. |
| ready006d-chunk1536-negative | `runs/20260709-231742` | No-warmup run14 + chunk1536 + prefill-max-requests4 + mixed chunk | `n/a` | `failed/hung` | `n/a` | `n/a` | Rejected: server returned 500/timeouts and metrics stayed empty until manual stop. |
| ready006e-lpm-res64-steps1 | `runs/20260709-234511` | SGLang LPM, `max_running_requests=10`, auto KV, `--num-reserved-decode-tokens=64 --num-continuous-decode-steps=1 --mamba-backend=flashinfer --mamba-ssm-dtype=bfloat16` | `0.524547` | `120/120` | `3896.43 ms` | `12.29 ms` | New provisional local ERS best. TTFT<=1.5s `23/120`, TPOT p95 `17.036 ms`; repeat before submission because gain over previous best is small. |
| ready006e-lpm-res128-steps2 | `runs/20260709-234709` | Same hybrid shape with `--num-reserved-decode-tokens=128 --num-continuous-decode-steps=2` | `0.523722` | `120/120` | `3923.63 ms` | `12.44 ms` | Strong watchlist but lower ERS than res64_steps1. TTFT<=1.5s `25/120`. |
| ready006e-fcfs-negative | `runs/20260709-235432` | `max_running_requests=12`, `schedule_policy=fcfs`, `reserved=128`, `steps=1`, Mamba FlashInfer + BF16 SSM | `0.448436` | `120/120` | `8390.09 ms` | `18.93 ms` | Rejected: extreme tail improved to `>=30s = 0`, but TTFT<=1.5s fell to `3/120` and TPOT p95 worsened to `39.355 ms`. |
| ready006e-res64-steps2-negative | `runs/20260709-235042` | LPM, `reserved=64`, `steps=2`, Mamba FlashInfer + BF16 SSM | `n/a` | `report_failed` | `19744.25 ms salvaged` | `13.13 ms salvaged` | Rejected: metrics JSONL/report failed; salvaged unique records show poor TTFT p95 `50371.084 ms` and TPOT p95 `28.830 ms`. |
| ready006f-schedule-lpm-repeat | `runs/20260710-001909-744278453-1027569` | Current hybrid LPM shape repeat | `0.521266` | `120/120` | `4142.42 ms` | `14.61 ms` | Confirms LPM still beats LOF/DFS/random in this matrix, but session score below current best. |
| ready006f-schedule-lof-negative | `runs/20260710-002107-783303383-1030524` | Current hybrid shape with `schedule_policy=lof` | `0.467596` | `120/120` | `10824.33 ms` | `15.79 ms` | Rejected: TTFT<=1.5s only `3/120`, TPOT p95 `31.271 ms`. |
| ready006f-chunk4096-mixed | `runs/20260710-003503-480268505-1113333` | LPM plus `--chunked-prefill-size=4096 --enable-mixed-chunk` | `0.519818` | `120/120` | `3772.96 ms` | `14.11 ms` | Best mixed-chunk candidate but below current best; TPOT p95 `20.334 ms`. |
| ready006f-prefill-4096-req2 | `runs/20260710-004237-697317545-1124835` | LPM plus `--max-prefill-tokens=4096 --prefill-max-requests=2` | `0.524073` | `120/120` | `3865.70 ms` | `12.80 ms` | Best prefill-limit candidate, close but not better than current best. |
| ready006f-conservativeness01 | `runs/20260710-005328-245551845-1140587` | LPM plus `--schedule-conservativeness=0.1` | `0.524510` | `120/120` | `3785.04 ms` | `12.87 ms` | Closest challenger to current best; not promoted because ERS is still lower by `0.000037`. |
| ready006f-disable-prefill-graph | `runs/20260710-010315-672403691-1157353` | LPM plus `--disable-prefill-cuda-graph` | `0.520807` | `120/120` | `3708.28 ms` | `13.25 ms` | Improves TTFT p50 and TPOT p95 but loses ERS distribution; not promoted. |
| ready007-proxy-a | `runs/20260710-011648-354154746-1172415` | One SGLang backend plus `two_policy_proxy.py`, `prefill_limit=8`, `total_limit=12`, backend `max_running_requests=12` | `0.488320` | `120/120` | `8215.97 ms` | `13.87 ms` | Rejected: HTTP proxy queueing loses fast TTFT bucket and TPOT p95 rises to `22.355 ms`. |
| ready007-proxy-b | `runs/20260710-011843-880410752-1175410` | One SGLang backend plus `two_policy_proxy.py`, `prefill_limit=10`, `total_limit=12`, backend `max_running_requests=12` | `0.491543` | `120/120` | `8427.25 ms` | `14.25 ms` | Rejected: best TTFT<=1.5s count in proxy round was only `6/120`, still far below current best. |
| ready007-proxy-c | `runs/20260710-012038-904173099-1178337` | One SGLang backend plus `two_policy_proxy.py`, `prefill_limit=10`, `total_limit=14`, backend `max_running_requests=14` | `0.493432` | `120/120` | `6618.47 ms` | `15.32 ms` | Best proxy round candidate, but below current best `0.524547`; TPOT p95 `21.721 ms`. |
| ready007-proxy-d | `runs/20260710-012230-178180437-1181214` | One SGLang backend plus `two_policy_proxy.py`, `prefill_limit=12`, `total_limit=16`, backend `max_running_requests=16` | `0.472062` | `120/120` | `5213.21 ms` | `16.09 ms` | Rejected: higher total limit improves p50 but pushes TPOT p95 to `26.566 ms` and hurts ERS. |
| ready006g-vllm-control | `runs/20260710-112329-709123855-61104` | vLLM 0.22.1 native hybrid control, auto KV, prefix/chunked, seq4 | `0.502427` | `120/120` | `17104.95 ms` | `8.93 ms` | Fresh exact-image hybrid control. |
| ready006g-vllm-text-only | `runs/20260710-112657-450574687-65716` | Control plus `--language-model-only` | `0.503230` | `120/120` | `16918.59 ms` | `8.84 ms` | Best vLLM candidate; operational fallback only. TPOT p95 `10.001 ms`, TTFT<=1.5s `4/120`. |
| ready006g-vllm-bf16-align | `runs/20260710-113017-176728767-70089` | Text-only plus BF16 Mamba state, requested FlashInfer GDN/Mamba, align cache | `0.502822` | `120/120` | `16991.33 ms` | `8.93 ms` | GDN prefill fell back to Triton/FLA; TPOT improved but ERS did not. |
| ready006g-vllm-interactive8192 | `runs/20260710-113956-010932590-82324` | Text-only, interactivity mode, max batched tokens 8192, seq4 | `0.501255` | `120/120` | `19103.02 ms` | `8.63 ms` | Rejected: larger prefill iteration worsened TTFT and throughput. |
| ready006g-vllm-run6-relaxed | `runs/20260710-114322-638917549-86794` | Text-only, interactivity, seq6, batched tokens 8192, no full-ISL reservation | `0.501262` | `120/120` | `13519.59 ms` | `10.42 ms` | Tail/throughput improved but TTFT<=1.5s dropped to `2/120`; reject for ERS. |
| ready006g-vllm-partial-prefill-negative | `subround_006g_vllm_hybrid_bakeoff/outputs/vllm_e_hybrid_scheduler_interactive_20260710-112036.log` | Concurrent partial prefill probe | `n/a` | `server_not_ready` | `n/a` | `n/a` | vLLM V1 reports concurrent partial prefill unsupported. |
| ready006-linear-flashinfer-negative | `round_006_hybrid_architecture_audit/outputs/hybrid_linear_flashinfer_bfloat16_hybrid_audit_20260709-214415.log` | `--linear-attn-backend=flashinfer` with/without `--mamba-ssm-dtype=bfloat16` | `n/a` | `backend_not_ready` | `n/a` | `n/a` | Rejected under current prefill CUDA graph policy: SGLang requires BF16 state, then FlashInfer GDN prefill expects float32 state and fails graph capture. |
| ready008a-native-control | `runs/20260710-125934-078367875-175689` | Pinned SGLang 0.5.14 digest; exact Ready 006e native config without telemetry | `0.512927` | `120/120` | `4182.81 ms` | `14.00 ms` | Same-session instrumentation control. Below historical best, so current baseline remains Ready 006e. |
| ready008a-instrumented | `runs/20260710-130137-147122260-179160` | Exact native control plus read-only scheduler/Policy/MambaRadixCache telemetry overlay | `0.511176` | `120/120` | `4223.35 ms` | `13.68 ms` | Telemetry accepted: ERS delta `-0.001751`, TPOT p95 `17.918 ms`, complete 120-request lifecycle and zero malformed telemetry lines. Measurement layer only, not a new serving baseline. |
| ready008b-shadow-v1 | `runs/20260710-135443-331069470-241059` | Native LPM plus read-only static prefix-rescue shadow v1, strict rolling decode-result p95 guard `22.1 ms` | `0.519800` | `120/120` | `4148.14 ms` | `13.22 ms` | Runtime overhead passed (`-0.000592` ERS), support/precision `14/0.9286`, but early support was zero. Guard was too jitter-sensitive; recalibrated, not promoted. |
| ready008b-shadow-v2-verbose-negative | `runs/20260710-140509-894031243-254599` | Shadow v2 with p97.5-derived guard `23.4 ms`, before state-change-only event emission | `0.510481` | `120/120` | `4916.43 ms` | `14.28 ms` | Rejected for observer overhead: ERS delta `-0.010026`, wall throughput delta `-45.952 tok/s`, and `5941` repeated EXPIRED events. |
| ready008b-shadow-v2-optimized | `runs/20260710-141228-940615768-265178` | Same frozen v2 policy with state-change-only labels and skipped cost prediction for terminal/ineligible requests | `0.512351` | `120/120` | `4395.90 ms` | `14.34 ms` | Observer accepted for research: ERS delta `+0.002358`, TPOT p95 `19.417 ms`, 210 labels and mean evaluator cost `108 us`. Static rule rejected because support collapsed to `1` with early/late `1/0`. Current serving baseline unchanged. |
| ready008c-adaptive-control | `runs/20260710-152043-342611197-340267` | Native LPM telemetry-only control using isolated no-outbound server/client network namespaces | `0.512949` | `120/120` | `4404.49 ms` | `13.98 ms` | Same-session control for adaptive threshold shadow; current serving baseline remains Ready 006e. |
| ready008c-adaptive-shadow | `runs/20260710-152248-479638027-340267` | Native LPM plus read-only queue-quantile bootstrap, rolling supervised threshold search, hysteresis and adaptive decode guard | `0.512680` | `120/120` | `4397.98 ms` | `14.10 ms` | Adaptive shadow accepted: ERS delta `-0.000269`, TPOT p95 `18.686 ms`, support/precision/lift `27/0.6667/3.3333x`, early/late `15/12`, no forbidden features. Measurement policy only, not a serving baseline. |
| ready008d-adaptive-shadow-control | `runs/20260710-162631-988525749-418467` | Ready 008d active-capable evaluator with environment kill switch off; exact native LPM order | `0.514307` | `120/120` | `4308.09 ms` | `14.00 ms` | Valid same-session causal control. Zero active rescue applied; current serving baseline remains Ready 006e. |
| ready008d-active-one-rescue-negative | `runs/20260710-162836-677120032-418467` | Native LPM followed by at most one adaptive global-front promotion, positions `1..8`, cooldown/decode guard/kill switch | `0.512578` | `120/120` | `4447.02 ms` | `15.04 ms` | Rejected: six safe unique interventions gained two fast TTFT requests but lost `0.001728` ERS. TTFT score component `-0.002143` outweighed TPOT component `+0.000415`; do not retry global-front promotion. |
| ready006j-native-tree-v2-shadow | `round_006_hybrid_architecture_audit/subround_006j_native_tree_gated_v2/outputs/tree_gated_v2.latest.md` | C++ boosted-stump TTFT ranker trained from native scheduler snapshots; queue age excluded from learned features; RAM/VRAM and rejuvenation are replay labels only | `n/a` | shadow dataset | `n/a` | `n/a` | Research gate passed: actionable AUC `0.900176`, native whole-queue top-K p99 `23.867us`. No serving baseline promotion because real migration and causal ERS were not measured. |
| ready006k-exact-cache-cohort-shadow-negative | control `runs/20260711-000210-105135682-945375`; shadow `runs/20260711-000407-378098813-945375` | Same safe local SGLang config at `mem_fraction_static=0.82`; shadow scores Tree v2 and direct heuristic only inside identical non-root cache-node cohorts and never mutates native LPM | `0.524547` control / `0.518136` shadow | `120/120` each | `3949.55 / 3805.95 ms` | `12.69 / 12.39 ms` | Rejected for active use. Both policies had `0/263` would-reorder support; evaluator p99 `333.063us`, ERS delta `-0.006411`. Baseline unchanged. |
| ready006l-parent-cache-nonposition-replay-negative | `round_006_hybrid_architecture_audit/subround_006l_parent_cache_nonposition_replay/outputs/parent_cache_replay.md` | Offline reconstruction of parent/fixed-hop/depth-normalized cache ancestry plus ten transparent policies that exclude native position | `n/a` | offline `120/120` outcomes | `n/a` | `n/a` | Rejected without a GPU run. No broader structural cohort existed; exact-node `service_only` changed 126/216 validation selections but had 0 wins/6 losses and lost all native positive capture. |

## Score 100 Integration Baselines

This compact table supersedes the older SGLang/ready-round current-best label
without deleting those historical controls. Detailed A/B/A evidence remains in
`local_workspaces/rtx5080/research_rounds/score_100_program/scoreboard.md`.

| Round | Candidate | ERS | TTFT p50/p95 | TPOT p95/p99 | Decision |
|---|---|---:|---:|---:|---|
| R000 | exact context/token audit | `n/a` | `n/a` | `n/a` | context 32768 fits; keep |
| R001 | context32768 seq6 FP8 KV | directional | `11033/17631ms` | `12.004/12.192ms` | enabling baseline |
| R002 | seq20 BF16 weights + FP8 KV | `0.401866` | `3828/5811ms` | `27.641/30.504ms` | reject; decode overloaded |
| R003 | seq20 online FP8 weights/KV | `0.573075` | `1497/4014ms` | `19.950/21.778ms` | promote integration |
| R004 | R003 + exact graph20 | `0.567559` | `1579/4123ms` | `20.689/22.578ms` | keep causal graph gain |
| R005 | custom prefix hold/release | `0.562907` | `1626/4088ms` | `20.929/22.417ms` | reject; zero compute saved |
| R006b | I003 + BF16 Mamba SSM cache | `0.586594` | `1289/3587ms` | `18.597/20.326ms` | promote I004 |
| R007 | lifecycle dual token budget | `0.580253` | `1354/3641ms` | `19.252/21.083ms` | reject; native 2048 kept |
| R008 | forced TRITON_ATTN | `0.526927` | `2153/4737ms` | `23.707/25.441ms` | reject; auto backend kept |
| R009 | I004 + synthetic shape warmup | `0.617657` | `1020/2454ms` | `17.364/19.688ms` | promote runtime-only I005 |
| R010 | I005 + pre-readiness package | `0.619885` | `1008/2414ms` | `17.117/19.269ms` | promote packaged I006 |
| R011a | I004 seq18 + graph18 | `0.593090` | `1195/3461ms` | `17.208/18.457ms` | conditional decode guard |
| R011b | seq17 + graph17 | `0.579116` | `1364/3710ms` | `17.733/18.739ms` | reject; concurrency descent closed |
| R012 | hybrid hotspot profile | n/a | cold prefill `157.491ms` GPU | pure decode `3.9795ms/step` | measurement; tied head opens R013 |
| R013 | FP8 tied head top-k4 | `0.532424` | `2010/4362ms` | `23.158ms` p95 | historical provisional result; promoted by R021a |
| R014 | native norm/quant fusion | n/a | startup only | zero replacements | reject |
| R015a | GDN launch geometry | n/a | microbench | `1.0015x` batch20 | reject |
| R015b/c | FP8/INT8 GDN state | n/a | microbench | `0.9554x/0.9570x` batch20 | reject speed and drift |
| R016a | FLA chunk32/64/128 | n/a | 6,931-token layer | chunk32 `0.8984x` | keep chunk64 |
| R016b | GDN backend capability | n/a | 6,931-token layer | local Triton `0.7919ms` | keep auto; alternates unavailable |
| R017 | mixed state layers | n/a | n/a | dependency negative | skip integration |
| R018 | HybridRadixState upper bound | n/a | ideal block16 largest-wave bound `180.0ms` | n/a | reject; compute reduction only `5.89%`, dense snapshots OOM |
| R021a | packaged FP8 tied head top-k4 | `0.626978` | `962/2866ms` | `14.492/15.805ms` | promote local I007; GPQA penalty proxy 1 |
| R021b | seq15/16/17 balance | selected seq16 repeat `0.620628` | selected `999/2955ms` | selected `14.896/16.217ms` | keep seq16; seq17 delta below drift |
| R021c | I006 head-off seq17 | `0.604517` vs midpoint `0.607739` | `1108/3204ms` | `16.319/18.651ms` | reject; fast bucket unchanged, ERS `-0.530%` |
| R022a | I006 FP8 linear backend screen | n/a | startup only | n/a | auto and explicit CUTLASS are identical; all other candidates fail capability |
| R034 | GPTQ W4A16/G128 Marlin vs I006 FP8 | `0.542724` vs midpoint `0.609622` | `1978/4308ms` | `22.209/23.213ms` | reject; quality penalty 1 but ERS `-10.974%`, W/T/L `0/37/83` |
| R035 | R034 timed-JIT accounting audit | n/a | n/a | pre/timed JIT `5/0` in all three cells | close without GPU; exact snapshot correction, R034 unchanged |
| R036 | local/H200 transfer residual audit | nine local runs recomputed exactly | component bounds from 11 Portal panels | two decode-dominant H200 comparisons; four median traps | close without GPU; local total ERS is not a target selector, R037 offline only |
| R037 | exact I006 vLLM 0.22.1 capability audit | n/a | n/a | DS layout ceiling `1.296%` decode / `1.196%` prefill; MRV2 no-cache `5.299x` compute | close without GPU; `14` capabilities, zero eligible candidates, I006 unchanged |
| R038 | causal first-token staging opportunity | six immutable R031 process panels | offline mean/worst ERS opportunity `+0.016036/+0.013850` | `6/6` panels, `0/36` negative waves, modeled TPOT p95/p99 max `18.183/19.032ms` | open gap-aware design only; requires 75% queue removal and 10% interference, no benchmark/config promotion |
| R039 | gap-aware policy + exact-image shadow AsyncScheduler | n/a | native/shadow/native 20-request probe; no official score | `20/20` each valid run; shadow vs native-B TTFT p50 `+2.125ms`, TPOT p95 `+0.057ms` | close no-op shadow; no stage action or causal gain; retain I006 |
| R040 | parked hybrid state + first-token swap | `0.454380` vs native midpoint `0.466172` on 20-request proxy | candidate `2686/5933ms`; native midpoint `2743/5900ms` | candidate TPOT `26.491/27.351ms`; native p95 midpoint `24.434ms` | state continuity passes, policy loses `2.529%` and adds no TTFT-supported request; reject without full120 |
| R041 | first-wave slope + static/row-wise FP8 backends | n/a | queue slope `158.037ms/request` at gmu0.82; whole-prefill gate `6.5%` | FlashInfer/Torch MM deltas `-8.762%/-3.954%`; static CUTLASS e2e `+1.319%` | numerical capability passes but every alternate backend misses the `9.467%` GEMM gate; keep native CUTLASS, no full120 |
| R042 | conditional global 2048/4096 budget audit | 2048/4096 process means `0.608403/0.597467` | wave 2 mean `+0.006770`, robust q25 `-0.008325`; only `6/9` process pairs positive | executable wave oracle `+0.001128`, robust oracle `0`; noise floor `0.004843` | close without GPU; fixed 2048 retained, no Tree-Gated/global budget overlay |
| R043 | native fused RMSNorm + dynamic FP8 | n/a | weighted prefill raw-op path `-156.311%`; projected whole-prefill `-10.058%` | M20 native fused `3.822us` vs actual compiled trace `2.832us`; NRMSE `1.622e-2` | close without engine/full120; current Triton reductions retained |
| R044 | remaining attention/GDN hotspot capability audit | n/a | full-attention/GDN shares `11.092%/9.945%` | H200 auto resolves FlashAttention-3/FlashInfer; `0/8` candidates pass | keep automatic backends; no GPU/full120/Attempt 17 |
| H200 I006-16 | I006 seq16 native head | Portal `51.59` | `4783/6800ms` | TBT median `16ms` | score and complete-panel champion; accuracy drop `0` |
| H200 A12 | I007 seq16 tied-head top-k4 | Portal `51.37` | `4148/7130ms` | TBT median `15ms` | complete panel and quality-safe; score `0.22` below I006 |

## Update Rule

After every completed round:

1. Add a row to the baseline table.
2. Update `Current Best Local Baseline` only if the round improves the decision baseline.
3. Link the run folder.
4. Keep local RTX 5080 numbers separate from official H200 baseline.
