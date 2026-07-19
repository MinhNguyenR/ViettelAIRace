# Ready Round Answers

## Ready Round 001 - Technique Synthesis

```text
Status: completed
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_001_technique_synthesis
Output: README.md
What happened: consolidated useful and rejected techniques from Warmup Round 001-012, including subrounds.
Decision: use this synthesis as the input inventory for integrated ready candidates.
```

## Ready Round 002 - Integrated All Techniques

```text
Status: completed
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_002_integrated_all_techniques
Performance summary: outputs/sweep_summary.latest.md
GPQA summary: outputs/gpqa_comparison.latest.md

Baseline candidate: baseline_raw_vllm
Baseline run: local_workspaces/rtx5080/runs/20260709-123358
Baseline ERS: 0.17178449688299366
Baseline TTFT p50/p95: 52560.5905 / 103815.88875 ms
Baseline TPOT p50/p95: 31.4386825 / 36.89197975 ms
Baseline GPQA proxy accuracy: 0.38

Integrated candidate: integrated_all_seq8_treegated
Integrated run: local_workspaces/rtx5080/runs/20260709-123806
Integrated ERS: 0.5024600596516284
Integrated score proxy after GPQA penalty: 50.24600596516284
Integrated TTFT p50/p95: 7890.5385 / 13361.92895 ms
Integrated TPOT p50/p95: 10.5100425 / 13.61329425 ms
Integrated wall tokens/s: 581.0440063045078
Integrated GPQA proxy accuracy: 0.35
Integrated official-style GPQA delta: 0.05
Integrated GPQA penalty proxy: 1.0

What happened: applying FP8 E4M3 KV, explicit prefix cache, explicit chunked prefill, high VRAM 0.88, MAX_NUM_SEQS=8, mixed endpoint warmup and Tree-Gated proxy scheduling dramatically improved performance over the raw no-technique baseline.
Technical conclusion: full-stack integration works and remains locally quality-safe by GPQA proxy, but it is still TTFT-capped. The current score region is about 50 because TTFT p50 remains much higher than the official 1500 ms ceiling.
Decision: keep integrated_all_seq8_treegated as the current full-stack ready baseline. Do not submit as final yet.
Next: run an ablation for Tree-Gated proxy overhead/value, then test integrated_all_seq4_score_safe.
```

## Ready Round 003 - SGLang, Native Tree-Gated, Triton Probe

```text
Status: subround 001, 002, 003a, 003b, 003c, 003d, 003e and 003f completed
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate

What happened: created one complete ready-round umbrella folder, probed the SGLang Docker image, fixed the CUDA graph flag policy, benchmarked SGLang raw baseline, benchmarked SGLang LPM/CUDA graph and SGLang FlashInfer/FP8 KV, ran GPQA proxy for both SGLang best candidates, completed a scheduler/memory ablation around `mem_fraction_static`, `max_running_requests`, and KV dtype, then ran TTFT bottleneck diagnostics before native Tree-Gated implementation.

Key structure:
- subround_001_sglang_raw_baseline
- subround_002_sglang_flashinfer_fp8kv
- subround_003_tree_gated_native
- subround_003b_ttft_bottleneck_diagnostics
- subround_003c_controlled_admission_scheduler
- subround_003d_native_prefix_priority_scheduler
- subround_003e_prefill_chunk_scheduler_sweep
- subround_003f_decode_protection_scheduler_sweep
- subround_004_triton_server_bridge
- subround_005_integrated_submission_candidate

Runnable configs:
- configs/sglang_raw_baseline.env
- configs/sglang_radix_lpm_cuda_graph.env
- configs/sglang_flashinfer_fp8kv.env
- configs/sglang_treegated_native_probe.env

Performance:
- sglang_raw_baseline: run local_workspaces/rtx5080/runs/20260709-131417, ERS 0.4990905198296, TTFT p50/p95 40827.872 / 92345.5271 ms, TPOT p50/p95 10.301955 / 15.85216925 ms, GPQA 0.37.
- sglang_radix_lpm_cuda_graph: run local_workspaces/rtx5080/runs/20260709-131931, ERS 0.5078617541914796, score proxy 50.78617541914796, TTFT p50/p95 4655.2425 / 33923.2610 ms, TPOT p50/p95 9.6436925 / 13.23437775 ms, GPQA 0.37, official-style penalty 1.0.
- sglang_flashinfer_fp8kv: run local_workspaces/rtx5080/runs/20260709-132112, ERS 0.5074004130700978, score proxy 50.740041307009776, TTFT p50/p95 3846.1750 / 31894.95925 ms, TPOT p50/p95 9.33773 / 12.7615515 ms, GPQA 0.34, official-style penalty 1.0.
- sglang_ablate_auto_mem086_run10: run local_workspaces/rtx5080/runs/20260709-134625, ERS 0.5177333696770363, score proxy 51.77333696770363, TTFT p50/p95 4318.3145 / 36102.52215 ms, TPOT p50/p95 13.2460375 / 18.549398 ms, GPQA 0.37, official-style penalty 1.0.
- sglang_ablate_fp8_mem088_run6: run local_workspaces/rtx5080/runs/20260709-134954, ERS 0.45086075605694304, success 105/120, zero-token/error 15. Rejected.

TTFT diagnostics:
- full_trace_max_tokens1: run local_workspaces/rtx5080/runs/20260709-143843, TTFT p50/p95 1903.889 / 4679.989 ms, 45/120 requests under 1500 ms.
- length_sweep_single: run local_workspaces/rtx5080/runs/20260709-144028, TTFT p50/p95 767.822 / 1246.430 ms, 6/6 requests under 1500 ms. Longest synthetic prompt 22796 words reached TTFT 1299.363 ms.
- burst_shared_cold: run local_workspaces/rtx5080/runs/20260709-144314, TTFT p50/p95 3265.370 / 5306.342 ms, 3/20 requests under 1500 ms.
- burst_shared_warm: run local_workspaces/rtx5080/runs/20260709-144435, TTFT p50/p95 2839.234 / 4588.891 ms, 4/20 requests under 1500 ms.
- burst_no_shared_cold: run local_workspaces/rtx5080/runs/20260709-144556, TTFT p50/p95 5386.083 / 9372.418 ms, 2/20 requests under 1500 ms.
- tokenization CPU in SGLang Docker under 3 CPU / 8g: encode p50/p95/p99 20.130 / 26.671 / 28.059 ms, total 120-record encode time 2.300 s.

Controlled admission sweep:
- same-session baseline sglang_ablate_auto_mem086_run10: run local_workspaces/rtx5080/runs/20260709-150608, ERS 0.515398, success 120/120, TTFT p50/p95 4610.932 / 36570.01585 ms, TPOT p95 18.121595 ms, TTFT<=1500 25/120.
- sglang_admission_fcfs_firstchunk_cap6: run local_workspaces/rtx5080/runs/20260709-150809, ERS 0.481412, success 120/120, TTFT<=1500 3/120, proxy queue-wait p95 18240.224 ms. Rejected.
- sglang_admission_fcfs_firstchunk_cap8: run local_workspaces/rtx5080/runs/20260709-151017, ERS 0.481840, success 120/120, TTFT<=1500 3/120, proxy queue-wait p95 16921.119 ms. Rejected.
- sglang_admission_ttftv0_firstchunk_cap6: run local_workspaces/rtx5080/runs/20260709-151229, ERS 0.490972, success 120/120, TTFT<=1500 8/120, proxy queue-wait p95 29064.152 ms. Rejected.

Native prefix/priority scheduler sweep:
- same-session baseline for prefill-delayer sweep: run local_workspaces/rtx5080/runs/20260709-152600, ERS 0.516041, success 120/120, TTFT<=1500 23/120, TTFT p50/p95 4564.696 / 36093.005 ms, TPOT p95 17.974 ms.
- sglang_native_prefill_delayer_q025_250: run local_workspaces/rtx5080/runs/20260709-152801, ERS 0.513646, success 120/120, TTFT<=1500 23/120, TTFT p50/p95 4493.901 / 36595.625 ms, TPOT p95 18.311 ms. Rejected.
- sglang_native_prefill_delayer_q050_500: run local_workspaces/rtx5080/runs/20260709-153003, ERS 0.515090, success 120/120, TTFT<=1500 23/120, TTFT p50/p95 4476.439 / 36670.492 ms, TPOT p95 18.265 ms. Rejected.
- same-session baseline for priority sweep: run local_workspaces/rtx5080/runs/20260709-153323, ERS 0.514740, success 120/120, TTFT<=1500 25/120, TTFT p50/p95 4561.498 / 36611.789 ms, TPOT p95 18.003 ms.
- sglang_priority_shortprefill_nocap: run local_workspaces/rtx5080/runs/20260709-153524, ERS 0.439851, success 120/120, TTFT<=1500 3/120, TTFT p50/p95 12830.630 / 41138.169 ms, TPOT p95 36.360 ms, proxy queue-wait p95 20.141 ms. Rejected.
- sglang_priority_prefixlpm_nocap: run local_workspaces/rtx5080/runs/20260709-153751, ERS 0.451969, success 120/120, TTFT<=1500 22/120, TTFT p50/p95 9811.921 / 66516.090 ms, TPOT p95 38.781 ms, proxy queue-wait p95 19.582 ms. Rejected.

Prefill chunk scheduler sweep:
- broad same-session baseline: run local_workspaces/rtx5080/runs/20260709-154949, ERS 0.512680, success 120/120, TTFT<=1500 23/120, TTFT p50/p95 4605.225 / 36691.048 ms, TPOT p95 18.161 ms.
- sglang_prefill_chunk1024: run local_workspaces/rtx5080/runs/20260709-155147, ERS 0.509090, success 120/120, TTFT<=1500 23/120, TTFT p50/p95 5553.023 / 32702.072 ms, TPOT p95 20.019 ms. Rejected.
- sglang_prefill_chunk4096: run local_workspaces/rtx5080/runs/20260709-155355, ERS 0.511333, success 120/120, TTFT<=1500 23/120, TTFT p50/p95 4625.593 / 37374.638 ms, TPOT p95 18.619 ms. Rejected.
- sglang_prefill_maxprefill8192 broad: run local_workspaces/rtx5080/runs/20260709-155557, ERS 0.515851, success 120/120, TTFT<=1500 25/120, TTFT p50/p95 4576.188 / 35990.431 ms, TPOT p95 17.969 ms. Refined because it beat same-session baseline.
- sglang_prefill_maxreq4: run local_workspaces/rtx5080/runs/20260709-155758, ERS 0.514130, success 120/120, TTFT<=1500 21/120, TTFT p50/p95 4487.104 / 36496.902 ms, TPOT p95 18.168 ms. Watchlist only.
- sglang_prefill_mixed_chunk: run local_workspaces/rtx5080/runs/20260709-160003, ERS 0.514428, success 120/120, TTFT<=1500 26/120, TTFT p50/p95 3913.834 / 35987.500 ms, TPOT p95 20.708 ms. Rejected as ERS baseline because TPOT p95 crossed the official floor region.
- refine same-session baseline: run local_workspaces/rtx5080/runs/20260709-160310, ERS 0.517014, success 120/120, TTFT<=1500 25/120, TTFT p50/p95 4533.439 / 37077.954 ms, TPOT p95 18.368 ms.
- sglang_prefill_maxprefill8192 refine: run local_workspaces/rtx5080/runs/20260709-160513, ERS 0.513922, success 120/120, TTFT<=1500 25/120, TTFT p50/p95 4543.819 / 36306.676 ms, TPOT p95 18.154 ms. Rejected as new baseline because pass 1 win did not repeat.
- sglang_prefill_maxprefill6144: run local_workspaces/rtx5080/runs/20260709-160713, ERS 0.515615, success 120/120, TTFT<=1500 23/120, TTFT p50/p95 4621.256 / 36264.169 ms, TPOT p95 17.916 ms. Watchlist only.
- sglang_prefill_maxprefill12288: run local_workspaces/rtx5080/runs/20260709-160915, ERS 0.513384, success 120/120, TTFT<=1500 21/120, TTFT p50/p95 4716.497 / 36470.563 ms, TPOT p95 18.076 ms. Rejected.
- sglang_prefill_maxprefill8192_mixed: run local_workspaces/rtx5080/runs/20260709-161120, ERS 0.512677, success 120/120, TTFT<=1500 26/120, TTFT p50/p95 4015.333 / 35982.761 ms, TPOT p95 20.345 ms. Rejected as ERS baseline.
- sglang_prefill_maxprefill8192_maxreq4: run local_workspaces/rtx5080/runs/20260709-161319, ERS 0.514780, success 120/120, TTFT<=1500 23/120, TTFT p50/p95 4572.905 / 36547.706 ms, TPOT p95 17.988 ms. Watchlist only.

Decode protection scheduler sweep:
- broad same-session baseline: run local_workspaces/rtx5080/runs/20260709-162322, ERS 0.516420, success 120/120, TTFT<=1500 25/120, TTFT p50/p95 4528.508 / 36718.905 ms, TPOT p95 18.093 ms.
- sglang_decode_reserve256: run local_workspaces/rtx5080/runs/20260709-162523, ERS 0.515200, success 120/120, TTFT<=1500 23/120, TTFT p50/p95 4439.511 / 36372.634 ms, TPOT p95 18.007 ms. Watchlist only.
- sglang_decode_reserve200: run local_workspaces/rtx5080/runs/20260709-162725, ERS 0.517276, success 120/120, TTFT<=1500 24/120, TTFT p50/p95 4407.224 / 36286.383 ms, TPOT p95 18.124 ms. Refined because it beat same-session baseline.
- sglang_decode_steps2: run local_workspaces/rtx5080/runs/20260709-162925, ERS 0.513750, success 120/120, TTFT<=1500 24/120, TTFT p50/p95 4585.756 / 36513.066 ms, TPOT p95 18.016 ms. Rejected.
- sglang_decode_steps4: run local_workspaces/rtx5080/runs/20260709-163125, ERS 0.515807, success 120/120, TTFT<=1500 27/120, TTFT p50/p95 4403.766 / 36173.404 ms, TPOT p95 18.348 ms. Refined as combo.
- sglang_schedule_conservative050: run local_workspaces/rtx5080/runs/20260709-163324, ERS 0.516384, success 120/120, TTFT<=1500 24/120, TTFT p50/p95 4334.474 / 36281.985 ms, TPOT p95 18.248 ms. Watchlist only.
- sglang_schedule_conservative200: run local_workspaces/rtx5080/runs/20260709-163524, ERS 0.514763, success 120/120, TTFT<=1500 25/120, TTFT p50/p95 4461.715 / 36393.750 ms, TPOT p95 18.041 ms. Rejected.
- refine same-session baseline: run local_workspaces/rtx5080/runs/20260709-163902, ERS 0.514151, success 120/120, TTFT<=1500 23/120, TTFT p50/p95 4449.644 / 36863.899 ms, TPOT p95 18.035 ms.
- sglang_decode_reserve200 refine: run local_workspaces/rtx5080/runs/20260709-164103, ERS 0.516315, success 120/120, TTFT<=1500 24/120, TTFT p50/p95 4613.804 / 36124.047 ms, TPOT p95 17.992 ms. Watchlist.
- sglang_decode_reserve224: run local_workspaces/rtx5080/runs/20260709-164304, ERS 0.516981, success 120/120, TTFT<=1500 25/120, TTFT p50/p95 4407.977 / 36529.803 ms, TPOT p95 17.985 ms. Watchlist.
- sglang_decode_reserve200_steps4: run local_workspaces/rtx5080/runs/20260709-164504, ERS 0.517185, success 120/120, TTFT<=1500 25/120, TTFT p50/p95 4499.346 / 35970.708 ms, TPOT p95 18.008 ms. Refined further.
- sglang_decode_reserve200_conservative050: run local_workspaces/rtx5080/runs/20260709-164705, ERS 0.515647, success 120/120, TTFT<=1500 23/120, TTFT p50/p95 4570.929 / 36143.026 ms, TPOT p95 17.873 ms. Watchlist only.
- final same-session baseline: run local_workspaces/rtx5080/runs/20260709-173008, ERS 0.516250, success 120/120, TTFT<=1500 26/120, TTFT p50/p95 4300.724 / 36187.187 ms, TPOT p95 18.308 ms.
- sglang_decode_reserve200_steps4 final: run local_workspaces/rtx5080/runs/20260709-173207, ERS 0.519726, success 120/120, TTFT<=1500 23/120, TTFT p50/p95 3762.269 / 33819.652 ms, TPOT p95 16.959 ms. Strong candidate.
- sglang_decode_reserve224_steps4 final: run local_workspaces/rtx5080/runs/20260709-173405, ERS 0.523150, success 120/120, TTFT<=1500 23/120, TTFT p50/p95 3754.602 / 33606.856 ms, TPOT p95 17.002 ms. New current local best.
- sglang_decode_reserve224_steps4 GPQA proxy: accuracy 0.37, total 100, parsed 100, errors 0, official-style delta 0.03, penalty 1.0.

Technical conclusion:
- SGLang is now proven locally and beats Ready 002 by ERS.
- The best ERS candidate is sglang_ablate_auto_mem086_run10.
- The best TTFT/TPOT profile is sglang_flashinfer_fp8kv, but it has FP8 KV quality risk and slightly lower ERS.
- The probed SGLang image does not support --enable-cuda-graph; use per-phase CUDA graph flags instead.
- Even the best Ready 003 candidate is not submission-ready because TTFT p50 is still 4318 ms, above the official 1500 ms ceiling.
- TTFT is primarily burst-queue/prefix-scheduling bound, not CPU-tokenizer bound and not raw single-request-prefill bound. Isolated long prompts are under the official TTFT ceiling, while 20-request bursts fail badly.
- External HTTP request-level admission control is not enough. It moves queueing into the proxy, and that waiting time still counts toward TTFT. Coarse `first_chunk` release does not solve burst prefill.
- Native SGLang prefill-delayer is a valid knob but did not beat baseline in this trace. Whole-request native priority scheduling is worse than LPM/radix baseline even when the proxy does not queue, so the next scheduler work must target prefix-build/prefill chunks rather than request-level priority.
- SGLang prefill chunk/token knobs did not produce a repeatable new baseline. Lowering max-prefill-tokens can reduce TTFT/TPOT tail in some runs, but the ERS win was not repeatable. Mixed chunk improves TTFT p50 and TTFT<=1500 count, but hurts TPOT p95 enough to lose ERS.
- Decode reserve is a real local lever because the public trace uses max_tokens=200 for every request. `num-reserved-decode-tokens=224 + num-continuous-decode-steps=4` is the new local best and passed GPQA proxy with accuracy 0.37.

Decision: promote sglang_decode_reserve224_steps4 as the current local SGLang ERS baseline. Keep sglang_ablate_auto_mem086_run10 as stable rollback and auto_mem082_run8 as safer-VRAM fallback. Keep Triton Inference Server as a measured subround, not the default path.
Next: package/validate sglang_decode_reserve224_steps4 as the current SGLang submission candidate. Continue TTFT work only if it preserves TPOT p95 near 17 ms.
```

## Ready Round 004 - SGLang Self Optimization

```text
Status: completed
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_004_sglang_self_optimization
Output:
- outputs/selfopt_summary_20260709-175739.md
- outputs/selfopt_summary_20260709-181541.md

What happened: kept SGLang as the runtime and tested built-in optimization knobs around RadixAttention, FlashInfer, CUDA graph, dynamic chunking, decode reserve, continuous decode steps, lower active concurrency and stronger continuous batching.

Best ERS:
- selfopt_control_reserve224_steps4: run local_workspaces/rtx5080/runs/20260709-175848, ERS 0.523052, success 120/120, TTFT p50/p95 3787.462 / 34046.056 ms, TPOT p50/p95 12.447 / 17.286 ms.

Lowest TPOT:
- selfopt_dynamic_run4_steps8: run local_workspaces/rtx5080/runs/20260709-181302, ERS 0.507828, success 120/120, TTFT p50/p95 20565.496 / 58843.374 ms, TPOT p50/p95 7.064 / 9.355 ms.

Best balanced TPOT rescue:
- selfopt_dynamic_run6_steps6: run local_workspaces/rtx5080/runs/20260709-181057, ERS 0.511819, success 120/120, TTFT p50/p95 4681.498 / 42216.217 ms, TPOT p50/p95 8.373 / 11.666 ms.

Continuous batching result:
- selfopt_contbatch_run14_steps6: run local_workspaces/rtx5080/runs/20260709-181849, ERS 0.501104, TTFT p50 3132.908 ms, wall tokens/s 553.047, but TPOT p95 23.741 ms. It improves TTFT p50 and throughput but is not scoring-safe because TPOT crosses the official 20 ms floor.

Technical conclusion:
- Current ERS candidate does not change.
- Dynamic chunking alone did not help this trace.
- Explicit FlashInfer backend works but is not better than SGLang auto backend.
- Lower active concurrency is the strongest SGLang TPOT lever.
- Aggressive continuous batching is a throughput/TTFT lever, not a TPOT/ERS lever under current scoring.

Decision: keep selfopt_control_reserve224_steps4 / sglang_decode_reserve224_steps4 as the local ERS baseline. Keep run6_steps6 and run4_steps8 only as TPOT rescue/floor candidates. Do not promote dynamic chunking, mixed chunk or aggressive continuous batching by default.
Next: if we want more score, attack TTFT/prefix burst without letting TPOT p95 rise above 20 ms.
```

## Ready Round 005 - TPOT Floor With Compatible TTFT Overlays

```text
Status: completed
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay
Outputs:
- outputs/ttft_overlay_summary_20260709-185002.md
- outputs/ttft_overlay_summary_20260709-191334.md

What happened: backed up the lowest-TPOT Ready Round 004 config
`selfopt_dynamic_run4_steps8`, froze its TPOT policy, then added only TTFT
overlays that did not remove the TPOT-floor policy. A second mini sweep tested
chunked prefill and mixed-chunk interleaving.

TPOT-floor control replay:
- run local_workspaces/rtx5080/runs/20260709-185111
- ERS 0.508153
- success 120/120
- TTFT p50/p95 20444.543 / 58445.085 ms
- TPOT p50/p95 6.998 / 9.629 ms

Best first-pass compatible overlay:
- candidate tpotfloor_full_ttft_overlay
- run local_workspaces/rtx5080/runs/20260709-191017
- ERS 0.512098
- success 120/120
- TTFT p50/p95 10445.564 / 50899.531 ms
- TPOT p50/p95 8.767 / 11.029 ms

Best chunked-prefill/interleaving overlay:
- candidate tpotfloor_chunk1024_prefillreq4_mixed
- run local_workspaces/rtx5080/runs/20260709-192607
- ERS 0.513911
- success 120/120
- TTFT p50/p95 12178.541 / 49009.192 ms
- TPOT p50/p95 9.107 / 11.549 ms
- config additions: chunked-prefill-size=1024, prefill-max-requests=4,
  enable-mixed-chunk

Chunk 256 follow-up:
- candidate tpotfloor_chunk256_prefillreq4_mixed
- run local_workspaces/rtx5080/runs/20260709-194516
- ERS 0.504052
- success 120/120
- TTFT p50/p95 27566.901 / 59758.971 ms
- TPOT p50/p95 11.093 / 13.209 ms
- decision: rejected; chunk 256 is too small under this policy and hurts both
  TTFT and TPOT versus the 1024 winner.

Chunk 256 one-request prefill follow-up:
- candidate tpotfloor_chunk256_prefillreq1_mixed
- run local_workspaces/rtx5080/runs/20260709-195432
- ERS 0.501992
- success 120/120
- TTFT p50/p95 27984.350 / 59794.267 ms
- TPOT p50/p95 10.718 / 12.465 ms
- decision: rejected; limiting prefill admission to 1 improves TPOT p95 versus
  chunk256 req4 but worsens TTFT and ERS, so it does not rescue chunk 256.

Chunk 1024 one-request prefill follow-up:
- candidate tpotfloor_chunk1024_prefillreq1_mixed
- run local_workspaces/rtx5080/runs/20260709-195859
- ERS 0.511664
- success 120/120
- TTFT p50/p95 12381.904 / 48774.064 ms
- TPOT p50/p95 8.983 / 11.087 ms
- wall tokens/s 379.044
- decision: not promoted over req4 for ERS. It improves TPOT p95 versus
  `tpotfloor_chunk1024_prefillreq4_mixed`, but loses ERS, so keep it as a
  TPOT-tail variant only.

Failure:
- candidate tpotfloor_chunk1024_full_interleave crashed during endpoint warmup.
- error: PCG capture stream is not set.

Technical conclusion:
- The TPOT-floor policy can accept TTFT overlays while keeping TPOT p95 near
  11 ms.
- Mixed chunk is the main TTFT lever in this lane.
- Chunk size is not monotonic: 256/512/768 were worse; 1024 with prefill admission
  cap 4 was best by ERS.
- At chunk 1024, prefill admission cap 1 is stable and TPOT-friendlier, but not
  the ERS winner.
- Max-prefill-tokens=8192 was not the winning component in the mini sweep.
- This lane is useful as a TPOT-safe module, but it still does not beat the
  global Ready 003f ERS baseline of 0.523150.

Decision: keep `tpotfloor_chunk1024_prefillreq4_mixed` as the best TPOT-floor
TTFT overlay. Do not promote it over the global SGLang baseline yet.
Next: if using a two-policy/phase-aware pipeline, use this as the TPOT-safe
backend module candidate and keep Ready 003f as the global ERS control.
```

### Ready Round 005b - TTFT Floor Hunt

```text
Status: completed
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/subround_005b_ttft_floor_hunt
Output: outputs/ttft_hunt_summary_20260709-203017.md

What happened: created a TTFT-only subround and tested aggressive SGLang
policies that do not preserve TPOT-floor safety. All candidates completed
120/120 requests.

Best ERS:
- ttft_control_global_003f
- run local_workspaces/rtx5080/runs/20260709-203130
- ERS 0.513092
- TTFT p50/p95 4338.697 / 37869.424 ms
- TPOT p95 18.660 ms

Lowest TTFT p50:
- ttft_run14_chunk1536_req4_mixed
- run local_workspaces/rtx5080/runs/20260709-204342
- ERS 0.404494
- TTFT<=1500 30/120
- TTFT<=3000 65/120
- TTFT p50/p95 2765.807 / 30741.190 ms
- TPOT p95 25.691 ms

Highest TTFT<=1500 count:
- ttft_run16_steps6
- run local_workspaces/rtx5080/runs/20260709-203754
- TTFT<=1500 32/120
- TTFT p50/p95 2975.257 / 20053.167 ms
- TPOT p95 44.433 ms
- Decision: reject as too aggressive for TPOT.

Engine check:
- No evidence of torch-native fallback.
- SGLang logs showed FlashInfer attention/sampling, Triton linear attention,
  and CUDA graph prefill/decode capture with `cuda graph: True` in batch logs.

Bound check:
- Live snapshot during benchmark: GPU util about 80%, power 223 W, VRAM
  15.6/16.3 GiB; container CPU about 142% on 3-core quota and RAM 5.2/8 GiB.
- Reading: not clearly CPU hard-bound. TTFT remains burst queue/prefill
  scheduling plus decode-interference bound.

Technical conclusion:
- A single global policy cannot currently minimize TTFT and TPOT together.
- MAX_RUNNING_REQUESTS around 14 plus mixed chunk/chunked prefill can reduce
  TTFT p50 to about 2.8s, but TPOT p95 rises above 25 ms.
- MAX_RUNNING_REQUESTS=16 improves TTFT tail count but destroys TPOT tail.
- Do not attach this as the final TTFT module yet; use the evidence to design a
  phase-aware TTFT lane that hands off to TPOT-safe decode before long decode.

Follow-up CPU/chunk/radix/overlap checks:
- Host `CUDA_LAUNCH_BLOCKING` was empty, so no debug synchronization was active.
- Benchmark concurrency is `CONCURRENCY=120` with `TIME_SCALE=1.0`.
- Focused chunk8192/thread/radix/overlap sweep output:
  `outputs/ttft_hunt_summary_20260709-205023.md`.
- All chunk8192 variants failed before readiness with prefill CUDA graph capture
  CUDA OOM. Thread cap `OMP_NUM_THREADS=1`, disabling radix, and disabling
  overlap did not change that startup failure class.
- `chunked-prefill-size=-1` started but produced 0/120 successful requests with
  HTTP 500 / connection refused, so it is not usable in this image/config.
- Next chunk test should use 3072/4096/6144 or cap prefill CUDA graph batch /
  disable prefill CUDA graph for a controlled test.
```

### Ready Round 005c - Trace Prefix Prime

```text
Status: completed
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/subround_005c_trace_prefix_prime
Output: outputs/prefix_prime_summary_20260709-211315.md

What happened: verified the public trace has one identical system prefix across
all 120 requests, length 38956 characters. Added an endpoint warmup mode that
sends that exact system message with max_tokens=1 before the timed benchmark,
then A/B tested it against the normal synthetic warmup on the Ready 003f global
SGLang baseline.

Control:
- candidate prefixprime_control_global_003f
- run local_workspaces/rtx5080/runs/20260709-211424
- ERS 0.515480
- success 120/120
- TTFT<=1500 28/120
- TTFT<=3000 49/120
- TTFT p50/p95 4413.646 / 35326.715 ms
- TPOT p95 18.243 ms
- wall tokens/s 478.204

Trace-system prime:
- candidate prefixprime_trace_system_global_003f
- run local_workspaces/rtx5080/runs/20260709-211620
- ERS 0.512806
- success 120/120
- TTFT<=1500 20/120
- TTFT<=3000 33/120
- TTFT p50/p95 10040.703 / 40916.335 ms
- TPOT p95 19.655 ms
- wall tokens/s 449.405

Log check:
- SGLang used MambaRadixCache and CUDA graph prefill remained active.
- The prime request itself ran 2048-token chunks with #cached-token 0.
- Timed benchmark logs had non-zero #cached-token entries in both control and
  prime candidates, so prefix caching was active even when the prime regressed.

Technical conclusion:
- Exact system-only trace prefix priming did not reduce TTFT; it regressed ERS,
  TTFT p50, TTFT p95 and TPOT p95 versus same-session control.
- This does not disprove Radix caching. It rejects this specific warmup shape.
- Likely causes are chat-template/prefix-boundary mismatch, missing real user
  prefix continuation, cache residency side effects, or suffix/queue pressure
  dominating the benchmark after warmup.

Decision: reject `trace_system_prefix` warmup as a carry-forward optimization.
Keep it as local research only because it uses public trace content and has
submission-safety risk unless BTC explicitly allows workload-specific cache
priming.
Next: do not spend more time on system-only priming. If prefix warmup is revisited,
first add explicit cache-hit instrumentation or compare tokenized chat-template
prefixes; otherwise continue with scheduler/prefill controls.
```

## Ready Round 006 - Hybrid Architecture Audit

```text
Status: completed
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit
Outputs:
- outputs/model_artifact_audit_20260709-213647.md
- outputs/hybrid_audit_summary_20260709-213647.md
- outputs/hybrid_audit_summary_20260709-214415.md

Why: local model artifact is not pure full-attention Transformer. `config.json`
shows 18 linear-attention layers and 6 full-attention layers. SGLang logs allocate
Mamba Cache and initialize MambaRadixCache.

Static audit:
- architecture Qwen3_5ForConditionalGeneration
- layer_type_counts {'linear_attention': 18, 'full_attention': 6}
- mamba_ssm_dtype float32
- full_attention_interval 4

Benchmark:
- First full sweep control crashed during trace and returned 0/120. This is
  treated as transient server crash, not a valid control.
- Replay control: run local_workspaces/rtx5080/runs/20260709-214526, ERS
  0.513978, success 120/120, TTFT p50/p95 4206.157 / 36970.491 ms, TPOT p95
  18.145 ms.
- hybrid_mamba_flashinfer: run local_workspaces/rtx5080/runs/20260709-213936,
  ERS 0.514223, success 120/120, TTFT p50/p95 4377.759 / 35469.702 ms, TPOT
  p95 17.475 ms, wall tokens/s 471.170. Watchlist only.
- hybrid_ssm_bfloat16: run local_workspaces/rtx5080/runs/20260709-214138, ERS
  0.519408, success 120/120, TTFT p50/p95 4172.921 / 36443.485 ms, TPOT p95
  19.004 ms. Best round-006 ERS, but quality-risk and still below global Ready
  003f ERS 0.523150.
- hybrid_linear_flashinfer failed startup because SGLang requires BF16 SSM state
  for FlashInfer linear attention on SM100+.
- hybrid_linear_flashinfer_bfloat16 failed during prefill CUDA graph capture:
  FlashInfer GDN prefill expected float32 input_state but got bfloat16.

Technical conclusion:
- We were optimizing a hybrid architecture, not a pure Transformer. The runtime
  has both full-attention KV and Mamba/linear-attention state.
- Pure Transformer KV-cache conclusions need revalidation.
- SGLang hybrid knobs are real and measurable, but no tested hybrid knob beat
  the existing global SGLang baseline yet.
- `mamba_ssm_dtype=bfloat16` is the strongest hybrid signal so far, but it must
  run GPQA/output-drift before promotion because it changes recurrent state
  precision.

Decision: keep current global baseline unchanged. Carry forward
`mamba_backend=flashinfer` and `mamba_ssm_dtype=bfloat16` as watchlist knobs.
Reject FlashInfer linear-attention backend under current prefill CUDA graph
policy.
Next: if continuing hybrid-specific work, run GPQA proxy for BF16 SSM and test
linear FlashInfer only with an explicit prefill-CUDA-graph policy subround.
```

### Ready Round 006a - Mamba State Quality Gate

```text
Status: completed
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006a_mamba_state_quality_gate
Output: outputs/gpqa_comparison_20260709-215832.md

Why: Round 006 found `--mamba-ssm-dtype=bfloat16` was the strongest
hybrid-specific performance knob, but it changes recurrent/linear-attention
state precision. This subround checked local GPQA proxy quality before any
promotion.

Control:
- candidate control_global_003f_quality
- SSM dtype default float32
- status ok
- accuracy 0.370000
- correct 37/100
- parsed 100/100
- official-style proxy penalty 1.000000

Candidate:
- candidate mamba_ssm_bfloat16_quality
- SSM dtype bfloat16
- status ok
- accuracy 0.370000
- correct 37/100
- parsed 100/100
- official-style proxy penalty 1.000000

Movement:
- changed_prediction 0/100
- changed_correctness 0/100
- baseline_correct_candidate_wrong 0
- baseline_wrong_candidate_correct 0

Technical conclusion:
- `--mamba-ssm-dtype=bfloat16` passed the local GPQA proxy against the
  default-float32 SSM control.
- Quality risk is reduced, but not fully eliminated because BTC uses its own
  evaluator/subset.
- Do not promote yet: Round 006 BF16 SSM trace ERS was 0.519408, still below
  Ready 003f global local best 0.523150.

Runner note:
- The first 006a invocation only produced missing_config rows because shared
  `round_lib.sh` overwrote subround CONFIG_DIR/OUTPUT_DIR. It did not run
  Docker/model evaluation. The runner was fixed and the successful run is
  timestamp 20260709-215832.

Next:
- Keep all Transformer/KV optimizations untouched.
- Treat BF16 SSM as a safer hybrid watchlist overlay.
- Next hybrid run should replay BF16 SSM against the current Ready 003f baseline
  in the same session, then optionally test `mamba_backend=flashinfer` combined
  with BF16 SSM.
```

### Ready Round 006b - BF16 SSM Same-Session Replay

```text
Status: completed
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit
Output: outputs/hybrid_audit_summary_20260709-220521.md

Why: after BF16 SSM passed local GPQA proxy, replay it directly against the
matching Ready 003f control in one benchmark session to reduce run-to-run noise.

Control:
- candidate hybrid_control_global_003f
- run local_workspaces/rtx5080/runs/20260709-220632
- ERS 0.512065
- success 120/120
- TTFT<=1500 21/120
- TTFT p50/p95 4518.249 / 36521.461 ms
- TPOT p50/p95 13.662 / 18.157 ms
- wall tokens/s 458.378

Candidate:
- candidate hybrid_ssm_bfloat16
- run local_workspaces/rtx5080/runs/20260709-220837
- ERS 0.513613
- success 120/120
- TTFT<=1500 21/120
- TTFT p50/p95 4112.222 / 37804.537 ms
- TPOT p50/p95 13.814 / 18.188 ms
- wall tokens/s 468.291

Technical conclusion:
- `--mamba-ssm-dtype=bfloat16` repeated as a positive hybrid-specific signal.
- Same-session ERS improved by +0.001548.
- TTFT p50 improved by about 406 ms and wall tokens/s improved by about 9.9.
- Trade-off: TTFT p95 and TPOT p95 worsened slightly.
- This does not promote BF16 SSM as current baseline because global Ready 003f
  remains higher at ERS 0.523150.

Decision:
- Keep BF16 SSM as a high-priority watchlist overlay.
- Next hybrid-specific experiment should test `--mamba-backend=flashinfer`
  combined with BF16 SSM in a separate same-session replay.
```

### Ready Round 006c - BF16 SSM + Mamba FlashInfer Replay

```text
Status: completed
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit
Output: outputs/hybrid_audit_summary_20260709-221222.md

Why: test whether `--mamba-backend=flashinfer` adds to the BF16 SSM state
overlay after BF16 SSM passed GPQA proxy and one same-session replay.

Control:
- candidate hybrid_control_global_003f
- run local_workspaces/rtx5080/runs/20260709-221331
- ERS 0.523891
- success 120/120
- TTFT<=1500 25/120
- TTFT p50/p95 3941.145 / 33750.322 ms
- TPOT p50/p95 12.917 / 17.627 ms
- wall tokens/s 480.635
- VRAM max 15623 MiB

BF16 SSM:
- candidate hybrid_ssm_bfloat16
- run local_workspaces/rtx5080/runs/20260709-221531
- ERS 0.522122
- success 120/120
- TTFT<=1500 23/120
- TTFT p50/p95 3776.410 / 34827.156 ms
- TPOT p50/p95 12.358 / 17.078 ms
- wall tokens/s 505.430
- VRAM max 15642 MiB

BF16 SSM + Mamba FlashInfer:
- candidate hybrid_mamba_flashinfer_ssm_bfloat16
- run local_workspaces/rtx5080/runs/20260709-221727
- ERS 0.518213
- success 120/120
- TTFT<=1500 22/120
- TTFT p50/p95 3762.957 / 34622.083 ms
- TPOT p50/p95 12.381 / 16.824 ms
- wall tokens/s 507.813
- VRAM max 15422 MiB

Technical conclusion:
- Combo BF16 SSM + Mamba FlashInfer did not improve ERS.
- It is the best TPOT/throughput/VRAM profile in this replay, but the scoring
  objective still prefers plain Ready 003f control.
- The control run reached ERS 0.523891, so the current best observed local run
  is the existing Ready 003f config, not a new hybrid overlay.

Decision:
- Do not promote `--mamba-backend=flashinfer --mamba-ssm-dtype=bfloat16`.
- Keep it as a TPOT/throughput/VRAM watchlist module.
- Update current best observed run for the Ready 003f config to
  local_workspaces/rtx5080/runs/20260709-221331.
```

### Ready Round 006d - Hybrid TTFT Combo Hunt

```text
Status: completed
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006d_hybrid_ttft_combo_hunt
Artifact: outputs/ttft_combo_artifacts_20260709.md
Per-request TTFT: outputs/ttft_per_request_20260709-221727.csv

Why: continue from hybrid Mamba/linear-attention insight and search for a combo
that lowers TTFT the most, while keeping earlier Transformer/KV optimizations
intact.

Important failure:
- First 006d sweep inherited synthetic endpoint warmup and crashed/timed out
  before benchmark. This is logged in failed_actions_log.md.
- Rerun disabled endpoint warmup for aggressive TTFT candidates.

Valid no-warmup candidates:
- control_no_warmup: run local_workspaces/rtx5080/runs/20260709-231156,
  ERS 0.487992, TTFT p50/p95 5361.742 / 39092.105 ms, TPOT p95 28.015 ms.
- run14_steps6: run local_workspaces/rtx5080/runs/20260709-231354,
  ERS 0.459727, TTFT p50/p95 11794.261 / 37790.917 ms, TPOT p95 30.631 ms.
- run14_steps6_bf16_flashinfer: run local_workspaces/rtx5080/runs/20260709-231546,
  ERS 0.461250, TTFT p50/p95 4351.329 / 32825.488 ms, TPOT p95 34.649 ms.
- run16_steps6_bf16_flashinfer: run local_workspaces/rtx5080/runs/20260709-232305,
  ERS 0.408281, TTFT p50/p95 4381.736 / 32022.485 ms, TPOT p95 40.979 ms.

Failed candidate:
- chunk1536_req4_mixed: run folder local_workspaces/rtx5080/runs/20260709-231742
  had empty metrics.jsonl; server returned 500/timeouts and the run was stopped.

Decision:
- Best stable TTFT combo remains the replay-006c candidate
  `hybrid_mamba_flashinfer_ssm_bfloat16`:
  run local_workspaces/rtx5080/runs/20260709-221727,
  ERS 0.518213, TTFT p50/p95 3762.957 / 34622.083 ms, TPOT p95 16.824 ms.
- It improves TTFT p50 versus Ready 003f control and gives better TPOT/throughput,
  but does not beat control ERS.
- Aggressive run14/run16 TTFT policies are not currently useful; they hurt TPOT
  and do not beat the stable hybrid combo.
```

### Ready Round 006e - Decode Fairness Hunt

```text
Status: completed
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006e_decode_fairness_hunt
Primary outputs: round_006_hybrid_architecture_audit/outputs/hybrid_audit_runs_20260709-234008.tsv
FCFS rerun output: round_006_hybrid_architecture_audit/outputs/hybrid_audit_runs_20260709-235324.tsv

Why: test the diagnosis that current TTFT loss is caused by decode blocking
and scheduling/fairness under CONCURRENCY=120. Keep the hybrid overlay
`--mamba-backend=flashinfer --mamba-ssm-dtype=bfloat16`, then vary
continuous decode steps, reserved decode tokens and schedule policy.

Control:
- candidate hybrid_mamba_flashinfer_ssm_bfloat16
- run local_workspaces/rtx5080/runs/20260709-234116
- ERS 0.519671
- success 120/120
- TTFT buckets: <=1.5s 22, 1.5-3s 24, 3-5s 23, 5-10s 12, 10-30s 26, >=30s 13
- TTFT p50/p95 3815.775 / 34693.195 ms
- TPOT p95 16.771 ms

Best candidate:
- candidate hybrid_fair_lpm_res64_steps1
- run local_workspaces/rtx5080/runs/20260709-234511
- config delta: `--num-reserved-decode-tokens=64 --num-continuous-decode-steps=1`
- ERS 0.524547
- success 120/120
- TTFT buckets: <=1.5s 23, 1.5-3s 24, 3-5s 20, 5-10s 14, 10-30s 25, >=30s 14
- TTFT p50/p95 3896.429 / 35006.865 ms
- TPOT p95 17.036 ms

Other valid LPM candidates:
- res128_steps1: run local_workspaces/rtx5080/runs/20260709-234313,
  ERS 0.522856, TTFT<=1.5s 25/120, TPOT p95 17.358 ms.
- res128_steps2: run local_workspaces/rtx5080/runs/20260709-234709,
  ERS 0.523722, TTFT<=1.5s 25/120, TPOT p95 17.022 ms.

Rejected candidates:
- res64_steps2: run local_workspaces/rtx5080/runs/20260709-235042 wrote all
  benchmark results but produced corrupted JSONL/report output. Salvaged parse
  recovered only 89 unique ok records; TTFT p50/p95 19744.251 / 50371.084 ms
  and TPOT p95 28.830 ms. Reject and log as harness failure plus bad candidate.
- fcfs_run12_res128_steps1: run local_workspaces/rtx5080/runs/20260709-235432,
  ERS 0.448436, success 120/120, TTFT<=1.5s 3/120, TTFT p50/p95
  8390.090 / 23842.449 ms, TPOT p95 39.355 ms. It removed the >=30s TTFT
  bucket but moved too many requests into 5-30s and damaged TPOT.

Technical conclusion:
- The best signal is not FCFS fairness. FCFS reduces extreme tail but destroys
  near-floor TTFT count and pushes TPOT close to the 45 ms ceiling.
- Reducing decode blocking inside LPM is mildly positive. `reserved=64,
  steps=1` is the best local ERS observed so far, but only by a small margin:
  0.524547 versus previous best 0.523891.
- Promote `hybrid_fair_lpm_res64_steps1` as a provisional local ERS candidate,
  but repeat/quality-gate before treating it as submission-safe.
```

### Ready Round 006f - Internal Scheduler Prefill Hunt

```text
Status: completed
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006f_internal_scheduler_prefill_hunt

Why: test SGLang internal scheduler/prefill knobs before writing a custom proxy.
Base is the current provisional best shape: LPM, max-running-requests=10,
reserved=64, continuous-decode-steps=1, Mamba FlashInfer, BF16 SSM.

Round I scheduler policy:
- LPM repeat: ERS 0.521266, TTFT<=1.5s 29/120.
- LOF: ERS 0.467596, TTFT<=1.5s 3/120, TPOT p95 31.271 ms.
- LOF + low-priority-first: ERS 0.467617, essentially same as LOF.
- DFS-weight: ERS 0.515851.
- Random: ERS 0.505599.

Round J chunked prefill:
- best J was `chunk4096 + mixed`: ERS 0.519818, TTFT<=1.5s 24/120,
  TPOT p95 20.334 ms.
- `chunk8192 + mixed` was backend_not_ready.
- `chunked-prefill-size=-1` produced 0/120 success, so chunked prefill is
  required for this workload.

Round K prefill limits:
- best K was `max-prefill-tokens=4096, prefill-max-requests=2`: ERS 0.524073.
- `4096/req4` and `8192/req2` were close but below best.

Round L graph/conservativeness:
- best L was `schedule-conservativeness=0.1`: ERS 0.524510,
  just below current best 0.524547.
- `disable-prefill-cuda-graph`: ERS 0.520807. It improved TTFT p50 and TPOT
  p95 but lost scoring distribution.

Technical conclusion:
- Current best remains `hybrid_fair_lpm_res64_steps1`, run
  local_workspaces/rtx5080/runs/20260709-234511, ERS 0.524547.
- LPM is still the best internal scheduler for this public trace.
- LOF/FCFS-style fairness hurts the near-floor TTFT bucket and is not a direct
  solution.
- Mixed chunk, dynamic chunking, manual prefill limits, and disabling prefill
  graph do not beat the current best.
- Short/medium prompts still appear in the >=30s TTFT tail, while >=20k-word
  prompts often do not. The bottleneck is admission/cache/order interaction, not
  raw prompt length alone.

Decision:
- Stop broad internal SGLang flag sweeps for now.
- Repeat and quality-gate the exact current best.
- Next new direction should be a lightweight admission/priority proxy or a
  priority-forwarding proxy into SGLang, not more random flag search.
```

### Ready Round 007 - Logical Two-Policy Proxy

```text
Status: completed
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_007_logical_two_policy_proxy
Summary: outputs/proxy_round_summary_20260710-011540.md

What happened:
- Implemented `local_workspaces/rtx5080/scripts/two_policy_proxy.py`, a
  single-model OpenAI-compatible streaming proxy.
- Backend SGLang ran on port 8001; proxy ran on port 8000.
- The proxy used separate lifecycle limits:
  `PROXY_PREFILL_LIMIT` for requests not yet past first token and
  `PROXY_TOTAL_LIMIT` for all active forwarded requests.
- The proxy released a prefill slot as soon as the first streamed `data:` event
  arrived, while the same request kept decoding through the same SGLang backend.

Benchmarked candidates:
- proxy_a_prefill8_total12: ERS 0.488320, TTFT<=1.5s 4/120, TTFT p50/p95
  8215.968 / 14608.821 ms, TPOT p95 22.355 ms.
- proxy_b_prefill10_total12: ERS 0.491543, TTFT<=1.5s 6/120, TTFT p50/p95
  8427.246 / 15442.091 ms, TPOT p95 22.147 ms.
- proxy_c_prefill10_total14: ERS 0.493432, TTFT<=1.5s 3/120, TTFT p50/p95
  6618.465 / 11752.298 ms, TPOT p95 21.721 ms. Best in this round.
- proxy_d_prefill12_total16: ERS 0.472062, TTFT<=1.5s 4/120, TTFT p50/p95
  5213.210 / 9308.726 ms, TPOT p95 26.566 ms.

Technical conclusion:
- The single-model two-policy concept was implemented correctly at HTTP
  admission level, but it is the wrong control layer for this trace.
- It removed the >=30s TTFT tail, but sacrificed the <=1500ms TTFT bucket and
  pushed TPOT p95 above the 20ms floor.
- Queue wait in the proxy still counts toward TTFT, so HTTP-level admission
  mostly moves waiting from SGLang into the proxy.

Decision:
- Reject Ready Round 007 candidates.
- Current provisional local best remains Ready 006e
  `hybrid_fair_lpm_res64_steps1`, run
  local_workspaces/rtx5080/runs/20260709-234511, ERS 0.524547.
- Keep `two_policy_proxy.py` only as a research harness.
- Next two-policy attempt must move closer to the engine scheduler or use
  backend-native priority/admission hooks without front-proxy queueing.
```

### Ready Round 006g - vLLM Hybrid-Aware Bake-Off

```text
Status: completed
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006g_vllm_hybrid_bakeoff
Summary: outputs/vllm_hybrid_summary.latest.md

Why: re-evaluate the exact BTC vLLM 0.22.1 image after discovering that
Qwen3.5-2B is a hybrid GDN/Mamba plus full-attention architecture.

Best vLLM result:
- candidate vllm_b_language_model_only
- run local_workspaces/rtx5080/runs/20260710-112657-450574687-65716
- ERS 0.503230
- success 120/120
- TTFT<=1.5s 4/120
- TTFT p50/p95 16918.594 / 31162.146 ms
- TPOT p95 10.001 ms
- wall tokens/s 396.996

Key capability findings:
- Native Qwen3_5ForConditionalGeneration loaded successfully; no Transformers fallback.
- vLLM used its V1 engine and experimental Mamba prefix cache align mode.
- language-model-only reduced model load memory 4.25 -> 3.63 GiB and improved TPOT/throughput.
- Mamba cache all is unsupported for this model and falls back to align.
- FlashInfer GDN prefill is unavailable on RTX 5080 and falls back to Triton/FLA.
- Concurrent partial prefill is unsupported and fails startup.
- run6 relaxed admission improved TTFT p50/p95 to 13519.590 / 21791.069 ms
  and removed >=30s tail, but TTFT<=1.5s fell to 2/120 and ERS was only 0.501262.

Decision:
- Close vLLM as a primary-framework challenger for this local trace.
- Keep vllm_b_language_model_only as an operational fallback for the exact BTC image.
- SGLang hybrid_fair_lpm_res64_steps1 remains current best at ERS 0.524547.
- Continue engine-native two-policy work in SGLang.
```

### Ready Round 008a - Engine-Native Scheduler Telemetry

```text
Status: completed; telemetry accepted
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_008_engine_native_ers_scheduler/subround_008a_scheduler_telemetry
Summary: outputs/telemetry_analysis.latest.md

Why:
- Ready 007 proved that HTTP-layer two-policy queueing is too late in the
  request path.
- Ready 006f exhausted the useful broad SGLang flag sweep.
- This subround needed engine-level labels before changing native scheduling.

Source contract:
- SGLang 0.5.14.
- Image digest
  lmsysorg/sglang@sha256:5027e95bf6ec536856b1b52a91d1f35ff5c564ab83e8a94758a169ff09bb8df3.
- Hooks wrap native Scheduler, SchedulePolicy and MambaRadixCache methods.
- The overlay is opt-in through sitecustomize/import hooks so scheduler spawn
  processes receive it without modifying the image or queue policy.

A/B result:
- control_native: run local_workspaces/rtx5080/runs/20260710-125934-078367875-175689,
  ERS 0.512927, success 120/120, TTFT<=1.5s 23, TTFT p50/p95
  4182.815 / 37873.315 ms, TPOT p95 19.392 ms.
- instrumented_native: run
  local_workspaces/rtx5080/runs/20260710-130137-147122260-179160,
  ERS 0.511176, success 120/120, TTFT<=1.5s 24, TTFT p50/p95
  4223.352 / 37162.208 ms, TPOT p95 17.918 ms.
- ERS overhead delta: -0.001751, inside the accepted -0.005 budget.

Telemetry completeness:
- 7147 valid timed events and 0 malformed JSONL lines.
- 120 unique queued, 120 first-token and 120 finished requests.
- Runtime cache is MambaRadixCache.
- 237 prefill batches, 2800 decode batches and no mixed batch.
- Mean prefill/decode batch size: 1.447 / 8.571.
- Queue depth max/mean: 56 / 26.251.
- Scheduler TTFT<=1.5s: 26/120; p50/p95 4003.140 / 36616.804 ms.
- First-prefill prefix match ratio mean/p95: 0.7767 / 0.8930.
- Cache events: 120 finished commits, 343 unfinished chunk commits,
  51 evictions, 114461 full-KV tokens evicted, 0 Mamba states evicted.
- No request retraction occurred.

Scheduler cost:
- Full prefill scheduling path mean/p50/p95:
  73.910 / 77.781 / 90.429 ms.
- Decode scheduling path mean/p50/p95:
  0.212 / 0.207 / 0.266 ms.
- LPM policy calculation mean/p95: 2.134 / 4.334 ms.
- The prefill measurement includes native cache/chunk bookkeeping and possible
  tensor synchronization; it is not pure Python sorting time.

Technical answer:
- Current behavior is a wave scheduler: small prefill batches feed dense
  decode waves, then waiting requests wait for another prefill window.
- Prefix reuse is strong, so replacing LPM with FCFS/LOF remains wrong.
- The first engine-native policy should rescue only high-prefix requests that
  can plausibly cross the 1500ms TTFT boundary while protecting TPOT.
- Global mixed chunking is already negative from Ready 006f; use a conditional
  rescue lane instead.

Decision:
- Accept telemetry as a reusable measurement layer.
- Do not change the current provisional best Ready 006e config.
- Phase B: static conditional prefix/deadline rescue budget in the native
  scheduler, one rescue at a time, with TPOT p95 guard <=20ms.
- Do not add XGBoost/C++ until static control beats the baseline and telemetry
  supplies repeated training labels.
```

### Ready Round 008b - Shadow Conditional Prefix Rescue

```text
Status: completed; static policy rejected
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_008_engine_native_ers_scheduler/subround_008b_shadow_conditional_prefix_rescue
Summary: outputs/live_shadow_analysis.latest.md

Why:
- Validate a generic high-prefix TTFT rescue rule before granting it native
  scheduling authority.
- Keep native LPM, queue order, batch construction and MambaRadixCache state
  unchanged during all measurements.

Offline calibration:
- 120 requests and 4752 policy observations from accepted Ready 008a telemetry.
- 192 generic static configurations swept without prompt text, response,
  request ID or trace index as policy features.
- Frozen v2 rule: match ratio >=0.75, remaining prefill <=3072 tokens,
  projected TTFT <=1000ms, running requests <=6, rolling decode-result p95
  guard 23.4ms.

Live attempt 1:
- control/shadow ERS 0.520392 / 0.519800, delta -0.000592.
- support 14, precision 0.9286, lift 3.8424x.
- rejected because early support was zero. Decode guard was only 0.059ms
  below observed rolling p95 despite request TPOT p95 18.863ms.

Live attempt 2:
- guard recalibrated from native p97.5 plus 1ms; static gates unchanged.
- control/shadow ERS 0.520507 / 0.510481, delta -0.010026.
- support 12, precision 0.7500, early/late support 6/6.
- rejected for observer overhead. Repeated terminal labels produced 5941
  EXPIRED events and starved the 3-core scheduler path.

Final optimized-observer attempt:
- request labels emit only when action/reason changes; expensive service-cost
  prediction is skipped for terminal/ineligible requests.
- event count fell from 6628 to 210.
- evaluator mean/max CPU 108/643us; profiled total 24.2ms.
- control run local_workspaces/rtx5080/runs/20260710-141021-958105244-261814:
  ERS 0.509993, success 120/120, TPOT p95 19.664ms.
- shadow run local_workspaces/rtx5080/runs/20260710-141228-940615768-265178:
  ERS 0.512351, success 120/120, TPOT p95 19.417ms.
- overhead delta +0.002358 passed, but rescue support collapsed to 1 with
  early/late support 1/0.

Technical answer:
- High-prefix/small-suffix state is predictive when the cohort exists, but a
  frozen one-run threshold is not stable under native LPM wave/cache timing.
- Shadow instrumentation can be made cheap enough for 3 CPU cores, but that
  does not make its policy causally useful.
- Static correlation did not prove scheduler control authority.

Decision:
- Reject the frozen static rescue policy and do not open active 008c.
- Keep the optimized telemetry/shadow/replay tools for multi-run dataset work.
- Keep Ready 006e as the provisional local serving best.
- Keep XGBoost/C++ disabled. The next policy study must use multiple native
  runs with leave-one-run-out validation before any active queue mutation.
```

### Ready Round 008c - Adaptive Threshold Shadow

```text
Status: completed; adaptive shadow accepted
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_008_engine_native_ers_scheduler/subround_008c_adaptive_threshold_shadow
Replay summary: outputs/adaptive_replay.latest.md
Live summary: outputs/live_adaptive_analysis.latest.md

Why:
- Replace Ready 008b's brittle frozen thresholds with online calibration.
- Preserve one SGLang model, native LPM and one shared KV/Mamba state pool.
- Validate prediction and overhead before any active scheduler mutation.

Algorithm:
- Ignore six observed pre-benchmark startup/warmup outcomes.
- Bootstrap thresholds from bounded waiting-queue quantiles.
- Join generic pre-first-token snapshots to native first-token outcomes.
- Recalibrate every four labeled outcomes over a rolling 48-request window.
- Search support/precision/lift/old-new-split objective.
- Limit one update to match 0.05, suffix 512 tokens, projected TTFT 100ms and
  running-request limit 1.
- Compare recent decode pressure with an older baseline window.
- Never use request ID, trace row, prompt text or response as policy features.

Cross-run replay:
- Ready 008a primary: support/precision/lift 16/0.8750/4.0385x.
- Ready 008b v1 primary: 26/0.9231/3.8196x.
- Ready 008b v2 optimized primary: 24/0.7917/3.8000x.
- Ready 008b verbose stress: 32/0.5938/2.9688x.
- Every timeline had 120 labels/outcomes, early and late coverage, at least 11
  supervised updates on primary runs and no forbidden feature.

Live A/B:
- control run local_workspaces/rtx5080/runs/20260710-152043-342611197-340267:
  ERS 0.512949, success 120/120, TTFT<=1.5s 19, TPOT p95 18.383ms.
- adaptive run local_workspaces/rtx5080/runs/20260710-152248-479638027-340267:
  ERS 0.512680, success 120/120, TTFT<=1.5s 20, TPOT p95 18.686ms.
- observer ERS delta -0.000269.
- live support/precision/lift 27/0.6667/3.3333x.
- early/late support 15/12; 120 labels and 120 outcomes.
- evaluate CPU mean 183us; supervised calibration mean 3.59ms.

Convergence:
- minimum match ratio 0.50;
- maximum remaining suffix 2944 tokens;
- projected TTFT target 800ms;
- maximum running requests 5.

Decision:
- Accept adaptive shadow and permit design of one conservative active probe.
- Do not claim an ERS gain: shadow never changed scheduling.
- Active probe must preserve native LPM by default, move at most one request,
  retain decode guard and expose an immediate kill switch.
- Ready 006e remains the serving baseline; XGBoost/C++ remains disabled until
  the simple active controller proves causal value.
```

### Ready Round 008d - Conservative Active Rescue

```text
Status: completed; active probe rejected
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_008_engine_native_ers_scheduler/subround_008d_conservative_active_rescue
Replay summary: outputs/active_opportunity_replay.latest.md
Live summary: outputs/live_active_analysis.latest.md
Score decomposition: outputs/paired_score_delta.latest.md

Why:
- Test causal value of the adaptive cohort accepted in Ready 008c.
- Keep native LPM as the default and change one request at most.
- Compare against an adaptive-shadow control with identical observer cost.

Safety implementation:
- Environment and runtime-file kill switches.
- Minimum 12 supervised outcomes before active authority.
- Native position window 1..8, one move per request, maximum eight moves.
- 150ms cooldown and adaptive decode guard.
- Native LPM executes first; all non-candidate requests preserve relative order.
- Any planner/enforcement failure falls back to the already-computed native order.

Offline replay:
- Ready 008a / 008b v1 / 008b optimized primary timelines planned 3 / 8 / 6
  unique interventions; verbose stress planned 8.
- All positions were inside 1..8 and no prebenchmark intervention occurred.
- Decision: accepted for one live active A/B; replay did not claim latency gain.

Live A/B:
- adaptive_shadow_control run
  local_workspaces/rtx5080/runs/20260710-162631-988525749-418467:
  ERS 0.514307, success 120/120, TTFT<=1.5s 21, TTFT p50/p95
  4308.092 / 38513.853ms, TPOT p95 18.353ms.
- active_one_rescue run
  local_workspaces/rtx5080/runs/20260710-162836-677120032-418467:
  ERS 0.512578, success 120/120, TTFT<=1.5s 23, TTFT p50/p95
  4447.015 / 40489.134ms, TPOT p95 19.430ms.
- Active delta: ERS -0.001728, fast bucket +2, TTFT p50 +138.923ms,
  TTFT p95 +1975.281ms and wall throughput -24.359 tok/s.
- Six unique interventions committed at native positions [3,1,1,2,1,1].
- Applied-request TTFT<=1.5s precision was 0.8333; no policy error,
  malformed telemetry or forbidden feature occurred.

Score decomposition:
- TTFT component delta -0.002143.
- TPOT component delta +0.000415.
- Request scores improved / worsened / unchanged: 10 / 19 / 91.

Technical answer:
- The controller identified fast candidates correctly, but global-front
  promotion crossed LPM locality boundaries.
- More neighboring requests lost TTFT score than selected requests gained.
- The first move also used a bootstrap-loosened suffix threshold and narrowly
  missed the ceiling at 1549ms.

Decision:
- Reject global-front one-rescue and do not open Ready 008e repeatability.
- Keep Ready 008c as diagnostic shadow and Ready 006e as serving baseline.
- Before another active attempt, add a locality-preserving cache-node or
  equal-prefix cohort boundary and freeze active authority until thresholds
  are no looser than the accepted 008c convergence.
- Keep XGBoost/C++ disabled.
```

### Ready Subround 006j - Native Tree-Gated V2

```text
Status: completed; accepted for shadow research only
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006j_native_tree_gated_v2
Summary: outputs/tree_gated_v2.latest.md

Why:
- Replace Brain1's circular heuristic RAM/VRAM target and constant GPQA
  priority with observed SGLang scheduler outcomes.
- Measure the actual native scorer and whole-queue top-K overhead before any
  SGLang integration.
- Replay bounded aging/context rejuvenation without moving real state.

Evidence:
- Dataset full/train rows: 54,729 / 11,574 from 11 timelines and 6 grouped
  sessions; no raw prompts or responses are stored.
- C++ model test/actionable AUC: 0.910670 / 0.900176.
- Actionable score standard deviation: 0.137673.
- Native / ctypes / whole-queue p99: 0.088 / 0.454 / 23.867us.
- Observed native SGLang policy calculation p99: 5409.762us.
- Prediction and top-K parity: exact.

Rejuvenation:
- Gentle locality policy rescued 161 placements, retained 100% positive
  coverage and reduced selected age p50 by about 517ms.
- It did not improve all metrics: precision fell 0.000155 absolute, prefix
  ratio fell 0.001168 and unselected age p95 rose about 17.5ms.
- Strong soft-aging and forced-deadline variants damaged precision/locality.

Decision:
- Native scoring cost passes the research budget.
- Keep the model and gentle rejuvenation shadow-only. No baseline/config or
  Portal candidate changes.
- The learned model uses only four effective features and mostly compresses
  native LPM position, remaining prefill and pressure. Compare it against an
  equally cheap explicit heuristic before granting learned-policy authority.
- Real RAM/VRAM promotion requires a valid KV plus Mamba-state handle and live
  causal ERS evidence; replay labels are not physical migration.
```

### Ready Subround 006k - Cache-Local Tree Versus Heuristic

```text
Status: completed; active exact-cache-node policy rejected
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006k_cache_local_tree_vs_heuristic
Offline summary: outputs/offline_comparison.md
Live summary: outputs/live_ab_analysis.latest.md

Offline:
- Validation selected the explicit four-feature heuristic.
- Validation AUC Tree/heuristic: 0.953414 / 0.974930.
- Actionable test AUC Tree/heuristic: 0.900176 / 0.916530.
- Test top-K selected-set agreement: 0.864407.

Live read-only A/B:
- Control/shadow ERS: 0.524547 / 0.518136; delta -0.006411.
- Success: 120/120 each.
- TTFT<=1.5s: 25 -> 20; TPOT p95: 16.982 -> 16.731ms.
- 263 exact non-root cache-node cohorts, 60 unique node tags.
- Tree/heuristic would-reorder counts: 0 / 0.
- Evaluator p99 333.063us; full wrapper max call 378.180us.
- Policy errors/order changes/rescues/mutations: 0/0/0/0.

Technical answer:
- Exact `last_node.id` is a valid opaque locality boundary and exposes no
  prompt/token data, but it is too strict for the current four-feature rankers.
- Both Tree and heuristic merely selected the native LPM head in every cohort.
- Live AUC was based on only 2 positive outcomes among 57 and is not evidence
  for active authority.
- The observer/performance gates failed, but optimizing overhead cannot create
  policy support where the candidate action set is empty.

Decision:
- Reject active Tree/heuristic scheduling inside exact-node cohorts.
- Preserve tools and data for research; do not alter current baseline/Portal.
- Any next learned scheduler must introduce new pre-decision information and
  validate a broader locality-preserving cohort offline before live mutation.
```

### Ready Subround 006l - Parent-Cache Non-Position Replay

```text
Status: completed; structural cohort and non-position policy rejected
Folder: local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006l_parent_cache_nonposition_replay
Summary: outputs/parent_cache_replay.md

Scope:
- Offline-only replay of accepted 006k telemetry; zero GPU/server runs.
- Reconstructed 146 cache-node parent edges with zero conflicting parents.
- Policy features excluded native position, request ID, trace index, prompt
  text, response text and future outcome.

Topology answer:
- Direct-parent same-key groups: 276; distinct current nodes/branches: 0/0.
- Fixed-hop ancestor groups at hops 2/3/4/5: 248/217/49/11; every distinct
  current-node and branch count was zero.
- Depth-normalized ancestor groups at depths 3/4/5: 165/164/160; every
  distinct-node count was zero.
- Current last-node ancestry therefore cannot form a broader cache-local
  action cohort on this trace.

Exact-node no-position diagnostic:
- Best validation policy: service_only.
- Cohorts/changes: 216/126; unique changed selected requests: 4.
- Native/candidate positive capture: 1.0/0.0.
- Support wins/losses: 0/6.
- Positive precision delta: -0.027778.
- Observed TTFT-component delta: -0.002374.
- Held-out cohorts/changes: 172/81, but no held-out positive cohort existed.

Decision:
- Reject parent/ancestor expansion and static no-position exact-node ranking.
- Do not run a live shadow/active candidate from 006l.
- Keep native LPM, Ready 006e baseline and Portal artifacts unchanged.
- Future work needs temporal/causal state or resource-budget optimization,
  not another static ranker over the same cache cohort.
```
## Part 2 R003 Pointer - 2026-07-19

The completed R003 `8192` full-attention-window study is recorded at
`local_workspaces/rtx5080/research_rounds/part2_lfm25/round_003_hybrid_attention_window/`.
It wins the inferred append-only replay but is research-only pending
long-context quality and rules review. It is not a Ready-round submission
candidate yet.
