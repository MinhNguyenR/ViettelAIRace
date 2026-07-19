# Warmup Round Answers

This file is the detailed answer ledger for all warmup rounds.

Update this file immediately after each round finishes.

Also update the quick tracker:

```text
local_workspaces/rtx5080/docs/round_baselines.md
```

Minimum required fields per completed round:

```text
Run folder:
Config:
ERS:
Score if accuracy gate passes:
TTFT p50/p95/p99:
TPOT p50/p95/p99:
Success/error/timeout/zero-token:
GPU/RAM observation:
What happened:
Technical conclusion:
Decision:
Next:
```

## Round 001 - Baseline Contract Replay

```text
Status: completed
Run folder: local_workspaces/rtx5080/runs/20260707-152939
Config: /model mount, vLLM v0.22.1, max_model_len=32768, gpu_memory_utilization=0.65, max_num_seqs=16, cpu=3, ram=8g
ERS: 0.055487335894078
Score if accuracy gate passes: 5.548733589407799
TTFT p50/p95/p99: 39293.1475 / 81471.43095 / 87360.31638 ms
TPOT p50/p95/p99: 52.53402 / 66.84395325 / 69.4248873 ms
Success/error/timeout/zero-token: 120 / 0 / 0 / 0
GPU/RAM observation: GPU memory max 12760 MiB, avg 12639 MiB; GPU util max 100%, avg 80.375%; power max 299.48 W.
What happened: vLLM loaded Qwen/Qwen3.5-2B from /model successfully. /v1/models reported Qwen3.5-2B with max_model_len=32768. Full trace replay completed with 120 successful requests.
Technical conclusion: The BTC-like /model path is valid in the RTX 5080 lab. Startup is non-trivial: engine init took about 105.30 s and server became ready after model load, compile, CUDA graph capture and multimodal warmup. Performance remains dominated by TTFT/queueing; TPOT is also above the official 45 ms ceiling for the median request.
Decision: Mark Round 001 complete. This is a contract-stability baseline, not a performance win.
Next: Run Round 002 MAX_NUM_SEQS sweep, prioritizing 8 and 12 before wider values.
```

## Round 002 - MAX_NUM_SEQS Sweep

```text
Status: completed
Run folders:
04: local_workspaces/rtx5080/runs/20260707-155728
08: local_workspaces/rtx5080/runs/20260707-160209
12: local_workspaces/rtx5080/runs/20260707-160636
16: local_workspaces/rtx5080/runs/20260707-161046
20: local_workspaces/rtx5080/runs/20260707-161513
24: local_workspaces/rtx5080/runs/20260707-161948
32: local_workspaces/rtx5080/runs/20260707-162424
Best config: MAX_NUM_SEQS=4, GPU_MEMORY_UTILIZATION=0.65, MAX_MODEL_LEN=32768, cpu=3, ram=8g
ERS: 0.49998662019716433
Score if accuracy gate passes: 49.99866201971643
TTFT p50/p95/p99: 49884.6015 / 95260.58495 / 100869.97238 ms
TPOT p50/p95/p99: 13.905705 / 19.78444175 / 20.000106 ms
Success/error/timeout/zero-token: 120 / 0 / 0 / 0
GPU/RAM observation: best candidate GPU memory max 12400 MiB. All candidates stayed inside the local 16GB RTX 5080 limit and the 8GB RAM docker limit.
What happened: The sweep tested MAX_NUM_SEQS values 4, 8, 12, 16, 20, 24 and 32. All candidates completed the full trace. Lower concurrency increased queue waiting and TTFT, but greatly reduced decode contention and TPOT.
Technical conclusion: On the current RTX 5080 lab, ERS is dominated by TPOT because TTFT is already far above the official 1500 ms ceiling for all candidates. MAX_NUM_SEQS=4 wins by pushing TPOT close to or below the official 20 ms floor for most requests. Increasing MAX_NUM_SEQS reduces TTFT somewhat but makes TPOT degrade fast enough to lose much more ERS.
Decision: Promote MAX_NUM_SEQS=4 as the current local decision baseline. Keep MAX_NUM_SEQS=8 as a fallback because it has lower TTFT than 4 and may become relevant after prefill/queue improvements.
Next: Run Round 003 GPU_MEMORY_UTILIZATION sweep using MAX_NUM_SEQS=4, then revisit prefill/TTFT-focused rounds.
```

## Round 003 - GPU Memory Utilization Sweep

```text
Status: completed
Run folders:
0.58: local_workspaces/rtx5080/runs/20260707-190523
0.62: local_workspaces/rtx5080/runs/20260707-190945
0.65: local_workspaces/rtx5080/runs/20260707-191407
0.68: local_workspaces/rtx5080/runs/20260707-191819
0.72: local_workspaces/rtx5080/runs/20260707-192232
Best config: GPU_MEMORY_UTILIZATION=0.72, MAX_NUM_SEQS=4, MAX_MODEL_LEN=32768, cpu=3, ram=8g
ERS: 0.5
Score if accuracy gate passes: 50.0
TTFT p50/p95/p99: 38655.347 / 82054.69825 / 87662.08744 ms
TPOT p50/p95/p99: 10.02582 / 19.67941375 / 19.73705065 ms
Success/error/timeout/zero-token: 120 / 0 / 0 / 0
GPU/RAM observation: best candidate GPU memory max 13359 MiB, avg 13196.625 MiB. This stays under the local 16GB RTX 5080 limit and should be inside the official 18GB H200 MiG budget, subject to runtime differences.
What happened: The sweep tested GPU_MEMORY_UTILIZATION values 0.58, 0.62, 0.65, 0.68 and 0.72 while keeping MAX_NUM_SEQS=4. All candidates completed the full trace. 0.58, 0.62, 0.65 and 0.72 all reached ERS 0.5; 0.72 had the best tie-break profile with lower TTFT, lower TPOT p50 and higher wall tokens/s.
Technical conclusion: More memory is useful after concurrency is controlled, but only up to the point where TPOT is fully scored. Local ERS is now capped at 0.5 because TTFT remains far above the official 1500 ms ceiling and contributes zero score.
Decision: Promote GPU_MEMORY_UTILIZATION=0.72 as the current local config baseline. Do not spend more warmup time only improving TPOT; prioritize TTFT/prefill/queueing next.
Next: Run Round 004 max batched tokens using MAX_NUM_SEQS=4 and GPU_MEMORY_UTILIZATION=0.72 as the current comparison baseline.
```

## Round 004 - Max Batched Tokens

```text
Status: completed
Run folders:
default: local_workspaces/rtx5080/runs/20260707-194404
8192: local_workspaces/rtx5080/runs/20260707-194826
16384: local_workspaces/rtx5080/runs/20260707-195253
24576: local_workspaces/rtx5080/runs/20260707-195718
32768: local_workspaces/rtx5080/runs/20260707-200210
49152: local_workspaces/rtx5080/runs/20260707-200734
Best config inside this round: MAX_NUM_BATCHED_TOKENS=16384, GPU_MEMORY_UTILIZATION=0.72, MAX_NUM_SEQS=4, MAX_MODEL_LEN=32768, cpu=3, ram=8g
ERS: 0.5
Score if accuracy gate passes: 50.0
TTFT p50/p95/p99: 40307.049 / 80776.9483 / 86523.88984 ms
TPOT p50/p95/p99: 8.241745 / 16.00821325 / 16.20319795 ms
Success/error/timeout/zero-token: 120 / 0 / 0 / 0
GPU/RAM observation: best round candidate GPU memory max 15283 MiB, close to the local 16GB RTX 5080 limit. 24576, 32768 and 49152 also used roughly 15.5GB and had worse throughput.
What happened: The sweep tested default, 8192, 16384, 24576, 32768 and 49152. All candidates completed the full trace. 16384 reached ERS 0.5 with the best TPOT profile, but TTFT p50 was worse than the current Round 003 baseline. Default had the best TTFT p50 in Round 004 but ERS was 0.499731 due to TPOT p99 crossing 20 ms.
Technical conclusion: Tuning MAX_NUM_BATCHED_TOKENS alone is not the TTFT fix for this local trace. Too-large values make TTFT and wall throughput much worse. 16384 is useful as a TPOT-biased watchlist config, not as the global local baseline.
Decision: Do not promote Round 004 over Round 003. Keep current global local baseline as GPU_MEMORY_UTILIZATION=0.72, MAX_NUM_SEQS=4, no explicit MAX_NUM_BATCHED_TOKENS.
Next: Run Round 005 chunked prefill, then revisit warmup/JIT if TTFT still dominates.
```

## Round 005 - Chunked Prefill

```text
Status: completed
Run folders:
default: local_workspaces/rtx5080/runs/20260707-202635
explicit_on: local_workspaces/rtx5080/runs/20260707-204049
Not-ready candidates: no_chunked, partial2_long1, partial4_long1, partial4_long2
Best config: ENABLE_CHUNKED_PREFILL=true, GPU_MEMORY_UTILIZATION=0.72, MAX_NUM_SEQS=4, MAX_MODEL_LEN=32768, cpu=3, ram=8g
ERS: 0.5
Score if accuracy gate passes: 50.0
TTFT p50/p95/p99: 37711.228 / 81008.25135 / 86573.74042 ms
TPOT p50/p95/p99: 9.8382825 / 19.72278175 / 19.76110045 ms
Success/error/timeout/zero-token: 120 / 0 / 0 / 0
GPU/RAM observation: explicit_on GPU memory max 13059 MiB, lower than Round 003 best 13359 MiB. Concurrent partial-prefill candidates did not become ready.
What happened: default chunked behavior and explicit-on both served the trace. explicit_on kept ERS at 0.5 and improved TTFT tail/TPOT tail versus Round 003. no_chunked and max_num_partial_prefills variants failed readiness within 10 minutes.
Technical conclusion: Chunked prefill must stay enabled for this model/workload. Explicitly enabling it is safer than relying on default. Concurrent partial-prefill knobs are not viable under the current local resource limits.
Decision: Promote explicit chunked prefill into the current local baseline and local compose. Do not carry partial-prefill concurrency forward.
Next: Run Round 006 prefix cache analysis or Round 012 warmup/JIT to continue targeting TTFT.
```

## Round 005a - Synthetic GPU Occupancy Ramp Subround

```text
Status: completed
Run folders:
baseline_repeat: local_workspaces/rtx5080/runs/20260707-214204
torch_ramp_5s: local_workspaces/rtx5080/runs/20260707-214615
torch_ramp_10s: local_workspaces/rtx5080/runs/20260707-215032
Best subround config: torch BF16 matmul ramp for 10 seconds before vLLM server startup, then Round 005 baseline flags.
ERS: 0.5
Score if accuracy gate passes: 50.0
TTFT p50/p95/p99: 36720.108 / 79852.61455 / 85461.35852 ms
TPOT p50/p95/p99: 9.5829525 / 19.4611985 / 19.84801735 ms
Success/error/timeout/zero-token: 120 / 0 / 0 / 0
GPU/RAM observation: best candidate GPU memory max 13518 MiB, near the same-session control 13559 MiB and above the original Round 005 explicit_on run 13059 MiB. All candidates stayed inside local RTX 5080 16GB and Docker 8GB RAM constraints.
What happened: baseline_repeat, torch_ramp_5s and torch_ramp_10s all completed. The 10-second ramp improved same-session TTFT p50/p95/p99 versus control and also improved TPOT p50/p95 while keeping TPOT p99 under 20 ms.
Technical conclusion: Synthetic occupancy ramp is a promising TTFT/startup stabilization hypothesis. It should be treated as a warmup technique, not as power/clock control. The signal needs a repeat because startup and trace timing can be noisy.
Decision: Do not promote as global baseline yet. Add torch_ramp_10s to the TTFT watchlist and repeat it before any submission-facing wrapper is built.
Next: run a small repeat A/B: baseline_repeat vs torch_ramp_10s with docker log capture, then test endpoint warmup prompts if the signal repeats.
```

## Round 005a Extended - Synthetic Ramp Intensity Sweep

```text
Status: completed
Run folders:
baseline_repeat: local_workspaces/rtx5080/runs/20260707-220306
torch_ramp_2s: local_workspaces/rtx5080/runs/20260707-220721
torch_ramp_10s_repeat: local_workspaces/rtx5080/runs/20260707-221139
torch_ramp_15s: local_workspaces/rtx5080/runs/20260707-221603
torch_ramp_heavy_10s: local_workspaces/rtx5080/runs/20260707-222022
Best extended config: torch_ramp_heavy_10s, SYNTHETIC_RAMP_SECONDS=10, START_N=6144, END_N=2048, STEPS=5, SYNC_EVERY=2
ERS: 0.4999949098888375
Score if accuracy gate passes: 49.99949098888375
TTFT p50/p95/p99: 37771.7815 / 81368.66895 / 87015.48359 ms
TPOT p50/p95/p99: 9.8974875 / 19.76824375 / 19.95766825 ms
Success/error/timeout/zero-token: 120 / 0 / 0 / 0
GPU/RAM observation: best extended candidate GPU memory max 13901 MiB. Extended baseline used 12926 MiB, so ramp candidates cost roughly 0.9-1.1GB more observed serving memory in this session.
What happened: baseline_repeat was much slower than the earlier control, confirming high startup variance. 2s and heavy 10s recovered TTFT/TPOT strongly versus same-session baseline. 10s repeat and 15s were worse, so ramp benefit is not monotonic with duration.
Technical conclusion: Synthetic ramp is a useful TTFT/startup stabilization lever, but config choice matters. The original 10s config has the best absolute result, while heavy 10s is the best extended-sweep stability candidate because TPOT p99 stays under 20 ms.
Decision: Do not promote globally yet. Keep `torch_ramp_10s` and `torch_ramp_heavy_10s` as important TTFT watchlist configs. Technical baseline and configs are recorded in `round_005a_synthetic_occupancy_ramp/technical_note.md`.
Next: repeat only `baseline_repeat`, `torch_ramp_2s`, and `torch_ramp_heavy_10s`; then test endpoint/vLLM prompt warmup behind a healthcheck gate.
```

## Round 005b - Synthetic Ramp + Batched Tokens

```text
Status: completed
Run folders:
ramp2s_base: local_workspaces/rtx5080/runs/20260707-223853
ramp2s_batched16384: local_workspaces/rtx5080/runs/20260707-224306
heavy10_base corrected: local_workspaces/rtx5080/runs/20260707-225648
heavy10_batched16384 corrected: local_workspaces/rtx5080/runs/20260707-230107
heavy2s_base: local_workspaces/rtx5080/runs/20260707-231252
heavy2s_batched16384: local_workspaces/rtx5080/runs/20260707-231704
Invalid/superseded:
heavy10_base initial: local_workspaces/rtx5080/runs/20260707-224721, invalid due MAX_NUM_BATCHED_TOKENS leak
heavy10_batched16384 initial: local_workspaces/rtx5080/runs/20260707-225134, superseded by corrected pair
Best overall config: ramp2s_base, SYNTHETIC_RAMP_SECONDS=2, START_N=3072, END_N=1024, no explicit MAX_NUM_BATCHED_TOKENS
Best TPOT-safe combo inside 005b only: heavy2s_batched16384, SYNTHETIC_RAMP_SECONDS=2, START_N=6144, END_N=2048, MAX_NUM_BATCHED_TOKENS=16384
ERS best overall: 0.5
Score if accuracy gate passes: 50.0
Best overall TTFT p50/p95/p99: 37716.1155 / 81793.6352 / 87386.88347 ms
Best overall TPOT p50/p95/p99: 9.91622 / 19.82607825 / 19.8656681 ms
TPOT-safe combo TTFT p50/p95/p99: 39524.6715 / 78690.2476 / 84317.13289 ms
TPOT-safe combo TPOT p50/p95/p99: 8.0768725 / 15.570625 / 15.7746871 ms
Success/error/timeout/zero-token: all valid candidates 120 / 0 / 0 / 0
GPU/RAM observation: startup max reached GPU 100% and power 321.84-354.92 W. Benchmark GPU avg stayed around 78.7-80.0% because serving is not always compute-saturated. MAX_NUM_BATCHED_TOKENS=16384 increased VRAM significantly, e.g. ramp2s 13481 -> 15358 MiB and heavy2s_batched16384 reached 15505 MiB.
What happened: Adding 16384 is a strong TPOT rescue lever, but it worsens TTFT p50 and VRAM. ramp2s_base already kept TPOT p99 under 20 ms in this run, so the combo was not needed for best overall score. The heavy2s follow-up showed heavy2s_base is a valid tail-biased resource-orchestration variant: it improved TTFT p95/p99 and TPOT p50 slightly, while paying TTFT p50 and VRAM. heavy2s_batched16384 is the best TPOT-safe 005b combo so far.
Technical conclusion: This is a new combo technique: ramp for startup/TTFT resource orchestration, 16384 for TPOT rescue. It should be applied conditionally, not by default. In 005b, heavy2s_batched16384 looked better than heavy10_batched16384 because it kept the shorter 2s startup shape and slightly improved TPOT/tail TTFT, but Round 005c later showed this was not stable enough to promote.
Decision: Keep ramp2s_base as best 005b overall watchlist. Keep heavy2s_batched16384 as a TPOT-safe hypothesis from 005b only; it must be revalidated because Round 005c later showed a negative repeat. Do not promote globally yet.
Next: repeat Round 005 baseline vs ramp2s_base. Use 16384 only if a candidate's TPOT tail rises above 20 ms.
```

## Round 005c - Ramp Stability Repeat

```text
Status: completed
Run folders:
round005_baseline: local_workspaces/rtx5080/runs/20260707-233726
ramp2s_base: local_workspaces/rtx5080/runs/20260707-234135
heavy2s_base: local_workspaces/rtx5080/runs/20260707-234540
heavy2s_batched16384: local_workspaces/rtx5080/runs/20260707-234953
Best same-session config: ramp2s_base, SYNTHETIC_RAMP_SECONDS=2, START_N=3072, END_N=1024, no explicit MAX_NUM_BATCHED_TOKENS
ERS best: 0.5
Score if accuracy gate passes: 50.0
Best TTFT p50/p95/p99: 37838.209 / 79865.15485 / 85332.10361 ms
Best TPOT p50/p95/p99: 10.03989 / 19.29603775 / 19.58641895 ms
Success/error/timeout/zero-token: all candidates 120 / 0 / 0 / 0
GPU/RAM observation: ramp2s_base startup power max 330.61 W, VRAM max 13682 MiB. heavy2s_base startup power max 342.66 W but did not improve tail. heavy2s_batched16384 used 15483 MiB VRAM.
What happened: Same-session repeat favored the light 2s ramp. round005_baseline lost ERS because TPOT p99 was 20.47 ms. ramp2s_base restored ERS to 0.5 and improved TTFT p95/p99 versus baseline. heavy2s_base did not repeat its 005b tail advantage and had TPOT p99 20.41 ms. heavy2s_batched16384 was a negative repeat: TPOT p50 improved to 8.73 ms, but TPOT p99 worsened to 23.53 ms and ERS dropped to 0.482821.
Technical conclusion: Resource orchestration still looks useful, but the stable form is light ramp2s, not heavy2s. `MAX_NUM_BATCHED_TOKENS=16384` is not a reliable TPOT guard in the heavy2s combo; it can improve TPOT p50 while hurting TPOT tail badly.
Decision: Keep ramp2s_base as the strongest repeated ramp watchlist candidate. Downgrade heavy2s_base to experimental tail-ramp watchlist. Downgrade heavy2s_batched16384 from TPOT-safe fallback to unstable watchlist. Do not promote any ramp to global baseline yet.
Next: Run Round 006 prefix-cache / prompt-overlap analysis before creating more ramp variants.
```

## Round 006 - Prefix Cache Analysis

```text
Status: completed
Prefix overlap: strong_prefix_cache_candidate
Run folders:
default/no flag: local_workspaces/rtx5080/runs/20260708-010415
explicit on: local_workspaces/rtx5080/runs/20260708-010904
explicit off: local_workspaces/rtx5080/runs/20260708-011310
Best config: ENABLE_PREFIX_CACHING=true, ENABLE_CHUNKED_PREFILL=true, GPU_MEMORY_UTILIZATION=0.72, MAX_NUM_SEQS=4, MAX_MODEL_LEN=32768, cpu=3, ram=8g
ERS:
default/no flag: 0.473541172225333
explicit on: 0.5
explicit off: 0.48469975919703284
Score if accuracy gate passes:
default/no flag: 47.3541172225333
explicit on: 50.0
explicit off: 48.46997591970329
TTFT p50/p95/p99:
default/no flag: 63676.7575 / 122216.30045 / 129201.33677 ms
explicit on: 38407.929 / 81091.56095 / 86517.85889 ms
explicit off: 60645.566 / 116282.74235 / 122825.36382 ms
TPOT p50/p95/p99:
default/no flag: 16.96639 / 23.6333445 / 23.97885395 ms
explicit on: 9.9978825 / 19.040615 / 19.17246175 ms
explicit off: 16.42129 / 22.341848 / 22.6878871 ms
Success/error/timeout/zero-token: all candidates 120 / 0 / 0 / 0
GPU/RAM observation: explicit-on used GPU memory max 13058 MiB, lower than default 13228 MiB and explicit-off 13367 MiB in this session. GPU util avg was lower for explicit-on because the trace finished faster; wall tokens/s improved to 209.56 vs 152.17 default and 158.60 off.
What happened: Offline analysis found 0 exact full prompt duplicates, but 120/120 exact duplicate system messages and very large previous-prefix overlap. 119/120 requests had at least 4096 previous prefix tokens, 100/120 had at least 8192 and 60/120 had at least 16384. The Docker sweep confirmed that no-flag/default vLLM behavior was effectively unsafe here; explicit --enable-prefix-caching was much faster and restored ERS to 0.5.
Technical conclusion: Prefix caching is not optional for this trace. It attacks repeated long-prefill work directly and improves TTFT, TPOT and wall throughput. Round 006 does not replace the global Round 005 baseline because Round 005 already used the normal compose path with prefix cache enabled; Round 006 proves the flag must remain explicit in every future vLLM config and submission compose.
Decision: Mark --enable-prefix-caching as mandatory carry-forward. Do not use vLLM no-flag/default behavior unless a round intentionally tests it. Keep current global local baseline as Round 005, but update registry to include explicit prefix caching as a required baseline flag.
Next: Run Round 007 KV cache dtype / FP8 KV cache with prefix caching and chunked prefill explicitly enabled.
```

## Round 007 - KV Cache Dtype

```text
Status: completed
Run folders:
kv_auto_control: local_workspaces/rtx5080/runs/20260708-105935
kv_fp8: local_workspaces/rtx5080/runs/20260708-110321
kv_fp8_e4m3: local_workspaces/rtx5080/runs/20260708-110611
kv_fp8_e4m3_calc: local_workspaces/rtx5080/runs/20260708-110846
kv_fp8_e5m2_calc: local_workspaces/rtx5080/runs/20260708-111206
Best measured config: KV_CACHE_DTYPE=fp8_e4m3, CALCULATE_KV_SCALES=true requested, ENABLE_PREFIX_CACHING=true, ENABLE_CHUNKED_PREFILL=true, GPU_MEMORY_UTILIZATION=0.72, MAX_NUM_SEQS=4, MAX_MODEL_LEN=32768, cpu=3, ram=8g
Important caveat: vLLM logged that calculate_kv_scales is disabled for this hybrid model because recurrent layers make calibration unreliable, so the effective FP8 KV scale is default 1.0.
ERS:
auto control: 0.49989897083397283
fp8: 0.5
fp8_e4m3: 0.5
fp8_e4m3_calc: 0.5
fp8_e5m2_calc: 0.5
Score if accuracy gate passes:
auto control: 49.98989708339728
fp8_e4m3_calc: 50.0
TTFT p50/p95/p99:
auto control: 37489.7365 / 80690.717 / 86269.81319 ms
fp8: 17097.774 / 29514.59775 / 31525.48382 ms
fp8_e4m3: 17196.7785 / 29686.13545 / 31728.24623 ms
fp8_e4m3_calc: 17052.643 / 29474.97195 / 31486.14336 ms
fp8_e5m2_calc: 17218.052 / 32239.9426 / 37777.68691 ms
TPOT p50/p95/p99:
auto control: 9.85686 / 19.7540235 / 20.0762521 ms
fp8: 8.2461025 / 9.5331315 / 9.6129659 ms
fp8_e4m3: 8.3049475 / 9.58167975 / 9.66480005 ms
fp8_e4m3_calc: 8.236585 / 9.53991225 / 9.6675083 ms
fp8_e5m2_calc: 8.289415 / 14.92777475 / 19.14814095 ms
Success/error/timeout/zero-token: all candidates 120 / 0 / 0 / 0
Memory effect:
auto control: available KV 4.94 GiB, GPU KV cache 388325 tokens, max concurrency 11.85x, VRAM max 12738 MiB
fp8: available KV 4.58 GiB, GPU KV cache 661559 tokens, max concurrency 20.19x, VRAM max 12260 MiB
fp8_e4m3: available KV 4.58 GiB, GPU KV cache 661559 tokens, max concurrency 20.19x, VRAM max 12256 MiB
fp8_e4m3_calc: available KV 4.62 GiB, GPU KV cache 666873 tokens, max concurrency 20.35x, VRAM max 12194 MiB
fp8_e5m2_calc: available KV 4.45 GiB, GPU KV cache 641190 tokens, max concurrency 19.57x, VRAM max 12248 MiB
Quality/output risk: local output validity passed, but GPQA accuracy is not tested. FP8 KV cache is a lossy serving optimization; do not mark it official submission-safe until quality is measured. The calculate_kv_scales flag is not a mitigation here because vLLM disables it for this hybrid model.
What happened: FP8 KV cache roughly doubled effective KV token capacity and cut wall duration from 114.34s to about 58.59s in the best run. TTFT p50 dropped from 37.49s to 17.05s and TPOT p95 dropped from 19.75ms to 9.54ms. E5M2 completed but had worse tail latency and throughput than E4M3.
Technical conclusion: FP8 E4M3 KV cache is the strongest local serving-performance improvement so far. The likely mechanism is larger KV/cache block capacity and less memory pressure, which lets prefix-cache/chunked-prefill serving finish the trace with much less queueing and decode contention. However, quality risk remains open.
Decision: Promote FP8 E4M3 KV cache to performance-candidate status. This decision was temporarily downgraded by Round 007a output drift, then partially rescued by Round 007b GPQA proxy. FP8 is now a provisional submission candidate, with auto/BF16 KV as rollback.
Next: Use Round 007b GPQA proxy as the main local quality signal before carrying FP8 forward. Repeat quality checks when changing precision, scheduler behavior, or prompt/eval format materially.
```

## Round 007a - FP8 KV Output Drift Proxy

```text
Status: completed
Subround folder: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_007_kv_cache_dtype/subrounds/round_007a_fp8_kv_output_drift
Outputs:
auto_a: outputs/auto_a_outputs.jsonl
auto_b: outputs/auto_b_outputs.jsonl
fp8_e4m3: outputs/fp8_e4m3_outputs.jsonl
Comparison summary: outputs/drift_summary_20260708-112328.md
Cases: 8 public-trace prompts selected by prompt-length quantiles
Auto repeat drift:
exact matches: 8/8
similarity min/avg/max: 1.0 / 1.0 / 1.0
FP8 vs auto drift:
exact matches: 0/8
similarity min/avg/max: 0.068627 / 0.385726 / 0.817468
low similarity cases <0.98: 8/8
Success/error: all captures completed with 0 request errors
What happened: auto KV was deterministic across two independent server starts. FP8 E4M3 KV completed all requests with the same completion token counts, but generated different text for every sampled prompt.
Technical conclusion: FP8 E4M3 KV cache is a high-performance but high-quality-risk optimization. The output drift is large enough that it cannot be considered submission-safe without GPQA or a stronger local multiple-choice quality check.
Decision: Downgrade FP8 E4M3 from current submission candidate to performance-only/high-risk candidate. Keep auto KV as quality-safe control.
Next: Build a local GPQA-like multiple-choice quality harness if we want to rescue FP8; otherwise continue performance rounds with auto KV or clearly mark FP8-only experiments as quality-risk research.
```

## Round 007b - GPQA Diamond KV Cache Accuracy

```text
Status: completed
Subround folder: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_007_kv_cache_dtype/subrounds/round_007b_gpqa_diamond_accuracy
Dataset: input/quality/private/gpqa_diamond_100.jsonl
Dataset note: deterministic 100-question local proxy from GPQA Diamond public records, seed 20260708. This is not the BTC hidden/fixed subset.
Candidates:
kv_auto_gpqa: KV_CACHE_DTYPE=auto
kv_fp8_e4m3_gpqa: KV_CACHE_DTYPE=fp8_e4m3
Outputs:
comparison: outputs/gpqa_comparison_20260708-115628.md
auto summary: outputs/kv_auto_gpqa_summary.md
fp8 summary: outputs/kv_fp8_e4m3_gpqa_summary.md
Auto KV:
accuracy: 0.38
correct/total: 38/100
ok/errors: 100/0
parsed/parse failures: 100/0
latency mean/p50/p95/p99: 243.17846 / 28.6565 / 45.45355 / 461.89317 ms
FP8 E4M3 KV:
accuracy: 0.34
correct/total: 34/100
ok/errors: 100/0
parsed/parse failures: 100/0
latency mean/p50/p95/p99: 40.99416 / 29.208 / 35.4052 / 257.04686 ms
Comparison:
relative drop auto - fp8: 0.04
changed prediction: 9
changed correctness: 4
both correct: 34
auto correct, fp8 wrong: 4
auto wrong, fp8 correct: 0
both wrong: 62
Official-style proxy:
assumed baseline accuracy: 0.4
auto delta: 0.02
fp8 delta: 0.06
auto penalty: 1.0
fp8 penalty: 1.0
What happened: FP8 E4M3 lost 4 correct answers versus auto on the local GPQA Diamond proxy. It did not recover any case auto got wrong. However, the measured drop is only 0.04, which is below the official no-penalty allowance of delta <= 0.10 if the proxy is representative.
Technical conclusion: The 8-prompt free-form drift from Round 007a was real, but it did not translate into a catastrophic multiple-choice accuracy collapse on this 100-question proxy. FP8 E4M3 KV is no longer blocked by local quality evidence; it remains a provisional candidate because BTC hidden questions and exact evaluator may differ.
Decision: Promote FP8 E4M3 KV to provisional submission candidate. Keep auto KV as rollback/control and rerun GPQA proxy after any major serving or precision change.
Next: Continue performance work from the FP8 E4M3 config, but do not combine it with other lossy optimizations until a paired GPQA check is repeated.
```

## Round 007c - Full Score Replay

```text
Status: completed
Subround folder: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_007_kv_cache_dtype/subrounds/round_007c_full_score_replay
Run folder: local_workspaces/rtx5080/runs/20260708-121041
Candidate: KV_CACHE_DTYPE=fp8_e4m3, GPU_MEMORY_UTILIZATION=0.72, MAX_NUM_SEQS=4, prefix cache on, chunked prefill on
Quality proxy source: Round 007b GPQA Diamond proxy
Quality accuracy proxy: 0.34
Official baseline accuracy: 0.4
Accuracy delta proxy: 0.06
Accuracy penalty proxy: 1.0
ERS: 0.5
Score proxy: 50.0
Target: 95.0
Status vs target: FAIL
Success/error/timeout/zero-token: 120 / 0 / 0 / 0
TTFT p50/p95/p99: 16440.635 / 28315.94975 / 30294.24758 ms
TPOT p50/p95/p99: 8.0753675 / 9.3355985 / 9.36906665 ms
Wall tokens/s: 418.35204530920987
GPU memory max: 13231 MiB
GPU util avg/max: 91.67% / 100%
What happened: Full trace replay repeated the Round 007 performance behavior. TPOT is excellent, accuracy proxy stays in the no-penalty region, and all requests succeed. However, TTFT is still far above the official 1500 ms ceiling for the whole trace.
Technical conclusion: Current candidate is not score-ready. It is TPOT/throughput strong but TTFT-capped: with TTFT score at 0 for every request, the best possible per-request score is approximately 0.5 from the TPOT half. Reaching score >=95 requires a TTFT-focused architecture change, not another TPOT-only tweak.
Decision: Do not submit this candidate yet under the >=95 target. Keep FP8 E4M3 KV as the performance base, but next work must reduce TTFT/queueing below the official ceiling.
Next: Run a TTFT-focused round: admission/scheduling based on trace arrival, prefill splitting, larger/lower concurrency comparison under FP8, or a runtime path that reduces queue TTFT. Do not spend the next round on TPOT unless TTFT is also protected.
```

## Round 007d - TTFT Scheduler Sweep

```text
Status: completed partial sweep / early stopped
Subround folder: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_007_kv_cache_dtype/subrounds/round_007d_ttft_scheduler_sweep
Sweep summary: outputs/sweep_summary_20260708-122850.md
Fixed resource limits: CPU_LIMIT=3, MEM_LIMIT=8g
Fixed base: KV_CACHE_DTYPE=fp8_e4m3, GPU_MEMORY_UTILIZATION=0.72, MAX_MODEL_LEN=32768, prefix cache on, chunked prefill on
Quality proxy: Round 007b accuracy 0.34, penalty 1.0
Candidates run:
seqs_04: MAX_NUM_SEQS=4, run local_workspaces/rtx5080/runs/20260708-123026
seqs_08: MAX_NUM_SEQS=8, run local_workspaces/rtx5080/runs/20260708-123318
seqs_12: MAX_NUM_SEQS=12, run local_workspaces/rtx5080/runs/20260708-123614
Best score:
seqs_04, score proxy 50.0, ERS 0.5
Metrics:
seqs_04: TTFT p50/p95/p99 18810.9795 / 31400.71595 / 33420.80994 ms; TPOT p50/p95/p99 8.37152 / 9.98833775 / 11.8701543 ms; score 50.0
seqs_08: TTFT p50/p95/p99 9813.8335 / 25580.64285 / 30447.37839 ms; TPOT p50/p95/p99 12.2161125 / 38.07663175 / 38.38804 ms; score 44.299415
seqs_12: TTFT p50/p95/p99 6280.0605 / 14946.32745 / 19812.97953 ms; TPOT p50/p95/p99 15.6488725 / 29.919245 / 51.1144617 ms; score 44.942615
What happened: Raising scheduler concurrency reduced TTFT p50 substantially, from 18.8s to 9.8s and then 6.3s. However, TTFT remained far above the official 1.5s ceiling, while TPOT tail worsened enough to reduce ERS below the control.
Technical conclusion: Plain `MAX_NUM_SEQS` increase is not a winning TTFT fix. It trades decode stability for less queueing but does not cross the TTFT scoring threshold. Score stays best at `MAX_NUM_SEQS=4`.
Decision: Stop the rest of the planned 16/20/32 sweep for now. Keep `MAX_NUM_SEQS=4` as current FP8 candidate. Future TTFT work should target prefill/admission/runtime behavior rather than raw scheduler concurrency.
Next: Continue with Round 008 Brain 1 shadow dataset, or create another Round 007 subround focused on prefill/admission simulation rather than increasing concurrency.
```

## Round 008 - Brain 1 Shadow Dataset

```text
Status: completed
Dataset CSV: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_008_brain1_shadow_dataset/outputs/shadow_dataset.csv
Dataset JSONL: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_008_brain1_shadow_dataset/outputs/shadow_dataset.jsonl
Summary: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_008_brain1_shadow_dataset/outputs/shadow_summary.md
Balanced RAM/VRAM CSV: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_008_brain1_shadow_dataset/outputs/shadow_dataset_balanced_ram_vram.csv
Balanced RAM/VRAM JSONL: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_008_brain1_shadow_dataset/outputs/shadow_dataset_balanced_ram_vram.jsonl
Rows: 7080
Runs: 59
Requests: 120
Features: 77 columns
Tree-Gated canonical features: sem_gravity, temp_decay, info_density, neuro_interest, surviv_index
Supporting heuristic features: entity_count_proxy, number_count, code_symbol_count, has_instruction, lexical_diversity, entropy_score, vram_usage_proxy, ram_pressure_proxy, scheduler_pressure
Labels: request_score, ttft_score, tpot_score, ttft_over_ceiling, tpot_over_ceiling, zero_token, is_error, score_proxy_bad, ram_vram_priority_score, target_tier_binary_name
Key counts:
TTFT over ceiling rows: 7080 / 7080
TPOT over ceiling rows: 1014 / 7080
Score proxy bad rows: 1756 / 7080
RAM/VRAM target tiers: RAM 3540, VRAM 3540
Request score p50/p95: 0.5 / 0.5
TTFT p50/p95: 37331.214 / 86499.2373 ms
TPOT p50/p95: 13.710985 / 65.350517 ms
What happened: Built a Brain 1 shadow dataset by joining the official trace with all local RTX 5080 metrics runs. Each row represents one request within one run. It includes arrival/burst features, prompt length features, prefix overlap opportunity, Tree-Gated semantic/system pressure proxies, run metadata, active-at-start estimate and observed TTFT/TPOT/request-score labels. No raw prompt text is stored.
Technical conclusion: The first 57-column dataset was too thin versus the Tree-Gated spec. The enriched 77-column dataset now contains the canonical five Brain 1 signals plus a balanced RAM/VRAM-only target. SSD/offload was removed from the active dataset because we cannot assume useful SSD behavior in the BTC environment. It still confirms that all local runs miss the official TTFT ceiling, so Round 009 should learn relative risk/ranking or RAM/VRAM placement behavior rather than a TTFT-pass classifier.
Decision: Round 008 completed as the data foundation for Brain 1. Do not use it to change serving yet.
Next: Round 009 should train/evaluate a lightweight ranker or XGBoost-style model to predict bad score/TPOT/relative TTFT risk, then Round 010 can simulate admission/scheduling policies.
```

## Round 009 - Brain 1 XGBoost Shadow

```text
Status: completed
Dataset: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_008_brain1_shadow_dataset/outputs/shadow_dataset_balanced_ram_vram.csv
Summary: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_009_brain1_xgboost_shadow/outputs/round009_summary.md
Metrics: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_009_brain1_xgboost_shadow/outputs/round009_metrics.tsv
Primary model path: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_009_brain1_xgboost_shadow/models/request_id.trace_pressure.ram_vram_target.json
Primary model: pure-NumPy boosted-stump fallback, because xgboost/sklearn/pandas are not installed locally
Primary split: request_id
Primary feature set: trace_pressure
Primary target: ram_vram_target
Primary validation metric: test ROC AUC 0.973690, balanced accuracy 0.932974, F1 0.931900
Primary top features: timestamp_ms, vram_usage_proxy, surviv_index
Correlation with bad requests: score_proxy_bad trace_pressure ROC AUC 0.871762; tpot_over_ceiling trace_pressure ROC AUC 0.905334
Best diagnostic config-set: ram_vram_target ROC AUC 0.989514, but it uses fixed config signals such as dtype/max_num_seqs so it is not the preferred production-safe model.
What happened: trained 18 usable boosted-stump shadow models across request_id/run_id group splits, feature sets and targets. Six TTFT-over-ceiling experiments were skipped because every local row exceeds the official TTFT ceiling.
Technical conclusion: Brain 1 can learn the Round 008 RAM/VRAM heuristic target and can rank TPOT/score risk when online pressure features are included. This is a shadow predictor, not proof that real KV migration or scheduler intervention improves ERS.
Decision: Round 009 completed. Use request_id + trace_pressure RAM/VRAM probabilities in Round 010 simulation before changing serving behavior.
Next: Round 010 should replay the trace offline with policy thresholds based on model probability and estimate whether admission/placement changes could improve ERS.
```

### Round 009b - Brain 1 Hyperparameter Sweep

```text
Status: completed
Folder: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_009_brain1_xgboost_shadow/subrounds/round_009b_hyperparameter_sweep
Summary: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_009_brain1_xgboost_shadow/subrounds/round_009b_hyperparameter_sweep/outputs/round009b_summary.md
Sweet spot model: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_009_brain1_xgboost_shadow/subrounds/round_009b_hyperparameter_sweep/models/sweetspot.request_id.trace_pressure.ram_vram_target.json
Broad configs: 48
Deep shortlist evals: 60
Sweet spot config: n60.lr0.200.thr8.leaf24
n_estimators: 60
learning_rate: 0.2
max_thresholds: 8
min_leaf: 24
Request primary ROC AUC / balanced acc / F1: 0.972375 / 0.932974 / 0.931900
Run primary ROC AUC / balanced acc / F1: 0.946675 / 0.909237 / 0.900405
Risk AUC mean: 0.854707
Primary top features: message_count, vram_usage_proxy, surviv_index, entropy_score
Performance-heavy fallback: n60.lr0.200.thr16.leaf24
What happened: ran a two-stage hyperparameter sweep. First, 48 configs were trained on request_id split for RAM/VRAM target. Then the top 10 were evaluated on request_id/run_id splits and risk targets score_proxy_bad/tpot_over_ceiling.
Technical conclusion: Round 009 default was already close, but 009b gives a smaller, more regularized model with the same balanced-accuracy/F1 region and a documented stability check.
Decision: Round 010 should use the 009b sweet spot model first and optionally compare the performance-heavy fallback.
Next: Run Round 010 offline policy simulation.
```

## Round 010 - Brain 1 Admission Simulation

```text
Status: completed
Folder: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_010_brain1_admission_simulation
Summary: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_010_brain1_admission_simulation/outputs/round010_summary.md
Script: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_010_brain1_admission_simulation/scripts/simulate_brain1_policy.py
Model: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_009_brain1_xgboost_shadow/subrounds/round_009b_hyperparameter_sweep/models/sweetspot.request_id.trace_pressure.ram_vram_target.json
Best baseline: safe_seq4 / run 20260708-123026
Baseline ERS / score proxy: 0.500000 / 50.000000
Baseline TTFT p50/p95: 18810.979500 / 31400.715950 ms
Baseline TPOT p50/p95: 8.371520 / 9.988338 ms
Best policy quality: threshold p_vram >= 0.15
Policy balanced accuracy: 0.976190
Policy VRAM precision/recall/F1: 0.975000 / 1.000000 / 0.987342
Best hybrid proxy: top 40% p_vram uses aggressive_seq12, remainder uses fp8_full_score
Hybrid proxy ERS / score proxy: 0.500000 / 50.000000
Hybrid proxy TTFT p50/p95: 8016.931000 / 28315.949750 ms
Hybrid proxy TPOT p50/p95: 9.168675 / 15.872813 ms
Oracle best observed request-score ERS: 0.500000
Oracle min-TTFT replay ERS: 0.450347
Risk: weak counterfactual proxy only. No vLLM serving behavior changed.
What happened: computed p_vram from the Round 009b sweet spot model, tested threshold/top-fraction policies, and replayed hybrid policies using observed local run metrics.
Technical conclusion: Brain 1 identifies request priority well and the best proxy policy cuts TTFT p50 from 18.8s to 8.0s while preserving ERS 0.5. However TTFT is still above the official 1500 ms ceiling and ERS does not improve over baseline.
Decision: Round 010 completed as a negative/guardrail result. Do not submit or implement Brain 1 admission yet.
Next: move to TTFT-first work: real scheduler intervention, prefill/warmup/JIT/runtime change, or a serving experiment where Brain 1 can actually affect request order.
```

## Round 011 - Memory Tiering Simulation

```text
Status: pending
Tier policy:
Simulated behavior:
Memory effect:
What happened:
Technical conclusion:
Decision:
Next:
```

## Round 012 - Warmup and JIT

```text
Status: completed
Primary summary: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_012_warmup_jit/outputs/sweep_summary_20260708-233249.md
Stability fallback summary: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_012_warmup_jit/outputs/sweep_summary_20260708-234237.md
0.95 full-card attempt: failed before measured sweep. vLLM saw only 12.91/15.44 GiB free while 0.95 required 14.67 GiB because desktop/app processes were already using about 2.1GB VRAM.
Near-max 0.83 runs:
- cold_fullvram_fp8: local_workspaces/rtx5080/runs/20260708-233429
- torch_warmup_2s_fullvram_fp8: local_workspaces/rtx5080/runs/20260708-233715
- torch_warmup_10s_fullvram_fp8: local_workspaces/rtx5080/runs/20260708-234003
- endpoint_warmup_short_fullvram_fp8: not_ready because free VRAM dipped to 12.76 GiB while vLLM required 12.81 GiB for 0.83
Stable 0.82 control run: local_workspaces/rtx5080/runs/20260708-234412
Stable 0.82 endpoint warmup run: local_workspaces/rtx5080/runs/20260708-234652
Best measured ERS: endpoint_warmup_short_usablevram82_fp8, ERS 0.5006135105048746, score proxy 50.06135105048746
Best near-max 0.83 TTFT p50: torch_warmup_10s_fullvram_fp8, TTFT p50 17421.6925 ms, but TTFT p95 31319.2015 ms and wall tokens/s 395.4835 were worse than cold 0.83
Cold 0.83 metrics: ERS 0.5, TTFT p50/p95/p99 17673.4095 / 30627.24865 / 32710.97207 ms, TPOT p50/p95/p99 8.4574325 / 9.740635 / 9.84815615 ms, VRAM max 14132 MiB
Endpoint 0.82 metrics: ERS 0.5006135105048746, TTFT p50/p95/p99 17849.3315 / 31575.4577 / 33710.74882 ms, TPOT p50/p95/p99 8.6846975 / 10.096045 / 10.1379647 ms, VRAM max 13957 MiB
Endpoint warmup cost: 3 synthetic streaming requests. TTFTs 938.286 ms, 23.262 ms, 22.321 ms.
Success/error/zero-token: all completed runs 120 / 0 / 0
GPU/RAM observation: all completed runs used Docker cpu=3 and memory=8g. 0.83 used up to 14383 MiB observed VRAM; 0.82 endpoint used 13957 MiB. 0.83 is not stable when desktop VRAM overhead fluctuates.
What happened: raising usable VRAM from 0.72 to 0.83 increased KV capacity and kept TPOT around 8.4-9.8 ms. Pure Torch warmup did not improve full-trace score. Endpoint warmup at 0.82 improved same-session cold control by ERS +0.0006135 and reduced TTFT p50/p95 by about 690 ms.
Technical conclusion: endpoint warmup is a real but small TTFT/JIT-shape optimization. Synthetic Torch occupancy ramp is weaker in this round. The remaining bottleneck is full-trace queue/prefill behavior; warmup alone cannot break the official 1500 ms TTFT ceiling.
Decision: mark Round 012 complete. Do not submit. Carry endpoint warmup forward as a healthcheck-gated warmup candidate. Treat 0.83 as experimental max usable VRAM and 0.82 as stable high-VRAM fallback on this desktop.
Next: run Round 012b endpoint shape warmup with synthetic medium/long prompts and concurrent warmup requests, then proceed to Round 013 only after deciding whether warmup is worth carrying into submission candidates.
```

## Round 012a - High VRAM Sweep Subround

```text
Status: completed
Folder: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_012_warmup_jit/subrounds/round_012a_high_vram_sweep
Summary: local_workspaces/rtx5080/research_rounds/warmup_rounds/round_012_warmup_jit/subrounds/round_012a_high_vram_sweep/outputs/sweep_summary_20260709-000300.md
Candidates: GPU_MEMORY_UTILIZATION=0.82, 0.84, 0.86, 0.88, 0.90
Fixed config: CPU=3, RAM=8g, MAX_MODEL_LEN=32768, MAX_NUM_SEQS=4, prefix caching on, chunked prefill on, KV_CACHE_DTYPE=fp8_e4m3
Best score/latency control: vram82_control, run local_workspaces/rtx5080/runs/20260709-000436
Best capacity candidate: vram90, run local_workspaces/rtx5080/runs/20260709-001453
vram82_control ERS: 0.500000
vram82_control TTFT p50/p95/p99: 16462.7585 / 28407.40815 / 30388.72997 ms
vram82_control TPOT p50/p95/p99: 8.09667 / 9.375602 / 9.41371405 ms
vram82_control KV cache: 6.13 GiB, 883850 tokens, 26.97x max concurrency for 32768-token requests
vram90 ERS: 0.500000
vram90 TTFT p50/p95/p99: 16504.6455 / 28455.75355 / 30437.12442 ms
vram90 TPOT p50/p95/p99: 8.0987875 / 9.40156925 / 9.43272585 ms
vram90 KV cache: 7.36 GiB, 1062745 tokens, 32.43x max concurrency for 32768-token requests
Capacity delta 0.82 -> 0.90: +178895 KV tokens, about +20.24%; 32K max concurrency +5.46x
Success/error/zero-token: all completed candidates 120 / 0 / 0
What happened: higher VRAM allocation clearly increases KV capacity, and 0.90 starts successfully when local desktop VRAM pressure is low enough. But this exact trace with MAX_NUM_SEQS=4 does not convert the extra KV capacity into higher ERS or lower latency; 0.82 remained slightly best on TTFT/TPOT.
Technical conclusion: VRAM higher is useful for capacity, not automatically useful for score. To exploit H200 MiG 18GB, extra VRAM should be paired with scheduler/concurrency/prefill changes that can use the larger KV budget.
Decision: keep 0.90 as H200/capacity watchlist. Keep 0.82 as stable local score control. Do not promote high VRAM alone.
Next: if we continue this branch, test 0.90 plus endpoint warmup and/or MAX_NUM_SEQS 6-8 to see whether extra capacity can reduce TTFT without TPOT collapse.
```

## Round 013 - FP8 Quantization Path

```text
Status: pending
Variant:
Run folder:
ERS:
Memory effect:
Quality/output risk:
What happened:
Technical conclusion:
Decision:
Next:
```

## Round 014 - Weight Quantization

```text
Status: pending
Variant:
Run folder:
ERS:
Accuracy risk:
Memory effect:
What happened:
Technical conclusion:
Decision:
Next:
```

## Round 015 - Speculative Decode

```text
Status: pending
Draft setup:
Run folder:
ERS:
TPOT effect:
Memory effect:
What happened:
Technical conclusion:
Decision:
Next:
```

## Round 016 - SGLang Runtime

```text
Status: pending
Runtime:
Run folder:
ERS:
Compatibility:
What happened:
Technical conclusion:
Decision:
Next:
```

## Round 017 - TensorRT-LLM Feasibility

```text
Status: pending
Feasibility:
Blocking issues:
What happened:
Technical conclusion:
Decision:
Next:
```

## Round 018 - TensorRT-LLM Engine Build

```text
Status: pending
Engine path:
Precision:
Max sequence/context:
Build time:
What happened:
Technical conclusion:
Decision:
Next:
```

## Round 019 - TensorRT-LLM Baseline Benchmark

```text
Status: pending
Run folder:
ERS:
TTFT p50/p95/p99:
TPOT p50/p95/p99:
Success/error/timeout/zero-token:
Compatibility:
What happened:
Technical conclusion:
Decision:
Next:
```

## Round 020 - TensorRT-LLM FP8 / Quantization

```text
Status: pending
Variant:
Run folder:
ERS:
Memory effect:
Accuracy risk:
What happened:
Technical conclusion:
Decision:
Next:
```

## Round 021 - TensorRT-LLM Submission Risk

```text
Status: pending
Startup time:
Image/package risk:
API compatibility:
Recommended action:
What happened:
Technical conclusion:
Decision:
Next:
```

## Round 022 - Submission Candidate Replay

```text
Status: pending
Recommended candidate:
Rollback candidate:
Run folders:
ERS:
What happened:
Technical conclusion:
Decision:
Next:
```
## Part 2 R003 Pointer - 2026-07-19

The completed R003 `8192` full-attention-window study is recorded at
`local_workspaces/rtx5080/research_rounds/part2_lfm25/round_003_hybrid_attention_window/`.
It wins the inferred append-only replay but is research-only pending
long-context quality and rules review. Do not copy its score into legacy
warmup-round baselines.

## Part 2 R003c Pointer - 2026-07-20

Sparse full-attention layer placement is complete at
local_workspaces/rtx5080/research_rounds/part2_lfm25/round_003_hybrid_attention_window/subrounds/subround_003c_sparse_window_layers/.
The early2 placement is a local research reference only: two full replays
averaged 58.0922 versus 56.5450 for control, with no failures, but the short
quality proxy cannot certify the 4096-token semantic boundary.
