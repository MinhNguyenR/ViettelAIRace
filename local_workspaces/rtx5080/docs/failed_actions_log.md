# Failed Actions Log

Purpose: keep a separate memory of actions that failed, regressed metrics, or
were rejected after measurement.

This file is different from `technique_ledger.md`:

- `technique_ledger.md` stores reusable technical lessons and candidate levers.
- `failed_actions_log.md` stores concrete actions to avoid repeating blindly.

## Update Rule

After each round or subround, add an entry here if an action:

- failed to start
- crashed, OOMed, timed out, or returned zero-token/error responses
- reduced ERS or damaged TTFT/TPOT/quality enough to reject
- violated or nearly violated the local resource/official submission contract
- looked promising but produced no measurable win

Each entry should include:

```text
Date:
Round:
Action:
Expected:
Observed:
Root cause / best hypothesis:
Decision:
Do not repeat unless:
Related files:
```

## Entries

### 2026-07-09 - Ready 006 - FlashInfer Linear Attention Backend

Date: 2026-07-09

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/
```

Action:

- Tested `--linear-attn-backend=flashinfer`.
- Then tested `--linear-attn-backend=flashinfer --mamba-ssm-dtype=bfloat16`
  after SGLang reported BF16 SSM was required on SM100+.

Expected:

- Use FlashInfer GDN kernels for Qwen3.5 linear-attention layers and improve
  TTFT/TPOT.

Observed:

```text
hybrid_linear_flashinfer: backend_not_ready
Reason: --linear-attn-decode-backend flashinfer on SM100+ requires --mamba-ssm-dtype bfloat16.

hybrid_linear_flashinfer_bfloat16: backend_not_ready
Reason: prefill CUDA graph capture failed because FlashInfer GDN expected float32 input_state but got bfloat16.
```

Root cause / best hypothesis:

- Current SGLang/FlashInfer path has a dtype-policy conflict for this model on
  local RTX 5080/SM100-class hardware under `tc_piecewise` prefill CUDA graph.
- SGLang requires BF16 SSM to select FlashInfer linear attention, but FlashInfer
  prefill kernel asserts float32 recurrent state during graph capture.

Decision:

- Rejected under current graph policy.

Do not repeat unless:

- Testing a dedicated variant with prefill CUDA graph disabled or changed.
- Or a newer SGLang/FlashInfer image fixes the dtype conflict.
- Then rerun GPQA/output-drift if the candidate starts.

Related files:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/outputs/hybrid_linear_flashinfer_hybrid_audit_20260709-213647.log
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/outputs/hybrid_linear_flashinfer_bfloat16_hybrid_audit_20260709-214415.log
```

### 2026-07-09 - Ready 005c - Exact System-Only Prefix Prime

Date: 2026-07-09

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/subround_005c_trace_prefix_prime/
```

Action:

- Extracted the identical public-trace system message shared by all 120 requests
  and sent it as a `max_tokens=1` OpenAI chat warmup before the timed benchmark.
- Compared against same-session Ready 003f global baseline with normal synthetic
  endpoint warmup.

Expected:

- Prime Radix cache with the 38956-character shared system prefix and reduce
  timed TTFT for the 120-request burst.

Observed:

| Candidate | ERS | TTFT<=1500 | TTFT p50 | TTFT p95 | TPOT p95 |
|---|---:|---:|---:|---:|---:|
| Control, no trace prime | `0.515480` | `28/120` | `4413.646 ms` | `35326.715 ms` | `18.243 ms` |
| Exact system-prefix prime | `0.512806` | `20/120` | `10040.703 ms` | `40916.335 ms` | `19.655 ms` |

Root cause / best hypothesis:

- SGLang Radix cache was active, but this exact warmup shape did not match or
  help the timed request path enough.
- Likely contributors: OpenAI chat-template boundary mismatch, no real user
  prefix continuation in the warmup, cache residency side effects, or suffix and
  burst queue pressure dominating after the prime.

Decision:

- Rejected as a carry-forward optimization.
- This is not evidence against Radix itself; it only rejects system-only trace
  prefix priming as implemented.

Do not repeat unless:

- Tokenized prompt-prefix equality is verified before benchmark.
- Cache-hit metrics are available.
- The experiment remains local research or BTC explicitly allows
  workload-specific cache priming in submission.

Related files:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/subround_005c_trace_prefix_prime/outputs/prefix_prime_summary_20260709-211315.md
local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/subround_005c_trace_prefix_prime/outputs/prefixprime_trace_system_global_003f_prefix_prime_20260709-211315.log
```

### 2026-07-09 - Ready 005b - MAX_RUNNING_REQUESTS=16 As Global Policy

Date: 2026-07-09

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/subround_005b_ttft_floor_hunt/
```

Action:

- Increased SGLang active requests to `MAX_RUNNING_REQUESTS=16` with continuous
  decode steps `6`, dynamic chunking and scheduler receive interval `1`.

Expected:

- Reduce TTFT queueing by admitting more active requests.

Observed:

```text
Candidate: ttft_run16_steps6
Run: local_workspaces/rtx5080/runs/20260709-203754
ERS: 0.455608
TTFT<=1500: 32/120
TTFT p50/p95: 2975.257 / 20053.167 ms
TPOT p95: 44.433 ms
```

Root cause / best hypothesis:

- Higher active requests reduce queue wait and improve TTFT counts, but decode
  contention pushes TPOT p95 near the official 45 ms ceiling.

Decision:

- Rejected as a global policy.
- Keep only as proof that TTFT improves when admission pressure is increased.

Do not repeat unless:

- A real phase-aware handoff moves long decode to a TPOT-safe lane.

Related files:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/subround_005b_ttft_floor_hunt/outputs/ttft_hunt_summary_20260709-203017.md
```

### 2026-07-09 - Ready 005b - Chunk8192 With Prefill CUDA Graph

Date: 2026-07-09

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/subround_005b_ttft_floor_hunt/
```

Action:

- Tested `--chunked-prefill-size=8192 --prefill-max-requests=4
  --enable-mixed-chunk` under the aggressive TTFT lane shape.
- Also tested variants with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `DISABLE_RADIX_CACHE=true`, and `--disable-overlap-schedule`.

Expected:

- Larger chunk size might reduce prefill scheduler round trips and lower TTFT.

Observed:

```text
All chunk8192 variants: backend_not_ready
Root error: Capture prefill CUDA graph failed: CUDA out of memory.
```

Root cause / best hypothesis:

- SGLang `tc_piecewise` prefill CUDA graph tries to capture shapes up to 8192.
- At local mem fraction `0.86`, RTX 5080 had only about 190-213 MiB free during
  capture and failed a 192 MiB allocation.
- Thread cap, no-radix and no-overlap cannot fix this because the failure occurs
  during CUDA graph memory capture.

Decision:

- Rejected for local RTX 5080 with current memory fraction and prefill CUDA graph.

Do not repeat unless:

- Testing chunk `3072/4096/6144` first.
- Or explicitly setting a smaller `--cuda-graph-max-bs-prefill`.
- Or disabling prefill CUDA graph as a separate measured experiment.

Related files:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/subround_005b_ttft_floor_hunt/outputs/ttft_hunt_summary_20260709-205023.md
```

### 2026-07-09 - Ready 005b - Chunked Prefill Size -1

Date: 2026-07-09

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/subround_005b_ttft_floor_hunt/
```

Action:

- Tested `--chunked-prefill-size=-1` as a possible chunk-disable sentinel.

Expected:

- Disable chunked prefill or force one-pass prefill behavior.

Observed:

```text
Run: local_workspaces/rtx5080/runs/20260709-205642
Success: 0/120
ERS: 0.0
Errors: HTTP 500, remote connection closed, connection refused
```

Root cause / best hypothesis:

- This SGLang image/config does not support `-1` as a safe disable path with the
  rest of the aggressive TTFT flags.

Decision:

- Rejected.

Do not repeat unless:

- The SGLang flag semantics are verified from the image help/source and tested
  first with a small health request.

Related files:

```text
local_workspaces/rtx5080/runs/20260709-205642
```

### 2026-07-09 - Ready 005 - Chunked Prefill Size 256

Date: 2026-07-09

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/
```

Action:

- Took the best TPOT-floor overlay shape and changed prefill chunk size down to
  `256`.
- Config: `--chunked-prefill-size=256 --prefill-max-requests=4
  --enable-mixed-chunk`.

Expected:

- Smaller prefill slices might reduce prefill blocking and improve TTFT.

Observed:

| Candidate | ERS | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | Wall tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Best 1024 overlay | `0.513911` | `12178.541 ms` | `49009.192 ms` | `9.107 ms` | `11.549 ms` | `376.143` |
| Chunk 256 test | `0.504052` | `27566.901 ms` | `59758.971 ms` | `11.093 ms` | `13.209 ms` | `282.158` |

Root cause / best hypothesis:

- Chunk `256` creates too many prefill fragments.
- Scheduler/kernel overhead dominates the benefit of smaller slices.
- GPU utilization average fell to `66.71%`, suggesting poor batching/occupancy
  rather than useful interleaving.

Decision:

- Rejected.
- Follow-up with `--prefill-max-requests=1` also rejected:
  - ERS `0.501992`
  - TTFT p50/p95 `27984.350 / 59794.267 ms`
  - TPOT p50/p95 `10.718 / 12.465 ms`
  - It improves TPOT p95 versus req4, but worsens TTFT and ERS.

Do not repeat unless:

- The scheduler/interleaving implementation changes or prefill CUDA graph policy
  changes enough to make tiny chunks cheap.

Related files:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/configs/tpotfloor_chunk256_prefillreq4_mixed.env
local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/outputs/ttft_overlay_summary_20260709-194409.md
```

### 2026-07-09 - Ready 005 - Chunk1024 One-Request Prefill As ERS Winner

Date: 2026-07-09

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/
```

Action:

- Kept the winning chunk size `1024`, but lowered prefill admission from
  `--prefill-max-requests=4` to `--prefill-max-requests=1`.
- Kept `--enable-mixed-chunk`, `MAX_RUNNING_REQUESTS=4`,
  `num-continuous-decode-steps=8` and `num-reserved-decode-tokens=224`.

Expected:

- One request at a time might reduce prefill contention and improve TPOT/tail
  while preserving the chunk1024 TTFT benefit.

Observed:

| Candidate | ERS | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | Wall tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Chunk1024 req4 winner | `0.513911` | `12178.541 ms` | `49009.192 ms` | `9.107 ms` | `11.549 ms` | `376.143` |
| Chunk1024 req1 test | `0.511664` | `12381.904 ms` | `48774.064 ms` | `8.983 ms` | `11.087 ms` | `379.044` |

Root cause / best hypothesis:

- Request1 reduces decode/preemptive pressure slightly, which helps TPOT p95.
- It does not admit enough useful prefill parallelism to improve overall ERS.
- The small TTFT p95 gain is not enough to offset the request-score distribution
  loss.

Decision:

- Do not promote as the Round 005 ERS overlay.
- Keep as a TPOT-tail variant if a future TTFT module needs extra decode safety.

Do not repeat unless:

- The objective is explicitly TPOT-tail protection rather than ERS.
- A new TTFT policy changes the admission/queueing trade-off enough to retest.

Related files:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/configs/tpotfloor_chunk1024_prefillreq1_mixed.env
local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/outputs/ttft_overlay_summary_20260709-195751.md
```

### 2026-07-09 - Ready 005 - Full Chunk1024 Interleave With Long Warmup

Date: 2026-07-09

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/
```

Action:

- Started from the TPOT-floor SGLang policy:
  `MAX_RUNNING_REQUESTS=4`, `num-continuous-decode-steps=8`,
  `num-reserved-decode-tokens=224`.
- Added `--chunked-prefill-size=1024`, `--max-prefill-tokens=8192`,
  `--prefill-max-requests=4`, `--enable-mixed-chunk` and long endpoint warmup.
- Kept prefill CUDA graph `tc_piecewise`.

Expected:

- Combine the first-pass full TTFT overlay with smaller prefill slices and
  interleaving.
- Reduce TTFT while preserving TPOT-floor policy.

Observed:

- Backend started, but endpoint warmup triggered a SGLang scheduler crash.
- Error:

```text
AssertionError: PCG capture stream is not set, please check if runtime recompilation happened
```

Root cause / best hypothesis:

- The combination of mixed chunking, explicit 1024 prefill chunks, prefill
  admission cap and `tc_piecewise` prefill CUDA graph caused runtime
  recompilation or graph replay outside the captured stream.

Decision:

- Rejected as unstable.
- The measured winner is the smaller compatible overlay
  `tpotfloor_chunk1024_prefillreq4_mixed`, without `max-prefill-tokens=8192`
  and without long warmup.

Do not repeat unless:

- Prefill CUDA graph is disabled or changed for this candidate.
- SGLang image/runtime changes and the candidate is re-validated from startup
  through endpoint warmup before trace benchmark.

Related files:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/outputs/tpotfloor_chunk1024_full_interleave_ttft_overlay_20260709-191334.log
local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/outputs/ttft_overlay_summary_20260709-191334.md
```

### 2026-07-09 - Ready 003c - External HTTP Admission Cap

Date: 2026-07-09

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/subround_003c_controlled_admission_scheduler/
```

Action:

- Put a Tree-Gated proxy in front of SGLang.
- Limit backend concurrency at the HTTP request level.
- Try FIFO cap6, FIFO cap8 and `ttft_v0` cap6.
- Release proxy slot on first streamed chunk.

Expected:

- Reduce burst pressure.
- Improve TTFT by classifying requests before they hit SGLang.

Observed:

| Candidate | ERS | TTFT <=1500 | TTFT p50 | TPOT p95 | Proxy queue p95 |
|---|---:|---:|---:|---:|---:|
| Same-session baseline | `0.515398` | `25/120` | `4610.932 ms` | `18.122 ms` | n/a |
| FIFO first-chunk cap6 | `0.481412` | `3/120` | `10643.210 ms` | `26.420 ms` | `18240.224 ms` |
| FIFO first-chunk cap8 | `0.481840` | `3/120` | `10507.946 ms` | `22.590 ms` | `16921.119 ms` |
| `ttft_v0` first-chunk cap6 | `0.490972` | `8/120` | `11190.256 ms` | `21.879 ms` | `29064.152 ms` |

Root cause / best hypothesis:

- Proxy waiting time still counts toward TTFT.
- Request-level admission only moved the queue from SGLang to the proxy.
- First streamed chunk can still arrive many seconds later, so
  `release_on=first_chunk` is not a true prefill-lane release.

Decision:

- Rejected.

Do not repeat unless:

- The control point moves below whole HTTP request admission, for example
  prefix-build singleflight, prefill chunk scheduling or native scheduler hooks.

Related files:

```text
local_workspaces/rtx5080/docs/technique_ledger.md
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/subround_003c_controlled_admission_scheduler/README.md
```

### 2026-07-09 - Ready 003d - Wrong SGLang Priority Schedule Policy

Date: 2026-07-09

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/subround_003d_native_prefix_priority_scheduler/
```

Action:

- Started SGLang with `--schedule-policy=priority` and
  `--enable-priority-scheduling`.

Expected:

- Use native SGLang priority queue while proxy only injects request priority.

Observed:

- Backend failed to become ready.
- SGLang asserted:

```text
To use priority scheduling, schedule_policy must be 'fcfs' or 'lof'.
'priority' is not supported.
```

Root cause / best hypothesis:

- The help output lists `priority` as a schedule policy, but this image's
  runtime validation rejects priority scheduling unless the base schedule policy
  is `fcfs` or `lof`.

Decision:

- Fixed configs to use `SCHEDULE_POLICY=fcfs` with
  `--enable-priority-scheduling`.

Do not repeat unless:

- A newer SGLang image changes this validation, and startup is re-verified.

Related files:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/configs/sglang_priority_shortprefill_nocap.env
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/configs/sglang_priority_prefixlpm_nocap.env
```

### 2026-07-09 - Ready 003d - Whole-Request Native Priority Scheduling

Date: 2026-07-09

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/subround_003d_native_prefix_priority_scheduler/
```

Action:

- Inject OpenAI request `priority` through the proxy.
- Let SGLang schedule whole requests with native priority.
- Test `short_prefill_v0` and `prefix_lpm_v0` priority heuristics.
- Do not cap proxy concurrency, to avoid repeating 003c's queue problem.

Expected:

- Preserve low proxy queue wait while letting SGLang prioritize likely cheaper
  or prefix-reusing requests.

Observed:

| Candidate | ERS | TTFT <=1500 | TTFT p50 | TTFT p95 | TPOT p95 | Proxy queue p95 |
|---|---:|---:|---:|---:|---:|---:|
| Same-session baseline | `0.514740` | `25/120` | `4561.498 ms` | `36611.789 ms` | `18.003 ms` | n/a |
| Priority short-prefill | `0.439851` | `3/120` | `12830.630 ms` | `41138.169 ms` | `36.360 ms` | `20.141 ms` |
| Priority prefix-LPM | `0.451969` | `22/120` | `9811.921 ms` | `66516.090 ms` | `38.781 ms` | `19.582 ms` |

Root cause / best hypothesis:

- Proxy queue was not the issue because queue p95 stayed near `20 ms`.
- Whole-request priority likely disturbed SGLang's LPM/radix-oriented behavior.
- Priority at full-request granularity is too coarse and can damage decode tail.

Decision:

- Rejected.

Do not repeat unless:

- Priority can be applied below request granularity, especially to prefix-build
  or prefill chunks.
- A same-session test proves it preserves TPOT tail.

Related files:

```text
local_workspaces/rtx5080/docs/technique_ledger.md
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/subround_003d_native_prefix_priority_scheduler/README.md
```

### 2026-07-09 - Ready 003d - Native Prefill Delayer As Standalone Win

Date: 2026-07-09

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/subround_003d_native_prefix_priority_scheduler/
```

Action:

- Enable SGLang native prefill delayer.
- Test queue ratio `0.25` with max delay `250ms`.
- Test queue ratio `0.5` with max delay `500ms`.

Expected:

- Batch prefill fragments and reduce TTFT under burst.

Observed:

| Candidate | ERS | TTFT <=1500 | TTFT p50 | TTFT p95 | TPOT p95 |
|---|---:|---:|---:|---:|---:|
| Same-session baseline | `0.516041` | `23/120` | `4564.696 ms` | `36093.005 ms` | `17.974 ms` |
| Delayer 0.25/250ms | `0.513646` | `23/120` | `4493.901 ms` | `36595.625 ms` | `18.311 ms` |
| Delayer 0.5/500ms | `0.515090` | `23/120` | `4476.439 ms` | `36670.492 ms` | `18.265 ms` |

Root cause / best hypothesis:

- The knob slightly improved TTFT p50 but worsened tail/TPOT enough to lose ERS.
- The trace bottleneck is larger than small queue-delayer batching.

Decision:

- Rejected as a standalone score win.
- Keep only as a small watchlist knob for later same-session sweeps.

Do not repeat unless:

- It is paired with a stronger prefill-chunk or decode-protection change and
  tested same-session against baseline.

Related files:

```text
local_workspaces/rtx5080/docs/technique_ledger.md
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/subround_003d_native_prefix_priority_scheduler/README.md
```

### Prior Local Notes - High VRAM And FP8 Instability Cases

Date: before 2026-07-09

Action:

- Tried very high local VRAM allocation and aggressive FP8/memory variants in
  earlier warmup/ready rounds.

Observed:

- Local `GPU_MEMORY_UTILIZATION=0.95` failed startup when desktop/app VRAM left
  insufficient free memory.
- Local `0.83` could be flaky when free memory was just below requested memory.
- Ready 003 `sglang_ablate_fp8_mem088_run6` returned only `105/120` success with
  `15` zero-token/error requests, ERS `0.45086075605694304`.

Decision:

- Treat high VRAM as a capacity lever, not a default score lever.
- Keep safer local baseline around the measured stable config.
- Do not assume FP8/high-memory variants are valid without success count and
  GPQA proxy check.

Do not repeat unless:

- VRAM is cleanly available before startup.
- Same-session success is `120/120`.
- GPQA proxy is rerun for lossy KV/model changes.

### 2026-07-09 - Ready 003e - Prefill Chunk Sweep Did Not Produce A Baseline

Date: 2026-07-09

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/subround_003e_prefill_chunk_scheduler_sweep/
```

Action:

- Swept SGLang native prefill/chunk knobs:
  - `--chunked-prefill-size=1024`
  - `--chunked-prefill-size=4096`
  - `--max-prefill-tokens=8192`
  - `--prefill-max-requests=4`
  - `--enable-mixed-chunk`
- Refined around `max-prefill-tokens=6144/8192/12288` and combinations with
  `mixed_chunk` / `prefill-max-requests=4`.

Expected:

- Reduce TTFT by giving the scheduler better prefill granularity.
- Preserve TPOT tail under the official `20 ms` floor.

Observed:

Broad sweep:

| Candidate | ERS | TTFT <=1500 | TTFT p50 | TTFT p95 | TPOT p95 |
|---|---:|---:|---:|---:|---:|
| Same-session baseline | `0.512680` | `23/120` | `4605.225 ms` | `36691.048 ms` | `18.161 ms` |
| Chunk `1024` | `0.509090` | `23/120` | `5553.023 ms` | `32702.072 ms` | `20.019 ms` |
| Chunk `4096` | `0.511333` | `23/120` | `4625.593 ms` | `37374.638 ms` | `18.619 ms` |
| Max prefill tokens `8192` | `0.515851` | `25/120` | `4576.188 ms` | `35990.431 ms` | `17.969 ms` |
| Mixed chunk | `0.514428` | `26/120` | `3913.834 ms` | `35987.500 ms` | `20.708 ms` |

Refine sweep:

| Candidate | ERS | TTFT <=1500 | TTFT p50 | TTFT p95 | TPOT p95 |
|---|---:|---:|---:|---:|---:|
| Same-session baseline | `0.517014` | `25/120` | `4533.439 ms` | `37077.954 ms` | `18.368 ms` |
| Max prefill tokens `8192` | `0.513922` | `25/120` | `4543.819 ms` | `36306.676 ms` | `18.154 ms` |
| Max prefill tokens `6144` | `0.515615` | `23/120` | `4621.256 ms` | `36264.169 ms` | `17.916 ms` |
| `8192` + mixed chunk | `0.512677` | `26/120` | `4015.333 ms` | `35982.761 ms` | `20.345 ms` |

Root cause / best hypothesis:

- Single prefill chunk knobs are too weak to solve the burst/shared-prefix
  bottleneck.
- `max-prefill-tokens=8192` can improve tail in one session but was not
  repeatable enough to replace the baseline.
- `enable-mixed-chunk` helps TTFT p50 but damages TPOT tail, which matters
  heavily under the official scoring formula.

Decision:

- Rejected as new baseline.
- Keep `max-prefill-tokens=6144/8192` and `enable-mixed-chunk` only as
  conditional knobs for future same-session sweeps.

Do not repeat unless:

- It is paired with a stronger decode-protection or prefix-build/singleflight
  change.
- The candidate is validated against same-session baseline and keeps TPOT p95
  safely below `20 ms`.

Related files:

```text
local_workspaces/rtx5080/docs/technique_ledger.md
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/subround_003e_prefill_chunk_scheduler_sweep/README.md
```

### 2026-07-09 - Ready 003f - Decode/Scheduler Variants Not Worth Repeating Alone

Date: 2026-07-09

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/subround_003f_decode_protection_scheduler_sweep/
```

Action:

- Tested standalone decode/scheduler knobs:
  - `--num-continuous-decode-steps=2`
  - `--num-continuous-decode-steps=4`
  - `--schedule-conservativeness=0.5`
  - `--schedule-conservativeness=2.0`

Expected:

- Protect TPOT/decode lane while improving TTFT enough to raise ERS.

Observed:

| Candidate | ERS | TTFT <=1500 | TTFT p50 | TTFT p95 | TPOT p95 |
|---|---:|---:|---:|---:|---:|
| Same-session baseline | `0.516420` | `25/120` | `4528.508 ms` | `36718.905 ms` | `18.093 ms` |
| Decode steps `2` | `0.513750` | `24/120` | `4585.756 ms` | `36513.066 ms` | `18.016 ms` |
| Decode steps `4` | `0.515807` | `27/120` | `4403.766 ms` | `36173.404 ms` | `18.348 ms` |
| Schedule conservativeness `0.5` | `0.516384` | `24/120` | `4334.474 ms` | `36281.985 ms` | `18.248 ms` |
| Schedule conservativeness `2.0` | `0.514763` | `25/120` | `4461.715 ms` | `36393.750 ms` | `18.041 ms` |

Root cause / best hypothesis:

- Decode steps and conservativeness alone are too weak/noisy.
- Decode steps `4` helps TTFT<=1500 count, but not enough to beat baseline ERS
  unless paired with a better decode reserve.

Decision:

- Do not repeat these as standalone candidates.
- Keep `decode_steps4` only as a combo with `num-reserved-decode-tokens`.

Do not repeat unless:

- Paired with decode reserve or another candidate that already beats
  same-session baseline.

Related files:

```text
local_workspaces/rtx5080/docs/technique_ledger.md
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/subround_003f_decode_protection_scheduler_sweep/README.md
```

### 2026-07-09 - Ready 004 - Dynamic Chunking and Aggressive Continuous Batching Not Baseline

Date: 2026-07-09

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_004_sglang_self_optimization/
```

Action:

- Tested SGLang built-in self-optimization knobs:
  - `--enable-dynamic-chunking`
  - explicit FlashInfer attention/sampling backends
  - `num-continuous-decode-steps=6/8`
  - lower `MAX_RUNNING_REQUESTS=8/6/4`
  - stronger continuous batching with `MAX_RUNNING_REQUESTS=12/14`
  - `--enable-mixed-chunk`

Observed:

| Candidate | ERS | TTFT p50 | TPOT p50 | TPOT p95 | Reading |
|---|---:|---:|---:|---:|---|
| Control | `0.523052` | `3787.462 ms` | `12.447 ms` | `17.286 ms` | Best ERS |
| Dynamic chunking | `0.516477` | `4338.047 ms` | `13.985 ms` | `18.190 ms` | No win |
| Explicit FlashInfer | `0.515243` | `4435.709 ms` | `13.663 ms` | `17.996 ms` | Valid but worse than auto |
| Run 6 + steps 6 | `0.511819` | `4681.498 ms` | `8.373 ms` | `11.666 ms` | TPOT rescue only |
| Run 4 + steps 8 | `0.507828` | `20565.496 ms` | `7.064 ms` | `9.355 ms` | TPOT floor, TTFT too costly |
| Continuous batch run 14 | `0.501104` | `3132.908 ms` | `15.479 ms` | `23.741 ms` | TTFT/throughput win, TPOT not scoring-safe |
| Mixed run 12 | `0.492521` | `3335.134 ms` | `19.642 ms` | `22.473 ms` | Reject |

Decision:

- Do not promote dynamic chunking alone, explicit FlashInfer forcing,
  `enable-mixed-chunk`, or aggressive continuous batching as baseline.
- Keep lower active concurrency only as TPOT rescue/floor probes.
- Keep `MAX_RUNNING_REQUESTS=14` continuous batching only as a TTFT/throughput
  research path, not as a scoring path while TPOT p95 is above `20 ms`.

Related files:

```text
local_workspaces/rtx5080/docs/technique_ledger.md
local_workspaces/rtx5080/research_rounds/ready_rounds/round_004_sglang_self_optimization/README.md
```

### 2026-07-09 - Ready 006a - Subround Runner Path Override

Date: 2026-07-09

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006a_mamba_state_quality_gate/
```

Action:

- First invocation of the 006a GPQA runner attempted to use subround configs,
  but `round_lib.sh` overwrote `CONFIG_DIR` and `OUTPUT_DIR` back to the parent
  Round 006 paths.

Observed:

- Output file:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/outputs/gpqa_runs_20260709-215748.tsv
```

- Both candidates were marked `missing_config`.
- No Docker/model evaluation ran in this failed attempt.

Fix:

- Restore `CONFIG_DIR` and `OUTPUT_DIR` to subround paths immediately after
  sourcing the shared round library.
- Successful fixed run:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006a_mamba_state_quality_gate/outputs/gpqa_comparison_20260709-215832.md
```

Decision:

- Ignore parent-output `gpqa_runs_20260709-215748.tsv`; it is a harness setup
  failure, not an experiment result.

### 2026-07-09 - Ready 006d - Synthetic Endpoint Warmup Crash Before TTFT Sweep

Date: 2026-07-09

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006d_hybrid_ttft_combo_hunt/
```

Action:

- Started the TTFT combo sweep with the normal `ENDPOINT_WARMUP_MODE=mixed`
  synthetic warmup inherited from Ready 003f.

Observed:

- The first control candidate timed out during endpoint warmup before the trace
  benchmark started.
- SGLang scheduler then crashed with CUDA OOM in a path involving
  `sglang/srt/models/qwen3_vl.py` and TorchInductor autotune.
- No valid trace benchmark result was produced by this failed attempt.

Related log:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/outputs/hybrid_control_global_003f_hybrid_audit_20260709-230222.log
```

Decision:

- Do not use this failed run as performance evidence.
- Rerun TTFT combo hunt with `ENDPOINT_WARMUP_MODE=none` to measure the trace
  path directly and avoid the synthetic warmup crash.

### 2026-07-09 - Ready 006e - JSONL Corruption In High-Concurrency Trace Report

Date: 2026-07-09

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006e_decode_fairness_hunt/
```

Action:

- Ran `hybrid_fair_lpm_res64_steps2` with `reserved=64`, `steps=2`, LPM and
  the Mamba FlashInfer + BF16 SSM overlay.

Observed:

- Trace client wrote 120 benchmark results to:

```text
local_workspaces/rtx5080/runs/20260709-235042/metrics.jsonl
```

- Report generation failed with `json.decoder.JSONDecodeError`.
- The JSONL file contained interleaved/corrupted fragments such as adjacent
  `{"request_id"...` records spliced together.
- Salvaged parse recovered only 89 unique ok records and showed a bad latency
  distribution: TTFT p50/p95 `19744.251 / 50371.084 ms`, TPOT p95 `28.830 ms`.

Decision:

- Reject this run as both a harness/report failure and a poor candidate.
- The Python benchmark writer serializes writes in the main thread, so the most
  likely root cause is run-directory collision from second-resolution timestamps
  while multiple benchmark scripts were alive in the same session.
- Fixed `local_workspaces/rtx5080/scripts/run_trace.sh` to create run folders
  with nanosecond timestamp plus PID and a collision fallback suffix.

### 2026-07-09 - Ready 006e - FCFS Rerun Start Failed Due Container Removal Race

Date: 2026-07-09

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006e_decode_fairness_hunt/
```

Action:

- Attempted to run `hybrid_fair_fcfs_run12_res128_steps1` immediately after the
  failed `res64_steps2` run.

Observed:

- Docker reported that removal of `phase1-rtx5080-sglang` was still in progress
  and then refused the new container name as already in use.
- This was a transient harness/container cleanup race, not a model/runtime
  result.

Decision:

- Ignore the failed start attempt.
- Reran FCFS after cleanup; valid run is:

```text
local_workspaces/rtx5080/runs/20260709-235432
```

- The valid FCFS run is rejected on metrics, not on startup failure.

### 2026-07-10 - Ready 006f - Partial L4 Run After Lost Session

Date: 2026-07-10

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006f_internal_scheduler_prefill_hunt/
```

Action:

- During Round L, the tool session lost the running process while
  `hybrid_graph_l4_lpm_disable_prefill_graph` was active.

Observed:

- Container remained running and the run folder existed:

```text
local_workspaces/rtx5080/runs/20260710-005900-768098124-1149012
```

- `metrics.jsonl` contained only `90/120` rows and no `summary.json`.
- This is a partial harness/session interruption, not a valid benchmark result.

Decision:

- Ignore the partial run.
- Valid rerun:

```text
local_workspaces/rtx5080/runs/20260710-010315-672403691-1157353
```

- The valid rerun completed `120/120` with ERS `0.520807`.

### 2026-07-10 - Ready 007 - HTTP-Layer Logical Two-Policy Proxy Rejected

Date: 2026-07-10

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_007_logical_two_policy_proxy/
```

Action:

- Implemented a single-model two-policy proxy in:

```text
local_workspaces/rtx5080/scripts/two_policy_proxy.py
```

- The proxy forwarded to one SGLang backend and released `prefill_limit` slots
  at first streamed token while keeping `total_limit` slots until decode
  completion.

Observed:

- Best candidate was `proxy_c_prefill10_total14`:

```text
run: local_workspaces/rtx5080/runs/20260710-012038-904173099-1178337
ERS: 0.493432
TTFT <=1500: 3/120
TTFT p50/p95: 6618.465 / 11752.298 ms
TPOT p95: 21.721 ms
```

- All four candidates were below the current local best `0.524547`.
- The proxy removed `TTFT >=30s` tail but made the fast bucket and TPOT worse.

Decision:

- Reject HTTP-layer lifecycle splitting as a submission path.
- Queue wait in the proxy is still part of TTFT, so this is too late in the
  stack.
- If revisiting the two-policy architecture, implement it inside the engine
  scheduler or through native priority/admission hooks.

### 2026-07-10 - Ready 006g - vLLM Hybrid Capability Limits

Date: 2026-07-10

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006g_vllm_hybrid_bakeoff/
```

Observed failures and fallbacks:

1. `--mamba-cache-mode=all` is not supported for Qwen3.5 prefix caching in
   vLLM 0.22.1. The engine starts but explicitly falls back to `align`.
2. `--gdn-prefill-backend=flashinfer` cannot use its kernel on RTX 5080 for
   this artifact. The engine explicitly falls back to Triton/FLA.
3. `--max-num-partial-prefills=4` fails startup. The V1 engine raises
   `NotImplementedError: Concurrent Partial Prefill is not supported`.
4. Text-only interactivity with `max_num_batched_tokens=8192` completed but
   worsened TTFT p50 to `19.103s` and ERS to `0.501255`.
5. Run6 with relaxed full-ISL admission improved TTFT tail but reduced the
   official fast bucket to `2/120`, leaving ERS at `0.501262`.

Decision:

- Do not present fallback behavior as an active optimization.
- Do not add concurrent partial prefill to vLLM 0.22.1 submission configs.
- Do not promote larger prefill iterations or relaxed admission based only on
  p50/p95; verify the `<=1500ms` bucket and ERS.
- Keep `--language-model-only` as the only positive vLLM carry-forward from
  this subround.

### 2026-07-10 - Ready 008a - Historical Run Is Not a Valid Instrumentation Control

Date: 2026-07-10

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_008_engine_native_ers_scheduler/subround_008a_scheduler_telemetry/
```

Observed:

- Historical best for the same config was ERS `0.524547`.
- Fresh uninstrumented control in Ready 008a reached only ERS `0.512927`.
- Fresh instrumented replay reached ERS `0.511176`.
- Comparing telemetry only to the historical best would incorrectly assign an
  overhead of `-0.013371`; the valid same-session delta is `-0.001751`.

Decision:

- Never judge scheduler instrumentation or a small policy change only against
  a prior-session best.
- Always run a matching native control in the same session and preserve the
  historical best only as a promotion threshold.
- Repeat any apparent gain around `0.005` or less before updating the current
  best config.

### 2026-07-10 - Ready 008b - Frozen Static Prefix Rescue Rejected

Date: 2026-07-10

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_008_engine_native_ers_scheduler/subround_008b_shadow_conditional_prefix_rescue/
```

Action:

- Calibrated a generic read-only shadow scorer from native queue age, prefix
  match ratio, remaining prefill cost, running requests and decode pressure.
- Kept native LPM ordering, batch construction and cache state unchanged.
- Ran three same-session control/shadow attempts before deciding whether to
  grant active scheduling authority.

Observed failure 1 - brittle guard:

```text
control/shadow ERS: 0.520392 / 0.519800
support/precision: 14 / 0.9286
early/late support: 0 / 14
rolling decode-result p95/guard: 22.159 / 22.1 ms
request TPOT p95: 18.863 ms
```

The baseline-p95-plus-1ms guard rejected a valid early cohort because of only
`0.059 ms` of scheduler-result jitter. It was recalibrated generically from
baseline p97.5 plus 1 ms, not from request IDs or trace positions.

Observed failure 2 - observer feedback loop:

```text
control/shadow ERS: 0.520507 / 0.510481
ERS delta: -0.010026
repeated EXPIRED labels: 5941
wall throughput delta: -45.952 token/s
```

Including a 100 ms queue-age bucket and running count in every label signature
caused unchanged terminal states to be serialized repeatedly. On 3 CPU cores,
the resulting slowdown created more age transitions and more logging.

Fix:

- Emit labels only when `(action, reason)` changes.
- Skip service-cost prediction for expired, low-prefix and oversized requests.
- Add one aggregate CPU profile per 32 evaluations.

The fix reduced label events from `6628` to `210`; mean evaluator CPU was
`108 us`, and final ERS delta was `+0.002358`, so the observer itself passed.

Observed failure 3 - policy instability:

```text
live support across attempts: 14 -> 12 -> 1
final early/late support: 1 / 0
```

Decision:

- Reject the frozen static rescue policy and do not open active Ready 008c.
- Keep the optimized observer/replay code as research infrastructure.
- Do not fit and promote a scheduler rule from one native timeline. Collect
  multiple runs and require leave-one-run-out stability first.
- Keep XGBoost/C++ disabled until a simple cross-run policy proves control
  authority.

### 2026-07-10 - Ready 008c - Missing Direct-Admission Outcome Snapshots

Date: 2026-07-10

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_008_engine_native_ers_scheduler/subround_008c_adaptive_threshold_shadow/
```

Initial replay failure:

- Adaptive labels covered all 120 requests, but only `95-104` requests per
  timeline had supervised outcomes.
- `16-25` requests reached first token without first appearing in a captured
  `policy_order` event because native SGLang admitted them directly.
- Treating missing snapshots as negative labels would have biased the online
  controller.

Fix:

- Add a read-only `request_queued` callback that seeds a generic snapshot at
  queue admission.
- Allow later policy observations to replace it only when expected TTFT score
  gain per estimated service cost improves.
- Keep the hashed request tag as a join key, never a feature.

Result:

- All four replays and the live run reached `120/120` adaptive outcomes.
- Cross-run and live policy gates passed.

Infrastructure note:

- A Docker internal network blocked host ingress as well as outbound access.
- The safe working topology is server `--network none` plus a separate client
  container using `--network container:<server>`. The client reaches only the
  server loopback while retaining separate CPU resources.

Decision:

- Always instrument queue admission as the earliest lifecycle boundary for
  future adaptive policies.
- Keep the no-outbound two-container benchmark topology for third-party runtime
  images.

### 2026-07-10 - Ready 008d - Adaptive Global-Front Rescue Rejected

Date: 2026-07-10

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_008_engine_native_ers_scheduler/subround_008d_conservative_active_rescue/
```

Action:

- Ran native LPM first, then promoted at most one adaptive candidate from
  native position `1..8` to global queue front.
- Kept one model/cache pool, 12-outcome minimum, 150ms cooldown, one move per
  request, eight-move cap, decode guard and kill switches.
- Compared against an adaptive-shadow control with identical observer cost.

Expected:

- Convert Ready 008c's precise fast-request cohort into more TTFT scores below
  1500ms without damaging TPOT.

Observed:

```text
control/active ERS: 0.514307 / 0.512578
ERS delta: -0.001728
TTFT<=1500: 21 -> 23
TTFT p50/p95 delta: +138.923 / +1975.281 ms
TPOT p95: 18.353 -> 19.430 ms
wall throughput delta: -24.359 tok/s
interventions / unique / commits: 6 / 6 / 6
active positions: [3, 1, 1, 2, 1, 1]
```

Paired score analysis:

```text
TTFT component delta: -0.002143
TPOT component delta: +0.000415
request scores improved / worsened / unchanged: 10 / 19 / 91
```

Root cause / best hypothesis:

- The adaptive selector was locally precise: five of six moved requests reached
  first token in `287-821ms`.
- Moving them to global front crossed native LPM prefix-locality boundaries.
  The selected request often improved, but more neighboring requests lost TTFT
  score.
- The first intervention also used a bootstrap-loosened suffix threshold
  (`3968` instead of accepted convergence `2944`) and reached `1549ms`.
- Therefore this is a policy externality, not a mutation-safety failure.

Decision:

- Reject global-front rescue and do not open Ready 008e repeatability.
- Keep Ready 008c shadow only and Ready 006e as serving baseline.

Do not repeat unless:

- A generic cache-node/equal-prefix cohort boundary is available.
- Promotion stays inside that cohort and preserves all order outside it.
- Active authority is blocked while adaptive thresholds are looser than Ready
  008c convergence.
- The redesigned policy passes multi-run shadow/replay first.

Related files:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_008_engine_native_ers_scheduler/subround_008d_conservative_active_rescue/outputs/live_active_analysis.latest.md
local_workspaces/rtx5080/research_rounds/ready_rounds/round_008_engine_native_ers_scheduler/subround_008d_conservative_active_rescue/outputs/paired_score_delta.latest.md
```

### 2026-07-10 - Unscaled SGLang FP8 KV Rejected

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006h_attempt02_kv_fp8_gate/
```

E4M3 quality failure:

```text
auto accuracy A/B: 0.37 / 0.38
E4M3 accuracy A/B: 0.36 / 0.33
auto/E4M3 repeat prediction changes: 2 / 9
worst paired accuracy drop: 0.05
```

E5M2 diagnostic result:

```text
accuracy A/B: 0.35 / 0.36
repeat prediction changes: 10
ERS auto/E5M2: 0.495794 / 0.487801
ERS delta: -0.007994
KV tokens auto/E5M2: 241871 / 494760
```

Root cause / limitation:

- SGLang defaulted E4M3 scaling factors to `1.0` and warned about accuracy.
- The pinned SGLang `0.5.14` Qwen3.5 class has no
  `load_kv_cache_scales` implementation, so a scale file cannot be enabled by
  Compose alone.
- E5M2 doubled capacity but worsened both repeatability and ERS.

Decision:

- Do not submit unscaled FP8 KV.
- Do not confuse lower TTFT p50 with a scoring gain; TTFT<=1500 stayed `16`.
- Require a custom calibrated implementation before repeating this branch.

### 2026-07-10 - Aggressive Tree-Gated Rejuvenation Rejected

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006j_native_tree_gated_v2/
```

Failed variants:

```text
soft_rejuvenation:
  positive coverage 1.0000 -> 0.9964
  selected precision 0.049592 -> 0.039478
  prefix ratio 0.578529 -> 0.567179

deadline_rejuvenation:
  positive coverage 1.0000 -> 0.1159
  selected precision 0.049592 -> 0.006502
  prefix ratio 0.578529 -> 0.517823
```

Root cause:

- A strong age bonus overwhelms TTFT-hit and prefix-locality signals.
- Forced promotion at 1300ms frequently selects work that is already too close
  to or beyond the 1500ms score ceiling.
- Aging can prevent starvation, but global placement of expired work creates
  the same neighboring-request externality seen in Ready 008d.

Decision:

- Do not use strong soft-aging or forced-deadline promotion.
- Keep only gentle locality rejuvenation for shadow study; it must remain
  bounded by cache cohort, context-state validity and a decode-pressure guard.
- Never describe rejuvenation as restored context unless KV and Mamba state
  are actually retained or recomputed and measured.

### 2026-07-11 - Exact Cache-Node Tree/Heuristic Active Path Rejected

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006k_cache_local_tree_vs_heuristic/
```

Action:

- Compared Native Tree-Gated v2 with a transparent heuristic using the same
  four effective features.
- Evaluated both only within identical non-root SGLang cache-node cohorts.
- Kept native LPM order, cache state and arrival timestamps unchanged.

Observed:

```text
offline actionable AUC Tree/heuristic: 0.900176 / 0.916530
live exact cohort observations: 263
Tree would-reorder: 0
heuristic would-reorder: 0
control/shadow ERS: 0.524547 / 0.518136
ERS delta: -0.006411
evaluator p99: 333.063us
safety errors/mutations: 0/0
```

Root cause:

- Exact `last_node.id` preserves locality but makes cohort members too similar.
- Native position and normalized position dominate both rankers, while matched
  prefix is equal by construction; the learned policy reproduces native LPM.
- High live AUC came from only two positives among 57 joined outcomes and did
  not produce a different action.

Decision:

- Do not activate Tree or heuristic under this exact-node boundary.
- Do not rerun merely after reducing Python/telemetry overhead; zero action
  support remains zero.
- Do not add this hook to Compose or Portal submissions.

Do not repeat unless:

- held-out offline replay shows nonzero alternative selections;
- new features add information beyond native position/prefix ordering;
- the new cohort remains locality-safe without requiring exact node identity;
- observer cost and same-session ERS gates then pass.

### 2026-07-11 - Parent-Ancestry And No-Position Ranking Rejected

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006l_parent_cache_nonposition_replay/
```

Failed hypothesis 1 - broaden exact-node locality through ancestry:

- Direct parent, fixed hops `2..5` and normalized depths `3..6` were rebuilt
  from an internally consistent 146-node graph.
- Every same-ancestor group still contained one current node and one branch.
- There was no structurally broader candidate to rank.

Failed hypothesis 2 - remove native position inside exact nodes:

```text
validation cohorts/changed: 216/126
support wins/losses: 0/6
positive capture native/candidate: 1.0/0.0
positive precision delta: -0.027778
TTFT-component delta: -0.002374
```

Root cause:

- Static service-cost features repeatedly selected a small set of short/new
  requests that did not become score-positive under the native timeline.
- Native LPM head position retained arrival/locality information that the
  no-position heuristic discarded.
- The final 60 arrivals had no positive exact-node cohort, so late held-out
  precision could not provide a rescue signal.

Decision:

- Do not run a live 006l policy.
- Do not retry parent/ancestor expansion from current `last_node` fields.
- Do not remove native position and promote suffix-shortest requests inside
  exact cache nodes.
- Reopen only with a new temporal/causal feature family or a resource-budget
  optimization, not another static snapshot ranker.

### 2026-07-11 - Direct SGLang Portal Launcher Failed Twice

Attempts:

```text
Attempt 02: direct SGLang, mem-fraction-static=0.86
Attempt 03: direct SGLang, mem-fraction-static=0.88
Portal result: failed for both; no detailed log or score returned
```

What is known:

- The same SGLang image/policy runs locally, so the Portal result does not prove
  the engine or model configuration is invalid.
- Both failed attempts changed the protected sample launcher from
  `python3 -m vllm.entrypoints.openai.api_server` to direct SGLang.
- The official vLLM sample launcher previously produced score `14.98000`.

Decision:

- Do not submit another direct-SGLang Compose.
- Preserve the exact BTC external module path through a custom compatibility
  image and validate the complete 120-request path before submission.
- Do not infer anything from the `0.86` versus `0.88` memory delta because both
  attempts failed at an unknown pre-benchmark stage.

### 2026-07-11 - Cold Custom SGLang Startup Rejected

Observed:

```text
ERS: 0.492667
TTFT <=1.5s: 13/120
TTFT p50: 9441.774 ms
TPOT p95: 25.862 ms
success: 120/120
```

Root cause evidence:

- The exact same image/policy with a legal synthetic warmup before readiness
  repeated at ERS `0.515174` and `0.519746`.
- TPOT p95 improved to `18.397/16.916 ms`, and TTFT p50 improved to
  `4134.943/3731.020 ms`.

Decision:

- Do not ship the cold compatibility image.
- Keep startup warmup inside the image; do not use trace-derived prompt content
  or precompute any benchmark response.

### 2026-07-11 - Attempt 04 Rejected Portal-Injected Flag

Observed:

```text
container exit: 2
unsupported compatibility argument: --trust-remote-code
```

Root cause:

- Portal appended a vLLM flag not shown in the public sample.
- The v1 compatibility boundary intentionally rejected every unknown flag.

Failed remediation:

- v2 forwarded `--trust-remote-code` to SGLang.
- The model loaded, but SGLang selected a multimodal startup path and generic
  image warmup OOMed at local memory fraction `0.86`.

Decision:

- Retire Attempt 04 and do not publish v2.
- Accept the Portal flag as a validated optional no-op in v3 because the fixed
  model already loads through the built-in SGLang path.
- Require full local trace before publishing/new submission.

Resolution:

- Correct v3 completed `120/120`, ERS `0.511706`, TPOT p95 `18.201 ms`.
- Public v3 digest `sha256:11a780...35a03` passed anonymous pull/dry-run.
- At that checkpoint Attempt 06 superseded Attempt 04; the next entry records
  why Attempt 06 was subsequently retired.

### 2026-07-11 - Attempt 06 Rejected Second Portal-Injected Flag

Observed:

```text
container exit: 2
unsupported compatibility argument: --disable-uvicorn-access-log
```

Root cause:

- v3 handled only the previously observed `--trust-remote-code` flag.
- Portal appended another undocumented vLLM option, so per-flag patching did
  not form a stable compatibility boundary.

Decision and resolution:

- Retire Attempt 06 and v3 digest `sha256:11a780...35a03`.
- v4 keeps required BTC arguments strict, accepts known Portal no-ops and
  ignores future unknown long options without logging their values.
- v4 passed 11/11 tests, constrained trace `120/120`, anonymous digest pull and
  public future-flag dry-run. Promote only Attempt 07.

### 2026-07-11 - Attempt 07 Timed Out Waiting For Pod Readiness

Observed:

```text
timed out waiting for contestant pod to be ready: context deadline exceeded
```

What did not reproduce locally:

- Same v4 image at context `262144`, 3 CPU, 8GB RAM, CUDA graphs and synthetic
  warmup reached `/v1/models` in `67s`.
- Server log showed normal model load, 42-shape prefill compile/capture and
  warmup completion; no OOM or crash.

Remaining hypotheses:

- Cold pull of the `13.20GB` local-size SGLang image exceeded orchestration
  time on a fresh node.
- SGLang child runtime stalled only on the H200/CUDA/driver environment.
- Pod scheduling or readiness infrastructure failed independently of code.

Decision:

- Do not blame `max-model-len=262144` or remove performance policy from this
  evidence alone.
- Preserve Attempt 07 and its digest as failure evidence.
- Isolate delivery/startup in the next attempt; do not combine it with ERS
  tuning.

### 2026-07-11 - Historical Tree-Gated Proxy Was Not The 18s-To-8s Cause

The old integrated result combined Tree-Gated with `seq8`, FP8 KV, higher VRAM,
prefix cache, chunked prefill and warmup. Its proxy also ran outside the Docker
CPU/RAM cgroup.

After repackaging proxy/backend into one official-vLLM-derived `3 CPU/8GB`
container, the historical profile showed:

```text
direct TTFT p50: 8937.661ms
FCFS proxy TTFT p50: 9300.832ms
Brain1 proxy TTFT p50: 9272.304ms
```

The historical model output only two priority values, exactly partitioned as
arrival sequences `0..39` and `40..119`. It therefore made no useful reorder;
the large historical gain came from the bundled serving changes rather than
Tree-Gated itself.

Decision: do not publish the corrected image and do not spend a Portal attempt
on this model. Reopen only after a TTFT-SLO-trained policy proves nonzero active
reorder under the same cgroup.

### 2026-07-11 - Attempt 09 Rejected/Blocked Candidates

- `xxhash` prefix hashing was rejected before GPU execution because
  `vllm/vllm-openai:v0.22.1` does not include the optional `xxhash` module.
- `performance-mode=interactivity` was negative at seq8: TTFT p50/p95 worsened
  to `9519.416/15444.876ms` and throughput fell to `547.629 tok/s`.
- Seq12 reduced TTFT but crossed TPOT `20ms` on `10/120` requests and reduced
  local ERS to `0.497559`; do not use as the default candidate.
- Seq11 repeat attempts were blocked before model load by desktop VRAM growth,
  not engine behavior. The observed startup errors explicitly reported free
  memory below the requested local utilization. Do not record these as seq11
  benchmark failures or copy lower local memory values into H200 Compose.

### 2026-07-11 - Attempt 09 TPOT-Guard Runner Path Error

The first seq6 A/B/A invocation supplied `FINAL_CANDIDATE_OUTPUT_DIR` as a
relative path. Both first engines reached Ready, but the benchmark client could
not bind the output directory because Docker interpreted the path as an invalid
named volume. The third startup was stopped to avoid wasting GPU time.

Decision:

- This was a harness error, not an engine or candidate failure.
- Always pass an absolute output path to `run_final_candidate_ab.sh`.
- The immediate retry with an absolute path completed all three candidates at
  `120/120`; only that retry is valid evidence.

### 2026-07-11 - Seq6 Batched-Tokens 16384 Rejected

Against the midpoint of two plain seq6+FP8 controls, explicit
`--max-num-batched-tokens=16384`:

```text
TPOT p50/p95: improved 7.3%/7.0%
TPOT p99: worsened 5.7%
TTFT p50/p95: worsened 15.7%/8.5%
throughput: worsened 3.2%
peak VRAM: increased 11.2%
TTFT <=3s: 2/120 versus 5/120 and 6/120 controls
```

Decision: do not add this flag to Attempt 09. It makes median decode look
better while degrading the per-request tails and TTFT headroom that ERS needs.

### 2026-07-12 - R003 Duplicate Control ID Aborted

The first planned BF16/FP8/BF16 bracket reused the same candidate ID for both
BF16 controls. The shared runner derives run directories from candidate ID, so
control B would overwrite control A.

Decision:

- Abort before any benchmark result was produced.
- Add distinct `_a` and `_b` aliases with identical runtime configuration.
- Do not use the partial startup output as performance evidence.

### 2026-07-12 - R003 Quality Script Executable Bit

The first direct invocation of the newly created quality shell script returned
permission denied because `apply_patch` created it without an executable bit.

Decision: invoke the script explicitly with `bash`. No server or quality case
ran during the failed invocation, so it is tooling-only and has no benchmark
meaning.

### 2026-07-12 - R005 Custom Singleflight Scheduler Rejected

The first custom class inherited synchronous `Scheduler`. Specifying a custom
class bypassed vLLM's automatically selected `AsyncScheduler`, so TPOT p95
regressed to `27.247ms` and ERS to `0.460717`. This run is implementation
failure evidence, not a fair singleflight comparison.

The corrected class inherited `AsyncScheduler` and completed `120/120`, but:

```text
native/candidate local compute tokens: 495102 / 495102
native/candidate local cache hits:     1924240 / 1924240
native/candidate ERS:                  0.565709 / 0.562907
paired TTFT wins/losses:               39 / 81
paired TPOT wins/losses:               52 / 68
```

Holding 19 followers for about `204ms` saved no measured prompt work and
perturbed queue/batch cadence. Do not retry threshold tuning on this mechanism
without a new implementation that first proves a lower local-compute counter.

The decisive follow-up was cache granularity: this hybrid runtime aligns both
scheduler hashes and attention pages to a `1072`-token Mamba state block. The
runtime-aligned tokenizer trie predicts exactly `495102` compute tokens, equal
to native. The earlier 16-token theoretical bound was not executable.

The AsyncScheduler run's tool session was interrupted after all 120 metrics
were written. GPU sampling was empty, so the comparator was updated to render
missing resource telemetry as `n/a`. Prometheus and event files were recovered
from the still-running container before cleanup; request latency evidence is
complete, while GPU telemetry for this one run is intentionally absent.

### 2026-07-12 - R006b Relative Docker Output Mount

The first A/B/A invocation passed a relative `FINAL_CANDIDATE_OUTPUT_DIR`.
Docker interpreted the run path as a named volume and rejected `/` characters.
Control A had reached Ready, but the benchmark client never started and no
request metrics were produced.

The running candidate startup was interrupted immediately, the container was
confirmed removed, and the exact A/B/A was rerun with an absolute output path.
Treat timestamp `20260712-120155` as tooling-only partial output. All promoted
R006b evidence comes from timestamp `20260712-120436`.

### 2026-07-12 - R007 Concurrent Partial Prefill Unsupported

I004 resolves `max_num_partial_prefills=1`. R007 attempted native values `2`
and `4` (with matching long-prefill limits), but vLLM 0.22.1 rejected every
candidate before model load:

```text
NotImplementedError: Concurrent Partial Prefill is not supported.
```

This is a framework/architecture feature boundary, not a latency result. No
benchmark request ran. The profiles were removed and R007 moved to the valid
`max_num_batched_tokens` sweep while keeping one partial prefill.

### 2026-07-12 - R007 Lifecycle Dual-Budget Scheduler Rejected

The custom scheduler correctly inherited `AsyncScheduler`, reached Ready in
`102s`, completed `120/120` requests and emitted the expected lifecycle
transitions. It was therefore a valid performance test, not a fallback.

Against the A/B/A control midpoint:

```text
ERS:                    0.582919 -> 0.580253
paired TTFT wins/loss:  17/103
paired TPOT wins/loss:  39/81
local prompt compute:   456542 -> 461022
prefill-time sum:       29.316s -> 31.719s
peak VRAM:              14903MiB -> 15218MiB
```

The prefill-only 4096 budget changed cache/block timing and increased prompt
work before mixed mode could protect decode. Do not tune the 4096/2048
threshold further without first demonstrating lower compute/cache counters.
Keep fixed 2048 and native scheduling.

### 2026-07-12 - R008 Forced Hybrid Backends Rejected Or Non-Actionable

- `--gdn-prefill-backend=flashinfer` reached Ready on RTX 5080 but logged that
  the kernel is unsupported on the current platform and fell back to
  Triton/FLA. Do not treat that run as FlashInfer performance evidence.
- `--gdn-prefill-backend=triton` duplicated local automatic resolution, so a
  full trace would not isolate a different backend.
- `--attention-backend=TRITON_ATTN` was a real change but lost every paired
  TTFT and TPOT request, reducing ERS from `0.577521` to `0.526927` and raising
  TPOT p95 to `23.707ms`.
- The third control failed startup because only `11.34GiB` VRAM was free versus
  the local profile's `12.66GiB` request. Record this as desktop resource
  interference, not a control or candidate engine failure.

Decision: keep automatic backends. Do not rerun the large negative attention
candidate or lower the local memory contract to manufacture a third control.

### 2026-07-12 - Root Unit-Test Invocation Missing PYTHONPATH

Running `python3 -m unittest discover -s tests -v` directly could not import
`llm_race_lab`. This was an invocation error, not a test failure. The corrected
command `PYTHONPATH=src python3 -m unittest discover -s tests -v` passed `5/5`.
Use the corrected environment in future verification summaries.

### 2026-07-12 - R009 Repeat Runner Interrupted After Candidate

The second A/B/A tool session ended after control A and warmup candidate had
written complete summaries but before control B started. The runner did not
reach its cleanup trap, leaving `phase1-final-candidate-local` alive.

Recovery:

- inspect confirmed the leftover container was the completed warmup candidate;
- all 120 metrics, summary, endpoint report and server logs were already saved;
- the container was removed explicitly to release GPU memory;
- control B was run separately and a recovered paired report was generated.

The recovered repeat remains positive, but its control B has a time gap and is
supporting evidence only. The uninterrupted primary A/B/A is the authoritative
R009 promotion result. Never present the partial runner directory as a normal
three-run bracket.

### 2026-07-12 - R010 Relay Test Needed Unsandboxed Loopback

The first image test invocation inside the restricted execution sandbox passed
the five pure wrapper/warmup tests but could not open the TCP loopback socket.
That `EPERM` was an execution-environment restriction, not relay evidence.

Decision: rerun the same test suite with local loopback permission. Do not
mark a network relay tested from pure mocks or reinterpret sandbox denial as a
candidate failure.

### 2026-07-12 - R010 TCP Half-Close Bug Fixed Before Benchmark

The first real loopback test returned an empty response. The relay cancelled
both copy tasks as soon as the client finished sending its request, so an HTTP
request-side half-close could terminate backend-to-client response copying.

Resolution:

- propagate end-of-stream with `write_eof()` independently in each direction;
- wait for both copy tasks instead of cancelling the peer on normal EOF;
- close writers only after both directions settle;
- keep a byte-exact request/response regression test.

The corrected suite passed `6/6`, including the live loopback relay. Only the
corrected implementation entered Docker build, startup validation and A/B/A;
the broken relay never served a benchmark run or submission artifact.

### 2026-07-12 - R011a Held Before GPU Startup

R011a static gates passed, but resource preflight measured only `12600MiB`
free on the RTX 5080. The unchanged local `gpu_memory_utilization=0.82` profile
needs about `12965MiB`, leaving a shortfall of roughly `365MiB`. Docker/GPU
execution also became temporarily unavailable to the agent session before any
container started.

Decision:

- do not lower memory utilization because that would change I004 cache capacity
  and invalidate the seq20/seq18 comparison;
- do not terminate unrelated desktop processes;
- do not record a startup or candidate failure;
- resume the exact A/B/A only when at least about `13.0GiB` VRAM is free and
  Docker/GPU execution is available.

### 2026-07-12 - R013 Full-Trace Control A Desktop Outlier

One R013 control reached `14492MiB` peak VRAM while the stable control and
candidate runs were near `12.9GiB`. Its ERS `0.366598` was inconsistent with
the adjacent controls and concurrent desktop GPU use was visible. It is
retained in raw outputs but excluded from the stable control mean. R013 claims
only the midpoint of the two resource-consistent controls, not the outlier.

### 2026-07-12 - R014 Fusion Pass Replaced Zero Patterns

`fuse_norm_quant=True` resolved correctly and the server reached Ready, but
each compilation partition logged `Replaced 0 patterns`. The pass added
compile time without changing execution. Do not run or submit this flag unless
a future vLLM graph visibly matches at least one pattern.

### 2026-07-12 - R015 FP8 Mask Fill Compile Error

The first FP8-state kernel used `tl.load(..., other=0)` on an E4M3 pointer.
Triton attempted to cast the integer fill value to FP8 and rejected all six
candidates. Changing the fill to `0.0` fixed compilation. This was a harness
implementation error, not FP8 performance evidence.

### 2026-07-12 - R015 Invalid Python-Wrapper Speedup

The first successful FP8-state run compared a direct custom Triton launch with
the validation-heavy native Python wrapper and appeared `1.37x` faster. CUDA
event elapsed time included GPU idle gaps while Python prepared repeated native
launches. After both paths used direct Triton ABI, FP8 was `0.9554x` at batch
20. The earlier timestamp is preserved but must never be quoted as speedup.

### 2026-07-12 - R015 Quantized State Rejected

- FP8 dynamic row scaling accumulated output NRMSE p95 `0.0901` over 200
  updates and was slower than BF16.
- INT8 dynamic row improved NRMSE p95 to `0.0243`, but missed the strict gate
  and all INT8 candidates remained slower.
- Storage reduction alone does not justify allocator complexity or quality
  risk. Do not reopen mixed-layer state on the same kernel implementation.

### 2026-07-12 - R016 Unsupported Prefill Paths

- FLA chunk128 asserted for every tested length; current triangular orchestration
  supports the fixed 64 path, not an arbitrary larger chunk.
- CuteDSL failed with `mlir_global_dtors()` runtime error; vLLM's Cutlass DSL
  CU13 integrity check returned false.
- Direct FlashInfer GDN rejected device major 12 because the packaged launcher
  targets SM90. This is a local architecture boundary, not H200 performance
  evidence.
- Keep automatic backend selection and do not force a Blackwell-local result
  into the H200 Compose.

### 2026-07-13 - R018 GPU Copy Benchmark Approval Unavailable

The R018 Docker/GPU copy microbenchmark did not start. The privileged
execution approval layer rejected the launch because its usage limit had been
reached. No container, GPU kernel or candidate serving process ran.

Resolution:

- do not retry through a different GPU execution path;
- run only the deterministic trace/state/VRAM upper-bound analysis from frozen
  exact R005/R006 evidence;
- treat GPU copy as unmeasured, not failed;
- close R018 anyway because the independent benefit gate reaches only `5.89%`
  compute and `4.18%` of the Portal TTFT p50 anchor, both below `10%`.

### 2026-07-13 - R021 Analysis And Validator Environment Mismatches

Two tooling-only failures occurred after successful benchmark data had already
been saved:

- an ad-hoc paired script expected `request_score` inside raw metrics and
  raised `KeyError`; raw rows intentionally store latency fields, so the
  corrected script called `llm_race_lab.metrics.request_score` explicitly;
- the first Attempt 12 validator invocation used the root `.venv`, which does
  not contain PyYAML, and failed with `ModuleNotFoundError: yaml`.

Resolution: rerun the immutable validator with the RTX lab venv, where PyYAML
is pinned. The corrected paired analysis and Compose validator both pass. No
server run, image, Compose content or performance result changed.

### 2026-07-13 - Attempt 11 Historical Path Was Mutated To Seq16

The path named `attempt_11_vllm_i006_readiness_warmup.yml` no longer matched
its recorded immutable SHA `01fe58...`; its current SHA is `e28f0a...` and it
contains seq16. Deterministic reconstruction showed that changing only seq16
back to seq20 reproduces `01fe58...` exactly. Therefore the graded `46.03`
artifact was seq20, not seq18/seq16.

Resolution:

- do not revert the user's seq16 file;
- archive the exact original bytes as
  `attempt_11_frozen_original_i006_seq20.yml`;
- rerun the original frozen validator against that archive; it passes;
- associate the current seq16 SHA `e28f0a...` with the user-reported score
  `51` only provisionally until its Portal label and full metric panel arrive;
- require a new filename and SHA for every future submission artifact.

### 2026-07-13 - R022a Unsupported FP8 Linear Backends

The exact I006 image accepted the backend names at CLI parsing, but engine
initialization rejected four of them:

- `flashinfer_cutlass`: incompatible online-FP8 activation/weight scale
  contract;
- `flashinfer_cudnn`: no kernel for the layer type;
- `flashinfer_trtllm`: no kernel for the layer type;
- `triton`: no kernel for the layer type.

Explicit `cutlass` reached Ready but selected the same
`CutlassFP8ScaledMMLinearKernel` as auto. Treat it as a no-op, not a positive
performance result. Do not bypass capability errors or benchmark identical
resolver outcomes under different names.

### 2026-07-13 - R022b Harness Recovery And xxhash Boundary

Three tooling failures were kept separate from valid R022b evidence:

- the first seq20 invocation passed a relative output path that Docker treated
  as a named volume; no timed request ran;
- a later runner session died after starting the server and left the container
  idle after prime; no timed request ran, the exact container was removed, and
  the experiment restarted in a new directory;
- the initial A/B/A loop reused the same path for repeated candidate names, so
  control B overwrote control A. The harness now prefixes every run with a
  stable ordinal and the complete bracket was rerun.

The first xxhash candidate also failed readiness because vLLM exposes the flag
but its baseline image omits the optional `xxhash` module. Pinning
`xxhash==3.6.0` fixed startup. The corrected A/B/A then showed no gain:
candidate ERS `0.959593` versus SHA256 midpoint `0.961662` (`-0.215%`). Keep
the corrected negative result; never quote the dependency failure as a
performance result and never quote the overwritten bracket as A/B/A evidence.

### 2026-07-13 - R022c Telemetry Was Silent Then Off By One

The first startup smoke completed but the named Python logger emitted no cache
activation/stats under vLLM's logging configuration. No A/B/A was started from
that unproven smoke. Aggregate markers moved to stderr and the next smoke
proved 140 logged native-render misses before readiness.

The first A/B/A then reported 119 timed hits because telemetry emitted every
20 total calls. Startup ended at call 141 and timed traffic ended at call 261,
so the last hit was real but never printed. The implementation was unchanged;
logging was corrected to emit every 20 hits independently. A full bracket was
rerun and directly recorded 120 timed hits plus ERS `0.999870`. Do not quote
the 119 counter as a cache miss or lower the gate to fit it.

### 2026-07-13 - R022c Jitter Shortcuts Rejected

Two shortcuts were tested only after the corrected cache path qualified:

- disabling access logs scored `0.999816` against deep-copy midpoint
  `0.999975` and worsened TTFT p95;
- returning shared cached objects reached `1.000000`, but both deep-copy
  controls also reached `1.000000`, so it added mutation risk without score.

Resolution: keep default access logging and return independent deep copies.
The latest retained run scored exact ERS `1.000000` with TTFT maximum
`83.974ms` and TPOT maximum `9.071ms`. Record these as negative design
boundaries, not as reasons to reopen the closed jitter branch.

### 2026-07-13 - R022e Desktop VRAM Retries And MTP Performance Rejection

The first MTP capability probes at local memory fractions `0.82`, `0.81` and
`0.79` stopped before model weight load. In each case, vLLM's requested memory
quota exceeded the RTX 5080 memory free after desktop/display use by a small
amount. These are resource-precondition retries, not evidence that MTP, FP8 or
the fixed checkpoint is unsupported.

At `0.78`, capability passed: `Qwen3_5MTP` resolved, no second full model was
loaded, and bounded piecewise CUDA Graph capture completed. The valid
native/MTP/native performance bracket then rejected the technique itself:

```text
ERS:             0.552309 midpoint -> 0.466266
TTFT p50:        1696.6 -> 3336.9ms
TPOT p99:        22.147 -> 25.611ms
throughput:      768.512 -> 700.116 tok/s
MTP acceptance:  67.643%
```

Resolution: keep the precondition retries separate from the valid negative
performance result. Do not increase draft length, run dependent seq18/20 or
tied-head experiments, or spend GPQA time on this rejected candidate.

### 2026-07-13 - R023a Desktop VRAM And Recovered Control B

The first R023a A/B/A at fraction `0.79` failed pre-weight in all three cells
because desktop use left `11.31-11.59GiB`, below the requested model quota.
At `0.72`, control A and telemetry completed, but control B missed the
precondition by roughly `70MiB`. The analyzer originally assumed every
`summary.json` existed and raised an exception; it was fixed to mark an
incomplete bracket invalid.

Resolution: rerun only immutable control B at the same fraction and merge the
three valid run paths. Every valid run had exactly `456542/1962800`
compute/cache tokens. Failed preflight attempts are not benchmark evidence.

### 2026-07-13 - R023b One-Shot Classification Miss

The first R023b policy classified each waiting request once. RID 2-4 were
checked before the leader scheduled the next full block, then never checked
again; RID 5-20 were held after the opportunity appeared. The run completed
but saved zero compute.

Resolution: re-evaluate only when cache or full-block leader progress changes.
This correction caught RID 2-20 and exposed the deeper negative result below.

### 2026-07-13 - R023b Follower Publication Deadlock

The corrected policy held all `19` shared-prefix followers until cache commit.
No request advanced committed cache beyond `5040/6160`, so all holds timed out
near one second. Compute increased `1120`, duplicate work increased `50%`, and
TPOT p99 reached `19.910ms`.

Resolution: reject admission-only hybrid singleflight. The remaining state
requires framework-level atomic in-flight attention-KV plus Mamba-state
publication, whose maximum observed compute benefit is only `0.4906%`.

### 2026-07-13 - R023e Probe Drift And Loaded-Path Namespace Collision

The first deadline-scheduler analyzer confused an environment-selected overlay
path with the module's own loaded-path marker. That namespace collision made
the activation check ambiguous. The marker was renamed to
`_deadline_loaded_path`, startup/MRO was revalidated, and no timed evidence was
accepted before the correction.

The corrected candidate probe then looked positive at ERS `0.581264`, but the
required A/B/A scored only `0.567834` versus control midpoint `0.570884` and
reduced the fast-TTFT count `66 -> 54`.

Resolution:

- keep probe results diagnostic only;
- require exact activation markers plus same-session A/B/A before promotion;
- reject R023e based on the corrected bracket, not the positive probe.

### 2026-07-13 - R023g Static Memory Saving Did Not Become KV Capacity

The one-storage FP8 tied matrix passed static accounting and saved about
`484-485MiB`, but runtime reported fewer KV tokens than a nearby native run:
`1,304,166/1,274,423` versus `1,325,339`. Dynamic FP8 projection workspace
and graph/runtime allocation consumed the expected persistent-memory margin.
TPOT p99 also remained `21.480/20.553ms`.

Resolution:

- never promote a storage optimization from tensor byte counts alone;
- require startup KV capacity, peak workspace, full-trace TPOT and ERS;
- reject both variants before A/B/A/quality and keep the simpler native tied
  storage.

### 2026-07-13 - DR02 Stale Canonical Snapshot Path

All three DR02 GPU profiles completed successfully, but the first post-run
analysis exited because the harness referenced a non-existent
`outputs/canonical_20260712/hotspot_summary.json`. The canonical summary is at
the round root. The path was corrected, the existing traces were reanalyzed,
and no GPU case was rerun. Shell syntax, Python compile and `5/5` analyzer
tests pass. Treat the exit as an analysis-harness failure only, not invalid GPU
evidence.

### 2026-07-14 - DR07 Sparse Snapshots Underestimated Peak Pool Use

The first DR07 report computed peak pool use from activation and every-20-
completion snapshots only. It reported `12.02%`, but those sparse samples
missed the within-wave allocation peak. The observer had already recorded
free-block counts at every one of `2425` schedule calls, so the analyzer was
corrected to use all schedule events. The valid result is `25.323%` peak use,
with a minimum `2026/2713` blocks free. No GPU run was repeated.

Resolution: capacity reports must use the highest-frequency existing pool
telemetry, while sparse snapshots remain useful for ownership composition.
Both values are far below the `85%` opening gate, so the correction does not
change DR07's negative decision.

### 2026-07-14 - DR16 Exact-Image Package Audit Permission Block

The exact I006 image registry and selected source files were inspected, but a
later no-GPU package-inventory Docker call was rejected after the execution
approval layer reached its usage limit. This did not demonstrate a missing
runtime package or a quantization failure.

Resolution: mark package inventory and the offline converter environment as
open blockers, park GPTQ W4A16 before GPU, and do not infer technical failure.
All checkpoint, shape, source, Amdahl and converter-contract work remains valid.

### 2026-07-14 - DR20 Broad Overlay COPY Caught Before Build

The first package validator attempted to read every file under `overlay/` and
found generated Python bytecode from local `py_compile`. The Dockerfile's
initial broad `COPY overlay/` would also have included that environment-
dependent bytecode.

Resolution: copy exactly `continuation_tokenizer_cache.py`,
`dr20_hardening.py` and `sitecustomize.py`; validate a closed source list and
scan only source text. The corrected package audit passes with zero failures.
No image was built from the broad-COPY draft.

### 2026-07-14 - DR20 GPU Brackets Unrun

The hardened implementation, A/B/A runner and four-process GPQA runner are
ready, but Docker/GPU execution remains unavailable under the same approval
limit. No build or benchmark command was retried after that blocker.

Resolution: record `pending_execution_permission`, preserve exact resume
commands, and do not classify DR20 as passed or failed until fresh performance
and quality gates run.

### 2026-07-14 - DR20 Verification Runner And Gate Assumptions

The system Python does not include `pytest`; invoking it failed before test
collection. This was a runner availability issue, not a test failure. The
repository's `unittest` suites were used instead and passed.

One manual package-validator invocation omitted its required `--project`
argument and exited at argument parsing. Re-running with `--project .` passed
with zero contract failures; no package content was changed by the bad call.

An expanded mixed hit/miss test initially expected two cache entries, but the
correct lifecycle retains the old root and publishes both newly encoded rows,
for three entries total. Token IDs were exact; the test expectation was fixed.

The first process-quality analyzer compared only intersecting case IDs. That
could falsely report equality for incomplete or mismatched result sets. It now
fails closed unless every run has the same 100 IDs, every output hash exists and
both candidate telemetry snapshots cover at least 100 requests.

### 2026-07-14 - DR20 Quality Attribution Rejected

The fresh performance removal bracket passed, but the first exact four-process
quality gate failed because even the no-cache controls changed `19/100`
predictions between fresh processes. Candidate repeats changed `12/100`; all
cells were `100/100`, penalty proxy stayed `1`, and candidate cache hits were
zero. This was classified as process-noise inconclusive rather than a cache
failure.

Before any repeat, a two-panel process-noise gate was frozen and unit-tested:
at least 70 jointly stable cases, zero stable candidate-specific changes,
candidate repeat excess at most 5, and cross-drift excess at most 5. The single
authorized repeat had 80 jointly stable cases but two stable candidate-specific
changes, so it failed the zero-change guard. The threshold was not relaxed.

Across all eight processes, 67 cases were globally stable with zero globally
stable candidate-specific changes. This does not prove systematic accuracy
loss, but it also does not provide clean attribution for an inactive GPQA path.
Resolution: reject DR03b integration, retain its performance result as research,
preserve I006 and do not rerun unchanged.

Two non-candidate command issues occurred during verification. Sandboxed
`nvidia-smi` could not reach the driver; the approved host check succeeded with
`14651MiB` free. One manual `jq` arithmetic expression had incorrect grouping;
the corrected expression measured the fresh relative ERS gain at `+0.7436%`.

### 2026-07-14 - R024 Loopback Test Blocked By Sandbox

The first shared startup-wrapper test invocation ran inside the restricted
sandbox. Five tests passed, but the byte-exact TCP relay test could not create
a loopback socket and raised `PermissionError: Operation not permitted`.

Resolution: rerun the unchanged six-test suite outside the socket-restricted
sandbox. It passed `6/6` in `0.012s`. This was an execution-environment false
failure; no source or test expectation changed.

### 2026-07-14 - R025 Static Invocation Corrections

The first R025 analyzer smoke test assumed every historical summary contained
`ttft_buckets`; older summaries do not. The analyzer now reconstructs the
fast-TTFT count from `metrics.jsonl` and has a regression test for that schema.

The first tokenizer probe called `len()` on Qwen's `BatchEncoding` and
reported its two mapping keys instead of sequence length. Reading
`input_ids.shape[-1]` produced the correct `28,819` chat tokens and `3,947`
tokens of context headroom.

The first test invocation used `python3 -m pytest`, but neither system Python
nor the lab venv installs pytest. R025 uses `unittest`; the correct discovery
command passes `10/10`. None of these invocation errors produced a candidate
image or GPU result.

### 2026-07-14 - R025 GPU Guard Timed Out Before Startup Probe

The frozen runner requires at least `15000MiB` free because historical I007
peaked at `14837MiB`. A ten-minute watcher observed only `13.6-14.1GiB` free
and sustained `9-15%` utilization from unrelated redroid containers. It exited
with code `75` before starting the R025 server.

Resolution: do not stop unrelated containers and do not lower the memory guard
or accept a contaminated benchmark. Resume the startup-only probe after the
RTX 5080 is idle; no R025 container/session remains active.

### 2026-07-14 - R025 Guard Recalibration And Quality Retry

After explicit user authorization, both redroid containers stopped. GPU free
memory remained around `13.8GiB` because desktop/display allocations persist.
Historical peak `14,837MiB used` already included that baseline, so the old
`15000MiB free` requirement was dimensionally over-conservative. An unchanged
I006 startup probe at memory fraction `0.82` passed at `107s`, used
`14,495MiB` total and retained `1,314MiB` free. The guard was therefore
recalibrated to `13500MiB`; runtime memory settings were not reduced.

The first GPQA command then found `13,412MiB` free, 88MiB below the new guard,
and exited `75` before container creation. After 20 seconds of driver cleanup,
free memory reached `13,958MiB`; the unchanged retry passed all quality gates.
This is a preflight retry, not a selected quality repeat.

At that checkpoint R026 was not started because `BOMBANANA.exe` began consuming
about `1.2GiB` and total GPU utilization reached `42%`. No attempt was made to
stop or interfere with the desktop game; R026 was pending clean execution.

### 2026-07-14 - R026 Graph Padding Surrogate Did Not Predict Decode Gain

After the GPU became free, graph11 passed the literal first A/B, but a
reverse-order warm control matched it within `0.000097` ERS and had better
TPOT p99. The original `+0.005198` gain was mostly process/warm-state drift.
Do not cite the first pair alone or promote graph11 from local evidence.

Adding graph6 to graph11 reduced ERS by `0.001479`, produced W/T/L `5/28/87`
and worsened TPOT p99. The stage-3 graph13 run was correctly skipped. Static
padding-removal percentages are opportunity upper bounds, not measured kernel
speedups; graph memory, launch geometry and scheduling can erase or reverse
them. Do not rerun graph6+11 or graph6+11+13 unchanged.

### 2026-07-15 - R028 Fixed Prefill Budget 4096 Lost The Score Shape

The earlier I004 screen made budget `4096` look Pareto because it improved
headline percentiles, throughput and TTFT-under-ceiling count. Exact-I006
`2048/4096/2048` A/B/A showed why those summaries were insufficient.

Fixed `4096` increased fast TTFT `78 -> 84`, improved TTFT p95 and reduced raw
TPOT p95/p99, but ERS fell `1.43%`. It reduced the `<=500ms` cohort
`22.5 -> 18`, lowered TTFT score `0.221140 -> 0.203685`, and lost request score
in every wave. TPOT score was already saturated at `1.0`. R030 later corrected
the telemetry mechanism: raw lookup hits fell `33.87%` while lookup queries
fell `34.54%`; actual request cache attribution fell only `0.143%`. Prefill
phase wall time rose `48.14%`, but that is not GPU compute. There was still no
OOM, preemption or fallback to blame.

Resolution: reject fixed `4096`, keep resolved/default `2048`, and do not use
fast count, p95 or raw throughput alone as promotion criteria. Do not rerun
fixed `4096` or `16384` unchanged and do not create Attempt 17 from this branch.

### 2026-07-15 - R029 Tree-Gated Budget Selector Overfit

The first quick oracle sketch zipped JSONL rows by file order. Completion order
is not request order, so it reported a slightly incorrect oracle. The formal
analyzer joins exact request IDs, validates `0..119`, and produces the canonical
midpoint/robust oracle gains `0.457%/0.351%`.

A depth-2 legal-feature tree then fitted `+0.001523` ERS on all 120 observations
but reversed to `-0.000224` midpoint and `-0.000764` robust gain under
leave-one-wave-out. Root features changed across folds and only `2/6` waves
were positive. More XGBoost depth or C++ training would increase fit capacity,
not create causal scheduler signal.

Resolution: use request-ID joins and held-out wave evaluation for every future
counterfactual audit. Close Tree-Gated `2048/4096` selection without GPU and do
not interpret in-sample gain as serving evidence.

### 2026-07-15 - R030 Raw Prefix-Hit Counter Was Misinterpreted

R028 initially described a `33.87%` reduction in
`prefix_cache_hits_total` as lost prefix-cache reuse and treated cumulative
request prefill time as prefill compute. Exact I006 source disproved both
interpretations.

The query/hit counters are updated before slot allocation. A waiting request
can be looked up and counted again after an allocation failure, so their raw
magnitudes depend on scheduler iteration count. In R028, lookup queries fell
`34.54%` and lookup hit rate actually rose `0.800pp`. The one-per-request
source metric showed only `2,800` fewer local-cache tokens (`0.143%`). The
request prefill timer spans scheduled timestamp to first-token timestamp and
includes wall delay; it is not a CUDA kernel timer.

Resolution: use query/hit rates for lookup behavior,
`prompt_tokens_by_source_total` for unique request attribution, and GPU
profiling for compute claims. Correct the mechanism text but retain the R028
rejection because ERS and frozen gates are unchanged. Do not open a cache-loss
repair from a repeated lookup counter.

### 2026-07-15 - R031 Fixed 4096 Lifecycle Direction Did Not Reproduce

The forward `2048/4096/2048` bracket showed a small order-sensitive ERS
picture, so treating that one bracket as causal would have repeated the R026
process-drift error. The reverse `4096/2048/4096` bracket changed the process
ordering and produced the required three-process estimate per budget.

Combined evidence rejects `4096`: ERS fell `1.80%`, queue time improved
`138.716ms`, but scheduled-to-first-token service worsened `122.927ms`.
Service worsened in all six waves; score lost in five. Wave 2 remained positive
because its queue saving exceeded its service penalty, so the strict all-six
direction gate failed.

Resolution: do not implement a fixed or Tree-Gated `2048/4096` scheduler and
do not create Attempt 17. Preserve the lifecycle observer as measurement-only.
Any final-chunk/service-isolation idea must first pass an offline,
leave-one-wave-out opportunity audit on the existing six-run panel.

### 2026-07-15 - R032 Ideal Final-Chunk Splice Needed Too Much Recovery

The optimistic `100%` splice suggested `+3.804%` mean relative ERS, but it
removes service delay without scheduling the deferred peer work. Treating that
ceiling as implementable would overstate the candidate.

At the frozen `50%` net-recovery stress, only `6/9` process pairs won, waves 1
and 5 lost, and the `<=500ms` cohort lost `1.019%`. A process pair needed
`66.85%` recovery merely to clear `+0.5%` ERS. This is insufficient margin for
queue debt, step detection and scheduler overhead.

Resolution: close R032 without step telemetry or GPU. Do not reopen the fixed,
Tree-Gated or final-chunk `2048/4096` family from the ideal phase-splice number.

### 2026-07-15 - R033 DR03b Historical Gain Did Not Survive Same-Image Control

Two older brackets reported `+0.850%/+0.744%` ERS for continuation tokenizer
caching, but candidate and controls did not use identical images. R033 first
confirmed that quality was not the blocker under the official formula, then
ran one same-image off/on/off bracket.

The candidate lost `0.257%` ERS to the control midpoint, had W/T/L `32/35/53`,
only three positive waves and `81` fast-TTFT requests versus `83/83`. Its
TPOT p99, `99` cache hits and zero unsafe fallback passed. Saving `1.85M`
tokenizer tokens therefore did not guarantee end-to-end latency gain; scheduler
and process variance absorbed or reversed the frontend saving.

Resolution: close unchanged DR03b and do not repeat until it wins by a new
causal mechanism. Skip the conditional fresh GPQA run and Attempt 17 package
because the frozen performance dependency failed.

### 2026-07-15 - R034 GPTQ W4A16 Marlin Lost To Native FP8

The static DR16 screen predicted meaningful storage reduction, but it did not
prove that weight-only INT4 would execute faster. R034 built the missing exact
converter/runtime path. Converter-only GPTQModel required protobuf 7 and a
narrow Transformers compatibility shim; those dependencies were isolated from
the serving image. CPU packing touched the 8GiB cgroup ceiling, so the final
converter used GPU packing and completed without swap/OOM. A first local
`gpu-memory-utilization=0.90` smoke could not start because the active desktop
left less free VRAM than vLLM's requested reservation; `0.84` smoke proved
correctness, and the controlled full bracket used `0.82` for every cell.

Artifact, native Marlin startup and GPQA `37/37/35` all passed. Performance did
not: ERS `0.542724` versus FP8 midpoint `0.609622`, TPOT p99 `23.213ms`, fast
TTFT `41` versus `78/79`, W/T/L `0/37/83`, all waves negative. W4 created
`4.607%` more GPU blocks, but average sampled utilization/power increased
`44.9%/185.8W -> 86.1%/234.0W`. The trace had no memory-capacity pressure, so
Marlin unpack/dequant overhead dominated.

Resolution: reject all-core GPTQ/AWQ on this vLLM/RTX path. Do not package or
submit the artifact and do not infer speed from compressed size. Reopen only
for a distinct H200-native INT4/NVFP4 kernel with target-image microbenchmark
evidence.

### 2026-07-15 - R035 Complete-Log JIT Count Was Not Timed JIT

Failure: the first R034 analyzer counted every `JIT compilation during
inference` warning in `server.log` and labeled the result timed. This included
startup/readiness warmup, creating a false next-round target.

Evidence: all three `server_prebenchmark.log` files are exact byte prefixes of
their complete logs. Each prefix already contains all five warnings, followed
by warmup completion and public readiness. Every suffix contains zero JIT.

Resolution: count only the exact post-snapshot suffix and fail closed when the
snapshot is not a prefix. Add regression tests, correct R034 hashes/docs, and
close R035 without GPU. Do not repeat broad warmup or JIT shape hunting from a
complete-log count.

### 2026-07-15 - R036 Local Total ERS Failed As An H200 Selector

Failure: treating the best local total ERS as the next H200 candidate would
select the wrong direction for active-set 20 to 16, FP8 tied head, and
active-set 19 to 18. Local TPOT was already saturated in most runs, while the
Portal still had a score-bearing decode tail hidden behind median TBT.

Evidence: seq20 to seq16 lost `1.026` local score points but gained `5.56` on
Portal; seq19 to seq18 lost `0.250` locally but gained `0.18` on Portal. Four
Portal panels report TBT median `<=20ms` while score remains below 50, which
proves mean TPOT score is unsaturated.

Resolution: compare TTFT support, TTFT component and decode distribution
separately. Do not launch another scalar sweep from local total ERS. Keep I006
seq16 and require R037 to identify a distinct target-only primitive before any
GPU work.

### 2026-07-15 - R037 DS Layout And Easy MRV2 Paths Missed The Opening Gate

Date: 2026-07-15

Round: Score 100 R037

Action: audit `VLLM_SSM_CONV_STATE_LAYOUT=DS`, direct MRV2 and MRV2 with prefix
cache disabled as exact-I006 H200 decode candidates.

Expected: remove a hidden GDN stride cost or reduce host runtime enough to
repair the H200 TPOT residual.

Observed: DS is executable but causal convolution is only `1.296%` of pure
decode and `1.196%` of cold prefill. Direct MRV2 rejects Mamba `align`.
Disabling prefix caching would discard `81.129%` request-attributed reuse and
raise prompt compute to `5.299x` control.

Root cause / best hypothesis: DS changes only the small convolution-state
access path, while I006 already receives full TPOT-floor credit locally.
MRV2 0.22.1 does not implement align-state publication/restore semantics.

Decision: no standalone DS benchmark, no validator bypass, no no-prefix MRV2,
no image/Compose/Attempt 17.

Do not repeat unless: DS is bundled for free into an independently qualified
non-speculative candidate, or a newer exact runtime adds real MRV2 align
support with retained prefix reuse.

Related files:
`research_rounds/score_100_program/round_037_exact_i006_h200_decode_capability_audit/`.

### 2026-07-15 - Independent Queue-Removal Model Was Too Optimistic

Round: Score 100 R038

Action: initially model each selected request as independently removing a
fraction of queue delay while leaving the remaining selected prefills
unserialized.

Expected: obtain a first upper bound for logical first-token staging.

Observed: the model selected 64 requests, moved nearly all of them under the
TTFT ceiling and reported zero parking for the best cell. It produced 91
apparently robust candidates, an implausibly broad result.

Root cause: multiple staged prefills could overlap without competing for one
GPU staging lane, so queue reduction was counted independently and causality
was violated.

Resolution: reject that output. Serialize the staging lane, update decode
occupancy after every staged first token, charge parking and overlap to TPOT,
and require process plus 36-wave gates. The corrected result is narrow: only
three guarded candidates, all requiring 75% queue reduction and 10% decode
interference.

Do not repeat: never promote an offline queue-latency splice that does not
model shared service capacity and decode externality.

Related files:
`research_rounds/score_100_program/round_038_first_token_staging_opportunity/`.

### 2026-07-15 - R039 First Runtime Probe Crossed Readiness Boundary

Failure: the first exact-image R039 probe at
`outputs/runtime_probe/20260715-203439/` treated `/v1/models` reachability as
timed readiness. The client then overlapped the package's synthetic warmup and
startup Triton compilation.

Observed: the run was not a valid latency baseline because the public warmup
marker had not appeared before timed traffic. The server was not broken; the
measurement boundary was wrong.

Resolution: require `[startup-warmup] public relay ready` before launching any
timed client, retain the failed run for provenance, and exclude it from all
comparisons. The corrected native, shadow and native-repeat runs completed
`20/20` successfully under the same boundary.

Do not repeat: `/v1/models` is an availability check only, not the benchmark
readiness signal for this image.

Related files:
`research_rounds/score_100_program/round_039_gap_aware_scheduler_prototype/`.

### 2026-07-15 - R040b First-Token Swap Paid Decode Cost Without TTFT Support

Round: Score 100 R040b

Action: displace one old decoder, admit one waiting request through its first
token, defer the new decoder, and restore the old decoder immediately.

Expected: recover most late-request queue delay while bounding the old
decoder's TPOT penalty to one prefill/first-token interval.

Observed: two complete swap/resume pairs were state-safe, but ERS proxy was
`0.454380` versus same-resource native midpoint `0.466172`. TTFT support stayed
`5/20`, while TPOT p95 reached `26.491ms`. The staged requests improved about
`1.0-1.1s` but still had TTFT `4323/4583ms`.

Root cause / best hypothesis: the only requests outside native active seq16
were already too far beyond the 1500ms ceiling. Parking shifted latency from
queue to decode but could not make those requests score-bearing.

Decision: close direct hold and first-token swap. Do not sweep cap, age or
stage count unless a new opportunity model proves a waiting request can cross
the official ceiling with TPOT p99 inside guard.

### 2026-07-15 - R040 Local Probe Infrastructure Exclusions

Two non-candidate failures were retained but excluded from latency analysis:

- two `0.82` startup attempts exited before readiness because unrelated
  desktop processes left `11.59/15.44GiB` free versus vLLM's `12.66GiB`
  request; the harness was corrected to preserve startup logs;
- one native control reached warmup readiness but the client did not start
  because a relative output path was interpreted as an invalid Docker volume
  name.

Resolution: add a parameterized local GPU reservation, run candidate and both
controls at `0.72`, and use absolute output paths. Never classify startup
headroom or client-mount failures as scheduler performance results.

### 2026-07-15 - R041a Non-Root Audit Missing Passwd Entry

The first exact-image capability audit used host UID/GID `1000:1000`. Torch
called `getpass.getuser()` during import, but the image has no passwd entry for
that UID and raised `KeyError: getpwuid(): uid not found: 1000` before GPU work.

Resolution: keep non-root output ownership, set `USER/LOGNAME=msi`, and use an
explicit `/tmp` TorchInductor cache. The retry completed and passed every
capability gate. Treat the first invocation as harness failure only.

### 2026-07-15 - R041 Alternate FP8 Backends Missed Performance Gate

Static FlashInfer and Torch row-wise scaled-MM both executed with finite,
numerically matching output, but lost `8.762%` and `3.954%` shape-weighted MM
performance to native CUTLASS. The FlashInfer `~0.10ms` launch floor was
especially damaging for the `M=211` final chunk. Static CUTLASS improved the
standalone quant-plus-MM path only `1.319%`.

Resolution: no checkpoint conversion or backend overlay. Do not repeat on the
same image/device unless the backend implementation changes; capability alone
is not a performance opening.

### 2026-07-15 - R042 Conditional Budget Signal Failed Robust Action Gate

Action: revisit R031's mean-positive wave 2 with six processes, arrival/cache
features and a scheduler-global 2048/4096 decision.

Observed: wave 2 is positive in both bracket directions but only `6/9`
process pairs agree. Its mean delta is `+0.006770`; robust q25 is
`-0.008325`. The executable wave oracle adds `+0.001128` ERS and its robust
gain is zero.

Root cause: the apparent cohort is smaller than process drift and depends on
which complete process is compared. Per-request selection inflates the bound
but cannot be executed by vLLM's scheduler-wide token budget.

Resolution: keep fixed 2048. Do not train a deeper Tree-Gated/XGBoost model or
write a scheduler mutation for this branch. First demand a positive robust
global-action oracle above `0.005` ERS.

### 2026-07-15 - R043 First Probe Did Not Register vLLM CUDA Ops

Failure: the first exact-image microbenchmark imported top-level `vllm` only,
then called `torch.ops._C.rms_norm`. The extension namespace had not been
registered and the process exited before any timing.

Resolution: import `vllm._custom_ops` explicitly for its registration side
effect, retain the same resource/timing contract, and rerun. The corrected run
completed all shapes and seven trials. Exclude the first invocation from all
performance claims.

### 2026-07-15 - R043 Eager Control Created A False Decode Opening

Action: compare native fused RMSNorm/dynamic-FP8 with separate eager RMSNorm
and quant CUDA ops.

Observed: native fused appears about 31% faster at M1/M20, but R012's real
compiled Triton reduction averages `2.832us/call`, versus `3.822us` for native
fused M20. Large prefill M1680 also regresses `175.056%`.

Root cause: the eager control has two launches, while production Inductor has
already fused the reduction into one optimized Triton kernel. The comparison
did not represent serving execution.

Resolution: reject decode and prefill integration. Future kernel screens must
compare against the exact compiled trace kernel, not a decomposed eager path.

### 2026-07-15 - R044 Backend Flags Produced No Distinct Target Candidate

Action: screen forced FlashAttention, FlashInfer, Triton and CuteDSL paths for
the remaining attention/GDN prefill hotspots.

Observed: H200 auto already selects FlashAttention-3 and FlashInfer. Forced
Triton attention has prior `-8.76%` ERS evidence, CuteDSL GDN is
Blackwell-only, and no custom path exists to benchmark. Zero of eight
candidates passed all gates.

Root cause: hotspot share was mistaken for available optimization headroom;
the target resolver already uses its specialized kernels.

Resolution: keep backend selection automatic. Do not repeat forced-backend
runs unless the image implementation changes or a concrete target-beating
kernel is added.

### 2026-07-17 - Part 2 Round 002 `xxhash` Cache Hash Failed at Runtime

Action: replace the default SHA-256 prefix-cache hash with
`--prefix-caching-hash-algo=xxhash` on the FP8-KV plus async incumbent and run
the six-turn `session_prefix` workload.

Observed: server startup and healthcheck passed, but all `420/420` requests
failed during request preprocessing. The exact server exception was
`ModuleNotFoundError: xxhash is required for the 'xxhash' prefix caching hash
algorithms` and all `330` graded requests received zero output tokens.

Root cause: the immutable `vllm/vllm-openai:v0.22.1` image does not include the
optional Python `xxhash` package. Startup validation alone cannot detect this
because the module is imported lazily on the first cache-key calculation.

Resolution: reject `xxhash` for compose-only submission. Keep the default
SHA-256 hash; do not add runtime package installation or a custom image unless
BTC changes the packaging contract. Future hash candidates must pass a real
one-request preprocessing smoke test before a full trace.

### 2026-07-17 - Local Screen Command Used an Invalid Working Directory

Failure: the first seq10 `session_prefix` screen command referenced a
duplicated/nonexistent workspace path, so the shell process was never created.
No container, server request or benchmark output was affected; the command was
retried immediately with the round directory and completed successfully.

Resolution: treat this as harness invocation noise, not a model/runtime result.
Verify `workdir` against the round path before launching future screens.

### 2026-07-17 - Part 2 DBO Requires Unavailable DeepEP Backend

Action: start the incumbent with `--enable-dbo`, keeping async scheduling and
the default DBO thresholds (`decode=32`, `prefill=512`).

Observed: the API process parsed the flag but failed before engine startup with
the vLLM validation error that microbatching/DBO only supports the
`deepep_low_latency` or `deepep_high_throughput` all-to-all backends. The
current single-GPU image resolves to `allgather_reducescatter`, and no DeepEP
kernel package is installed. Healthcheck never became ready.

Resolution: reject DBO for the fixed compose image. Do not force a DeepEP
backend or install packages at runtime; either would be a submission startup
risk and is not a valid local comparison on this single-GPU path.

### 2026-07-17 - Explicit FlashInfer CUTLASS Cannot Consume W8A16 Artifact

Action: force `--linear-backend=flashinfer_cutlass` while retaining the
validated `--quantization=fp8_per_tensor` checkpoint path.

Observed: engine initialization failed before healthcheck. vLLM selected
`FlashInferFP8ScaledMMLinearKernel` but rejected it because that kernel requires
per-tensor activation and weight scales. The incumbent artifact is static
weight-only FP8 (W8A16), with activation scale `None`; no request was served.

Resolution: keep `--linear-backend=auto`, which selects executable CUTLASS for
the actual quantization contract. Do not change quantization merely to satisfy
an incompatible backend; that would be a different precision candidate with a
separate quality/performance gate.

### 2026-07-17 - Triton Linear Backend Has No Kernel for LFM2 W8A16 Layer

Action: force `--linear-backend=triton` on the same validated FP8 per-tensor
weights and FP8 KV configuration.

Observed: model construction failed before healthcheck with
`--linear-backend=triton was requested but no 'triton' kernel exists for this
layer type.` No request was served.

Resolution: retain automatic backend selection. The current W8A16 artifact has
an executable CUTLASS path only in this image; a Triton custom kernel would
require a new image/runtime implementation and cannot be introduced through
the compose-only contract.

### 2026-07-17 - Custom CUDA Graph Capture Sizes Caused Runtime OOM

Action: set `--compilation-config={"cudagraph_capture_sizes":[1,2,4,8]}` to
reduce graph inventory and recover KV capacity.

Observed: startup and healthcheck passed, and reported KV capacity increased to
`1,744,608` tokens. During the first graded wave, vLLM emitted four lazy Triton
JIT warnings and then crashed while executing two new prefills totaling 7,968
scheduled tokens. Inductor attempted to allocate another 64 MiB with only
83 MiB free and raised `torch.OutOfMemoryError`; the engine died and all 90
graded requests in the screen failed.

Root cause: startup KV profiling did not reserve the runtime workspace/JIT
shape cost introduced by the custom graph set. The apparent extra cache
capacity was not usable under the real prefill shape.

Resolution: reject explicit graph capture overrides. Keep the
`performance-mode=interactivity` automatic FULL_AND_PIECEWISE graph policy and
its default capture sizes. Do not infer safety from startup capacity alone.

### 2026-07-17 - FP8 E5M2 KV Regressed the LFM2 Decode Path

Action: replace the promoted `fp8_e4m3` attention KV storage with
`fp8_e5m2`, keeping weights, async scheduling and all scheduler settings
fixed.

Observed: startup passed, but the 90-graded screen scored `64.539` versus the
incumbent's roughly 67-point band. TPOT p50/p95 rose to `2.494/3.529ms`, TTFT
p95 rose to `166.221ms`, and startup extended to 62 seconds. No request failed,
but the performance gate failed decisively.

Resolution: reject E5M2 and retain E4M3. Do not spend a quality run on a
candidate that already loses both score components.

### 2026-07-17 - Round Script Was Not Executable

Failure: invoking `./scripts/start_server.sh` returned `Permission denied`
because the research script has no execute bit in the workspace.

Resolution: invoke it explicitly as `bash scripts/start_server.sh`; this is a
harness permission issue, not a server or model result.

### 2026-07-17 - Compile-Size 4096 Did Not Remove Lazy Helper JIT

Action: add `--compilation-config={"compile_sizes":[4096]}` to precompile the
dominant request shape.

Observed: startup rose to `86s`, but the first real workload still JIT-compiled
`_zero_kv_blocks_kernel`, `_compute_slot_mapping_kernel`,
`_causal_conv1d_fwd_kernel` and `batch_memcpy_kernel`. The screen completed,
but no causal performance gain was established.

Resolution: reject the override. The official 90-request primer already
provides a legal warmup boundary; shape compilation does not cover these
cache/block helpers in this image.

### 2026-07-17 - Throughput Mode Failed Full Repeatability

Action: repeat the promising throughput-mode full trace from a cold server.

Observed: first full run scored `68.029`; the clean repeat scored `66.490`
with zero failures. The screen gain was therefore process-order variance.

Resolution: retain interactivity mode as incumbent; do not package throughput.

### 2026-07-17 - Seq10 Failed Full Confirmation

Action: promote the promising seq10 screen to a cold full 420-request run.

Observed: peak active requests rose to `9`, but ERS fell to `64.789`, TTFT p95
reached `207.975ms`, and TPOT p95 reached `3.391ms`; no requests failed.

Resolution: retain `max-num-seqs=8`; reject the extra graph/concurrency cost.

### 2026-07-17 - GPU Utilization 0.86 Cannot Fit the Local Display GPU

Action: raise `--gpu-memory-utilization` from the local `.80` proxy toward the
BTC `.95` setting while holding the incumbent runtime fixed.

Observed: vLLM rejected startup before model initialization because only
`12.95/15.44 GiB` was free and `.86` requested `13.28 GiB`.

Resolution: this is a local hardware/display constraint, not an inference
failure. Test `.83` as the highest feasible local reservation; do not infer
that `.86` is unsafe on the BTC 18 GiB MIG without a BTC run.

### 2026-07-17 - Package Tests Initially Used the Wrong Runner

Failure: system `python3 -m pytest` failed because pytest is not installed;
`uv run pytest` then failed first on an unwritable external cache and again
because the default project environment does not include the dev extra.

Resolution: the tests are standard-library `unittest` suites. Run them with
`PYTHONPATH=src python3 -m unittest ...`; the corrected command passed all
`25/25` tests without installing packages or modifying the environment.

### 2026-07-19 - Part 2 R003 Boundary And Promotion Limits

The R003 `8192` attention-window candidate wins the inferred append-only
closed-loop replay, but an independent 4k screen is slightly negative. Do not
reuse the append-only gain for a workload where each API prompt is complete and
below 8192 tokens. The candidate also increases the available KV pool, so
attributing the full gain to attention alone is invalid. Short GPQA does not
cover the discarded long-context information, and the official request-history
semantics remain inferred from metadata. Keep R003 out of the Portal queue
until the `4096/8192/16384` follow-up, long-context quality check and rules
review are complete.

### 2026-07-19 - R003a Scalar Window Limits

The `16384` window retains only `40.47%` of the registered `8192` turn-5
saving and misses the `15ms` opening gate. Window `12288` passes isolated
profiling but scores only `57.0213` on full replay, demonstrating that one
conversation's GPU time is not a score proxy. Do not spend more full runs on
either size without a new mechanism.

One `6144` profile process was excluded because its turn-0 time rose to about
`81.9ms` where the window was inactive. Do not average process contamination
into a window effect. The retained A/C profile passes, but the only full replay
and one lost retrieval case do not justify carrying `6144` over `8192` or
`4096`.

Window `4096` is performance-positive in two full runs but loses two paired
retrieval cases. Do not infer quality safety from unchanged short GPQA, and do
not publish/package this branch before explicit rules and long-context review.

### 2026-07-19 - R003b Selective Global Attention Placement

Action: keep layer `14`, `2`, `8`, or the pair `2,14` globally causal while
windowing every other full-attention layer at `4096`.

Expected: recover the two retrieval cases lost by the all-layer `4096`
candidate without giving up its long-context latency gain.

Observed: all four profiles passed the performance gate and saved
`32.443..37.932ms` at turn 5, but every retrieval candidate answered only
`2/14` key cases versus `4/14` for the unbounded control. The same 5084- and
7109-token cases were lost in every placement; no case was gained.

Root cause / best hypothesis: a global layer after a bounded layer cannot
reconstruct information already discarded by the earlier stack. The quality
loss is distributed across the windowed layers, not localized to one layer.

Decision: reject all placements and skip full replay, image creation and
Compose work. Future work must preserve selected old information in each
affected layer.

Related files:

```text
local_workspaces/rtx5080/research_rounds/part2_lfm25/round_003_hybrid_attention_window/subrounds/subround_003b_sparse_global_attention/
```
