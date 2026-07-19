# RTX 5080 Technique Ledger

This is the shared technique notebook for local RTX 5080 rounds.

Round folders keep code, configs, logs and exact benchmark outputs. This file keeps the reusable technical lesson: what a technique tries to reduce/increase, when to use it, observed evidence and trade-offs.

## Update Rule

After each round, add or update one entry here when a technique changes our understanding of:

- TTFT
- TPOT
- throughput
- VRAM
- stability
- quality risk
- submission safety

Do not create separate per-round technical note files unless a round has unusually large analysis that cannot fit here.

## Technique Index

| Technique | Primary target | Status | Best evidence | Carry-forward rule |
|---|---|---|---|---|
| Scheduler concurrency cap | Lower TPOT | Current baseline component | Round 002 `MAX_NUM_SEQS=4` improved ERS to `0.499987`. | Keep `MAX_NUM_SEQS=4` unless a TTFT round proves larger concurrency is worth the TPOT cost. |
| Moderate GPU memory budget | Better KV/runtime headroom | Current baseline component | Round 003 `GPU_MEMORY_UTILIZATION=0.72` reached ERS `0.5`. | Keep `0.72` in local lab while inside VRAM limit. |
| Explicit prefix caching | Lower TTFT/TPOT on repeated prefixes | Required | Round 006 explicit-on ERS `0.5`; no-flag/default ERS `0.473541`. | Always pass `--enable-prefix-caching` for vLLM candidates unless testing prefix cache itself. |
| Explicit chunked prefill | Long-prefill stability | Required | Round 005 explicit-on ERS `0.5`; no-chunked candidates did not become ready. | Always pass `--enable-chunked-prefill` for vLLM candidates. |
| `MAX_NUM_BATCHED_TOKENS=16384` | Lower TPOT | Watchlist | Round 004 TPOT p50/p95 `8.24 / 16.01 ms`, but worse TTFT/VRAM. | Use only as a TPOT rescue with same-session A/B. |
| Synthetic occupancy ramp | TTFT/startup stabilization | Watchlist | Round 005c light 2s ramp reached ERS `0.5` and improved tail vs same-session control. | Do not promote until healthcheck-safe integration and repeat. |
| FP8 KV cache | Lower TTFT, TPOT, VRAM pressure and improve capacity | Provisional candidate | Round 007 FP8 E4M3 reached TTFT p50 `17.05s`, TPOT p95 `9.54ms`, wall tokens/s `409.65`. Round 007b GPQA proxy: auto `0.38`, FP8 `0.34`, drop `0.04`. | Use as current performance candidate, but keep auto KV as rollback and rerun GPQA proxy after major/lossy changes. |
| Online FP8 weights at wave20 | Repair decode bandwidth under high admission | Promoted in I002+ | Score 100 R003 beat BF16 on all paired TTFT/TPOT requests and reached ERS `0.573075`; local GPQA was `0.34/0.34`. | Keep with FP8 KV and hidden-quality guard; do not use BF16-weight wave20. |
| Exact full CUDA Graph size 20 | Protect TPOT at the actual active batch | Promoted in I003+ | R004 candidate beat control midpoint ERS by `0.005212`, with paired TTFT/TPOT wins `97/119`. | Keep exact size20 while `max-num-seqs=20`; do not add shapes without evidence. |
| BF16 Mamba SSM cache | Reduce hybrid page/block granularity | Promoted in I004+ | R006b block size `1072 -> 560`, ERS `0.586594`, paired TTFT/TPOT wins `99/112`, local GPQA flips `0/100`. | Keep align mode and FP32 rollback; retest quality after another lossy model/state change. |
| Endpoint synthetic warmup | Hide first-traffic JIT/allocator/shape latency | Promoted in I005/I006 | R009 ERS `0.617657` vs midpoint `0.581198`; R010 packaged ERS `0.619885` vs `0.589324`; five observed first-traffic JIT warnings moved before timing. R035 confirms current I006 timed suffix count is zero. | Use only trace-independent shapes and complete warmup before public readiness. H200 transfer remains required. |
| Readiness-gated one-model package | Make endpoint warmup submission-valid | Local packaged champion I006 | One native backend on 8001 warms before byte-transparent 8000 relay bind; readiness `102s`; relay-only ERS `0.594723` vs midpoint `0.592591`. | Preserve protected module, one model/context, public digest and immutable Attempt 11. Do not add request-aware relay logic. |
| Output-only FP8 tied head + exact BF16 top-k4 | Reduce decode vocabulary-projection bandwidth | H200-quality-safe watchlist | R021a packaged candidate ERS `0.626978` vs midpoint `0.623107`; Attempt 12 scored `51.37` with TBT `15ms`, accuracy drop `0`, penalty `1` and no failures, but I006 seq16 scored `51.59`. | Keep implementation and feature-off rollback, but disable it in the target score baseline until repeated H200 evidence beats native head. |
| Seq16 active-set balance | Balance TTFT admission and TPOT tail | Current H200 score champion setting | I006 seq16 scores `51.59`; R021c head-off seq17 lost ERS `0.530%`, added zero fast-TTFT requests and raised TPOT p99 `0.873ms`. | Keep I006 seq16. Close seq17/18 concurrency-only tuning and require a new prefill/admission mechanism. |
| High local VRAM budget | Increase KV capacity | Capacity watchlist, not standalone score win | Round 012a `0.82 -> 0.90` increased KV tokens `883850 -> 1062745` (+20.24%), but ERS stayed `0.5` and `0.82` had slightly better latency. | Use `0.90` for capacity/H200 experiments only when paired with scheduler/concurrency changes; keep `0.82` as local score control. |
| Integrated all techniques | Full-stack local serving candidate | Superseded local baseline | Ready Round 002 `integrated_all_seq8_treegated` reached ERS `0.502460`, TTFT p50 `7890.54 ms`, TPOT p95 `13.61 ms`, GPQA proxy `0.35` with no local penalty. | Keep as vLLM fallback and Tree-Gated comparison point. |
| SGLang LPM + CUDA graph | Lower TTFT and improve overall ERS | Superseded framework baseline | Ready Round 003a `sglang_ablate_auto_mem086_run10` reached ERS `0.517733`, TTFT p50 `4318.31 ms`, TPOT p95 `18.55 ms`, GPQA proxy `0.37` with no local penalty. | Preserve as SGLang reference only; Score 100 I007 is the current cross-framework champion. |
| TTFT bottleneck diagnostics | Identify TTFT target | Completed diagnosis | Ready Round 003b: isolated length sweep p95 `1246.43 ms`, but shared-prefix burst p95 `5306.34 ms`; tokenization p95 only `26.67 ms`. | Tree-Gated V0 should target burst-aware admission, prefix singleflight/residency and prefill chunk scheduling before semantic compression. |
| External HTTP admission cap | Reduce burst queueing | Rejected | Ready Round 003c: best proxy candidate ERS `0.490972` vs same-session baseline `0.515398`; proxy queue-wait p95 up to `29.1s`. | Do not use whole-request HTTP queueing as Tree-Gated V0. Move closer to scheduler/prefix-build control. |
| Logical two-policy HTTP proxy | Split TTFT admission and TPOT decode on one model | Rejected at HTTP layer | Ready Round 007 best `proxy_c_prefill10_total14` reached ERS `0.493432`, TTFT<=1500 `3/120`, TPOT p95 `21.721 ms`, below current best `0.524547`. | Do not use front-proxy queueing as the two-policy implementation. Move the split inside the engine scheduler or native priority/admission hooks. |
| SGLang native prefill delayer | Batch prefill fragments | Watchlist, not score win | Ready Round 003d best delayer ERS `0.515090` vs same-session baseline `0.516041`; TTFT p50 slightly lower but TTFT/TPOT tail worse. | Keep as a small tuning knob only; do not promote without same-session win. |
| Whole-request native priority | Request classification / scheduling | Rejected | Ready Round 003d priority-prefix ERS `0.451969`, priority-short ERS `0.439851` vs same-session baseline `0.514740`; TPOT p95 rose to `36-39ms`. | Do not replace SGLang LPM/radix scheduling with whole-request priority. Work at prefix-build/prefill-chunk level instead. |
| SGLang prefill token/chunk knobs | TTFT/prefill scheduling | Watchlist, not baseline | Ready Round 003e broad sweep found `max-prefill-tokens=8192` ERS `0.515851` vs baseline `0.512680`, but refine sweep baseline won `0.517014` vs `8192` `0.513922`. | Keep `max-prefill-tokens=6144/8192` and `enable-mixed-chunk` as same-session knobs only; do not promote as current baseline. |
| SGLang decode reserve | Decode memory/headroom | Historical SGLang component | Ready Round 003f final `reserve224_steps4` reached ERS `0.523150` vs same-session baseline `0.516250`; GPQA proxy `0.37`, penalty `1.0`. | Keep only when reproducing the SGLang branch; it is not part of I006. |
| SGLang TPOT protection / continuous batching stress | Lower TPOT vs batch-fill trade-off | Watchlist | Ready Round 004 lowest TPOT was `run4_steps8`, TPOT p50/p95 `7.064 / 9.355 ms`, but TTFT p50 `20565.496 ms`. Aggressive continuous batching `run14_steps6` improved TTFT p50 to `3132.908 ms` and throughput to `553.047 tok/s`, but TPOT p95 rose to `23.741 ms`. | Use `run6_steps6`/`run4_steps8` only as TPOT rescue. Do not promote aggressive continuous batching while TPOT p95 is above `20 ms`. |
| TPOT-floor compatible TTFT overlay | Reduce TTFT while preserving low TPOT | New watchlist | Ready Round 005 `tpotfloor_chunk1024_prefillreq4_mixed` reached ERS `0.513911`, TTFT p50/p95 `12178.541 / 49009.192 ms`, TPOT p50/p95 `9.107 / 11.549 ms`. | Use only when starting from the TPOT-floor module. It is not better than global Ready 003f ERS baseline. |
| Chunked prefill + mixed chunk interleaving | Reduce prefill blocking | New watchlist | Ready Round 005 showed chunk `1024` plus `prefill-max-requests=4` plus `enable-mixed-chunk` beat chunk-only and max-prefill-only variants. Chunk1024 request1 improved TPOT p95 to `11.087 ms` but lost ERS `0.511664` vs req4 `0.513911`. Chunk `256` regressed to ERS `0.504052`; one-request chunk256 regressed further to `0.501992`. | Favor `chunked-prefill-size=1024`, `prefill-max-requests=4`, `enable-mixed-chunk`; use request1 only as a TPOT-tail variant, avoid chunk `256` and full 1024+8192+maxreq4+long-warmup with prefill CUDA graph. |
| TTFT-aggressive SGLang lane | Reduce first-token queueing | Research candidate, not global policy | Ready Round 005b lowest TTFT p50 was `ttft_run14_chunk1536_req4_mixed`: TTFT p50/p95 `2765.807 / 30741.190 ms`, TTFT<=1500 `30/120`, but ERS `0.404494` and TPOT p95 `25.691 ms`. `run16_steps6` reached TTFT<=1500 `32/120` but TPOT p95 `44.433 ms`. | Use only as TTFT-lane design input. Do not use as global serving policy. Keep running requests near 14, not 16, unless TPOT handoff is real. |
| Large prefill chunk with CUDA graph | Reduce prefill round trips | Rejected locally at chunk8192 | Ready Round 005b focused follow-up: chunk8192 variants all failed readiness with prefill CUDA graph capture CUDA OOM. `OMP/MKL=1`, disabling radix, and disabling overlap did not fix startup. `chunked-prefill-size=-1` started but returned 0/120 successful requests. | Do not use chunk8192 with `tc_piecewise` prefill CUDA graph at local mem `0.86`. If retesting, try chunk 3072/4096/6144 or cap/disable prefill CUDA graph explicitly. |
| Exact system-prefix cache prime | Lower cold TTFT on shared trace prefix | Rejected as implemented | Ready Round 005c: control ERS `0.515480`, TTFT p50/p95 `4413.646 / 35326.715 ms`; exact system-prefix prime ERS `0.512806`, TTFT p50/p95 `10040.703 / 40916.335 ms`. Both logs showed non-zero cached-token batches during timed benchmark. | Do not use system-only trace prefix priming as a fix. If revisiting, verify tokenized chat-template boundary/cache metrics first and treat it as local research due submission-safety risk. |
| Qwen3.5 hybrid architecture audit | Correct optimization target | Active correction | Local `config.json` shows `18 x linear_attention` and `6 x full_attention`; SGLang logs allocate `Mamba Cache` and initialize `MambaRadixCache`. | Treat pure Transformer KV-cache assumptions as unverified. Optimize hybrid state/cache and scheduler behavior until BTC hash proves otherwise. |
| SGLang Mamba backend FlashInfer | Hybrid state runtime | Watchlist | Ready Round 006 `hybrid_mamba_flashinfer` reached ERS `0.514223`, TTFT p50/p95 `4377.759 / 35469.702 ms`, TPOT p95 `17.475 ms`, wall tokens/s `471.170`. Replay control ERS `0.513978`, TPOT p95 `18.145 ms`. | Keep as a same-session knob. It improves TPOT/throughput slightly but does not beat global Ready 003f. |
| SGLang Mamba SSM BF16 | Hybrid state precision | Watchlist | Ready Round 006 `hybrid_ssm_bfloat16` reached ERS `0.519408`; subround 006a GPQA proxy matched control exactly; replay 006c reached `0.522122` but lost to control `0.523891`. | Quality risk reduced, but not an ERS baseline. Keep as hybrid overlay watchlist. |
| SGLang Mamba FlashInfer + BF16 SSM | Hybrid state backend + precision | TPOT/throughput watchlist | Ready Round 006c `hybrid_mamba_flashinfer_ssm_bfloat16` reached ERS `0.518213`, TPOT p95 `16.824 ms`, wall tok/s `507.813`, VRAM max `15422 MiB`. | Do not promote for ERS. Useful as a TPOT/throughput/VRAM module candidate. |
| SGLang FlashInfer linear-attention backend | Hybrid linear attention kernels | Rejected with current graph policy | Ready Round 006 `hybrid_linear_flashinfer` failed startup without BF16 SSM; `hybrid_linear_flashinfer_bfloat16` failed prefill CUDA graph capture with FlashInfer GDN expecting float32 state but receiving BF16. | Do not use under current `tc_piecewise` prefill CUDA graph policy. Retest only in a dedicated subround with graph policy changed. |
| Phase-aware KV pipeline | Split TTFT and TPOT policies without copying KV | Architecture note | Current evidence shows TTFT-friendly continuous batching and TPOT-friendly low concurrency conflict under one global config. | Keep as next architecture candidate: one model, one KV arena, TTFT lane hands an internal KVHandle to TPOT lane. Do not build two HTTP servers unless KV handoff avoids re-prefill. |
| Tree-Gated proxy scheduling | Internal request prioritization / admission shaping | New watchlist | Ready Round 002 proxy processed 125 requests and integrated candidate beat raw baseline strongly, but no direct-vLLM ablation yet. | Keep only as ready-round experiment until direct-vLLM ablation proves proxy value over overhead. |

## TTFT Bottleneck Diagnostics

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/subround_003b_ttft_bottleneck_diagnostics/
```

Target:

- Determine whether TTFT is limited by raw single-request prefill, burst queue,
  prefix cache behavior, tokenization CPU or decode backlog.
- Decide what native Tree-Gated V0 should optimize.

Best local evidence:

| Scenario | TTFT p50 | TTFT p95 | Under 1500 ms | Reading |
|---|---:|---:|---:|---|
| Baseline 200-token trace | `4318.31 ms` | `36102.52 ms` | `26/120` | Current ERS baseline is TTFT-capped. |
| Full trace, `max_tokens=1` | `1903.89 ms` | `4679.99 ms` | `45/120` | Removing long decode helps, but burst first-token service still fails. |
| Isolated length sweep | `767.82 ms` | `1246.43 ms` | `6/6` | Raw single-request long prefill can fit under the official ceiling locally. |
| Shared-prefix cold burst | `3265.37 ms` | `5306.34 ms` | `3/20` | Burst queueing over long prefill is a primary bottleneck. |
| Shared-prefix warm burst | `2839.23 ms` | `4588.89 ms` | `4/20` | Prefix warming helps but does not solve scheduling. |
| No-shared-prefix cold burst | `5386.08 ms` | `9372.42 ms` | `2/20` | Prefix cache already helps; losing it is much worse. |

Tokenization CPU evidence:

```text
Docker image: lmsysorg/sglang:latest
CPU/RAM: 3 cores / 8g
Official trace records: 120
Total encode time: 2.300 s
Encode p50/p95/p99: 20.130 / 26.671 / 28.059 ms
Token count p50/p95/max: 20164 / 27381 / 27398
```

Final repeat:

| Candidate | ERS | TTFT <=1500 | TTFT p50 | TTFT p95 | TPOT p95 | Decision |
|---|---:|---:|---:|---:|---:|---|
| Same-session baseline | `0.516250` | `26/120` | `4300.724 ms` | `36187.187 ms` | `18.308 ms` | Control |
| Reserve `200` + decode steps `4` | `0.519726` | `23/120` | `3762.269 ms` | `33819.652 ms` | `16.959 ms` | Strong candidate |
| Reserve `224` + decode steps `4` | `0.523150` | `23/120` | `3754.602 ms` | `33606.856 ms` | `17.002 ms` | New local best |

GPQA proxy for `reserve224_steps4`:

```text
accuracy: 0.37
total: 100
parsed: 100
errors: 0
official-style delta vs 0.4 baseline: 0.03
penalty proxy: 1.0
```

Technical conclusion:

- TTFT is primarily burst-queue and prefix-scheduler bound.
- CPU tokenization is not the primary bottleneck.
- Raw prompt length alone is not the primary bottleneck on local RTX 5080,
  because isolated long prompts stayed under `1500 ms`.
- Decode length still interacts with TTFT indirectly because active 200-token
  decodes occupy scheduler slots and increase first-token waiting time.

Carry-forward:

```text
Tree-Gated V0 must be lossless first:
- burst-aware admission
- prefix singleflight
- prefix hot/warm residency
- prefill chunk scheduling
- limited decode protection

Do not start with semantic compression/pruning until the lossless scheduler path
is measured.
```

## External HTTP Admission Cap

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/subround_003c_controlled_admission_scheduler/
```

Target:

- Test whether classifying requests and queueing them before SGLang can reduce
  TTFT under burst traffic.
- Release proxy scheduler slots on the first streamed chunk rather than after
  full 200-token completion.

Evidence:

| Candidate | ERS | TTFT <=1500 | TTFT p50 | TPOT p95 | Proxy queue p95 | Decision |
|---|---:|---:|---:|---:|---:|---|
| Same-session SGLang baseline | `0.515398` | `25/120` | `4610.932 ms` | `18.122 ms` | n/a | Keep baseline |
| FIFO first-chunk cap6 | `0.481412` | `3/120` | `10643.210 ms` | `26.420 ms` | `18240.224 ms` | Reject |
| FIFO first-chunk cap8 | `0.481840` | `3/120` | `10507.946 ms` | `22.590 ms` | `16921.119 ms` | Reject |
| `ttft_v0` first-chunk cap6 | `0.490972` | `8/120` | `11190.256 ms` | `21.879 ms` | `29064.152 ms` | Reject |

Technical conclusion:

- The idea of classification is still valid, but request-level HTTP queueing is
  too coarse.
- Proxy waiting time is included in TTFT, so this implementation mostly moves
  the queue from SGLang into the proxy.
- First streamed chunk can still arrive many seconds later under burst, so
  `release_on=first_chunk` does not create a true prefill lane.

Carry-forward:

```text
Do not promote external HTTP admission cap.
Next Tree-Gated should work closer to SGLang scheduler/prefix cache:
- prefix builder/follower split
- singleflight shared prefix
- prefill chunk priority
- native scheduler hooks
```

## SGLang Native Prefill Delayer And Priority

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/subround_003d_native_prefix_priority_scheduler/
```

Target:

- Test the user's request-classification/queueing idea without moving queue
  time into an external HTTP admission cap.
- Let SGLang own the queue through native prefill delayer or native priority
  scheduling.

Native prefill-delayer evidence:

| Candidate | ERS | TTFT <=1500 | TTFT p50 | TTFT p95 | TPOT p95 | Decision |
|---|---:|---:|---:|---:|---:|---|
| Same-session baseline | `0.516041` | `23/120` | `4564.696 ms` | `36093.005 ms` | `17.974 ms` | Keep |
| Prefill delayer `ratio=0.25`, `delay=250ms` | `0.513646` | `23/120` | `4493.901 ms` | `36595.625 ms` | `18.311 ms` | Reject |
| Prefill delayer `ratio=0.5`, `delay=500ms` | `0.515090` | `23/120` | `4476.439 ms` | `36670.492 ms` | `18.265 ms` | Reject |

Native priority evidence:

| Candidate | ERS | TTFT <=1500 | TTFT p50 | TTFT p95 | TPOT p95 | Proxy queue p95 | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Same-session baseline | `0.514740` | `25/120` | `4561.498 ms` | `36611.789 ms` | `18.003 ms` | n/a | Keep |
| Priority short-prefill | `0.439851` | `3/120` | `12830.630 ms` | `41138.169 ms` | `36.360 ms` | `20.141 ms` | Reject |
| Priority prefix-LPM | `0.451969` | `22/120` | `9811.921 ms` | `66516.090 ms` | `38.781 ms` | `19.582 ms` | Reject |

Technical conclusion:

- SGLang priority scheduling requires `schedule_policy=fcfs` or `lof`; using
  `schedule_policy=priority` fails startup in this image.
- Corrected native priority scheduling runs, but degrades ERS and TPOT tail.
- Proxy queue time was not the issue in the corrected priority sweep: proxy
  queue p95 stayed near `20 ms`, while backend first chunk p95 rose to tens of
  seconds.
- This means the loss likely comes from disrupting SGLang's LPM/radix-oriented
  scheduler behavior, not from Python proxy admission.

Carry-forward:

```text
Do not promote whole-request native priority.
Keep native prefill-delayer only as a same-session tuning knob.
The next valid scheduler experiment should target:
- prefix builder/follower split
- shared-prefix singleflight
- prefill chunk priority
- limited decode protection
```

## SGLang Prefill Token And Chunk Knobs

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/subround_003e_prefill_chunk_scheduler_sweep/
```

Target:

- Test lower-level SGLang prefill/chunk scheduler knobs after request-level
  queueing and priority both failed.
- Determine whether `chunked-prefill-size`, `max-prefill-tokens`,
  `prefill-max-requests` or `enable-mixed-chunk` can reduce TTFT without
  damaging TPOT.

Broad sweep evidence:

| Candidate | ERS | TTFT <=1500 | TTFT p50 | TTFT p95 | TPOT p95 | Decision |
|---|---:|---:|---:|---:|---:|---|
| Same-session baseline | `0.512680` | `23/120` | `4605.225 ms` | `36691.048 ms` | `18.161 ms` | Control |
| Chunk `1024` | `0.509090` | `23/120` | `5553.023 ms` | `32702.072 ms` | `20.019 ms` | Reject |
| Chunk `4096` | `0.511333` | `23/120` | `4625.593 ms` | `37374.638 ms` | `18.619 ms` | Reject |
| Max prefill tokens `8192` | `0.515851` | `25/120` | `4576.188 ms` | `35990.431 ms` | `17.969 ms` | Refine |
| Prefill max requests `4` | `0.514130` | `21/120` | `4487.104 ms` | `36496.902 ms` | `18.168 ms` | Watchlist |
| Mixed chunk | `0.514428` | `26/120` | `3913.834 ms` | `35987.500 ms` | `20.708 ms` | TTFT-only watchlist |

Refine evidence:

| Candidate | ERS | TTFT <=1500 | TTFT p50 | TTFT p95 | TPOT p95 | Decision |
|---|---:|---:|---:|---:|---:|---|
| Same-session baseline | `0.517014` | `25/120` | `4533.439 ms` | `37077.954 ms` | `18.368 ms` | Keep baseline |
| Max prefill tokens `8192` | `0.513922` | `25/120` | `4543.819 ms` | `36306.676 ms` | `18.154 ms` | Reject as baseline |
| Max prefill tokens `6144` | `0.515615` | `23/120` | `4621.256 ms` | `36264.169 ms` | `17.916 ms` | Watchlist |
| Max prefill tokens `12288` | `0.513384` | `21/120` | `4716.497 ms` | `36470.563 ms` | `18.076 ms` | Reject |
| `8192` + mixed chunk | `0.512677` | `26/120` | `4015.333 ms` | `35982.761 ms` | `20.345 ms` | Reject for TPOT/ERS |
| `8192` + prefill max requests `4` | `0.514780` | `23/120` | `4572.905 ms` | `36547.706 ms` | `17.988 ms` | Watchlist |

Technical conclusion:

- `max-prefill-tokens=8192` looked good in the broad sweep but did not repeat
  in the refine sweep.
- `max-prefill-tokens=6144` and `8192+prefill-max-requests=4` improved TPOT p95
  versus the refine baseline but lost ERS.
- `enable-mixed-chunk` is a real TTFT p50 lever, but it moves TPOT p95 into the
  official floor danger zone around `20 ms`.
- Changing `chunked-prefill-size` alone is not enough for this trace.

Carry-forward:

```text
Do not promote 003e as baseline.
Keep these as conditional knobs:
- max-prefill-tokens=6144/8192 when TTFT tail is the target and TPOT has margin
- enable-mixed-chunk only when TTFT p50 matters more than TPOT tail
Next scheduler work should target decode protection or native prefix-build/singleflight.
```

## SGLang Decode Reserve

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/subround_003f_decode_protection_scheduler_sweep/
```

Target:

- Reduce over-reserved decode KV/headroom.
- The public trace uses `max_tokens=200` for all 120 requests, while SGLang
  default reserve is `512`.
- Test whether reducing reserve improves admission/prefill without breaking
  decode TPOT.

Broad sweep evidence:

| Candidate | ERS | TTFT <=1500 | TTFT p50 | TTFT p95 | TPOT p95 | Decision |
|---|---:|---:|---:|---:|---:|---|
| Same-session baseline | `0.516420` | `25/120` | `4528.508 ms` | `36718.905 ms` | `18.093 ms` | Control |
| Reserve `256` | `0.515200` | `23/120` | `4439.511 ms` | `36372.634 ms` | `18.007 ms` | Watchlist only |
| Reserve `200` | `0.517276` | `24/120` | `4407.224 ms` | `36286.383 ms` | `18.124 ms` | Refine |
| Decode steps `2` | `0.513750` | `24/120` | `4585.756 ms` | `36513.066 ms` | `18.016 ms` | Reject |
| Decode steps `4` | `0.515807` | `27/120` | `4403.766 ms` | `36173.404 ms` | `18.348 ms` | Refine as combo |
| Schedule conservativeness `0.5` | `0.516384` | `24/120` | `4334.474 ms` | `36281.985 ms` | `18.248 ms` | Watchlist only |
| Schedule conservativeness `2.0` | `0.514763` | `25/120` | `4461.715 ms` | `36393.750 ms` | `18.041 ms` | Reject |

Refine evidence:

| Candidate | ERS | TTFT <=1500 | TTFT p50 | TTFT p95 | TPOT p95 | Decision |
|---|---:|---:|---:|---:|---:|---|
| Same-session baseline | `0.514151` | `23/120` | `4449.644 ms` | `36863.899 ms` | `18.035 ms` | Control |
| Reserve `200` | `0.516315` | `24/120` | `4613.804 ms` | `36124.047 ms` | `17.992 ms` | Watchlist |
| Reserve `224` | `0.516981` | `25/120` | `4407.977 ms` | `36529.803 ms` | `17.985 ms` | Watchlist |
| Reserve `200` + decode steps `4` | `0.517185` | `25/120` | `4499.346 ms` | `35970.708 ms` | `18.008 ms` | Best 003f candidate |
| Reserve `200` + conservativeness `0.5` | `0.515647` | `23/120` | `4570.929 ms` | `36143.026 ms` | `17.873 ms` | Watchlist only |

Technical conclusion:

- Decode reserve is a real lever for this trace.
- `reserve200` repeated as a same-session improvement in early sweeps.
- `reserve224_steps4` is the strongest 003f candidate and beats the previous
  global best `0.517733` from Ready 003a.
- Decode steps alone and schedule conservativeness alone are weaker.

Carry-forward:

```text
Current candidate:
- num-reserved-decode-tokens=224
- num-continuous-decode-steps=4

Fallback/watchlist:
- num-reserved-decode-tokens=200 + num-continuous-decode-steps=4
- num-reserved-decode-tokens=224 without decode steps

Next:
- package/validate this as the current SGLang submission candidate
- continue TTFT work only if TPOT p95 remains near 17 ms
```

## SGLang TPOT Protection and Continuous Batching Stress

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_004_sglang_self_optimization/
```

Target:

- Keep SGLang and built-in runtime controls.
- Use RadixAttention, FlashInfer, CUDA graph, dynamic chunking, decode reserve
  and continuous decode steps.
- Push TPOT as low as possible, then measure the ERS/TTFT trade-off.
- Test whether stronger continuous batching helps this scoring workload.

Evidence:

| Candidate | ERS | TTFT p50 | TPOT p50 | TPOT p95 | Wall tok/s | Reading |
|---|---:|---:|---:|---:|---:|---|
| Current control | `0.523052` | `3787.462 ms` | `12.447 ms` | `17.286 ms` | `483.542` | Best ERS remains control |
| Dynamic chunking | `0.516477` | `4338.047 ms` | `13.985 ms` | `18.190 ms` | `456.383` | No win |
| Explicit FlashInfer | `0.515243` | `4435.709 ms` | `13.663 ms` | `17.996 ms` | `470.101` | Valid but not better than auto |
| Steps 6 | `0.514140` | `4496.191 ms` | `13.818 ms` | `18.009 ms` | `483.027` | No win at same concurrency |
| Steps 8 | `0.513908` | `4552.971 ms` | `13.841 ms` | `18.002 ms` | `484.347` | No win at same concurrency |
| Run 8 + steps 6 | `0.507424` | `4595.672 ms` | `9.644 ms` | `13.058 ms` | `489.674` | TPOT rescue candidate |
| Run 6 + steps 6 | `0.511819` | `4681.498 ms` | `8.373 ms` | `11.666 ms` | `441.759` | Best balanced TPOT rescue |
| Run 4 + steps 8 | `0.507828` | `20565.496 ms` | `7.064 ms` | `9.355 ms` | `324.222` | Lowest TPOT, TTFT too costly |
| Continuous batch run 12 | `0.496968` | `3433.746 ms` | `14.898 ms` | `24.262 ms` | `507.287` | TPOT too high |
| Continuous batch run 14 | `0.501104` | `3132.908 ms` | `15.479 ms` | `23.741 ms` | `553.047` | Best TTFT/throughput, TPOT too high |
| Continuous batch run 12 + steps 8 | `0.492020` | `3599.526 ms` | `15.141 ms` | `26.357 ms` | `490.887` | Reject |
| Mixed chunk run 12 | `0.492521` | `3335.134 ms` | `19.642 ms` | `22.473 ms` | `539.837` | Reject for TPOT |

Technique roles:

- RadixAttention: preserve shared-prefix reuse.
- FlashInfer: low-overhead attention/sampling kernel path; SGLang auto selection
  was better than explicit backend forcing in this run.
- CUDA graph decode/prefill: keep launch overhead low and stabilize common
  prefill/decode shapes.
- Dynamic chunking: did not help this trace when added to the current control.
- Decode reserve `224`: remains useful because public trace output cap is `200`.
- Continuous decode steps: helps only when paired with lower active concurrency;
  increasing steps alone at `MAX_RUNNING_REQUESTS=10` did not improve TPOT/ERS.
- Lower max running requests: strongest TPOT lever, but it shifts cost into TTFT.
- Aggressive continuous batching: improves TTFT p50 and throughput, but decode
  contention pushes TPOT p95 above the official floor.

Carry-forward:

```text
Do not replace the current ERS control.
For TPOT emergency rescue, try:
- MAX_RUNNING_REQUESTS=6, num-continuous-decode-steps=6
- MAX_RUNNING_REQUESTS=4, num-continuous-decode-steps=8 only to find TPOT floor

For TTFT/throughput-only research, keep:
- MAX_RUNNING_REQUESTS=14, num-continuous-decode-steps=6

Reject for current scoring:
- dynamic chunking alone
- mixed chunk
- aggressive continuous batching while TPOT p95 > 20 ms
```

## TPOT-Floor Compatible TTFT Overlay

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_005_tpot_floor_ttft_overlay/
```

Target:

- Start from the lowest-TPOT SGLang policy from Ready Round 004.
- Preserve the TPOT module:
  `MAX_RUNNING_REQUESTS=4`, `num-continuous-decode-steps=8`,
  `num-reserved-decode-tokens=224`, dynamic chunking, Radix cache and CUDA graph.
- Add only TTFT overlays that do not remove the TPOT-floor policy.

Evidence:

| Candidate | ERS | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | Decision |
|---|---:|---:|---:|---:|---:|---|
| TPOT-floor control replay | `0.508153` | `20444.543 ms` | `58445.085 ms` | `6.998 ms` | `9.629 ms` | TPOT floor, TTFT too costly |
| Chunk `256` + prefill max requests `1` + mixed chunk | `0.501992` | `27984.350 ms` | `59794.267 ms` | `10.718 ms` | `12.465 ms` | Reject; one-request prefill does not rescue chunk 256 |
| Chunk `256` + prefill max requests `4` + mixed chunk | `0.504052` | `27566.901 ms` | `59758.971 ms` | `11.093 ms` | `13.209 ms` | Reject; chunk too small |
| Full first-pass TTFT overlay | `0.512098` | `10445.564 ms` | `50899.531 ms` | `8.767 ms` | `11.029 ms` | Good but superseded |
| Chunk `1024` + prefill max requests `4` + mixed chunk | `0.513911` | `12178.541 ms` | `49009.192 ms` | `9.107 ms` | `11.549 ms` | Best within TPOT-floor lane |
| Chunk `1024` + max prefill `8192` + prefill max requests `4` + mixed chunk + long warmup | n/a | n/a | n/a | n/a | n/a | Crashed during warmup |

Technical conclusion:

- `enable-mixed-chunk` is the core TTFT overlay in this TPOT-floor lane.
- Manual chunk size is not monotonic: `256`, `512` and `768` were worse than `1024`.
- Lowering `prefill-max-requests` from `4` to `1` at chunk `256` improves TPOT
  p95 slightly but worsens ERS and TTFT, so chunk `256` itself is the bad lever.
- `chunked-prefill-size=1536` gave good TTFT p50, but `1024 + prefill-max-requests=4`
  gave the best ERS.
- `max-prefill-tokens=8192` is not the winning component in this lane.
- The best TPOT-floor overlay is still below the global Ready 003f ERS baseline
  because the TPOT-floor policy intentionally sacrifices TTFT.

Carry-forward:

```text
For TPOT-floor lane:
- MAX_RUNNING_REQUESTS=4
- num-continuous-decode-steps=8
- num-reserved-decode-tokens=224
- enable-dynamic-chunking
- chunked-prefill-size=1024
- prefill-max-requests=4
- enable-mixed-chunk

Avoid:
- chunked-prefill-size=256 under this policy

Do not combine chunk1024 + max-prefill8192 + prefill-max-requests=4 +
mixed chunk + long warmup under tc_piecewise prefill CUDA graph unless the
PCG capture crash is fixed or prefill CUDA graph policy is changed.
```

## FP8 KV Cache

Round:

```text
local_workspaces/rtx5080/research_rounds/warmup_rounds/round_007_kv_cache_dtype/
```

Target:

- Reduce VRAM pressure from KV cache.
- Increase useful KV token capacity.
- Reduce TTFT/queueing under long-context trace.
- Reduce TPOT tail by lowering memory pressure and decode contention.

Best local evidence:

| Candidate | KV dtype | Calc scales | ERS | TTFT p50 | TPOT p95 | Wall tokens/s | GPU KV tokens | Max 32K concurrency | VRAM max |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| auto control | auto | false | `0.499899` | `37489.74 ms` | `19.75 ms` | `209.55` | `388325` | `11.85x` | `12738 MiB` |
| FP8 | fp8 | false | `0.500000` | `17097.77 ms` | `9.53 ms` | `409.32` | `661559` | `20.19x` | `12260 MiB` |
| FP8 E4M3 | fp8_e4m3 | false | `0.500000` | `17196.78 ms` | `9.58 ms` | `407.90` | `661559` | `20.19x` | `12256 MiB` |
| FP8 E4M3 calc requested | fp8_e4m3 | true | `0.500000` | `17052.64 ms` | `9.54 ms` | `409.65` | `666873` | `20.35x` | `12194 MiB` |
| FP8 E5M2 calc requested | fp8_e5m2 | true | `0.500000` | `17218.05 ms` | `14.93 ms` | `364.89` | `641190` | `19.57x` | `12248 MiB` |

Technical conclusion:

- FP8 E4M3 is the strongest local performance candidate so far.
- The effect is large enough that it should be used in future performance rounds.
- E5M2 completed but had worse TPOT tail and throughput than E4M3.
- `fp8` alias and explicit `fp8_e4m3` behave similarly on this setup.

Quality and submission risk:

- FP8 KV cache is lossy.
- Round 007 only checked serving success and zero-token validity, not GPQA accuracy.
- vLLM logged that `--calculate-kv-scales` is disabled for this hybrid Qwen3.5 model because recurrent layers make calibration unreliable.
- Therefore, the best run should not be considered quality-safe just because `CALCULATE_KV_SCALES=true` was requested.
- Round 007a output drift proxy found auto KV is deterministic across two independent server starts: exact match `8/8`, similarity `1.0`.
- The same proxy found FP8 E4M3 differs from auto on every sampled public-trace prompt: exact match `0/8`, similarity min/avg/max `0.068627 / 0.385726 / 0.817468`.
- Round 007b GPQA Diamond proxy then measured local multiple-choice accuracy: auto KV `0.38`, FP8 E4M3 `0.34`, relative drop `0.04`.
- The official-style proxy with baseline `0.4` gives delta `0.02` for auto and `0.06` for FP8, both inside the no-penalty region.
- Therefore, free-form drift is real, but it did not translate into catastrophic local GPQA multiple-choice accuracy loss.

Carry-forward:

```text
Performance research rounds: KV_CACHE_DTYPE=fp8_e4m3
Quality/control rollback: KV_CACHE_DTYPE=auto
Submission candidates: KV_CACHE_DTYPE=fp8_e4m3 is provisionally allowed after Round 007b; rerun GPQA proxy after major or lossy changes
Do not use --calculate-kv-scales as a mitigation for this model.
```

## Endpoint Synthetic Warmup

Round:

```text
local_workspaces/rtx5080/research_rounds/warmup_rounds/round_012_warmup_jit/
```

Target:

- Reduce first endpoint TTFT from vLLM/Triton JIT and first-shape setup.
- Warm OpenAI streaming path before official trace replay.
- Keep warmup legal by using synthetic prompts only.

Best local evidence:

| Candidate | GPU memory util | Warmup | ERS | TTFT p50 | TTFT p95 | TPOT p95 | VRAM max |
|---|---:|---|---:|---:|---:|---:|---:|
| cold control | `0.82` | none | `0.500000` | `18539.6135 ms` | `32264.74145 ms` | `10.066987 ms` | `14247 MiB` |
| endpoint warmup | `0.82` | 3 short synthetic streaming requests | `0.5006135105048746` | `17849.3315 ms` | `31575.4577 ms` | `10.096045 ms` | `13957 MiB` |

Warmup cost:

```text
request 0 TTFT: 938.286 ms
request 1 TTFT: 23.262 ms
request 2 TTFT: 22.321 ms
```

Technical conclusion:

- Endpoint warmup is stronger than pure Torch warmup in Round 012.
- The gain is real but too small to make a submission candidate.
- The short warmup likely covers only part of the JIT/prefill shape space.

Carry-forward:

```text
Keep endpoint warmup in lab candidates.
Next test: medium prompt, long prompt, and 4-way concurrent synthetic warmup.
Do not use trace prompts or trace answers in warmup.
```

## High Local VRAM Budget

Rounds:

```text
local_workspaces/rtx5080/research_rounds/warmup_rounds/round_012_warmup_jit/
local_workspaces/rtx5080/research_rounds/warmup_rounds/round_012_warmup_jit/subrounds/round_012a_high_vram_sweep/
```

Observed behavior:

- `GPU_MEMORY_UTILIZATION=0.95` failed locally:
  - vLLM free startup memory: `12.91/15.44 GiB`
  - requested memory at 0.95: `14.67 GiB`
  - reason: active desktop/app processes used about `2.1GB` VRAM.
- `0.83` completed several runs and reached observed VRAM max `14383 MiB`.
- `0.83` is flaky because one candidate failed with free memory `12.76 GiB`
  while vLLM requested `12.81 GiB`.
- `0.82` completed the endpoint warmup stability fallback.
- Round 012a later ran a same-session sweep from `0.82` to `0.90` after more
  VRAM was free.
- All Round 012a candidates completed successfully under `3 CPU` and `8GB RAM`.

Round 012a capacity result:

| GPU memory utilization | KV GiB | KV tokens | Max concurrency for 32768-token requests | ERS | TTFT p50 | TPOT p95 |
|---:|---:|---:|---:|---:|---:|---:|
| `0.82` | `6.13` | `883850` | `26.97x` | `0.500000` | `16462.7585 ms` | `9.375602 ms` |
| `0.84` | `6.43` | `929017` | `28.35x` | `0.500000` | `16530.1135 ms` | `9.387417 ms` |
| `0.86` | `6.74` | `973298` | `29.70x` | `0.500000` | `16625.2230 ms` | `9.428332 ms` |
| `0.88` | `7.05` | `1017579` | `31.05x` | `0.500000` | `16597.7390 ms` | `9.431705 ms` |
| `0.90` | `7.36` | `1062745` | `32.43x` | `0.500000` | `16504.6455 ms` | `9.401569 ms` |

Technical conclusion:

- Higher VRAM allocation is proven to increase KV capacity.
- The capacity delta from `0.82` to `0.90` was `+178895` KV tokens, about
  `+20.24%`.
- This did not improve the trace score while `MAX_NUM_SEQS=4`, because the
  scheduler is not using the extra capacity.
- Therefore high VRAM is a capacity lever, not a standalone score lever.

Carry-forward:

```text
Desktop lab stable high-VRAM fallback: GPU_MEMORY_UTILIZATION=0.82
Capacity/H200 watchlist: GPU_MEMORY_UTILIZATION=0.90 when free VRAM allows it
Do not use 0.95 locally unless display/app VRAM is freed first.
Official H200 MiG 18GB may support a higher value, but it must be validated by submission logs.
Pair high VRAM with scheduler/concurrency or long-prefill work if the goal is score, not just capacity.
```

## Integrated All Techniques

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_002_integrated_all_techniques/
```

Target:

- Combine the useful warmup-round techniques into one serving stack.
- Compare against a raw no-technique baseline.
- Measure performance and GPQA proxy in the same round.

Integrated candidate:

```text
KV_CACHE_DTYPE=fp8_e4m3
ENABLE_PREFIX_CACHING=true
ENABLE_CHUNKED_PREFILL=true
GPU_MEMORY_UTILIZATION=0.88
MAX_NUM_SEQS=8
ENDPOINT_WARMUP_MODE=mixed
USE_TREE_GATE_PROXY=true
CPU_LIMIT=3
MEM_LIMIT=8g
```

Best local evidence:

| Candidate | ERS | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | Wall tokens/s | GPQA proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw baseline | `0.171784` | `52560.59 ms` | `103815.89 ms` | `31.44 ms` | `36.89 ms` | `172.22` | `0.38` |
| integrated all | `0.502460` | `7890.54 ms` | `13361.93 ms` | `10.51 ms` | `13.61 ms` | `581.04` | `0.35` |

Technical conclusion:

- Full-stack integration is much better than raw vLLM/no-technique baseline.
- Local GPQA proxy remains inside the no-penalty region:
  integrated accuracy `0.35`, official-style delta `0.05`, penalty `1.0`.
- This is still not final-submission-ready because TTFT remains far above the
  official `1500 ms` ceiling.

Carry-forward:

```text
Use integrated_all_seq8_treegated as the current full-stack local baseline.
Next required ablation: seq8 direct-vLLM without Tree-Gated proxy vs seq8 Tree-Gated proxy.
Also test integrated_all_seq4_score_safe for stability and GPQA.
```

## SGLang Runtime, RadixAttention, LPM Scheduling and CUDA Graph

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/
```

Target:

- Reduce TTFT by switching runtime from vLLM to SGLang.
- Use SGLang default RadixAttention/prefix cache behavior.
- Use LPM scheduling for prefix-aware request ordering.
- Use decode CUDA graph and prefill `tc_piecewise` CUDA graph where supported.
- Keep the same local contract: model `/model`, CPU `3`, RAM `8g`, trace `120`
  requests.

Best local evidence:

| Candidate | ERS | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | Wall tokens/s | GPQA proxy | VRAM max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SGLang raw | `0.499091` | `40827.872 ms` | `92345.5271 ms` | `10.301955 ms` | `15.852169 ms` | `188.61` | `0.37` | `13166 MiB` |
| SGLang LPM + CUDA graph | `0.507862` | `4655.2425 ms` | `33923.2610 ms` | `9.643693 ms` | `13.234378 ms` | `497.36` | `0.37` | `15519 MiB` |
| SGLang FlashInfer + FP8 KV | `0.507400` | `3846.1750 ms` | `31894.95925 ms` | `9.337730 ms` | `12.761552 ms` | `520.93` | `0.34` | `15456 MiB` |

Technical conclusion:

- SGLang is now the best local runtime by ERS.
- `sglang_radix_lpm_cuda_graph` was the first SGLang ERS baseline and is now
  superseded by `sglang_ablate_auto_mem086_run10`.
- `sglang_flashinfer_fp8kv` has the better TTFT/TPOT profile but slightly lower
  ERS and lower local GPQA proxy accuracy.
- The probed image does not support `--enable-cuda-graph`; use current per-phase
  flags:

```text
--cuda-graph-backend-decode=full
--cuda-graph-backend-prefill=tc_piecewise
--cuda-graph-tc-compiler=eager
```

Trade-off:

- SGLang LPM/CUDA graph uses very high local VRAM: max observed `15519 MiB`,
  leaving only about `290-300 MiB` free on the RTX 5080 run.
- FP8 KV remains quality-risky even though local official-style penalty is still
  `1.0`: raw SGLang GPQA `0.37`, FP8 KV GPQA `0.34`.
- Even Ready 003 best still misses the official TTFT target: p50 `4655 ms`
  versus ceiling `1500 ms`.

Carry-forward:

```text
Current local ERS baseline: sglang_ablate_auto_mem086_run10
TTFT/TPOT watchlist: sglang_flashinfer_fp8kv
Next work: native Tree-Gated around SGLang, or scheduler/memory ablation around max_running_requests and mem_fraction_static.
Do not use --enable-cuda-graph with this image.
```

## SGLang Scheduler and Memory Ablation

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_003_sglang_triton_treegated_candidate/subround_003a_sglang_scheduler_memory_ablation/
```

Target:

- Find a better SGLang sweet spot around `mem_fraction_static` and
  `max_running_requests`.
- Keep RadixAttention/default prefix cache and LPM scheduling.
- Compare auto KV versus FP8 KV under the same Ready 003 runtime path.

Best local evidence:

| Candidate | KV | Mem | Running | ERS | Success | TTFT p50 | TPOT p95 | VRAM max | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `sglang_ablate_auto_mem082_run8` | auto | `0.82` | `8` | `0.506578` | `120/120` | `4633.192 ms` | `13.156227 ms` | `14952 MiB` | Safe-VRAM fallback |
| `sglang_ablate_auto_mem084_run8` | auto | `0.84` | `8` | `0.504164` | `120/120` | `4101.1345 ms` | `12.849521 ms` | `15338 MiB` | Not promoted |
| `sglang_ablate_auto_mem086_run6` | auto | `0.86` | `6` | `0.513863` | `120/120` | `4602.235 ms` | `10.489755 ms` | `15371 MiB` | Lower-TPOT fallback |
| `sglang_ablate_auto_mem086_run10` | auto | `0.86` | `10` | `0.517733` | `120/120` | `4318.3145 ms` | `18.549398 ms` | `15624 MiB` | Current local ERS baseline |
| `sglang_ablate_fp8_mem086_run8` | fp8_e4m3 | `0.86` | `8` | `0.513875` | `120/120` | `4522.633 ms` | `14.189527 ms` | `15500 MiB` | FP8 watchlist |
| `sglang_ablate_fp8_mem088_run6` | fp8_e4m3 | `0.88` | `6` | `0.450861` | `105/120` | `3766.339 ms` | `11.363345 ms` | `15528 MiB` | Rejected |

Technical conclusion:

- Increasing `max_running_requests` from 8 to 10 at `mem_fraction_static=0.86`
  improves ERS to `0.517733`, despite TPOT p99 crossing `20 ms`.
- Reducing to 6 improves TPOT p95 but does not maximize ERS.
- Lower memory fraction `0.82` is safer for local VRAM and still competitive,
  but does not beat the score.
- FP8 KV at `0.88/run6` is unsafe for this trace: `15` errors/zero-token
  outputs.

Carry-forward:

```text
Current local baseline: sglang_ablate_auto_mem086_run10
Safer local fallback: sglang_ablate_auto_mem082_run8
Lower-TPOT fallback: sglang_ablate_auto_mem086_run6
Reject: sglang_ablate_fp8_mem088_run6
Next: native Tree-Gated/chunk scheduler should start from auto_mem086_run10, not FP8.
```

## Hybrid Qwen3.5 Mamba/Linear-Attention State Optimization

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/
```

Target:

- Correct the optimization model from pure Transformer KV-cache to hybrid
  full-attention KV + Mamba/linear-attention state.
- Test hybrid-specific SGLang knobs without deleting or invalidating existing
  Transformer/KV optimizations.

Evidence:

| Candidate | Purpose | ERS / Quality | Decision |
|---|---|---:|---|
| `hybrid_mamba_flashinfer` | `--mamba-backend=flashinfer` | ERS `0.514223` | Usable watchlist; improves TPOT/throughput slightly versus replay control, but not a baseline. |
| `hybrid_ssm_bfloat16` | `--mamba-ssm-dtype=bfloat16` | ERS `0.519408` | Strongest hybrid perf signal, below Ready 003f global best. |
| `hybrid_ssm_bfloat16` same-session replay | `--mamba-ssm-dtype=bfloat16` | ERS `0.513613` vs control `0.512065` | Repeated positive signal; improves TTFT p50 and wall tok/s, slightly worsens TTFT/TPOT tail. |
| `hybrid_mamba_flashinfer_ssm_bfloat16` | `--mamba-backend=flashinfer --mamba-ssm-dtype=bfloat16` | ERS `0.518213` vs control `0.523891` | Not an ERS baseline; best TPOT p95 and wall tok/s in replay 2, lower VRAM. |
| `mamba_ssm_bfloat16_quality` | GPQA proxy for BF16 SSM state | Accuracy `0.370000`, changed predictions `0/100` | Passed local quality proxy. |
| `hybrid_linear_flashinfer_bfloat16` | FlashInfer linear-attention backend | Startup/capture failed | Rejected under current prefill CUDA graph policy. |

Technical conclusion:

- `--mamba-ssm-dtype=bfloat16` is now quality-safer than before: on the local
  100-question GPQA proxy it exactly matched the default-float32 parsed answers
  and correctness.
- A same-session replay also showed a small ERS win over the matching control:
  `0.513613` vs `0.512065`, with TTFT p50 `4112.222 ms` vs `4518.249 ms`.
- A second same-session replay did not confirm BF16 SSM as an ERS winner:
  Ready 003f control reached `0.523891`, BF16 SSM reached `0.522122`, and
  BF16 SSM + Mamba FlashInfer reached `0.518213`.
- The combo is still useful for the TPOT/throughput module: TPOT p95 `16.824 ms`,
  wall tokens/s `507.813`, VRAM max `15422 MiB`.
- `--mamba-backend=flashinfer` remains a runtime backend watchlist knob.
- FlashInfer linear-attention backend likely needs a dedicated graph-policy
  subround because current `tc_piecewise` prefill CUDA graph captures fail.

Carry-forward:

```text
Keep all Transformer/KV optimizations intact.
Add Mamba-state knobs only as explicit candidate overlays.
BF16 SSM has passed local GPQA proxy and same-session trace replay, but remains a watchlist overlay until it beats global Ready 003f.
Mamba FlashInfer + BF16 SSM is a TPOT/throughput/VRAM watchlist, not an ERS baseline.
Next hybrid-specific test should target why the TTFT<=1500 count drops under hybrid overlays before trying to promote them.
Do not re-enter KV-only Tree-Gated design without accounting for Mamba/linear-attention state.
```

## Decode Fairness Under Hybrid SGLang

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006e_decode_fairness_hunt/
```

Target:

- Test whether TTFT is primarily hurt by decode blocking and head-of-line
  scheduling under `CONCURRENCY=120`.
- Keep the hybrid overlay `--mamba-backend=flashinfer --mamba-ssm-dtype=bfloat16`.
- Compare lower `num-continuous-decode-steps`, lower
  `num-reserved-decode-tokens`, and `schedule-policy=fcfs`.

Evidence:

| Candidate | ERS | TTFT <=1.5s | TTFT p50 | TTFT p95 | TPOT p95 | Decision |
|---|---:|---:|---:|---:|---:|---|
| Control `steps4/res224/lpm` | `0.519671` | `22/120` | `3815.775 ms` | `34693.195 ms` | `16.771 ms` | Control for 006e. |
| `res128_steps1_lpm` | `0.522856` | `25/120` | `3971.634 ms` | `35418.402 ms` | `17.358 ms` | Positive but not best. |
| `res64_steps1_lpm` | `0.524547` | `23/120` | `3896.429 ms` | `35006.865 ms` | `17.036 ms` | New provisional local ERS best. |
| `res128_steps2_lpm` | `0.523722` | `25/120` | `3923.631 ms` | `35207.870 ms` | `17.022 ms` | Strong watchlist. |
| `res64_steps2_lpm` | `n/a` | salvaged `1/89` | `19744.251 ms` | `50371.084 ms` | `28.830 ms` | Reject; report JSONL corrupted and salvaged distribution is bad. |
| `fcfs_run12_res128_steps1` | `0.448436` | `3/120` | `8390.090 ms` | `23842.449 ms` | `39.355 ms` | Reject. |

Technical conclusion:

- Lowering decode blocking inside LPM is mildly useful; `steps=1` beats the
  previous hybrid control in this session.
- `reserved=64, steps=1` is the best observed local ERS candidate, but the
  margin over the previous Ready 003f replay is small enough to require a repeat.
- FCFS is the wrong fairness lever for this trace. It removes the `>=30s` TTFT
  bucket but sharply reduces the number of near-floor TTFT requests and pushes
  TPOT p95 to `39.355 ms`, close to the official `45 ms` ceiling.
- Do not tune only by p50/p95. Keep TTFT bucket counts in every future report:
  `<=1.5s`, `1.5-3s`, `3-5s`, `5-10s`, `10-30s`, `>=30s`.

Carry-forward:

```text
Provisional local ERS candidate: hybrid_fair_lpm_res64_steps1.
Repeat before submission and run the local GPQA proxy on the exact config.
Do not promote FCFS for the current public trace.
Fix or avoid the benchmark JSONL writer corruption before trusting high-concurrency report failures.
```

## Internal SGLang Scheduler / Prefill Knob Sweep

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006f_internal_scheduler_prefill_hunt/
```

Target:

- Test SGLang's own scheduler/prefill controls before writing an external proxy.
- Keep the best local hybrid shape: LPM family, `reserved=64`, `steps=1`,
  `--mamba-backend=flashinfer`, `--mamba-ssm-dtype=bfloat16`.

Evidence:

| Group | Best / Key Candidate | ERS | TTFT <=1.5s | TPOT p95 | Decision |
|---|---|---:|---:|---:|---|
| Schedule policy | `lpm_repeat` | `0.521266` | `29/120` | `18.821` | LPM remains preferred. |
| Schedule policy | `lof` | `0.467596` | `3/120` | `31.271` | Reject. |
| Schedule policy | `dfs-weight` | `0.515851` | `19/120` | `17.661` | Not promoted. |
| Chunked prefill | `chunk4096 + mixed` | `0.519818` | `24/120` | `20.334` | Not promoted. |
| Chunked prefill | `chunked-prefill-size=-1` | `0.000000` | `0/120` | `n/a` | Reject; chunked prefill required. |
| Prefill limits | `max-prefill-tokens=4096, prefill-max-requests=2` | `0.524073` | `23/120` | `17.157` | Close watchlist. |
| Graph/conservativeness | `schedule-conservativeness=0.1` | `0.524510` | `23/120` | `16.983` | Closest watchlist, still below best. |
| Graph/conservativeness | `disable-prefill-cuda-graph` | `0.520807` | `22/120` | `16.692` | Better p50/TPOT, lower ERS. |

Technical conclusion:

- SGLang internal flags appear near a local plateau around ERS `0.524`.
- `--schedule-policy=lof` is not a replacement for a custom length-aware
  admission policy on this trace.
- `--enable-mixed-chunk` and dynamic chunking do not improve ERS here.
- Manual prefill limits are safe but only close, not better.
- `--schedule-conservativeness=0.1` is the only new near-miss and should be kept
  as a watchlist, not promoted.
- Prompt-length bucket analysis shows the `>=30s` TTFT tail often comes from
  `<20k-word` prompts, while `>=20k-word` prompts are not the sole tail source.
  The issue is admission/cache/order behavior, not prompt length alone.

Carry-forward:

```text
Current provisional best stays hybrid_fair_lpm_res64_steps1, ERS 0.524547.
Keep schedule-conservativeness=0.1 and max-prefill-tokens=4096/prefill-max-requests=2 as watchlists.
Next: repeat + quality-gate current best, then build a lightweight admission/priority proxy if still TTFT-capped.
```

## Logical Two-Policy HTTP Proxy

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_007_logical_two_policy_proxy/
```

Target:

- Test the user's single-model two-policy lifecycle split without duplicating
  the model and without manual KV/Mamba-state handoff.
- Policy A controls requests before first token through `prefill_limit`.
- Policy B preserves decode/TPOT through `total_limit` while SGLang handles
  all model execution on one backend.

Evidence:

| Candidate | ERS | TTFT <=1.5s | TTFT p50 | TTFT p95 | TPOT p95 | Decision |
|---|---:|---:|---:|---:|---:|---|
| `proxy_a_prefill8_total12` | `0.488320` | `4/120` | `8215.968 ms` | `14608.821 ms` | `22.355 ms` | Reject |
| `proxy_b_prefill10_total12` | `0.491543` | `6/120` | `8427.246 ms` | `15442.091 ms` | `22.147 ms` | Reject |
| `proxy_c_prefill10_total14` | `0.493432` | `3/120` | `6618.465 ms` | `11752.298 ms` | `21.721 ms` | Best in round, reject globally |
| `proxy_d_prefill12_total16` | `0.472062` | `4/120` | `5213.210 ms` | `9308.726 ms` | `26.566 ms` | Reject |

Technical conclusion:

- The lifecycle split worked mechanically: first streamed event released the
  prefill slot, while completion released the total active slot.
- The proxy removed the extreme `>=30s` TTFT tail, but that is not enough for
  scoring.
- It destroyed the scoring-critical `<=1500 ms` bucket and pushed TPOT p95
  above the official `20 ms` floor.
- Queue wait in the proxy is still TTFT-visible, so the implementation is too
  high in the stack.

Carry-forward:

```text
Do not use HTTP-level two-policy admission as a submission path.
Keep two_policy_proxy.py only as a research harness.
Future two-policy attempts must be engine-native: scheduler hooks, native priority/admission, or true in-engine prefill/decode lifecycle control.
```

## vLLM 0.22.1 Hybrid-Aware Serving

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_006_hybrid_architecture_audit/subround_006g_vllm_hybrid_bakeoff/
```

Evidence:

| Technique | Measured effect | Decision |
|---|---|---|
| `--language-model-only` | Model load memory `4.25 -> 3.63 GiB`, wall throughput `353.246 -> 396.996 tok/s`, TPOT p95 `15.788 -> 10.001 ms`, ERS `0.502427 -> 0.503230` | Keep in vLLM fallback |
| BF16 Mamba cache state | TPOT p95 improved to `9.475 ms`, ERS dropped to `0.502822` | TPOT-only; no promotion |
| `--mamba-cache-mode=all` | Unsupported for Qwen3.5; runtime falls back to `align` | Do not use as an optimization claim |
| FlashInfer GDN prefill | Unsupported on RTX 5080 path; runtime falls back to Triton/FLA | H200-only recheck at most |
| Concurrent partial prefill | V1 startup raises `NotImplementedError` | Do not configure in vLLM 0.22.1 candidate |
| Interactivity + `max_num_batched_tokens=8192` | TTFT p50 worsened to `19.103s`, ERS `0.501255` | Reject |
| Run6 + relaxed full-ISL admission | TTFT p50/p95 improved to `13.520/21.791s`, extreme tail removed, throughput `468.408 tok/s`; fast bucket fell to `2/120`, ERS `0.501262` | Tail-only watchlist; reject for ERS |

Technical conclusion:

- The exact vLLM image handles the hybrid model natively, but its available
  Qwen3.5 prefix-state and scheduler controls cannot reproduce SGLang's fast
  TTFT bucket on this trace.
- `--language-model-only` is the only unambiguous carry-forward technique for
  a vLLM fallback.
- Better p50/p95 is not sufficient. The official TTFT ceiling makes the
  `<=1.5s` request count more important than eliminating the extreme tail.

Carry-forward:

```text
Primary framework remains SGLang.
vLLM fallback should use language-model-only, native model implementation,
prefix caching, chunked prefill, auto quality-safe cache, max_num_seqs=4,
and the protected official max-model-len at submission time.
Do not carry concurrent partial prefill, Mamba cache all, or claimed FlashInfer GDN prefill into the fallback.
```

## Engine-Native Scheduler Telemetry and Conditional Prefix Rescue

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_008_engine_native_ers_scheduler/subround_008a_scheduler_telemetry/
```

Technique:

- Pin the exact SGLang image/source contract.
- Install an opt-in import overlay through `sitecustomize.py` so spawned
  scheduler processes are instrumented without rebuilding the image.
- Observe native admission, LPM order, prefill/decode batches, first-token
  transition, MambaRadixCache commits and eviction.
- Keep wrappers read-only and call every original method exactly once.

Parity evidence:

| Candidate | ERS | Success | TTFT <=1.5s | TPOT p95 | Decision |
|---|---:|---:|---:|---:|---|
| Native control | `0.512927` | `120/120` | `23/120` | `19.392 ms` | Same-session control |
| Read-only telemetry | `0.511176` | `120/120` | `24/120` | `17.918 ms` | Accepted measurement layer |

Telemetry evidence:

```text
Valid events: 7147
Malformed lines: 0
Lifecycle coverage: 120 queued / 120 first-token / 120 finished
Cache: MambaRadixCache
Batch modes: 237 prefill / 2800 decode / 0 mixed
Mean batch size: prefill 1.447 / decode 8.571
Queue depth max / mean: 56 / 26.251
First-prefill prefix match ratio mean / p95: 0.7767 / 0.8930
Unfinished chunk commits: 343
Evictions: 51 calls, 114461 full-KV tokens, 0 Mamba states
Retractions: 0
```

Scheduler cost evidence:

```text
Prefill schedule mean/p50/p95: 73.910 / 77.781 / 90.429 ms
Decode schedule mean/p50/p95: 0.212 / 0.207 / 0.266 ms
LPM calculation mean/p95: 2.134 / 4.334 ms
```

Technical conclusion:

- The native control forms small prefill waves followed by dense decode waves.
- Strong prefix reuse explains why LPM remains better than FCFS/LOF.
- LPM calculation is only a small part of measured prefill scheduling cost.
- Full-attention KV churn is visible; Mamba-state eviction is not the current
  capacity signal.
- Since there are no retractions, OOM recovery is not creating the TTFT tail.
- Global mixed chunking was already negative in Ready 006f, so the useful new
  policy is conditional.

Candidate technique for Phase B:

```text
Conditional prefix rescue:
1. Keep native LPM as default.
2. Detect a waiting request with high cached-prefix ratio and small estimated
   remaining prefill cost.
3. Admit at most one such request during a decode wave if its projected first
   token can cross the 1500ms score boundary.
4. Preserve decode service with a TPOT deadline guard around 20ms.
5. Add aging only to prevent starvation; expired requests should not consume
   rescue budget merely to improve p95.
```

Phase B shadow evidence:

| Attempt | ERS delta vs same-session control | Rescue support | Precision <=1500 | Early/Late support | Reading |
|---|---:|---:|---:|---:|---|
| v1 strict guard | `-0.000592` | `14` | `0.9286` | `0 / 14` | Decode guard was too jitter-sensitive |
| v2 verbose observer | `-0.010026` | `12` | `0.7500` | `6 / 6` | Labels useful; observer implementation too expensive |
| v2 optimized observer | `+0.002358` | `1` | `1.0000` | `1 / 0` | Observer cheap; static policy support unstable |

Observer implementation result:

```text
Repeated terminal labels: 6628 total label events, including 5941 EXPIRED
State-change-only labels: 210 total label events
Optimized evaluator mean/max: 108 / 643 us
Profiled evaluator CPU total: 24.2 ms
Final lifecycle: 120 queued / 120 first-token / 120 finished / 120 labeled
Shadow errors: 0
```

Technical conclusion for the candidate:

- Match ratio and remaining-prefill cost are useful predictive features, not
  a stable standalone policy.
- A rolling guard calibrated as baseline p95 plus 1 ms is too brittle; p97.5
  plus headroom avoided the first false block, but did not solve cross-run
  cohort movement.
- Frozen thresholds fitted to one native timeline must not be promoted to
  active scheduling.
- State-change-only emission is itself a reusable systems technique for
  low-core telemetry overlays.

Carry-forward:

```text
Use the telemetry overlay for engine-native A/B and dataset generation.
Do not call it a serving optimization; it is a measurement layer.
Keep the optimized shadow evaluator and offline replay tooling, but reject the
frozen Ready 008b policy as an active scheduler.
Collect multiple independent native timelines and use leave-one-run-out
validation before another policy attempt.
Do not add XGBoost/C++ before a simple cross-run policy proves control authority.
Do not retry HTTP proxy queueing, global FCFS/LOF or global mixed chunking.
Pin the SGLang digest before applying any scheduler overlay.
```

## Adaptive Threshold Calibration

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_008_engine_native_ers_scheduler/subround_008c_adaptive_threshold_shadow/
```

Technique:

- Seed generic request snapshots at queue admission, then replace them when a
  higher expected TTFT-score-gain-per-cost observation appears.
- Join each snapshot to its later native first-token outcome without using the
  request tag as a feature.
- Bootstrap from bounded queue quantiles before 12 labeled outcomes.
- Recalibrate every four outcomes over a rolling 48-outcome window.
- Search thresholds using posterior precision, old/new split precision,
  recall, target support and actual TTFT score.
- Limit threshold movement per update and compare recent decode pressure with
  an older baseline window.

Cross-run evidence:

| Timeline | Role | Support | Precision | Lift | Early/Late |
|---|---|---:|---:|---:|---:|
| Ready 008a | primary | `16` | `0.8750` | `4.0385x` | `9/7` |
| Ready 008b v1 | primary | `26` | `0.9231` | `3.8196x` | `10/16` |
| Ready 008b v2 optimized | primary | `24` | `0.7917` | `3.8000x` | `14/10` |
| Ready 008b verbose | stress | `32` | `0.5938` | `2.9688x` | `18/14` |

Live evidence:

```text
control/adaptive ERS: 0.512949 / 0.512680
observer delta: -0.000269
success: 120/120 each
TPOT p95: 18.686 ms
support/precision/lift: 27 / 0.6667 / 3.3333x
early/late support: 15 / 12
evaluate mean CPU: 183 us
calibration mean CPU: 3.59 ms
forbidden features: none
```

Converged live thresholds:

```text
match ratio >= 0.50
remaining suffix <= 2944 tokens
projected TTFT <= 800 ms
running requests <= 5
```

Technical conclusion:

- Adaptive calibration fixed the static policy's support collapse across
  scheduler timelines.
- Queue-admission snapshots are required because native direct-admit requests
  may reach first token without appearing in a `policy_order` event.
- Hysteresis is necessary; thresholds converged over multiple updates rather
  than jumping directly to one timeline's optimum.
- The policy is predictive and cheap enough for one conservative active test,
  but its shadow ERS is not a serving gain.

Carry-forward:

```text
Keep adaptive bootstrap, supervised updates, hysteresis and decode guard.
For the first active probe, move at most one request and preserve native LPM
for every other request. Add an immediate environment kill switch.
Do not introduce XGBoost/C++ until this simple controller proves causal ERS
value in repeated active A/B.
```

## Bounded Global-Front Active Rescue

Round:

```text
local_workspaces/rtx5080/research_rounds/ready_rounds/round_008_engine_native_ers_scheduler/subround_008d_conservative_active_rescue/
```

Technique:

- Run native LPM prefix matching and sorting first.
- Let the accepted adaptive controller select at most one candidate from
  native position `1..8`.
- Promote only that request to position zero and preserve every other
  request's relative order.
- Require 12 supervised outcomes, a 150ms cooldown, one move per request,
  eight moves maximum, an adaptive decode guard and environment/runtime-file
  kill switches.
- Compare against an adaptive-shadow control with the same evaluator cost.

Safety evidence:

```text
offline primary opportunity counts: 3 / 8 / 6
live interventions / unique / commits: 6 / 6 / 6
prebenchmark interventions: 0
native positions: [3, 1, 1, 2, 1, 1]
policy errors / malformed JSONL / forbidden features: 0 / 0 / 0
```

Performance evidence:

| Metric | Shadow control | Active | Delta |
|---|---:|---:|---:|
| ERS | `0.514307` | `0.512578` | `-0.001728` |
| TTFT <=1500 | `21` | `23` | `+2` |
| TTFT p50 | `4308.092 ms` | `4447.015 ms` | `+138.923 ms` |
| TTFT p95 | `38513.853 ms` | `40489.134 ms` | `+1975.281 ms` |
| TPOT p95 | `18.353 ms` | `19.430 ms` | `+1.076 ms` |
| Wall tokens/s | `460.727` | `436.369` | `-24.359` |

Paired official-score decomposition:

```text
TTFT component delta: -0.002143
TPOT component delta: +0.000415
improved / worsened / unchanged request scores: 10 / 19 / 91
```

Technical conclusion:

- The adaptive predictor remained useful: five of six promoted requests
  reached first token in `287-821 ms`, and applied-request precision was
  `0.8333`.
- Prediction precision is not enough for scheduler control. Global-front
  promotion crossed LPM prefix-locality boundaries and delayed more neighboring
  requests than it helped.
- The score loss was a TTFT externality, not primarily TPOT regression. TPOT
  p95 rose, but most TPOT values remained under the 20ms score floor.
- Bootstrap thresholds temporarily loosened suffix from the accepted `2944`
  to `3968`; the first position-three intervention reached `1549 ms` and was
  not score-positive.

Carry-forward:

```text
Reject global-front active rescue and do not run 008e repeatability.
Keep the kill-switch/fallback machinery as research infrastructure.
Before another active attempt:
- expose a generic cache-node or equal-prefix cohort identity;
- move only inside that cohort;
- preserve all order outside the cohort;
- deny active authority while thresholds are looser than 008c convergence;
- pass multi-run shadow/replay before live mutation.
Keep XGBoost/C++ disabled.
```

## Notes

- Local RTX 5080 evidence is directional only. Official target is H200 MiG 18GB.
- A technique is not submission-safe until it preserves request success, output validity and likely accuracy-gate behavior.
- Avoid hardcoding trace outputs or modifying tokenizer. Serving optimizations must remain legal under BTC rules.

## SGLang Unscaled FP8 KV Revalidation

Subround 006h retested KV compression on the exact Attempt 02 SGLang policy
shape with context `262144`, no external warmup and repeated GPQA passes.

| KV dtype | GPQA A/B | Repeat prediction changes | ERS | TTFT p50 | TPOT p95 | KV tokens | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `auto` | `0.37 / 0.38` | `2` | `0.495794` | `14136.840 ms` | `25.765 ms` | `241871` | Control |
| `fp8_e4m3` | `0.36 / 0.33` | `9` | not run | n/a | n/a | `502092` in quality server | Reject quality |
| `fp8_e5m2` | `0.35 / 0.36` | `10` | `0.487801` | `5326.200 ms` | `27.858 ms` | `494760` | Reject quality/performance |

Carry-forward:

- FP8 capacity gain is real, approximately `2.05x` KV tokens at fixed static
  memory.
- Unscaled FP8 is not submission-safe for this hybrid model.
- SGLang `0.5.14` lacks a Qwen3.5 KV-scale loader; calibrated FP8 requires a
  custom runtime patch and a new quality gate.

## Brain1 Tree-Gated GPQA Revalidation

Subround 006i isolated the Ready 003 Brain1 proxy against a direct path through
the same Attempt 02 SGLang engine. It ran sequential and 16-worker GPQA traffic,
twice per path.

| Mode | Direct accuracy A/B | Tree accuracy A/B | Tree repeat changes |
|---|---:|---:|---:|
| Sequential | `0.37 / 0.38` | `0.38 / 0.38` | `0` predictions |
| Concurrent | `0.38 / 0.38` | `0.38 / 0.37` | `3` predictions |

Evidence:

```text
worst direct-minus-Tree accuracy drop: 0.01
Tree minimum official-style penalty: 1.0
proxy rows/status: 400 / all ok
concurrent queued rows: 198/200
queue wait p50/p95: 25.244 / 132.308 ms
```

Limitation:

- Brain1 returned constant priority `0.879565` for every GPQA request, so these
  questions occupied one learned leaf/cohort.
- Quality safety passed for the proxy/admission path, but differentiated Tree
  ranking was not exercised.
- Keep Tree-Gated quality-safe on the watchlist, not submission-ready; trace
  ERS must offset its queue/dispatch overhead.

## Native Tree-Gated V2 And Bounded Rejuvenation

Subround 006j rebuilt Brain1 from SGLang scheduler telemetry instead of the
old circular RAM/VRAM heuristic label.

Dataset and target:

```text
source timelines: 11 across 6 grouped sessions
full scheduler snapshots: 54,729
training snapshots: 11,574
timeline/request pairs: 1,320
learned features: 25
target: observed positive TTFT score under 100/1500ms bounds
raw prompt/response features: none
```

Queue age and deadline progress remain available to the explicit aging
controller but are excluded from the model, preventing an inflated score from
simply recognizing requests that already missed the deadline.

Model evidence:

| Metric | Result |
|---|---:|
| Grouped-session test ROC AUC | `0.910670` |
| Actionable queue-age <=1000ms ROC AUC | `0.900176` |
| Actionable score standard deviation | `0.137673` |
| Pure C++ p99 | `0.088us` |
| Python to C ABI p99 | `0.454us` |
| Whole observed queue score plus top-K p99 | `23.867us` |
| Existing SGLang policy calculation p99 | `5409.762us` |

Only four features received stump gain: native LPM position, remaining prefill
tokens, scheduler pressure and normalized native position. This is useful
score separation, but it also means the learned ranker may only compress a
small direct heuristic. A future A/B must compare both at equal overhead.

Rejuvenation evidence:

| Policy | Positive coverage | Selected precision | Prefix ratio | Rescued |
|---|---:|---:|---:|---:|
| Base | `1.0000` | `0.049592` | `0.578529` | `0` |
| Gentle locality | `1.0000` | `0.049438` | `0.577361` | `161` |
| Prefix guarded | `1.0000` | `0.049438` | `0.578297` | `181` |
| Soft aging | `0.9964` | `0.039478` | `0.567179` | `191` |
| Forced deadline | `0.1159` | `0.006502` | `0.517823` | `211` |

Carry-forward:

- Native scorer overhead is acceptable, but live feature extraction is still
  unmeasured.
- Gentle locality rejuvenation is the only bounded shadow candidate. It
  reduced selected-age p50 by about `517ms` and churn by about 109 transitions,
  but did not improve every metric.
- Reject soft/forced aging. A request beyond the TTFT ceiling should not gain
  global priority merely because it is old.
- Rejuvenation cannot recreate context. Promotion requires a retained valid
  KV plus Mamba-state handle or paid recomputation.
- Do not claim RAM/VRAM migration or ERS improvement until a real native hook
  measures bytes moved, transfer/recompute latency, cache locality and causal
  full-trace score.

## Exact Cache-Node Tree Versus Heuristic Shadow

Subround 006k implemented the follow-up required by 006j. SGLang native LPM
runs first; requests are grouped by the same non-root `req.last_node.id`; both
rankers are evaluated; the original queue is returned unchanged.

Offline comparison:

| Metric | Tree v2 | Four-feature heuristic |
|---|---:|---:|
| Validation AUC | `0.953414` | `0.974930` |
| Test AUC | `0.910670` | `0.885298` |
| Actionable test AUC | `0.900176` | `0.916530` |
| Top-K selected positive rate | `0.090717` | `0.090717` |

The heuristic was selected from validation only. Live evidence:

```text
control/shadow ERS: 0.524547 / 0.518136
TTFT<=1500: 25 / 20
success: 120/120 each
exact cohorts / unique nodes: 263 / 60
Tree/heuristic would-reorder: 0 / 0
evaluator p99: 333.063us
queue mutations: 0
```

Technical conclusion:

- Exact native node identity is a safe, generic locality label; it is never a
  learned feature and no prompt/token content is logged.
- The four effective model features are too aligned with native LPM inside
  this boundary. Both policies reproduce the native cohort head exactly.
- Live AUC is misleading here because only `2/57` joined outcomes were
  positive and the candidate action set was empty.
- Do not activate, submit or further latency-tune this exact policy. Preserve
  it as negative research evidence and a source-audit instrument.
- Future learned scheduling must prove nonzero held-out action support before
  a costly live run, using new pre-decision features and a broader but still
  locality-preserving cohort definition.

## Parent-Ancestry And Non-Position Replay

Subround 006l reused 006k telemetry without launching a server. It reconstructed
the process-local cache graph from `146` observed nodes and found no conflicting
`child -> parent` relationship.

Structural support:

```text
direct-parent groups: 276; distinct current nodes/branches: 0/0
ancestor hops 2/3/4/5 groups: 248/217/49/11; distinct nodes: 0
normalized depth 3/4/5 groups: 165/164/160; distinct nodes: 0
```

This means ancestry labels do not produce a broader queue-action cohort for
the observed trace. Requests sharing reconstructable ancestry were already at
the same exact `last_node`.

The best exact-node policy with native position forbidden was `service_only`:

```text
validation cohorts/changes: 216/126
unique selected requests after change: 4
support wins/losses: 0/6
native/candidate positive capture: 1.0/0.0
TTFT-component delta: -0.002374
matched-ratio delta: +0.000377
```

Carry-forward:

- Reject static service/locality/deadline reranking inside these cache cohorts.
- Native position is not merely leakage here; it summarizes useful LPM
  locality/arrival state that protected the score-positive heads.
- Do not launch another live ranker until new offline evidence uses temporal or
  causal features and preserves positive capture.
- Prefer tuning prefill/decode budgets, admission capacity or engine execution
  over another snapshot-only ranker.

## Score 100 R012 Hybrid Hotspot Profile

Measured paths:

```text
cold prefill, 6931 tokens: 157.491ms GPU, 95.9% busy
pure batch-20 decode:      3.9795ms per step, 94.8% busy
cache branch hit rate:     96.8%, 219.5 residual tokens/request
```

Ranked evidence:

- Cold prefill: GEMM `68.7%`, full attention `11.1%`, GDN `9.9%`, norm/quant
  `6.4%`.
- Pure decode: tied BF16 LM head `28.6%`, GDN `8.4%`, norm/quant `6.4%`, full
  attention `1.2%`.
- CPU scheduling and launch gaps were not the dominant cold-compute limit.

Carry-forward:

- Optimize the tied output head before custom attention/decode work.
- Hybrid snapshots remain the causal prefix branch because native alignment
  still recomputes 219.5 tokens after a high cache hit.

## Score 100 R013 FP8 Tied LM Head

Implementation:

- Keep the tied BF16 `VocabParallelEmbedding` for input lookup.
- Create one output-only per-tensor E4M3 weight after loading.
- Quantize decode hidden rows per token, run CUTLASS FP8 GEMM, select top four,
  and recompute those four logits exactly with BF16 rows.
- Fall back to native for TP != 1, non-CUDA, non-BF16, non-2D,
  non-contiguous or bias-bearing calls.

Evidence:

```text
real hidden replay:          top1 500/500, first-token top1 100/100
stable control mean ERS:     0.514707
top-k4 full-trace ERS:       0.532424
control/candidate TPOT p95:  24.214/23.158ms
control/candidate throughput:797.417/803.248 tok/s
GPQA control/candidate:      35/100 and 33/100, deterministic repeats
persistent memory:           +485MiB
KV capacity:                 -6.63%
```

Carry-forward:

- Retain top-k4 as a provisional kernel candidate.
- Raw FP8 is not selected even though it is fast; top-k4 carries the stronger
  deterministic quality contract.
- Require packaged A/B/A before adding this to I006/I007.

## Score 100 R014-R017 Negative Kernel Boundaries

R014 native norm/quant fusion:

- Server and CUDA Graph reached Ready.
- Compiler logged `Replaced 0 patterns` in every partition.
- Reject because the flag changes no timed graph node.

R015 GDN decode/state:

- Best launch geometry validated at only `1.0015x` batch-20 speedup.
- FP8 state: `0.9554x` at batch 20 and failed 200-step drift quality.
- INT8 state: `0.9570x` at batch 20; dynamic row quality improved but still
  missed strict NRMSE/cosine gates.
- Keep native BF16 recurrent state. Mixed-layer R017 is skipped because each
  quantized layer is slower on this implementation.

R016 GDN long prefill:

- Chunk32 was `0.8984x`, chunk128 asserted, chunk64 remains best.
- CuteDSL failed local MLIR because Cutlass DSL CU13 integrity is false.
- Packaged FlashInfer GDN rejects SM120; vLLM auto uses FlashInfer on target
  H200 SM90, so automatic selection is retained.

Carry-forward:

- Do not add R014-R017 code or flags to a candidate.
- Preserve all harnesses as regression and architecture evidence.
- Move to R018 hybrid attention-KV plus Mamba-state snapshot economics.

## Score 100 R018 HybridRadixState Upper Bound

Exact state payload:

```text
18 linear-attention layers
conv state/layer:      [3, 6144] BF16
recurrent state/layer: [16, 128, 128] BF16
raw state/layer:       561,152 bytes
raw all-layer state:   10,100,736 bytes
```

Trace result:

```text
native block560 measured compute: 456,542
ideal block16 projected compute:  429,646
saved:                            26,896 / 5.89%
largest wave bound:               180.0ms / 4.18% Portal TTFT p50
```

Memory result:

- Endpoint-only conservative peak fits at `32.41-32.74%` of the local pool.
- State at every boundary uses `108.91%` at block560 and `2911.53%` at
  block16.

Carry-forward:

- Reject HybridRadixState for the current trace.
- Do not build R019 singleflight or R020 two-policy from this representation.
- Keep native BF16 align-mode endpoints.
- Package and qualify R013 top-k4 next in R021a.

## Score 100 R021 Packaged FP8 Head And Seq Balance

Package contract:

- exact I006 public digest as the parent image;
- one image for control/candidate/control;
- `VLLM_FP8_TIED_LM_HEAD=0/1/0`, top-k fixed at four;
- same seq16, warmup, graph sizes, context, memory and scheduler;
- one RTX 5080, three CPU cores, 8GB RAM and no serving-network egress.

Measured result:

```text
candidate/control midpoint ERS: 0.626978 / 0.623107
TTFT p50/p95:                   962.477 / 2866.026ms
TPOT p95/p99:                   14.492 / 15.805ms
throughput:                     819.557 tok/s
peak VRAM delta:                +95MiB
GPQA candidate/control:         35/100 x2 / 34/100 x2
```

Activation markers, native fallback guard, CUDA Graph guard, `120/120` and
penalty proxy `1` all passed. Promote this technique into local I007.

The seq balance sweep then measured:

```text
seq15/16/17 ERS:      0.620712 / 0.620628 / 0.621375
seq15/16/17 TPOT p99: 15.389 / 16.217 / 17.079ms
```

Carry-forward:

- Keep seq16. The seq17 gain is below session drift and consumes decode-tail
  margin; seq15 gives up TTFT capacity without an ERS return.
- Treat the user-reported H200 score increase `46 -> 51` at seq16 as the
  target-hardware tie-break, while keeping its exact metric panel pending.
- Preserve feature-off in the same image as rollback.
- Do not treat local score proxy `62.6978` as an H200 prediction.

## Score 100 R022a FP8 Linear Backend Capability

R012 justified this screen because GEMM consumed `68.7%` of cold-prefill GPU
time. The exact I006 resolver produced:

```text
auto / explicit cutlass: CutlassFP8ScaledMMLinearKernel, Ready 102/103s
flashinfer_cutlass:      incompatible online-FP8 scale layout
flashinfer_cudnn:        no kernel for layer type
flashinfer_trtllm:       no kernel for layer type
triton:                  no kernel for layer type
```

Carry-forward:

- vLLM auto already selects the retained local FP8 GEMM kernel;
- explicit CUTLASS is not a new technique and gets no full-trace credit;
- unsupported kernel names must fail at startup rather than be submitted;
- preserve the `18ms` TPOT-p99 hard gate before evaluating TTFT on future
  candidates;
- H200-specific kernel capability remains a separate target-image question.

## Score 100 R034 GPTQ W4A16 / Marlin

Implementation contract:

- GPTQModel `7.0.0` in an isolated converter image;
- W4A16/G128, symmetric, desc_act off, true sequential;
- `150` quantized text linears (`54` linear-attention, `24` full-attention,
  `72` MLP); vision/MTP/head/conv1d/small GDN projections retained;
- exact I006 image for FP8/GPTQ/FP8 serving, seq16, FP8 KV, BF16 Mamba state;
- three CPU cores, 8GiB RAM, no swap/network, one RTX 5080.

Measured result:

```text
candidate/control midpoint ERS: 0.542724 / 0.609622 (-10.974%)
candidate TTFT p50/p95:         1977.5 / 4307.9ms
candidate TPOT p95/p99:         22.209 / 23.213ms
candidate/control throughput:   791.048 / 804.989 tok/s
paired W/T/L:                   0/37/83
positive waves:                 0/6
GPU blocks:                     2827 / 2702.5 (+4.607%)
GPQA:                           37/37/35, penalty proxy 1
```

Carry-forward:

- reject all-core GPTQ/AWQ through current Marlin;
- keep native online FP8/CUTLASS in I006;
- do not trade TPOT for unused KV capacity;
- keep converter and artifact only as reproducible research evidence;
- require a distinct H200-native INT4/NVFP4 kernel microbenchmark before any
  target-only precision branch reopens.

## Score 100 R035 Timed JIT Accounting

Method:

- immutable R034 three-cell logs only; no server or GPU;
- exact byte-prefix requirement for every prebenchmark snapshot;
- separate kernel extraction before and after the snapshot boundary;
- exact I006 source ownership audit inside the immutable image.

Measured result:

```text
exact snapshot prefixes: 3/3
prebenchmark JIT:        5/5/5
timed suffix JIT:        0/0/0
audit gates:             7/7
GPU runs:                0
```

Technical conclusion:

- the old R034 counter included startup and mislabeled the warnings;
- packaged I006 already compiles KV, Mamba and GDN kernels before readiness;
- no warmup expansion or shape sweep is justified;
- this is an accounting correction, not a latency improvement;
- R034 GPTQ/Marlin remains rejected and I006 seq16 remains rollback.

## Score 100 R036 Component-Level H200 Transfer Audit

Method:

- recalculate official TTFT/TPOT component scores from nine immutable local
  request panels;
- join seven exact local/Portal anchors and four Portal-only context anchors;
- derive Portal component bounds from score and TTFT `passed_slo` support;
- compare direction per component instead of extrapolating local total ERS.

Measured result:

```text
decode-dominant H200 comparisons:       2
local/Portal score direction conflicts: 3
median-TBT traps:                       4
I006 support-limited score ceiling:     56.666667
support required for score 70 / 100:    48 / 120
```

Carry-forward:

- use local component/distribution direction, never local total ERS alone, to
  select H200 probes;
- do not treat TBT median `<=20ms` as a saturated TPOT component;
- retain I006 seq16 and open only an offline exact-image H200 decode capability
  screen;
- require a new compute-saving TTFT-support primitive for score 70+;
- create no GPU run or submission from this audit itself.

## Score 100 R037 Exact-Image Capability Audit

Method:

- freeze exact I006 digest and runtime versions;
- classify native H200 resolvers, decode kernels, cache layouts, MRV2,
  precision branches and frontend paths;
- validate evidence locators and prefix-cache counters mechanically;
- require distinctness, executability, contract safety and at least a
  `0.5`-point score ceiling before opening GPU work.

Result:

```text
capabilities:                 14
eligible GPU candidates:      0
DS causal-conv decode share:  1.296%
DS causal-conv prefill share: 1.196%
prefix reuse lost by MRV2 no-cache: 81.129%
prompt-compute multiplier:    5.299x
```

Carry-forward:

- retain DS layout as a valid low-cost companion knob only;
- never combine DS with align-mode speculative rollback on vLLM 0.22.1;
- do not bypass the MRV2 align validator or remove prefix caching;
- stop exact-image scalar screening and target first-token support causally;
- keep I006 seq16/default2048 unchanged.

## Score 100 R038 Gap-Aware First-Token Staging

Technique: one model and one hybrid state pool with 16 logical decode slots,
one serial prefill/first-token staging lane and bounded parked requests.

Offline six-process sensitivity result:

```text
stage slots / queue threshold: 8 / 500ms
queue removal / interference:  75% / 10%
mean/worst ERS opportunity:    +0.016036 / +0.013850
positive panels/waves:         6/6 and 36/36
worst wave ERS delta:          +0.002806
modeled TPOT p95/p99 max:      18.183 / 19.032ms
mean fast-TTFT gain:           +13.7 requests
```

Trade-off boundary:

- 50% queue removal yields only `+0.007925` mean ERS, below the drift gate;
- 25% decode interference keeps positive ERS but raises TPOT p95/p99 to
  `19.677/21.634ms`, outside the guard;
- native seq24 alone is not this technique; capacity must be split logically
  into decode and staged/parked lifecycle sets;
- do not repeat R007 global budget switching or R023e/f queue mutations.

Carry-forward: permit a static-tested gap-aware scheduler prototype only. It
must schedule decode first, spend measured slack on one stage chunk, preserve
KV/Mamba state in-engine and fail closed to I006. No serving candidate or
submission is promoted by the offline result.

## Score 100 R039 Gap-Aware Scheduler Shadow Boundary

Implementation status: mechanism study complete; no serving promotion.

```text
behavioral tests:              9 passed
policy boundary tests:         2 passed
shadow adapter boundary tests: 2 passed
AST violations:                0
valid runtime runs:            3 x 20/20
mutating stage actions:        0
official score:                not applicable
```

The policy remains bounded by 16 decode slots plus eight staged/parked slots.
The exact-image shadow `AsyncScheduler` reads only lifecycle metadata and calls
native `super().schedule()` without changing queues, budget, KV blocks or
hybrid state. Native/shadow/native timing is effectively tied. No stage action
occurs because the shadow boundary has no measured stage-cost signal and cannot
prove parked-state retention.

Carry-forward: keep the policy as a design specification only. A future
mutating fork must prove Transformer KV plus Mamba state retention/restoration,
then demonstrate queue gain and TPOT protection on a causal full-trace A/B/A.
Until that exists, keep I006 seq16/default2048 unchanged.

## Score 100 R040 Parked Hybrid-State Scheduling

Technique: park and resume complete Transformer KV plus Mamba request state,
then shorten displacement to the staged request's first-token interval.

Runtime result:

```text
state continuity:                 passed
complete swap/resume pairs:       2
candidate/native-midpoint ERS:    0.454380 / 0.466172
candidate relative ERS change:    -2.529%
TTFT-supported requests:          5/20 -> 5/20
candidate/native TPOT p95:        26.491 / 24.434ms
```

The mechanism is real, but both tested policies spend decode capacity after
the request is already too far outside the TTFT score ceiling. Direct hold and
first-token swap therefore lose TPOT/ERS without creating support.

Carry-forward: retain the proven hybrid-state continuity path only as research
infrastructure. Reject both active policies, create no full120/package, and
keep I006 seq16/default2048 unchanged.

## Score 100 R041 First-Wave Service Slope And FP8 Backends

Method:

- decompose seven native first-wave processes into queue and service slopes;
- derive the minimum causal reduction needed to add TTFT support;
- source-audit serialized static FP8 support in the exact vLLM image;
- benchmark static FlashInfer and Torch row-wise FP8 on the exact 114-GEMM
  Qwen chunk matrix before any engine integration.

Result:

```text
active slope at reservation 0.82:       158.037ms/request
active slope at reservation 0.72:       232.667ms/request
queue/service span:                     2510.046 / -117.523ms
minimum whole-engine reduction:         6.5%
required GEMM-category reduction:       9.467%
static FlashInfer MM vs CUTLASS:         -8.762%
static FlashInfer quant+MM vs current:   -10.270%
Torch row-wise MM vs CUTLASS:            -3.954%
Torch row-wise quant+MM vs current:      -0.867%
```

Serialized static FP8 is executable and unlocks FlashInfer, but executability
does not imply speed. Both backend alternatives are numerically valid and
slower than the native CUTLASS path; static quant/fusion alone also has only a
`6.435%` cold-prefill category ceiling, below the required GEMM reduction.

Carry-forward: close static FlashInfer, static-only conversion and Torch
row-wise integration. Preserve online FP8/CUTLASS and I006
seq16/default2048. Any later first-wave work must expose a new compute-saving
primitive or a generic, robust conditional policy before using the GPU.

## Score 100 R042 Global Budget Cohort Separability

Technique: use generic arrival/cache/inflight state to choose one global
prefill-token budget for an arrival burst, with six-process bidirectional
labels and leave-one-wave-out validation.

Result:

```text
fixed 4096 ERS delta:               -0.010936
wave 2 forward/reverse delta:       +0.013964 / +0.001507
wave 2 positive process pairs:       6/9
wave 2 robust q25:                  -0.008325
request-level oracle gain:          +0.002348
global-wave oracle gain:            +0.001128
global-wave robust oracle gain:     +0.000000
```

The request oracle is not implementable in vLLM because the budget belongs to
the scheduler iteration. The implementable wave oracle is below the `0.005`
gate, below process noise, and has no positive robust cohort. The robust
full-data rule and all held-out folds correctly choose fixed 2048.

Carry-forward: close conditional `2048/4096` routing. Do not increase model
depth or retrain Tree-Gated for this action; prediction cannot recover value
that the executable oracle does not contain. Preserve I006
seq16/default2048.

## Score 100 R043 Native Fused Reduction Envelope

Technique: call vLLM's native fused RMSNorm plus dynamic per-token FP8 op
directly and compare exact Qwen prefill/decode shapes before patching the graph
matcher.

Result:

```text
M211 reduction vs two raw ops:       +6.789%
M1680 reduction vs two raw ops:    -175.056%
weighted prefill reduction:        -156.311%
whole-prefill projection:           -10.058%
M20 native fused:                     3.822us
real compiled Triton RMS average:     2.832us
```

Small eager shapes create a misleading `~31%` win against two raw calls. The
serving graph already emits one Inductor-fused reduction, which is faster than
the native fused op. Large prefill shapes strongly regress.

Carry-forward: keep Inductor-generated RMSNorm/SiLU plus quant reductions.
Reject matcher forcing, manual native-op insertion and a custom activation
kernel derived from this path.

## Score 100 R044 Remaining Attention/GDN Hotspot Audit

Technique: resolve the exact target-H200 backend selection from vLLM source,
then gate every attention and hybrid GDN alternative by category share,
target executability, distinctness from the automatic path, prior evidence and
availability of a same-path benchmark.

Result:

```text
full attention prefill share:         11.092%
hybrid GDN/Mamba prefill share:        9.945%
H200 auto full-attention backend:     FlashAttention-3
H200 auto GDN-prefill backend:        FlashInfer
candidates passing all gates:          0/8
```

The categories are large enough to matter, but category size is not evidence
that a backend flag is an optimization. The target resolver already selects
the specialized SM90 implementations. Forced Triton attention has negative
serving evidence, CuteDSL GDN is restricted to Blackwell, and the remaining
forced choices are no-ops or have no executable benchmark.

Carry-forward: retain automatic attention and linear-attention backend
selection. Only reopen either category for a real custom implementation that
beats the exact compiled target on trace-representative shapes.

## Part 2 R002a Hybrid Prefix-Cache Semantics

Technique: bracket fixed-size, rolling-window and append-only six-turn request
construction with token-identical cache-on/cache-off workloads.

Result:

```text
append-only cache on/off:          57.464 / 20.736 ERS points
fixed-size full cache on/off:      68.458 / 72.068 ERS points
fixed-prefix cache hits/queries:   0 / 719,796 tokens
no-prefix full failures:           0 / 330 graded
no-prefix TTFT/TPOT p95:           175.722 / 3.166ms
```

Effect: prefix caching is decisive only when a later request is an exact
continuation of a retained hybrid-state endpoint. For fixed approximately 4k
turn prompts, Mamba `align` bookkeeping adds latency without usable hits and
removing it restores the low-contention TTFT fast path.

Trade-off: hidden prompt semantics are not observable from the redacted trace.
Keep separate prefix-on and prefix-off Portal artifacts; do not claim either
policy as universally best.

Carry-forward: package no-prefix b16384 as Part 2 Attempt 03 and preserve
prefix-enabled Attempt 02 as the opposite hedge. Reject b20480 and seq10.

## Part 2 R002b No-Prefix Scalar Sweep

Technique: independently vary prefill budget, chunking, active cap and runtime
mode after removing hybrid prefix-cache overhead.

```text
fixed control mean:             71.894
b20480 delta:                   -0.273
no-chunk b32768 delta:          +0.001
seq6 two-run mean delta:        +0.107
throughput delta:               -0.458
```

Effect: none survives the repeat/full gate. Budget20480 does not raise active
concurrency, no-chunk is exactly neutral, seq6 reverses on a cold repeat and
throughput worsens TTFT.

Carry-forward: preserve Attempt 03 seq8/interactivity/b16384 with chunked
prefill. Do not create a scalar-tuning follow-up submission.

## Part 2 R003 Hybrid Full-Attention Window

Technique: inject a trace-independent `8192` per-layer causal sliding window
into the six LFM2.5 full-attention modules while retaining the ten ShortConv
layers, hybrid cache coordinator and R002k decode policy.

Evidence:

```text
append-only control mean:       56.5450
append-only candidate mean:     61.1708
full-replay delta:               +4.6258 points
TTFT p50/p95 delta:              -18.91 / -45.67ms
TPOT p95 delta:                  -0.498ms
turn-5 score delta:              +9.373 points
independent 4k delta:            -0.4123 points on 90 graded requests
```

The gain is strongest at turns 3-5 because attention work grows with retained
history. The candidate also reserves a larger KV pool, so the result combines
attention work reduction with memory/admission relief. It is not a pure FA3
kernel speed claim and local RTX 5080 timing does not establish H200 timing.

Trade-offs and status:

- Short GPQA proxy is unchanged, but it does not test long-context reasoning.
- Contexts beyond 8192 are semantically different from the original model.
- The public trace exposes approximately 4k per-turn metadata while the
  append-only closed-loop protocol is inferred; independent 4k traffic gains
  nothing.
- Keep the overlay research-only. Follow-up `R003a` must compare 4096/8192/
  16384 and include a long-context quality decision before any packaging.

## Part 2 R003a Bounded-Attention Frontier

Technique: sweep the causal attention horizon independently on the six full-
attention layers while preserving ShortConv, precision, prefix cache,
scheduler, CUDA Graph and fused greedy-head behavior.

Evidence:

```text
window       full score mean     retrieval losses vs control
unbounded        56.5450                    baseline
4096             64.5504                       2
6144             61.7945 (n=1)                 1
8192             61.1708                       0
12288            57.0213 (n=1)                 0
16384            profile gate failed            0
```

Effect: `4096` is a stable score ceiling. It lowers repeated TTFT p50/p95 by
`29.78/81.27ms`, TPOT p95 by `0.818ms`, raises available KV capacity and
produces a turn-depth score gain from `+1.180` to `+13.514`. The repeated
score range is only `0.0766` point.

Trade-off: the same hard window removes information the unbounded control can
retrieve at 5.1k and 7.1k tokens. GPQA inputs below 1k tokens cannot validate
that boundary. Window `8192` is the balanced branch with no paired losses, but
it is still not proof of semantic equivalence or rules acceptance.

Carry-forward: preserve `4096` only as a performance upper bound and `8192`
as a semantics-aware research control. Close scalar window tuning. Keep R002k
as the submission incumbent and do not package a window until driver, rules
and target-H200 qualification pass.
