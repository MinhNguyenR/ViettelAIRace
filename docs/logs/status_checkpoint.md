# Status Checkpoint

Date: 2026-07-18

Active Part 2 checkpoint: R002k Attempt 04 is graded at `60.75`, above Attempt
02's `58.17` and Attempt 01's `56.53`. It retains prefix caching, Mamba
`align`, FP8 E4M3 KV, async scheduling and b16384, and changes only to the
qualified fused-greedy image. Attempt 04 is the H200 score champion; Attempt 02
is the native-image rollback and Attempt 01 remains the zero-failure rollback.

The unchanged Attempt 04 digest repeated at ERS `58.94`, TTFT p50/p95
`55/79ms`, TBT median `4ms` and seven failures. The first run was `60.75`,
`44/80ms`, `4ms` and six failures. Exact-artifact mean is `59.845`; observed
ERS range `1.81`, TTFT-p50 range `11ms`, and failure range one. Use `60.75` as
best score but `59.845` plus the range as the research attribution anchor.

Subround 002m is complete without a promotion. Its long exact-image
microbenchmark found only `+0.011%` GPU gain from direct output ownership and
`+0.489%` at the unsafe no-clone floor; transposed BF16 and manual GEMV lost.
No live trace, image or submission was opened, and R002k remains unchanged.

Both Attempt 02 and 03 report `warmup_count=0`, nonzero failures (`8/5`) and
the same visible `config_hash` despite different cache flags. Treat primer
accounting and the hash as incomplete Portal telemetry, not proof of identical
runtime policy.

Subround 002f then measured output-only FP8 on the Portal-calibrated append-only
path. Tensor no-repair FP8 scored `58.857` versus a matched native `56.729`,
but matched only `80/96` first tokens and `0/96` full sequences. Row-wise
top-k32 improved equivalence to `92/96` first tokens and `29/96` sequences but
remained unsafe. No custom image or Compose is promoted.

Subround 002b then re-bracketed the fixed-4k no-prefix policy. The closing
control reproduced `71.951`; b20480, no-chunk/b32768, seq6 and throughput had
drift-corrected effects `-0.273/+0.001/+0.107/-0.458` points. No factor opened
a full trace or changed Attempt 03.

Subround 002c rejected built-in GPU n-gram speculative decoding. Startup was
valid, but only `67/5820` draft tokens (`1.15%`) were accepted and first-token
draft acceptance was `40/2910` (`1.37%`). The path also downgraded FlashInfer
from full-and-piecewise to piecewise CUDA graphs. k3/n1-3, the performance
screen and a Portal artifact were closed by early-stop.

Latest Deep Cycle 2 checkpoint: the cycle is complete without a promoted
primitive. DR16 parked W4A16 before GPU. DR20 independently confirmed DR03b's
`+0.7436%` ERS gain and TPOT p99 `17.556ms`, but its single pre-registered
quality repeat produced two stable candidate-specific changes against a zero-
change guard. DR03b is excluded; I006 seq16 remains H200 champion and rollback.
R024 subsequently passed the conservative I006 freeze with no new candidate.

## Completed

- Official Phase 1 statement captured in project docs.
- Public trace added: `input/trace-round1.jsonl`.
- BTC baseline compose added: `input/docker-compose-baseline.yml`.
- Phase config added: `configs/phase1_round1.yaml`.
- Submission compose updated to match BTC baseline.
- Python 3.11 benchmark tooling is installed in `.venv`.
- Local harness supports official trace body format.
- Local metrics include ERS and score assuming Accuracy Gate passes.
- Rules and anti-cheating notes documented.
- Old synthetic/generated artifacts removed from the active project.
- Separate RTX 5080 local lab added at `local_workspaces/rtx5080/`.
- RTX 5080 lab has its own venv, Docker Compose, env file, run folder and local rules notes.
- Central note added: `docs/local/rtx5080_local_lab.md`.
- RTX 5080 lab venv created at `local_workspaces/rtx5080/.venv`.

## Verified

- Trace has 120 JSONL requests.
- Trace model name is `Qwen3.5-2B`.
- Trace `max_tokens` is 200 for every request.
- Trace `temperature` is 0 and `seed` is 42.
- Baseline compose uses `vllm/vllm-openai:v0.22.1`.
- Local code can summarize the trace without printing full prompts.
- RTX 5080 lab shell scripts pass syntax checks.
- RTX 5080 lab Docker Compose renders successfully with example env.
- Local dry-run trace/report/summary flow works for a small sample.
- RTX 5080 lab venv can run trace summary and dry benchmark sample.
- Submission skeleton remains structurally valid.
- RTX 5080 local baseline completed with `Qwen/Qwen3.5-2B` on `vllm/vllm-openai:v0.22.1`.
- Baseline note added: `local_workspaces/rtx5080/docs/baseline_20260707_111358.md`.
- Experiment 001 sweet spot search completed for tested candidates.
- Current RTX 5080 local baseline: `GPU_MEMORY_UTILIZATION=0.65`, `MAX_NUM_SEQS=16`.
- Experiment 001 note added: `local_workspaces/rtx5080/docs/experiment_001_sweetspot.md`.
- Local private quality set prepared: `input/quality/private/gpqa_diamond_100.jsonl`.
- Quality set generation script added: `scripts/prepare_gpqa_diamond_subset.py`.
- Round 007b GPQA proxy completed for auto KV vs FP8 E4M3 KV.
- Round 007c full score replay completed for FP8 E4M3 KV candidate.
- Round 007d TTFT scheduler sweep completed as an early-stop subround.
- Round 008 Brain 1 shadow dataset completed.
- Round 009 Brain 1 shadow ranker completed.
- Round 009b Brain 1 hyperparameter sweep completed.
- Round 010 Brain 1 admission simulation completed.
- Round 012 warmup/JIT and full usable VRAM sweep completed.
- Round 012a high VRAM sweep completed.
- Effective technique watchlist added at `docs/techniques/effective_technique_watchlist.md`.
- Ready SGLang rounds established `hybrid_fair_lpm_res64_steps1` as the
  provisional local best at ERS `0.524547`.
- Qwen3.5-2B local artifact audited as hybrid: 18 linear-attention layers and
  6 full-attention layers.
- Ready 007 rejected HTTP-layer two-policy queueing.
- Ready 006g retested the exact BTC vLLM 0.22.1 path; best vLLM ERS was
  `0.503230`, so SGLang remains primary.
- Ready 008a pinned SGLang 0.5.14 source and accepted a read-only engine
  scheduler telemetry overlay with complete 120-request lifecycle coverage.
- Portal Attempt 01 official vLLM control was graded at `14.98000`: TTFT
  p50/p95 `737/10066ms`, TBT median `59ms`, passed SLO `84/120`, failed `0`,
  accuracy drop `0` and penalty `1`. It is the TTFT-fast/decode-saturated H200
  endpoint, not the best rollback.
- Portal Attempts 02 and 03 direct-SGLang candidates both failed without a
  score or detailed log; their `0.86/0.88` memory comparison is inconclusive.
- Exact-entrypoint SGLang compatibility image implemented and validated. It
  preserves the BTC module path, runs native SGLang, performs legal synthetic
  startup warmup and completes `120/120` local trace requests.
- Warm SGLang repeated at ERS `0.515174/0.519746` with local memory `0.82`;
  the `0.86` memory probe reached `0.519207` and `15791 MiB` peak VRAM.
- Attempt 05 official vLLM text-only/seq4 fallback completed `120/120` locally
  at ERS `0.500000` and TPOT p95 `9.576 ms`.
- Attempt 05 then completed `120/120` on Portal at score `50.13`, with TTFT
  p50/p95 `24025/25721ms`, TBT median `13ms`, accuracy drop `0`, penalty `1`
  and `4/120` passed SLO. It is the operational H200 baseline.
- Historical Tree-Gated was repackaged on the official vLLM base so proxy and
  backend shared one `3 CPU/8GB` cgroup. All three A/B candidates completed
  `120/120`, but the Tree model emitted only two arrival-contiguous priority
  classes and produced FCFS-equivalent dispatch. The image was not published.
- Attempt 08 direct official-vLLM candidate was prepared with
  `max-num-seqs=8` and `kv-cache-dtype=fp8_e4m3`. Exact-vLLM GPQA repeats
  scored `0.35/0.35`, parsed `100/100` each and changed `0/100` predictions;
  local penalty proxy remains `1.0` with a five-question margin.
- Attempt 08 exact-context startup reached Ready in `75s`, reported
  `max_model_len=262144`, stayed within `3 CPU/8GB` and logged no OOM/fallback.
- Attempt 08 was graded `45.29`: TTFT p50/p95 `17507/19627ms`, TBT median
  `21ms`, passed SLO `3/120`, accuracy drop `0`, penalty `1`, failures `0`.
  FP8 quality passes, but seq8 concurrency is rejected on H200.
- Existing same-session seq4/6/8 FP8 evidence selected seq6: local TTFT p50/p95
  `10961.745/17544.772ms`, TPOT p50/p95 `9.816/11.986ms`, success `120/120`.
- Seq6 TPOT-guard A/B/A rejected explicit batched tokens `16384`; it improved
  TPOT center but worsened TPOT p99, TTFT, throughput and VRAM.
- Attempt 09 seq6+FP8 was graded `49.98`: TTFT p50/p95
  `21496/24498ms`, TBT median `18ms`, passed SLO `3/120`, accuracy drop `2`,
  penalty `1`, failures `0`. Its local TBT projection `17.94ms` transferred,
  but it did not beat Attempt 05; seq4/6/8 concurrency-only tuning is closed.
- Official Compose comments were re-audited: `max-model-len=262144` is unmarked
  and is now treated as a baseline/rollback value rather than a hard contract.
  Context `32768` is approved for research but still requires exact-template,
  full-trace, GPQA, startup and causal A/B gates before Portal promotion.
- Attempt 04 image was published and anonymously pull-verified, but Portal
  appended `--trust-remote-code`; v1 rejected it and exited `2`. Attempt 04 and
  digest `sha256:2cd330fe...` are retired.
- v2 passthrough of the Portal flag was rejected locally after it selected a
  multimodal startup path and OOMed at memory fraction `0.86`.
- v3 source accepts the Portal flag as an optional no-op; 8/8 tests and
  `py_compile` pass. Correct v3 full trace reached `120/120`, ERS `0.511706`
  and TPOT p95 `18.201 ms`.
- Attempt 06 still failed before model load because Portal appended the next
  unpublished flag, `--disable-uvicorn-access-log`; v3 digest is retired.
- v4 keeps required contract fields strict while tolerating Portal-only long
  options. Its 11/11 tests, constrained full trace, anonymous pull and public
  future-flag dry-run pass.
- v4 public digest is `sha256:34ed1a...23295`; Attempt 07 is `996` bytes but
  failed because the Portal Pod did not become Ready before its deadline.
- Exact-context local startup reached Ready in `67s` with context `262144`,
  3 CPU and 8GB RAM. No local OOM/crash reproduced the Portal timeout.
- Final Attempt 04/05 Compose files are `996/1052` bytes, below the Portal
  maximum of `1 MB`.
- Subround 006h rejected unscaled SGLang FP8 KV: E4M3 failed repeated quality,
  while E5M2 doubled KV capacity but reduced ERS by `0.007994` and changed
  `10/100` predictions between repeats.
- Subround 006i passed Brain1 Tree-Gated local GPQA quality across sequential
  and concurrent repeated tests, but all GPQA requests received one constant
  priority; performance promotion remains blocked.
- Subround 006j rebuilt Tree-Gated from `54,729` native scheduler snapshots.
  Its actionable AUC was `0.900176`; native whole-queue top-K cost was
  `23.867us p99`. The scorer is accepted for shadow research only.
- Subround 006k compared Tree-Gated v2 with a direct four-feature heuristic
  and instrumented exact non-root SGLang cache-node cohorts. The heuristic won
  actionable offline AUC (`0.916530`), but both policies had zero would-reorder
  support across `263` live cohort observations. The active branch is rejected.
- Subround 006l reconstructed parent/ancestor cache topology fully offline.
  No structurally broader cohort existed, while exact-node no-position ranking
  produced `0` wins and `6` losses on validation-positive cohort heads. The
  static cache-cohort ranking branch is closed without another GPU run.
- Score 100 R000-R010 completed locally. I004 remains the clean Portal
  calibration core. Packaged I006 adds trace-independent pre-readiness warmup
  and reaches `120/120`, ERS `0.619885`, TTFT p50/p95 `1008/2414ms`, TPOT
  p95/p99 `17.117/19.269ms`, and `827.780` output tok/s.
- R006 resolved the hybrid cache to three Mamba groups plus one full-attention
  group. Align-mode wave-6 active guard is only `545/1424` blocks; CPU/NVMe
  offload and Mamba all-mode are rejected.
- R006b changed Mamba SSM cache FP32 -> BF16, reducing block size `1072 ->
  560`, local compute `495102 -> 456542`, and queue time `24.774%`. A/B/A won
  TTFT `99/120` and TPOT `112/120`.
- R006b local GPQA A/B stayed `34/100` in four total runs, with `0/100`
  control/candidate prediction flips and penalty proxy `1`.
- R007 native token-budget screen rejected `1024` and `8192`; fixed `4096` is
  Pareto but lost ERS `0.001911` versus 2048. The lifecycle 4096/2048
  AsyncScheduler passed startup and `120/120`, but A/B/A ERS was `0.580253`
  versus midpoint `0.582919`; it increased prompt compute and prefill time.
  R007 is closed negative and I004 keeps native scheduling/fixed 2048.
- R008 is complete local negative. Explicit GDN Triton duplicates local auto;
  requested GDN FlashInfer falls back on RTX 5080. Forced full-attention
  `TRITON_ATTN` lost paired TTFT/TPOT `120/120` and reduced ERS
  `0.577521 -> 0.526927`. Keep automatic backend resolution; H200 may resolve
  a supported GDN path differently.
- R009 moved all five observed inference-time Triton JIT compilations into a
  `1.587s` trace-independent warmup. Primary A/B/A improved ERS from midpoint
  `0.581198` to `0.617657`; paired TTFT/TPOT wins were `96/92`, and timed JIT
  count was zero. A supplemental repeat remained ERS-positive. This is I005.
- R010 loads one native vLLM backend on internal port 8001, executes the I005
  warmup, then binds a transparent public relay on port 8000. Packaged A/B/A
  improved ERS from midpoint `0.589324` to `0.619885`, all runs completed
  `120/120`, and timed JIT stayed zero. Relay-only A/B/A showed no measurable
  regression. Promote local packaged I006.
- Attempt 10 I004 completed `120/120` on Portal at score `44.09`, TTFT p50/p95
  `4522/5422ms`, TBT `21ms`, accuracy drop `0`, penalty `1`, and `22/120`
  passed SLO.
- Attempt 11 I006 completed `120/120` on Portal at score `46.03`, TTFT p50/p95
  `4306/5334ms`, TBT `20ms`, accuracy drop `2`, penalty `1`, and `20/120`
  passed SLO. `warmup_count=0` confirms the internal synthetic warmup was not
  counted as trace traffic.
- Score 100 R011a seq18/graph18 reached ERS `0.593090`, TPOT p99 `18.457ms`
  and `814.132 tok/s`; retain it only as a conditional decode guard. R011b
  seq17 added only `0.000208` ERS over its midpoint and is rejected.
- The active A-Z execution roadmap now covers R012 hotspot profiling through
  R025 H200 finalization in `docs/planning/score_100_research_roadmap.md`.
- Score 100 R012 captured canonical cold-prefill, cache-branch and pure
  batch-20 decode profiles. Cold prefill is compute-bound; the tied BF16 LM
  head is the largest isolated decode hotspot at `28.6%`.
- R013 implemented output-only FP8 for the tied embedding plus exact BF16
  top-k4 repair. CUDA Graph, real hidden-state replay, full trace and repeated
  local GPQA passed. Candidate ERS `0.532424` beat stable control mean
  `0.514707`; GPQA drop was two points with penalty proxy `1`.
- R014 native RMSNorm/FP8 fusion is rejected because compilation reported zero
  pattern replacements.
- R015a launch geometry is rejected at only `1.0015x`. R015b FP8 state and
  R015c INT8 state halve storage but are `4-6%` slower than direct BF16 and
  fail the repeated-state quality gate.
- R016a keeps FLA chunk64; chunk32 was `0.8984x` at 6,931 tokens and chunk128
  is unsupported. R016b keeps automatic backend resolution: local CuteDSL
  package integrity fails and packaged FlashInfer is SM90-only.
- R017 mixed-state integration is skipped because every quantized state layer
  on the current path is slower.
- R018 HybridRadixState is complete and rejected. Exact block16 versus native
  block560 bounds prompt-compute reduction at `5.89%`; the largest wave is
  only `180.0ms`/`4.18%` of Portal TTFT p50. Endpoint state fits but dense
  boundary state exceeds the cache pool.
- R021a packages R013 on the frozen I006 digest with one baked image and an
  off/on/off feature flag bracket. Its Dockerfile, runner profiles and
  activation/performance/resource gate analyzer pass `5/5` static tests,
  `py_compile` and `shellcheck`.
- R021a Docker A/B/A completed under one RTX 5080, 3 CPU and 8GB RAM. Packaged
  top-k4 reached ERS `0.626978` versus control midpoint `0.623107`, improved
  TTFT p50/p95 `2.65%/3.66%`, improved TPOT p95 `3.80%`, and raised throughput
  to `819.557 tok/s` with only `95MiB` peak-VRAM delta versus midpoint.
- R021a repeated local GPQA completed: controls `34/100` twice, candidates
  `35/100` twice, zero within-config prediction changes and penalty proxy `1`.
  Package and quality gates pass; promote the path into local I007.
- R021b seq15/16/17 balance completed `120/120` for every run. Seq17 gained
  only `0.000747` ERS over the same-session seq16 repeat and worsened TPOT p99
  by `0.863ms`; cross-session drift was larger. Keep seq16.
- Attempt 12 graded exact public I007 seq16 at score `51.37`, ERC `0.133333`,
  TTFT p50/p95 `4148/7130ms`, TBT `15ms`, accuracy drop `0`, penalty `1`, and
  zero failures. It is the best complete Portal panel.
- The later I006 seq16/native-head rerun scored `51.59` at Compose SHA
  `e28f0a...`, making it the target score champion by `0.22`. Its full metric
  panel is recorded separately from original seq20 Attempt 11.
- Compared with fully recorded Attempt 11 seq20, Attempt 12 improves score by
  `5.34`, TTFT p50 by `158ms`, and TBT by `5ms`, but TTFT p95 worsens by
  `1796ms`. The remaining optimization target is TTFT tail/admission.
- R021c I006 head-off seq16/17/16 completed `120/120` for every run. Seq17
  loses ERS `0.530%`, leaves fast TTFT at `78`, and raises TPOT p99 `0.873ms`.
  Reject seq17 and close concurrency-only seq18 continuation.
- R022a screened six FP8 linear-backend settings on exact I006. Auto and
  explicit CUTLASS both selected `CutlassFP8ScaledMMLinearKernel`; all
  FlashInfer/TRT/Triton alternatives failed capability. No distinct candidate
  reached full trace, TPOT was not put at risk and I006 remains unchanged.
- R022b corrected A/B/A shows state-resident seq20 ERS `0.938972` versus
  seq16 midpoint `0.865220`, with `120/120` under the TTFT ceiling and TPOT
  p99 `9.491ms`.
- R022b all120 exact prime then scores `0.957285` versus leaf20 midpoint
  `0.937695`, reaches timed prefix hit `98.61%`, TTFT p50/p95/p99
  `84.9/398.3/445.5ms`, TPOT p99 `7.926ms`, and Ready in `113s`.
- Exact120 GPQA is deterministic at control/candidate `36/34`, local drop
  `0.02`, penalty proxy `1`. xxhash is rejected at `-0.215%` ERS versus
  SHA256 midpoint. The whole branch remains local under submission hold.
- R022c same-image off/on/off reaches ERS `0.999870` versus midpoint
  `0.942855`, TTFT p50/p95/p99 `42.9/80.4/96.6ms`, TPOT p99 `9.341ms`, and
  exactly `120` timed native-render cache hits. `119/120` are under 100ms;
  max TTFT is `122.095ms`.
- R022c GPQA off/on is exactly `34/100` in all four runs with zero prediction
  flips. It is the gray-area local champion, while I006 remains the Portal
  champion and every submission artifact stays unchanged.
- R022c jitter closure rejected access-log suppression (`0.999816` versus
  deep-copy midpoint `0.999975`) and shared cached objects (no ERS gain at
  saturation plus mutation risk). The retained deep-copy path reached exact
  ERS `1.000000` three times across the two follow-up brackets.
- Latest selected deep-copy run is `120/120`, TTFT p50/p95/p99/max
  `43.5/70.3/81.3/84.0ms`, TPOT p95/p99/max `8.600/9.000/9.071ms`, and local
  score `100.0`. R022c has no remaining local performance task.
- R022d packages the selected R022c path into a self-contained linux/amd64
  image with only the public 120-request trace embedded. Runtime audit passes
  `13/13`; no GPQA data, model weights, generated outputs or answer material
  are present in the added layer.
- R022d self-contained validation used no host trace mount and reached Ready
  in `118s`, ERS `1.000000`, success `120/120`, TTFT p50/p95/p99/max
  `42.4/78.6/89.9/93.2ms` and TPOT p95/p99/max
  `8.776/9.160/9.227ms`. Two packaged GPQA repeats scored `34/100` with zero
  flips versus the selected R022c reference.
- Public image digest
  `sha256:228c4fce4bdbcf973ebed3ded1d9c3413359e05320b3d6f0c80228d0cd97c229`
  resolves anonymously for linux/amd64. Attempt 13 Compose SHA-256 is
  `d5c9395beb5713b5e38c1278f5431a8b81bd3f4431bd22e08b4b87f75de28dd4`;
  Compose parse, immutable validator and Portal-injected-flag dry-run pass.
- Attempt 13 graded successfully at score/ERS `49.45`, ERC `0.166667`,
  `20/120` passed SLO, TTFT p50/p95 `4066/5367ms`, TBT median `21ms`, zero
  failures, accuracy drop `0` and penalty `1`.
- Versus I006 seq16, R022d improves TTFT p50/p95 by `717/1433ms` and adds four
  SLO passes, but TBT regresses `16 -> 21ms` and score falls `51.59 -> 49.45`.
  The local exact-preload/token-cache fast path did not transfer materially.
- The post-Attempt13 plan is frozen at
  `docs/planning/post_attempt13_research_plan.md`. R022e capability passed,
  R022f rejected MTP-1, and R023a generic arrived-request telemetry is next.
- R022e reached Ready in `127s` from the same fixed checkpoint, resolved the
  one-layer MTP architecture, shared target embedding/lm_head and loaded no
  second full model. Local memory fraction `0.78` was required by desktop VRAM.
- R022f MTP-1 completed `120/120` with `67.643%` acceptance but lost ERS
  `15.58%`, raised TTFT p50 `96.68%`, raised TPOT p99 `15.64%`, and reduced
  throughput `8.90%` versus native midpoint. R022g/R022h are closed.
- R023a measurement-only telemetry passed all privacy/behavior gates and
  proved `2,240` cross-request duplicate tokens, `0.4906%` of local compute.
- R023b startup resolved the intended async scheduler without fallback. Its
  first probe saved zero compute; the corrected probe timed out `19/19` holds,
  increased compute `456542 -> 457662`, increased duplicate work `2240 ->
  3360`, and violated TPOT p99 at `19.910ms`. It is rejected.
- R023c proves native retention serves all `100/100` later requests with full
  shared state and `5.842GiB` worst-active headroom; no custom retention policy
  is justified.
- R023d finds `0/120` exact post-arrival body/message hits; exact renderer
  caching is closed.
- R023e deadline-aware lifecycle scheduling completed exact-image A/B/A but
  lost ERS `0.567834` versus midpoint `0.570884` and fast TTFT `54` versus
  midpoint `66`; reject.
- R023f bounded `5ms` burst admission completed exact-image A/B/A but lost ERS
  `0.567317` versus midpoint `0.568643` and throughput; reject.
- R023g one-storage FP8 tied weights save `484-485MiB` statically but miss
  runtime KV/TPOT gates; reject before quality.
- R023h closes the remaining hybrid hotspot queue. No I008-I013 derivative or
  Attempt 14 exists; I006 seq16 remains champion at `51.59`.

## Known Limitations

- Local GPU is not the BTC MiG H200 environment.
- RTX 5080 numbers are directional only and must not be recorded as official baseline.
- Accuracy Gate cannot be reproduced without the hidden GPQA Diamond test.
- BTC model hash still needs to be confirmed.
- The compatibility layer uses `unittest`; all 11 translation/contract tests
  pass. This does not replace H200 Portal validation.
- `gpu_samples.csv` for the first RTX 5080 baseline was empty in the Codex-run script, so GPU values were observed manually from `nvidia-smi`.
- `MAX_NUM_SEQS=8` was not run in Experiment 001 because Docker/GPU execution hit the Codex usage limit.
- Local GPQA Diamond 100-question set is a public proxy, not the hidden/fixed BTC Accuracy Gate set.
- Raw GPQA examples must remain under `input/quality/private/` and must not be committed or printed in logs.
- FP8 E4M3 KV passed the official Attempt 08 hidden gate with accuracy drop
  `0`; the local GPQA proxy uncertainty is superseded for this exact dtype.
- Round 007c local score proxy is only `50.0`, below the `95.0` target; candidate is not submission-ready.
- Round 007d showed raw auto-KV concurrency hurt TPOT. Attempt 08 now confirms
  the H200 limit even with FP8: seq8 TBT is `21ms`; seq10/11 are rejected and
  Attempt 09 confirms seq6 TBT `18ms` but score only `49.98`. The lower branch
  is measured and rejected as an operational winner.
- Historical Round 008 Brain1 data had no TTFT-pass positives. Subround 006j
  supersedes it for scheduler research with grouped native decision snapshots
  containing both score-positive and score-zero TTFT outcomes.
- Brain 1 dataset was rebuilt to 77 columns with Tree-Gated canonical features and balanced RAM/VRAM-only labels.
- SSD/offload was removed from the active Brain 1 dataset because the BTC environment does not guarantee a useful SSD path.
- Round 009 primary production-safe ranker is `request_id.trace_pressure.ram_vram_target` with ROC AUC `0.973690`, balanced accuracy `0.932974`, F1 `0.931900`.
- Round 009 used a pure-NumPy boosted-stump fallback because `xgboost`, `sklearn` and `pandas` are not installed locally.
- Round 009 is still shadow-only; it does not prove real KV migration or scheduler intervention improves ERS.
- Round 009b sweet spot is `n60.lr0.200.thr8.leaf24`, with request ROC AUC / balanced accuracy / F1 `0.972375 / 0.932974 / 0.931900` and run ROC AUC / balanced accuracy / F1 `0.946675 / 0.909237 / 0.900405`.
- Round 009b sweet spot bundle is `local_workspaces/rtx5080/research_rounds/warmup_rounds/round_009_brain1_xgboost_shadow/subrounds/round_009b_hyperparameter_sweep/models/sweetspot.request_id.trace_pressure.ram_vram_target.json`.
- Round 009b performance-heavy fallback is `n60.lr0.200.thr16.leaf24`.
- Round 010 best policy quality is threshold `p_vram >= 0.15`, balanced accuracy `0.976190`, VRAM F1 `0.987342`.
- Round 010 best hybrid proxy keeps ERS `0.500000` and lowers TTFT p50 from `18810.979500 ms` to `8016.931000 ms`, but TTFT remains above the official `1500 ms` ceiling.
- Round 010 is a negative/guardrail result: do not implement or submit Brain 1 admission yet.
- Legacy warmup Round 011 memory-tiering simulation is parked; this is not
  Score 100 R011 concurrency calibration.
- Round 012 showed `GPU_MEMORY_UTILIZATION=0.95` cannot start on the active RTX 5080 desktop session because app/display VRAM overhead leaves insufficient free memory.
- Round 012 near-max usable desktop VRAM is `0.83`, but it is flaky; stable high-VRAM fallback is `0.82`.
- Round 012 endpoint warmup short at `0.82` produced the current best local ERS `0.5006135105048746`, but score proxy is still only `50.06135105048746`.
- Round 012 endpoint warmup is a small positive TTFT/JIT technique, not a submission-ready solution.
- Round 012a proved higher VRAM increases KV capacity: `0.82 -> 0.90` increased KV tokens `883850 -> 1062745` (+20.24%), but did not improve ERS/latency at `MAX_NUM_SEQS=4`.
- Ready 008a fresh native control scored `0.512927`, below the historical best
  `0.524547`; small scheduler gains must use same-session A/B and repeat runs.
- Ready 008a telemetry is a measurement layer, not a submission optimization.
- Current SGLang native behavior has no mixed batches in the best config;
  global mixed chunking was already negative and must not be promoted blindly.
- Tree-Gated v2 latency excludes field extraction from live SGLang request
  objects and does not measure physical RAM/VRAM transfers or Mamba-state
  restoration. Its high AUC is ranking evidence, not causal ERS evidence.
- Ready 006k measured live feature extraction plus dual scoring at
  `333.063us p99`, above its `250us` budget. More importantly, exact-node
  ranking produced no action different from native LPM; reducing overhead
  alone cannot make this policy useful.
- Ready 006l's last-60-arrival test half contained no score-positive exact-node
  cohort, so its positive precision is temporally uninformative. The branch is
  still rejected because the first-half validation had six positives and the
  candidate lost all six while topology offered no broader action boundary.
- Historical I006 seq20 is H200-operationally validated at `46.03`; the later
  I006 seq16 rerun is the current champion at `51.59`. Portal exposes only
  median TBT, so local tail guards remain necessary for custom-kernel promotion.
- RTX 5080 is Blackwell while H200 is Hopper. Triton must compile for the
  active target, and custom CUDA must not ship only an RTX-specific binary.
- HybridRadixState is rejected for the current trace before engine integration.
  Its benefit upper bound fails, so continuation/copy and H200 gates are not
  justified. FP8/INT8 recurrent state are also measured negative on the
  current kernel path.
- The R018 Docker/GPU copy microbenchmark could not start because the
  privileged execution approval layer hit its usage limit. This is recorded
  as an unrun downstream check, not as a candidate failure.
- I007 is public at immutable digest `sha256:d6d626e8...b7611`; anonymous
  linux/amd64 manifest inspection, Compose rendering, the frozen validator and
  H200 Attempt 12 all pass. Its RTX 5080 ERS `0.626978` remains directional;
  its authoritative Portal score is `51.37`, below I006 seq16 `51.59`.
- Portal exposes only median TBT. The complete I006 seq16 panel is score/ERS
  `51.59`, ERC `0.133333`, `16/120` passed SLO, TTFT p50/p95 `4783/6800ms`,
  TBT `16ms`, accuracy drop `0`, penalty `1`, and zero failures. R021c still
  uses the stricter local TPOT p99 `<=18ms` guard because the Portal gives no
  decode-tail metric.
- Exact-trace state priming remains unresolved by the supplied literal rules.
  The one risk-accepted R022d trial was graded `49.45` without a rule error,
  but that is not explicit BTC approval. No second embedded exact-trace
  derivative is planned.
- Native MTP uses the checkpoint's existing `mtp.*` tensors, so it does not
  change the fixed model hash. Its local performance is nevertheless decisively
  negative; quality was not rerun after the performance gate failed.
- DR07 native/observer/native completed `120/120` in every cell with identical
  `456542/1962800` compute/cache totals. Observer ERS `0.619399` remained
  within `-0.381%` of control midpoint `0.6217665`, validating the attribution
  while rejecting pinning, allocator/layout and native-scheduler work.
- DR16 audited 632 checkpoint tensors and 186 eligible text-linear matrices.
  W4A16 G128 covers `99.9141%`, estimates `632.438MiB` static saving and leaves
  36 small GDN matrices BF16. It is parked before GPU because exact converter
  and image-package gates remain open; no precision candidate was promoted.
- DR20 reconstructed DR03b's historical gate: `+0.850%` ERS, TPOT p99
  `16.641ms`, `120/120`, `99/100` continuation coverage and penalty proxy `1`.
  It added exact all-miss delegation, 22 passing DR20 tests, 6 passing inherited
  primitive tests and a passed closed-list package audit. The quality analyzer
  fails closed on missing/mismatched cases, absent output hashes or stale
  telemetry. Fresh A/B/A passed at `+0.7436%` ERS, TPOT p99 `17.556ms` and
  `120/120`. Quality panel 1 passed its process-noise interpretation with zero
  stable candidate changes; the only authorized repeat failed with two such
  changes. No systematic global change was observed, but attribution was not
  clean enough to ship, so DR03b is closed for integration.
- R024 audited four exact-I006 trace controls without dropping the outlier.
  All were `120/120`, local ERS ranged `0.609243-0.621927`, startup peaked at
  `107s`, and three passed TPOT p99 `<=18ms`; the fourth was `18.511ms`.
- Four fresh I006 GPQA controls were each `100/100` at accuracy `0.35-0.39`
  and retained penalty proxy `1`. Exact 3-CPU/8GB/network-none contracts, one
  EngineCore, Compose/image identity and the Portal `51.59` panel all passed.
- The dedicated seq16 validator, Compose parser, R024 audit tests `2/2` and
  readiness wrapper tests `6/6` pass. No Attempt 14 was created.

## Next Recommended Task

Proceed only to the R025 H200 decision ladder. Keep immutable I006 seq16 as
both candidate and rollback; exclude DR03b, W4A16, I007, scheduler and memory
overlays. Spend no Portal attempt unless it answers one attributable H200
uncertainty whose outcome can change the final artifact. No DR19 campaign or
Attempt 14 is currently authorized.

## R025 Active Checkpoint - 2026-07-14

- Aggressive synthetic startup warmup is implemented for immutable I006 and
  packaged I007 without trace or quality data.
- Both local images build and pass protected-entrypoint/module audit.
- The long chat input is exactly `28,819` tokens under the fixed tokenizer and
  has `3,947` tokens of context headroom after output reservation.
- I007 seq17-20 profiles use exact graph sets; unused seq18/19 graphs are no
  longer captured by unrelated candidates.
- The score analyzer, one-run GPQA gate, shell checks and `10/10` unit tests
  pass.
- No performance run has started: unrelated redroid GPU activity kept free
  VRAM below the frozen `15000MiB` guard for ten minutes.
- Current champion and rollback remain I006 seq16 at Portal score `51.59`.
- No Attempt 14 image, Compose or Portal claim exists.

## Next Recommended Task

Temporarily free the RTX 5080 from the redroid workload, run the startup-only
probe in `round_025_score_sprint/PLAN.md`, then execute the frozen score sweep.
Run GPQA and package only if the measured winner beats the historical I006
control while completing `120/120` with acceptable TPOT tail.

## R025 Completion Checkpoint - 2026-07-14

- Startup probe passed at `107s`; full bracket completed every cell `120/120`.
- I007 seq19 won locally at ERS `0.610256`, fast TTFT `84/120`, TTFT p50/p95
  `1076/2882ms` and TPOT p95/p99 `17.165/19.721ms`.
- Its relative local gain over selected I006 control is `+0.1663%`; the gain
  is TTFT-only while TPOT component remains `0.999995`.
- Aggressive warmup is rejected on I006 and I007; retain I005 startup warmup.
- GPQA seq19 passed `100/100` completion/parse at accuracy `0.34` and penalty
  proxy `1`.
- Attempt 14 Compose is frozen at `1897` bytes and SHA-256
  `6db580df...f025`, using unchanged public I007 digest.
- Strict validator, Docker Compose parser and protected-entrypoint dry-run all
  pass. No new image push is required.
- I006 seq16 at H200 score `51.59` remains conservative rollback.
- R026 exact-source analysis proves missing batches pad to the next graph.
- The graph18 prior is only `0.196%`; R026b ranks graph11 at `2.204%`,
  graph6+11 at `3.530%` and graph6+11+13 at `4.250%` nominal decode-wall
  padding reduction.
- All three fail-closed GPU stages and five static tests are ready but unrun
  while a desktop game occupies the GPU.

## Next Recommended Task

Submit Attempt 14 for H200 attribution. After the desktop game releases the
GPU, run R026b stage 1 control/graph11. Continue to graph6+11 and
graph6+11+13 only after each prior gate passes. Use any winner only for a later
artifact and never mutate the frozen Attempt 14 file.

## Attempt 14 Portal Checkpoint - 2026-07-14

- Portal score is `46.86`, with `19/120` passed SLO, TTFT p50/p95
  `4295/6586ms`, TBT median `19ms`, zero failures, accuracy drop `2` and
  penalty `1`.
- Relative to I006 seq16, TTFT and passed-SLO improve while decode/TBT worsens.
- Seq19 is retained as an H200-validated TTFT-tail mechanism; I006 remains the
  complete score champion at `51.59`.
- R026b stage 1 control/graph11 is now the direct next experiment. Its purpose
  is to recover decode performance without giving back seq19 TTFT gains.

## Next Recommended Task

Run R026b stage 1 on an uncontended RTX 5080. Continue to graph6+11 only if
graph11 passes ERS, TPOT, reliability, startup and memory gates.

## R026 Completion Checkpoint - 2026-07-14

- Attempt 14 remains TTFT-tail/SLO positive: `19/120` passed SLO and TTFT
  `4295/6586ms`; its TBT `19ms` prevents complete promotion.
- R026b graph11 passed the literal first A/B, but reverse-order confirmation
  reduced its ERS edge to `+0.000097` and showed worse TPOT p99.
- Graph6+11 lost `0.001479` ERS with W/T/L `5/28/87`; graph13 was skipped.
- Every R026 run completed `120/120` under the exact 3 CPU/8GB contract.
- Attempt 15 adds only graph11 to immutable Attempt 14, passes strict Compose
  validation and remains an optional low-confidence H200 probe.
- I006 seq16 remains Portal champion and rollback at score `51.59`.

## Next Recommended Task

Use at most one Portal calibration slot for Attempt 15. Interpret it only as a
target-specific graph11 A/B against Attempt 14. If it does not improve TBT and
score together, close graph11 and move the next score experiment to the
measured seq17/18 active-set boundary rather than adding more graph sizes.

## R027 Completion Checkpoint - 2026-07-14

- Seq19/seq18/seq19 completed `120/120` in all cells under 3 CPU/8GB.
- Seq18 reduced TPOT p95/p99 by `0.353/0.530ms` versus seq19 midpoint.
- It lost one fast-TTFT request, increased TTFT p95 `7.71%`, and lost only
  `0.400%` local ERS; all ten frozen H200-probe gates passed.
- Attempt 16 is immutable at `1897` bytes, SHA-256 `f0e5f647...082b3`, using
  the unchanged public I007 image digest.
- Compose validator, Docker Compose parser, analyzer tests and resource checks
  pass. Attempt 16 has not been submitted.
- I006 seq16 remains H200 champion and rollback at score `51.59`.

## Next Recommended Task

Submit Attempt 16 before optional Attempt 15. Read its full Portal panel as an
interpolation test: seek TBT `<=17-18ms`, roughly `18-19` passed SLO, zero
failures, penalty `1`, and score above `51.59`. Do not change another variable
until that result is recorded.

## Attempt 16 Portal Checkpoint - 2026-07-14

- Attempt 16 graded `47.04`, ERC `0.158333`, `19/120` passed SLO and
  `120/120` successful requests.
- TTFT p50/p95 was `4407/8799ms`; TBT median was `18ms`.
- Accuracy drop `0`, penalty `1`, failed count `0`, warmup count `0`.
- Versus Attempt 14, TBT improved `1ms` and score improved `0.18`, but TTFT
  p95 worsened `2213ms`; active-set interpolation is target-negative.
- I006 seq16 remains champion at `51.59`, TBT `16ms`, TTFT `4783/6800ms`.
- Attempt 15 is parked because graph11's confirmed local delta is nearly zero
  and slightly worsens TPOT p99.

## Next Recommended Task

Do not submit Attempt 15 now. The next proposed experiment is R028: exact I006
seq16 control `2048`, candidate `--max-num-batched-tokens=4096`, then a fresh
I006 seq16 control. R007's one-pass I004 screen made 4096 Pareto on fast TTFT,
headline TTFT/TPOT and throughput, but never confirmed it by A/B/A on the H200
champion architecture. Require a genuine ERS/TTFT win while preserving local
TPOT p99 `<=18ms`; do not reopen `16384` or change active-set size.

## R028 Completion Checkpoint - 2026-07-15

- Exact public I006 seq16 completed the frozen `2048/4096/2048` bracket at
  `120/120` for every cell under 3 CPU, 8GB RAM and network-none.
- Fixed `4096` scored ERS `0.601842` versus control midpoint `0.610570`, a
  `1.43%` loss.
- Fast TTFT rose `78 -> 84` and p95 improved, but the `<=500ms` cohort fell
  `22.5 -> 18`, TTFT p50 worsened and mean TTFT score fell
  `0.221140 -> 0.203685`.
- Every one of the six waves lost request score. TPOT score remained `1.0` in
  all cells, so lower TPOT p95/p99 could not compensate.
- R030 later showed raw lookup hits fell `33.87%` only while queries fell
  `34.54%`; true request cache attribution fell `0.143%`. Prefill-phase wall
  time rose `48.14%` but is not a GPU compute timer. There were zero
  preemptions and no OOM/error/fallback.
- Four frozen promotion gates failed. Default/resolved budget `2048` remains
  selected; fixed `4096` and `16384` are closed.
- I006 seq16 remains Portal champion and rollback at score `51.59`.
- No Attempt 17 image or Compose exists. Attempt 15 remains parked.

## Next Recommended Task

Do not run another scalar budget, active-set or CUDA-graph sweep. First build a
new opportunity screen against the R028 request-level score decomposition and
open a GPU round only for a mechanism that can preserve or enlarge the
`<=500ms` TTFT cohort while keeping TPOT score `1.0`. Until such a primitive is
identified, keep exact I006 seq16 as the submission rollback and make no new
Portal artifact.

## R029 Completion Checkpoint - 2026-07-15

- R028 outcomes and R023a telemetry passed alignment, completeness and privacy
  contracts for all 120 requests.
- Midpoint oracle ERS was only `0.613358` versus control `0.610570`; robust
  lower-bound oracle ERS was `0.612715` (`+0.351%`).
- The legal-feature stump retained `+0.000301` midpoint but only `+0.000046`
  robust gain under six held-out waves.
- Depth-2 overfit from `+0.001523` in-sample to `-0.000224` midpoint and
  `-0.000764` robust held-out gain.
- All six opening gates failed. No R029b scheduler, GPU run, image, Compose or
  Attempt 17 was created.
- Fixed and dynamic `2048/4096` budget switching are closed.
- I006 seq16/default2048 remains Portal champion and rollback at `51.59`.

## Next Recommended Task

Stop policy-model work on the existing budget action. Audit exact metric
semantics before interpreting the apparent prefix-hit reduction and prefill
phase increase. Open GPU work only for a mechanism with a static or measured
ceiling comfortably above process drift and implementation overhead.

## R030 Completion Checkpoint - 2026-07-15

- Exact public I006 source confirms prefix query/hit counters are recorded
  before slot allocation and can count a waiting request repeatedly.
- Corrected prebenchmark deltas show lookup queries/hits changed
  `-34.538%/-33.870%`; lookup hit rate increased `0.800pp`.
- One-per-request local-cache attribution changed only `-2,800` tokens
  (`-0.143%`), while local compute changed `+2,800` (`+0.613%`).
- Prefill-phase wall sum changed `+48.138%`; it measures scheduled timestamp to
  first-token timestamp and is not GPU compute.
- R028 ERS, request-score shape, all-wave losses and four failed gates are
  unchanged. R029 used request-level evidence and also remains closed.
- No cache-loss repair, GPU run, image, Compose or Attempt 17 was created.
- I006 seq16/default2048 remains Portal champion and rollback at `51.59`.

## Next Recommended Task

Do not optimize the raw lookup counter. If another GPU diagnostic is opened,
measure the per-request lifecycle boundary directly under exact
`2048/4096/2048`: queue-to-first-schedule, scheduled-to-first-token, chunk
count, decode interference and actual source-attributed compute. Require that
diagnostic to expose a different executable primitive; it must not reopen the
already rejected fixed or Tree-Gated budget policies.

## R031/R031b Completion Checkpoint - 2026-07-15

- The measurement-only lifecycle observer passed seven unit/static tests,
  import-only Docker, exact-resource startup and no-mutation/privacy checks.
- Forward `2048/4096/2048` and reverse `4096/2048/4096` brackets completed all
  six cells at `120/120` under exact I006, 3 CPU, 8GB RAM and network-none.
- Across three processes per budget, ERS changed `0.608403 -> 0.597467`
  (`-1.80%`); TTFT p50/p95 changed `1088.1/3106.5 -> 1141.3/3216.1ms`.
- Fixed `4096` reduced mean queue time `138.716ms` but increased mean
  scheduled-to-first-token service `122.927ms`.
- Lifecycle steps fell `3.54 -> 2.65`, while peer-prefill tokens sharing those
  steps rose by `3,025.714` per request. Local prompt compute rose only
  `23.333` tokens per request.
- Service worsened in `6/6` waves and total score lost in `5/6`; wave 2 won
  because its queue saving exceeded the service penalty.
- The strict direction gate failed. No scheduler, image, Compose or Attempt 17
  was created. I006 seq16/default2048 remains champion at `51.59`.

## Next Recommended Task

Run an offline-only, process-drift-aware opportunity audit for final-chunk or
first-token service isolation using the existing six-run lifecycle panel.
Estimate the ERS ceiling in the score-bearing `<=1500ms` cohorts and require
positive leave-one-wave-out gain before writing scheduler code or consuming
another GPU bracket. Do not reopen Tree-Gated `2048/4096` classification.

## R032 Completion Checkpoint - 2026-07-15

- Immutable R031/R031b data produced a `3 x 3` cross-process service-repair
  opportunity panel; no new runtime or GPU work was used.
- Full recovery was positive in `9/9` pairs, but it is an optimistic ceiling.
- At `50%` net recovery, mean relative gain was `+0.775%`, worst gain
  `-0.497%`, and only `6/9` pairs won.
- Waves 1 and 5 and the `<=500ms` cohort remained negative.
- Required recovery to exceed `+0.5%` reached `66.85%` in the worst pair.
- Four of six frozen gates passed; both distribution-stability gates failed.
- No R032b, GPU run, scheduler, image, Compose or Attempt 17 was created.

## Next Recommended Task

Do not continue the `2048/4096` family. Reassess a separate primitive only
when it already has repeated positive performance and evaluate quality against
the official penalty function, while recording any stricter reproducibility
guard separately. DR03b continuation tokenization is the first such audit
candidate; an offline requalification must precede any fresh GPU run.

## R033 Completion Checkpoint - 2026-07-15

- Offline score-first requalification passed `8/8`; historical candidate GPQA
  remains within official penalty `1`.
- Fresh same-image off/on/off completed `3 x 120/120` under
  `3 CPU/8GB/network-none`.
- Candidate ERS `0.612869` lost to control midpoint `0.614450` (`-0.257%`).
- Candidate TPOT p99 `16.911ms` and cache safety passed, but W/T/L
  `32/35/53`, fast TTFT `81 vs 83/83`, and only `3/6` positive waves failed.
- Fresh quality confirmation was skipped by the frozen dependency; this is not
  recorded as an accuracy failure.
- No image, Compose, push or Attempt 17 was created. I006 seq16 score `51.59`
  remains champion and rollback.

## Next Recommended Task

Do not rerun DR03b unchanged. Open the next round only around a distinct causal
primitive that can alter the score-bearing TTFT path while preserving local
TPOT p99 `<=18ms`; require same-image A/B/A before quality or packaging.

## R034 Completion Checkpoint - 2026-07-15

- GPTQModel `7.0.0` converted `150` exact text-linear modules to W4A16/G128
  under `3 CPU/8GiB/no-swap`; quantization completed in `101.177s` without OOM.
- The `2,397,719,368`-byte tensor artifact passed structural, tokenizer,
  target, dtype, scale and real-loss validation. Exact I006 selected native
  GPTQ Marlin and passed startup/smoke.
- Full GPQA repeats scored `37/100`, `37/100` and `35/100`; all three retained
  official penalty proxy `1.0`, with no HTTP or parse failures.
- The same-image FP8/GPTQ/FP8 bracket completed `3 x 120/120` under the frozen
  seq16, context32768, FP8-KV, BF16-Mamba, `3 CPU/8GB/no-swap/network-none`
  contract.
- Candidate ERS `0.542724` lost `10.974%` to control midpoint `0.609622`.
  TTFT p50/p95 regressed to `1978/4308ms`; TPOT p95/p99 regressed to
  `22.209/23.213ms`; W/T/L was `0/37/83`; all six waves lost.
- Extra W4 capacity added `4.607%` GPU blocks but did not help this
  non-capacity-bound workload. No image, Compose, push or Attempt 17 was
  created. I006 seq16 score `51.59` remains champion and rollback.

## Next Recommended Task

Run R035 as an attribution-first log-boundary audit before any GPU work. Only
open a warmup bracket if a warning remains in the exact timed suffix.

## R035 Completion Checkpoint - 2026-07-15

- All three immutable R034 prebenchmark logs are exact byte prefixes of their
  complete logs.
- Each prefix contains the same five JIT warnings, then the packaged warmup
  report and public-ready marker; each timed suffix contains zero warnings.
- Exact-image source maps the kernels to KV block handling and hybrid
  Mamba/GDN execution paths.
- The R034 analyzer now subtracts the exact snapshot and fails closed on a
  non-prefix. Its corrected report is `5/0` prebenchmark/timed in every cell.
- R034 ERS, TTFT, TPOT, quality and rejection remain unchanged.
- Four R035 tests and twenty corrected R034 tests pass. No server, GPU, image,
  Compose or Attempt 17 was created.

## Next Recommended Task

Do not run another warmup, scalar config or closed-policy sweep. First perform
an offline R036 H200-transfer residual audit across the complete local/Portal
attempt pairs. Require it to identify a configuration-independent,
score-bearing residual with a plausible executable primitive and a ceiling
above process drift before opening another RTX 5080 bracket.

## R036 Completion Checkpoint - 2026-07-15

- Seven paired anchors, four Portal context anchors and nine local runs passed
  the frozen data contract; local request coverage is `1,080/1,080`.
- Two independent H200 comparisons are decode-dominant: seq20 to seq16 gained
  `5.56` score points and seq19 to seq18 gained `0.18`, both while TTFT p50,
  p95 and support did not improve.
- Attempts 09, 11 seq20, 14 and 16 prove TBT median `<=20ms` can coexist with
  an unsaturated mean TPOT component.
- Local total score disagreed with Portal direction for three measured
  changes; it is not a standalone H200 selector.
- I006 seq16 is bounded at `56.666667` without raising its `16/120` TTFT
  support. Score 70 needs at least `48/120`; score 100 needs all 120.
- No distinct executable primitive passed the opening gate. No GPU, image,
  Compose or Attempt 17 was created.
- I006 seq16/default2048 remains Portal champion and rollback at `51.59`.

## Next Recommended Task

Run R037 offline against exact I006 help/source only. Inventory H200-native
decode-tail backends or custom-op paths that are distinct from closed
active-set, graph, W4 and warmup work. Open no RTX GPU bracket unless one path
has a plausible target-only mechanism and benefit ceiling above measurement
drift. In parallel, keep the larger architectural requirement explicit: a
score-70 candidate must raise TTFT support from 16 to at least 48 without
losing decode distribution.

## R037 Completion Checkpoint - 2026-07-15

- Exact I006 digest and versions are frozen in a `14`-capability manifest.
- Auto-active H200 paths, already-active local paths, closed negatives,
  startup incompatibilities and dominated alternatives are explicit.
- DS convolution-state layout is a real Qwen3.5 GDN mechanism, but its
  full-removal ceiling is only `1.296%` decode and `1.196%` cold prefill.
- MRV2 cannot run with I006 Mamba `align`; no-prefix MRV2 would raise prompt
  compute to `5.299x`; real align support remains unbounded engineering.
- Analyzer counters close, evidence files resolve and `5/5` tests pass.
- Decision is `close_r037_no_executable_primitive`; no GPU, image, Compose or
  Attempt 17 was created.
- Local GPU verification is currently unavailable because `nvidia-smi` cannot
  communicate with the NVIDIA driver; this is an environment blocker, not a
  negative candidate measurement.

## Next Recommended Task

Do not reopen exact-image scalar flags. Build an offline opportunity model for
a single-model, engine-native first-token staging policy: keep the proven
seq16 decode cohort, admit a bounded number of prefill/first-token jobs, then
quantify the TPOT cost of parking them before implementing scheduler code.
Require a robust positive ERS ceiling across existing process/wave panels and
local TPOT p99 `<=18ms` before any GPU run.

## R038 Completion Checkpoint - 2026-07-15

- Six R031 process panels and all 720 timing rows passed the immutable input
  contract; no prompt, token or generated content was consumed.
- The final simulator uses one serial staging lane, updates decode occupancy,
  and charges parking plus prefill overlap to TPOT.
- Guarded `slots8/gain0.75/impact0.10/queue500ms` gains `+0.016036` mean ERS
  and `+0.013850` in the worst process.
- All `6/6` processes and `36/36` waves are positive; worst wave delta is
  `+0.002806`.
- Modeled TPOT p95/p99 maximum is `18.183/19.032ms`; mean fast-TTFT gain is
  `+13.7` requests.
- The branch is conditional: 50% queue reduction misses the drift gate and
  25% decode interference exceeds the TPOT guard.
- Four tests, Python compilation, JSON validation and ASCII scan pass.
- No engine, GPU, image, Compose, champion or Attempt 17 changed.

## Next Recommended Task

Create R039 as a static-tested vLLM scheduler prototype only. Preserve native
queue ownership, reserve decode first, stage one bounded chunk from measured
slack and park first-token-complete state in-engine. Do not launch a measured
probe until `nvidia-smi` works. Then require exact-image startup, a 20-request
mechanism gate and same-image A/B/A before quality or packaging.

## R039 Static Checkpoint - 2026-07-15

- The pure gap-aware lifecycle policy passes `9/9` behavioral tests.
- The AST boundary suite passes `2/2` tests with zero forbidden imports,
  symbols or calls.
- Python compilation and `git diff --check` pass; no vLLM override, image,
  Compose or Portal artifact was created.
- The policy rejects non-finite measurements and interference outside `[0,1]`
  and falls back when queue, interference or TPOT gates fail.
- Runtime score, latency and quality are not measured because `nvidia-smi`
  cannot communicate with the local NVIDIA driver.
- I006 seq16/default2048 remains the Portal champion and rollback.

## Next Recommended Task

Repair or restore the NVIDIA driver, then run only the R039 gates in order:
exact-image startup, 20-request state/latency probe, queue/interference
telemetry and same-image A/B/A. If any gate fails, close R039 and retain
native I006 without packaging.

## R039 Runtime Completion Checkpoint - 2026-07-15

- Host-side `nvidia-smi` and Docker are healthy; the restricted sandbox cannot
  expose the GPU device, so probes were run through the approved host boundary.
- The exact I006 image loaded the R039 shadow `AsyncScheduler` class.
- The readiness-race probe was excluded; the corrected native, shadow and
  native-repeat runs all observed the warmup marker first and completed
  `20/20` requests with no errors, OOM or preemption.
- Native/shadow/native TTFT and TPOT are effectively tied. Shadow versus the
  second native run differs by `+2.125ms` TTFT p50 and `+0.057ms` TPOT p95.
- The shadow policy produced no stage action because it is intentionally
  non-mutating and lacks proven parked hybrid-state retention.
- R039 is closed without image, Compose, Portal artifact or champion change.

## Next Recommended Task

Do not promote the shadow adapter. If continuing the score hunt, open a new
source-audited scheduler-fork round only after specifying how a request keeps
both Transformer KV and Mamba recurrent state while temporarily leaving the
native decode set. Otherwise keep I006 seq16/default2048 and pursue a distinct
measured kernel or admission primitive.

## R040 Runtime Completion Checkpoint - 2026-07-15

- Exact-image source and runtime prove tested hybrid request state survives
  leaving and later re-entering the scheduler running set.
- R040a direct hold and R040b first-token swap each completed `20/20` outputs
  with zero preemption, error, OOM or state-continuity failure.
- R040b emitted `5/2/2/2` displace/bind/swap/resume events and three safe
  rollbacks, so the mechanism itself is exercised rather than inferred.
- At matched local GPU reservation `0.72`, candidate ERS proxy `0.454380`
  loses to native midpoint `0.466172`; support remains `5/20`.
- Candidate TPOT p95/p99 is `26.491/27.351ms`, outside the frozen guard.
- No full120, quality run, image, Compose or Attempt 17 was created.

## Next Recommended Task

Open R041 as a no-mutation first-wave service-slope audit. Attribute the
roughly `200-250ms` TTFT staircase across the first 16 active requests to
prefix/suffix prefill, scheduler token allocation, sampling and first decode.
Calculate the minimum batching or kernel speedup required for grouped suffix
prefill to beat native squared TTFT score without taking TPOT above its guard.
Do not implement another scheduler until this opportunity bound passes across
multiple native processes.

## R041 Completion Checkpoint - 2026-07-15

- Seven native first-wave inputs passed validation; three include exact
  frontend/queue/service decomposition.
- The TTFT staircase is queue-dominated, with minimum request-index
  correlation `0.999850` and no increasing service slope.
- Robust score gates require `11.5%` queue-only, `17.5%` service-only, or
  `6.5%` whole-prefill reduction.
- Static FP8 capability passes, but static/fusion alone has only a `6.435%`
  hard category ceiling.
- Shape-weighted static FlashInfer MM is `8.762%` slower than CUTLASS; Torch
  row-wise MM is `3.954%` slower. All numerical checks pass.
- Static CUTLASS's standalone quant-plus-MM gain is only `1.319%` and cannot
  justify checkpoint conversion under compiled-graph uncertainty.
- Unit suites pass `4/4` for each R041a/R041b/R041c analyzer; no full120,
  image, Compose or Attempt 17 was created.
- I006 seq16/default2048 remains the Portal champion and rollback.

## Next Recommended Task

Open R042 offline only. Revisit the one positive R031 fixed-4096 cohort using
features available at request arrival or native cache lookup, not request ID,
wave number, timestamp constants or payload. Require leave-one-wave/process
generalization, robust gain above drift and a scheduler-level feasibility
proof before any GPU code. If those gates fail, keep fixed 2048 and close
conditional budget routing again.

## R042 Completion Checkpoint - 2026-07-15

- Six valid R031 processes were grouped as three observations per budget.
- Fixed 4096 loses `0.010936` mean ERS to fixed 2048.
- Wave 2 is positive in both run-order brackets, but only `6/9` process pairs
  agree and robust q25 is `-0.008325`.
- The impossible request-level oracle gains `+0.002348`; the executable
  global-wave oracle gains only `+0.001128`, with robust gain zero.
- Process noise floor is `0.004843`, more than four times the executable
  oracle gain.
- The full-data robust rule and all leave-one-wave-out folds select 2048.
- Unit tests pass `5/5`; no GPU, image, Compose or Attempt 17 was created.
- I006 seq16/default2048 remains champion and rollback.

## Next Recommended Task

Do not reopen budget routing with a larger classifier. Start R043 as an
offline hotspot-fusion opportunity audit over R012/R041 evidence: quantify
whether fusing RMSNorm, FP8 quantization, MLP activation and adjacent layout
work can exceed the `6.5%` whole-prefill gate without replacing CUTLASS. Only
write a Triton/CUDA microkernel if the measured Amdahl bound and executable
shape coverage both pass.

## R043 Completion Checkpoint - 2026-07-15

- Exact-image source confirms native fused RMSNorm/dynamic-FP8 is callable.
- The pre-registered summary/unit suite passes `5/5`.
- Seven rotated trials use preallocated outputs under 3 CPU, 8GB RAM and one
  RTX 5080 GPU with no network.
- Native fused M1680 regresses `175.056%`; weighted prefill regresses
  `156.311%` and projects to `-10.058%` whole-prefill.
- Native fused M20 is about 35% slower than the real compiled Triton reduction,
  despite beating a two-raw-op eager control.
- Numerical NRMSE is `1.622e-2`, above the frozen `1e-2` gate.
- No engine, server, full120, image, Compose or Attempt 17 was created.
- I006 seq16/default2048 remains champion and rollback.

## Next Recommended Task

Do not write a norm/activation kernel from R043. Re-audit the remaining R012
cold-prefill categories by actual compiled kernel, not category labels. The
next offline round should rank attention and hybrid GDN kernels by removable
time, shape coverage and H200 portability, then open at most one exact-kernel
microbenchmark whose standalone Amdahl ceiling exceeds `6.5%`.

## R044 Completion Checkpoint - 2026-07-15

- R012's remaining full-attention and GDN/Mamba shares are
  `11.092%/9.945%`.
- Exact target source resolves H200 SM90 to FlashAttention-3 for full attention
  and FlashInfer for Qwen GDN prefill.
- CuteDSL GDN is Blackwell-only; forced Triton attention already has negative
  same-image serving evidence.
- The pre-registered capability analyzer passes `5/5` tests and finds `0/8`
  eligible backend/custom candidates.
- No GPU run, server, full120, image, Compose or Attempt 17 was created.
- I006 seq16/default2048 remains score champion and rollback.

## Next Recommended Task

Stop scalar/backend sweeps. The remaining queue has no source-audited built-in
candidate. Reopen GPU work only after a concrete custom attention or GDN
implementation beats the exact compiled target at trace-representative shapes
and projects above the `6.5%` whole-prefill gate with TPOT protection.

## Leaderboard Reset Checkpoint - 2026-07-16

- BTC reset the leaderboard and locked the round while preparing a revised
  problem statement.
- The replacement statement and trace are not available in this workspace.
- R045 was stopped before code changes or GPU work.
- The superseded contract is frozen at
  `competition_reset/rapid_requalification_kit/contracts/phase1_legacy.json`.
- The pending contract contains `47` unresolved required fields by design.
- Contract diff reports `55` critical changes and `54` critical values now
  unknown, affecting every optimization domain.
- The technique matrix blocks all `13` legitimate techniques and marks two
  loophole/answer paths prohibited.
- Benchmark authorization is false with four explicit blockers.
- The new test suite passes `18/18`; the full project suite passes `23/23`
  before final repository verification.

## Next Recommended Task

Wait for the exact BTC statement and trace, save both unchanged, hash them and
populate a new official contract file. Run the kit until authorization passes,
then establish a repeated native baseline. Do not resume R045 or package a
submission before those gates.

## Part 2 R002k Packaging Checkpoint - 2026-07-18

- The active contract is LFM2.5 Part 2; legacy Qwen checkpoints above are
  historical only.
- Portal Attempt 02 remains the H200 champion at ERS `58.17`, TTFT p50/p95
  `58/82ms`, TBT median `4ms` and accuracy drop zero.
- R002k's non-lossy Triton BF16 greedy head improves the matched local full
  replay `56.7287 -> 58.3989` with `330/330` graded successes.
- Same-hidden-state audit matches all argmax IDs over 20 live batches; random,
  active processors, grammar, speculative decode and logprobs retain native
  fallback.
- The clean vLLM `0.22.1` image starts independently, captures PIECEWISE 16
  and FULL 8 graphs, and adds only two overlay files.
- Strict-greedy and logprob smokes both pass without failures.
- Packaged GPQA completes `100/100`, retains `21/100` accuracy and changes
  zero correctness decisions relative to the incumbent process.

## Next Recommended Task

Create an immutable Part 2 R002k submission Compose that points to a published
Docker Hub digest, preserving the exact incumbent H200 flags and using
`gpu-memory-utilization=0.95`. Submit it as a new attempt and compare H200 ERS,
TTFT, TBT, failures and accuracy drop against Attempt 02. Continue kernel work
only after that transfer result identifies whether decode gain survives SM90.

## Part 2 Attempt 04 Artifact Checkpoint - 2026-07-18

- Public image and local audited image share digest
  `sha256:10f819c0e765bb579b89caccac4de5f17984c2fd5342335130832dc7ad93f6f5`.
- `part2_attempt_04_lfm25_r002k_fused_greedy_seq8_b16384.yml` differs from
  Attempt 02 only in its image reference.
- Docker Compose parse passes; file size is `1,216` bytes and SHA-256 is
  `edeb74cfb5a5f5777e8c315e3f4a4b17294a2d96e8022b35d07ae09d51db0698`.
- No Portal upload has been performed from this workspace.

## Next Recommended Task

Upload Part 2 Attempt 04 and wait for its H200 result before changing its
runtime flags. Compare against Attempt 02 on ERS, TTFT p50/p95, TBT median,
failure count and primer accounting. If the fused image does not beat `58.17`,
retain Attempt 02 and use the target result to decide whether the next custom
kernel should target SM90 decode or whether the local gain is RTX-specific.

## Part 2 R002l Completion Checkpoint - 2026-07-18

- Source, exact-op and graph-safe phase gates all pass.
- Weighted isolated prefill gain is `21.917%`, but only `0.149ms` absolute.
- Live audit proves large prefill deferral and native decode restoration.
- Full output SHA, 20 argmax audits and native logprob fallback pass.
- Full B/A/B scores are `55.8737 / 53.5857 / 52.7145` for candidate/control/
  candidate, all with zero graded failures.
- Mean candidate delta is `+0.7084` points, but candidate repeatability and
  TPOT guards fail; the second run reaches peak active 11 and TPOT p95
  `4.568ms`.
- R002l is closed without an image, Compose or Portal attempt.
- The benchmark container is stopped; no required process remains running.

## Next Recommended Task

Keep Attempt 04 frozen and wait for its H200 result. In parallel, profile or
source-audit a larger Part 2 primitive whose projected whole-request gain is
at least `1%`; do not reopen scalar scheduler flags or another sub-millisecond
normalization kernel. A fused residual/norm-aware greedy-head design may open
only if an exact microbenchmark beats native final norm plus R002k at both
decode and prefill shapes.

## Part 2 Attempt 04 H200 Checkpoint - 2026-07-18

- Artifact identity is confirmed by Compose SHA, immutable image digest,
  Dockerfile contents and creation timeline.
- Attempt 04 is R002k fused greedy only; R002l is not included.
- Portal ERS is `60.75`, `+2.58` over Attempt 02.
- TTFT p50/p95 is `44/80ms`, versus `58/82ms` for Attempt 02.
- Median TBT remains `4ms`; failures fall from eight to six.
- Attempt 04 is the current H200 incumbent. Attempt 02 and Attempt 01 remain
  native-image and zero-failure rollbacks.
- Final quality is still pending BTC's selected-submission GPQA stage.

## Next Recommended Task

Use best `60.75` for leaderboard comparison and two-run mean `59.845` plus
range `1.81` for research attribution. Investigate the six-to-seven failed
requests and TPOT/TTFT tails before changing concurrency or precision. The
next custom candidate must retain R002k and project clearly above two Portal
points; do not package R002l alone.

## Part 2 Attempt 04 Exact Repeat - 2026-07-18

- The unchanged Attempt 04 Compose was resubmitted at 16:19.
- Compose SHA, R002k image digest and all runtime flags remain identical.
- Purpose: measure Portal process/failure variance and possibly improve the
  retained score without introducing a new factor.
- Result: ERS `58.94`, TTFT `55/79ms`, TBT `4ms`, seven failures.
- Exact two-run mean/range: `59.845/1.81` ERS. No new repository submission
  artifact was created.

## Part 2 R002n Decode Hotspot Checkpoint - 2026-07-18

- Kineto capability and clean batch `1/2/4/8` decode profiles pass.
- ShortConv BF16 projections account for `19.499%` of child-kernel time; an
  opt-in FP8 constructor overlay was implemented without changing weights on
  disk or the native fallback.
- `out_proj` and all-projection FP8 are closed. All-projection full replay is
  `58.2706` with GPQA proxy `18/100`, below the R002k anchor/quality reference.
- `in_proj` only scores `58.9590` and `59.4062` in two local full replays,
  with GPQA proxy `22/100` twice and TPOT p50/p95 around `2.41/3.76-4.01ms`.
- The candidate changes output tokens materially (`268/330` first-token
  matches against the paired control) and misses the 400ms TTFT SLO once per
  run, so it is research-only despite the positive ERS signal.
- Local `.95` startup is blocked by display VRAM; no `.95` candidate result is
  claimed. No image, Compose or Portal artifact was created.
- A reversed-order candidate repeat was invalidated by a local CUDA OOM at a
  20.8k-token shape (`512MiB` requested, `409.69MiB` free); its ERS `0.0075`
  is excluded and no follow-up control was run.

## Next Recommended Task

Keep R002k frozen. When the GPU is clean, run one more paired control/in-only
full replay at `.80` or a clean `.95` smoke followed by the same quality and
token-drift gates. Promote only if the effect survives process variance and
the hidden-quality risk is acceptable; otherwise close R002n and profile a
different model-layer family.

## Part 2 R002o Completion Checkpoint - 2026-07-18

- One-model BF16-prefill/FP8-decode ShortConv routing is implemented and
  source-audited.
- Ten of ten layers pass shared-load and post-quantization storage invariants.
- Real prefill/decode transitions and PIECEWISE-16/FULL-8 CUDA Graph capture
  pass without eager fallback.
- The all-layer candidate reduces profiled non-output-head child-kernel time by
  `6.72%`; the selective `0,1` mask saves only `0.845%` and is stopped.
- The registered local quality screen is neutral at `21/100` twice.
- Exact full replay is negative: candidate `53.2023/52.9304`, control
  `55.4200`, with candidate mean delta `-2.3537` ERS.
- R002o is complete and closed. No image, Compose or Portal artifact exists;
  R002k remains incumbent.

## Next Recommended Task

Profile the existing FP8 GEMM path by CUDA-event wall time and isolate the cost
of dynamic activation quantization. Open a new implementation only if a static,
fused or reused-scale design removes launch/quantization overhead while keeping
R002k behavior and projecting above the observed two-point Portal envelope.
Do not reopen phase-aware ShortConv routing from child-kernel sums alone.

## Part 2 R002p Completion Checkpoint - 2026-07-19

- Corrected exact-shape microbenchmarks pass and repeat at about `6.56%`
  static quant-plus-MM gain.
- Live attribution bounds dynamic activation quantization at `4.6195%` of
  clean decode child-kernel time.
- Static startup loads 44 unique target scales, preserves R002k and captures
  PIECEWISE-16/FULL-8 graphs.
- The force-static A/B/A diagnostic gains `1.2296%` child-kernel time with
  `0.4006%` control drift; graph-captured phase routing does not switch.
- Repeated quality fails for both scale maps: `20/18` and `19/18` versus
  R002k `21/100`.
- R002p is complete and closed. Full replay, image, Compose and Portal work
  were skipped; no benchmark container remains running.

## Next Recommended Task

Keep R002k frozen. Open a source-and-profile round on vLLM's graph-native
dynamic FP8 fusion controls and linear backend dispatch. Require an actual
reduction in complete decode CUDA-event time without changing activation
scales; stop before full replay unless the gain clears `1%` and repeated
quality remains at least the R002k reference.

## Part 2 R002q Completion Checkpoint - 2026-07-19

- Exact-image help/source audit confirms the nested fusion contract.
- Norm fusion startup replaces five patterns and preserves required graph
  modes; activation fusion replaces zero; attention fusion replaces zero and
  forces FULL graph capture.
- Batch-8 A/B/A complete-step profile gain is `0.5970%` with `0.1812%` drift;
  dynamic quant producers fall `44 -> 22`, but total work is mostly moved.
- R002q is complete and closed below the 1% gate. No quality, replay, image,
  Compose or Portal artifact exists; no benchmark container remains running.

## Next Recommended Task

Screen the executable FP8 linear backends (`auto`, `cutlass` and any exact
image backend that passes startup) on the same batch-8 profile. Keep the
current `cutlass` control, reject fallback/nonexistent kernels, and require a
complete-step gain above 1% before any quality run.

## Part 2 R002r Completion Checkpoint - 2026-07-19

- Exact-image registry and SM120/SM90 capability audits are complete.
- Auto and explicit CUTLASS select the same kernel; Torch selects its
  channel-wise fallback. All three pass graph/head/request startup checks.
- FlashInfer and Marlin fail model initialization for exact documented
  constraints; Triton/DeepGEMM do not match this FP8 scale layout.
- Batch-8 A/B/A measures CUTLASS mean `2.2038956ms/step`, Torch
  `2.2047690ms/step`, `0.1555%` control drift and `-0.0396%` gain.
- R002r is complete and closed. No quality, replay, image, Compose or Portal
  artifact exists; no benchmark container remains running.

## Next Recommended Task

Keep R002k frozen. Audit a block-scaled FP8 representation only if the exact
conversion/runtime path can preserve model semantics and unlock an SM90-native
kernel; otherwise move directly to a custom dynamic-per-token FP8 kernel
feasibility study. Require source-level H200 portability and a measured
complete-step ceiling before any conversion or full replay.

## Part 2 R002s Completion Checkpoint - 2026-07-19

- Online block FP8 source, shape and target-dispatch audits are complete.
- All local startup cases preserve R002k, graph capture and `8/8` requests.
- Local block CUTLASS loses `4.2785%` complete batch-8 decode time in a
  `0.1632%`-drift A/B/A bracket.
- Repeated quality is `18/100` twice versus R002k `21/100`; the strict quality
  gate fails despite remaining within the broad official ten-point band.
- Diagnostic full replay is reliable at `330/330` but negative at `49.3582`
  ERS versus R002k `58.3989`; TTFT and TPOT tails both regress.
- R002s is complete and closed. No image, Compose, Portal artifact, or
  benchmark container remains.

## Next Recommended Task

Keep R002k frozen. Move to a custom-kernel feasibility round that preserves
the current per-tensor-weight/dynamic-per-token numerical representation and
targets the measured 44 FP8 linear calls without changing scales. First prove
an executable SM90-compatible interface and a >1% complete-step ceiling;
otherwise stop at source/microbenchmark and avoid another quality-risk branch.

## Part 2 R002v Completion Checkpoint - 2026-07-19

- R002v's packaged local gain came from online FP8 ShortConv `in_proj` only;
  the allocator-qualified local bracket completed `420/420` and retained
  aggregate GPQA `21/100`, but one correctness flip remains a risk.
- H200 Attempt 05 and its byte-identical repeat scored `60.19` and `59.34`.
  Their mean/range is `59.765/0.85`; R002k's exact pair is `59.845/1.81`.
- TTFT is `50/90ms` then `61/97ms`, TBT median is `4ms` in both runs, and
  failures are `6` then `5`. The isolated precision change is neutral on
  target and is not promoted.
- `expandable_segments:True` is retained as a reliability fix, not as proof
  of a score gain. The same digest must not be submitted again.

## Next Recommended Task

Open R002w as a decode-only ShortConv epilogue/state-fusion feasibility study.
Preserve BF16 state and the native prefill path; fuse `B*x`,
`causal_conv1d_update` and `C*...` before `out_proj` to remove intermediate
traffic and launches. Require distinct-weight working-set timing, numerical
parity, FULL/PIECEWISE graph capture and complete local A/B/A before building
an image. The registered plan is at
`subround_002w_shortconv_decode_epilogue_fusion/`.

## Part 2 R002w Completion Checkpoint - 2026-07-19

- Decode-only ShortConv epilogue/state fusion is implemented and source-
  audited for the exact LFM2.5 layout.
- Runtime-layout M1/M2/M4/M8 output and state are bit-exact after 16 updates.
- M8 ten-layer epilogue timing is `0.285780 -> 0.091605ms`; repeated live
  batch-8 decode median improves `2.86%`.
- Ten layers retain FULL_AND_PIECEWISE graph capture; prefill/mixed fallback,
  startup, memory and request-completion gates pass.
- Full candidate/control/candidate ERS is
  `50.5693/50.6229/50.1385`, zero failed graded requests in every run; mean
  candidate delta is `-0.2689`.
- R002w is complete and closed. No GPQA, public image, Compose or Portal
  artifact follows; no benchmark server remains running.

## Next Recommended Task

Keep R002k frozen. Do not tune the R002w epilogue further in isolation. The
next GPU branch should first profile a larger graph boundary and open only a
fusion that removes at least one projection/staging boundary in addition to
the recurrent update. Require a repeated whole-replay gain above the local
process envelope before quality or H200 packaging.

## Part 2 R002x Completion Checkpoint - 2026-07-19

- The production graph-step profiler, multi-batch harness, scheduler telemetry
  and A/B/A comparator are complete.
- All twelve valid panels record `64/64` pure-decode steps and zero request
  failures; one `.84` startup failed before profiling and is excluded.
- Full replay records `47,657/48,263` active steps as pure decode and
  `83,109/83,610` decode rows on the eligible path.
- Scheduler-weighted GPU mean gains `3.501%`, but M4 is equal to control drift
  and M8 regresses `1.687%`; the registered gate fails.
- Direct M8 ShortConv saving is `0.039981ms/step`, `4.86x` below the eager
  projection. No larger fusion, quality run, image, Compose or Portal artifact
  is created.
- R002x is complete and closed; R002k remains the H200 incumbent and no
  benchmark container remains required.

## Next Recommended Task

Open a cached-turn prefill/first-token production attribution round. Bucket
the six public turn shapes and measured cache-hit suffixes, profile the full
prefill-to-first-token graph boundary, and identify one kernel family with a
causal TTFT contribution above control drift. Keep decode policy, R002k head,
FP8 KV and TBT near `4ms` frozen while profiling. Do not implement or package
a candidate until the attribution proves an end-to-end TTFT opportunity.

## Part 2 R002y Completion Checkpoint - 2026-07-19

- The first-token profiler, append-only panel driver, A/Kineto/B comparator and
  static tests are complete.
- All six cached turns are valid after 90 unarmed primer requests; weighted
  complete-GPU control drift is `1.418%`.
- Cache fraction reaches `83.16%`, but complete GPU time rises
  `60.801 -> 114.685ms`; attention explains `92.64%` of the growth.
- The aligned second model step costs `2.296ms` when weighted over turns.
- Local SM120 uses FlashInfer with FP8 KV; exact image source proves H200 SM90
  uses packaged FA3. Local kernel timing is not an H200 speedup claim.
- R002y is complete without an overlay, image, Compose, quality run or Portal
  artifact. R002k remains incumbent and no benchmark container is required.

## Next Recommended Task

Continue with R003a's `4096/8192/16384` window-size screen on the long-context
panel and the 100-case quality proxy. Keep the R003 overlay research-only until
the prompt-history semantics and rules interpretation are resolved. Do not
force FlashInfer or TRT-LLM on H200.

## Part 2 R003 Completion Checkpoint - 2026-07-19

- The R003 overlay injects an `8192` window only into full-attention layers
  `[2,5,8,10,12,14]`; ten ShortConv layers and R002k hooks are verified
  unchanged.
- Full append-only replay is `420/420` in four runs. Candidate mean ERS is
  `61.1708` versus control mean `56.5450`; candidate TTFT p50/p95 is
  `95.35/269.89ms` versus `114.26/315.56ms`, and TPOT p95 is `3.661ms`
  versus `4.159ms`.
- Independent 4k traffic is neutral/slightly negative, which bounds the claim
  to retained long-history semantics.
- GPQA short proxy has no candidate-specific drop; long-context quality is
  untested. Candidate has a larger KV pool at the same memory fraction.
- R003 is a local research winner, not a Portal image or Compose. R002k stays
  the submission incumbent.

## Part 2 R003a Completion Checkpoint - 2026-07-19

- Profile, calibrated retrieval and full-replay phases are complete under the
  fixed 3-core/8-GiB/RTX-5080 envelope.
- `4096` repeats at `64.5121/64.5886`, zero failures, TTFT p50/p95 mean
  `84.49/234.29ms` and TPOT p95 mean `3.342ms`.
- The repeated control mean is `56.5450`; the `4096` gain is `+8.0054`
  points and exceeds process drift.
- `4096` loses two paired retrieval cases; `6144` loses one; `8192` and
  `12288` lose none. The probe is not an absolute accuracy benchmark.
- `8192` remains the balanced research branch. `4096` is score-max only.
- No benchmark container remains required, no image/Compose is created, and
  R002k remains the Portal incumbent.

## Next Recommended Task

Do not tune the scalar window further. Audit whether the bounded-attention
change is permissible and whether the hidden driver actually sends append-only
history. If the branch remains eligible, design a semantics-preserving
alternative that reduces old-token attention work without hard truncation,
using `8192` as the balanced control and `4096` only as the measured upper
bound on attainable local score.

## Part 2 R003b Completion Checkpoint - 2026-07-19

- Source and overlay checks passed for exact selective routing of full-
  attention layers `[2,5,8,10,12,14]`; ShortConv layers remained untouched.
- Four placements (`global14`, `global2`, `global8`, `global2_14`) all passed
  the isolated performance gate and saved `32.443..37.932ms` at turn 5.
- Every retrieval process completed `14/14`, but each candidate lost the same
  two control-pass cases at actual prompt lengths `5084` and `7109` tokens.
- No full replay, image, Compose or Portal candidate was created. The global
  layer repair hypothesis is closed; R002k remains the incumbent.

## Next Recommended Task

If bounded attention remains eligible after rules/semantics review, test a
retention pattern that preserves a small, deterministic old-token/sink set in
every affected layer. Do not reopen scalar window sizes or global-layer
placement without a new quality hypothesis and the paired retrieval gate.

## Part 2 R003d Completion Checkpoint - 2026-07-20

- Tiered horizon overlays were static-tested and verified against the exact six
  full-attention layers; all ten ShortConv layers and R002k hooks remained
  unchanged.
- `safe8192` passed profile and paired retrieval gates twice. `mid6144` lost
  one control-pass retrieval case and `front3` lost two, so both were closed.
- Two safe8192 full replays averaged `61.8876` versus `56.5450` for two
  unbounded controls. TTFT p50/p95 improved `114.262/315.560 ->
  90.966/245.763ms`; TPOT p95 improved `4.159 -> 3.612ms`; all graded runs
  had zero failures.
- The short GPQA proxy returned `23/100` versus a noisy `21..22/100` control
  range. It is not a long-context semantic gate because its prompts are below
  8192 tokens.
- R003d is complete as local research only. The measured gain includes a
  larger KV pool and a semantic horizon change; no image, Compose or Portal
  artifact was created. R002k remains the submission incumbent.

## Next Recommended Task

Design a request-local prefix-plus-recent retention experiment or qualify an
H200-specific full-attention path. Preserve selected old positions in every
affected layer, repeat the 14-case long-context retrieval gate, and keep
`safe8192` as the comparison reference. Do not perform another scalar horizon
sweep or package a semantic-changing overlay without a rules decision.
