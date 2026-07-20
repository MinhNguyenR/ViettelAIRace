# Decision Log

## Decision 001 - Use Python 3.11 for Tooling

Reason:

- Stable ecosystem for benchmark tooling.
- Compatible with local scripts and Docker benchmark image.
- Serving runtime is isolated in Docker, so Python tooling does not constrain the final inference engine.

Status: accepted.

## Decision 002 - Treat BTC Baseline Compose as Initial Baseline

Reason:

- It matches the official entrypoint requirements.
- It uses `vllm/vllm-openai:v0.22.1`, the published baseline image.
- It keeps `python3 -m vllm.entrypoints.openai.api_server`, which BTC explicitly says not to replace with `vllm-server`.

Status: accepted until measured.

## Decision 003 - Keep Official Trace Unmodified

Reason:

- Rules forbid changing arrival timestamp or concurrency.
- ERS must be measured on the exact request distribution.

Status: accepted.

## Decision 004 - Do Not Start Quantization Before Baseline

Reason:

- Accuracy Gate can zero out final score if GPQA drop is too large.
- Need BF16/vLLM baseline metrics before judging the latency/accuracy trade-off.

Status: accepted.

## Decision 005 - Keep SGLang Primary, vLLM Exact-Image Fallback

Reason:

- The local Qwen3.5-2B artifact is hybrid, with 18 linear-attention and 6
  full-attention layers.
- Ready 006g verified that vLLM 0.22.1 loads the model natively, but its best
  hybrid-aware candidate reached ERS `0.503230` versus SGLang best `0.524547`.
- `--language-model-only` is useful for fallback, but vLLM concurrent partial
  prefill is unsupported in this path.

Status: accepted for current local research; H200 submission still requires
packaging validation.

## Decision 006 - Move Two-Policy Control Into the SGLang Scheduler

Reason:

- Ready 007 HTTP-layer queueing reduced the extreme TTFT tail but destroyed the
  scoring-critical `<=1500 ms` bucket and worsened TPOT.
- Ready 008a accepted a read-only engine telemetry overlay with same-session
  ERS delta `-0.001751` and complete 120-request lifecycle coverage.
- Telemetry shows high prefix reuse, no retractions, and prefill/decode waves;
  a global FCFS/LOF/mixed policy is not justified.

Decision:

- Keep one model process and one shared KV/Mamba pool.
- Keep native LPM as default.
- Test a static conditional high-prefix/deadline rescue lane with TPOT guard.
- Delay XGBoost/C++ until static policy control works and repeated telemetry
  provides a valid training dataset.

Status: accepted for Ready Round 008 Phase B.

## Decision 007 - Reject Global-Front Adaptive Rescue

Reason:

- Ready 008d safely promoted six unique adaptive candidates after native LPM.
- Fast TTFT count improved by two, but ERS fell `0.001728` versus matching
  adaptive-shadow control.
- Paired scoring attributed `-0.002143` to TTFT and `+0.000415` to TPOT:
  global-front promotion harmed more neighboring LPM requests than it helped.

Decision:

- Do not include Ready 008d active mutation in a Portal submission.
- Keep Ready 008c as diagnostic shadow only.
- Require cache-local cohort evidence before another active scheduler test.

Status: rejected for serving and submission.

## Decision 008 - First Portal Attempt Uses Official vLLM Control

Reason:

- The first H200 result must establish a valid infrastructure and runtime
  control before comparing the local SGLang candidate.
- Exact baseline fields test image pull, `/model` mount, context `262144`,
  memory `0.95`, healthcheck, endpoint and Accuracy Gate on BTC hardware.
- The contest accepts a Docker Compose `.yml`/`.yaml`, allows five attempts per
  day and reports a 600-second wait.

Decision:

- Upload `portal_submissions/attempt_01_vllm_official_control.yml` unchanged.
- Record all returned metrics in `portal_submissions/submission_plan.md`.
- Open the stable SGLang two-phase attempt only after Attempt 01 produces a
  valid result or an actionable contract failure.

Status: completed. Portal returned final score `14.98000`; detailed summary
metrics were supplied later: TTFT p50/p95 `737/10066ms`, TBT median `59ms`,
passed SLO `84/120`, accuracy drop `0`, penalty `1`. Raw logs remain absent.

## Decision 009 - Attempt 02 Uses Native SGLang Without Two-Policy Logic

Reason:

- Attempt 01 established a valid official vLLM control score of `14.98000`.
- TPOT below the official `20 ms` floor receives no additional score, so the
  strict `7 ms` TPOT-floor config wastes latency budget while TTFT remains high.
- Ready 006e achieved local ERS `0.524547`, TPOT p95 `17.036 ms` and TTFT p50
  `3896.429 ms`, outperforming the strict TPOT-floor branch by ERS.
- HTTP two-policy and active scheduler overlays were negative locally.

Decision:

- Submit the pinned SGLang Ready 006e shape as Attempt 02.
- Use one native LPM scheduler, one model and one shared KV/Mamba cache pool.
- Keep RadixAttention and per-phase CUDA graphs; use Mamba FlashInfer and BF16
  SSM, whose local 100-question quality proxy matched the control exactly.
- Preserve official context length `262144` and auto KV dtype.
- Submit the locally tested `mem-fraction-static=0.86` value unchanged.
- Keep `max-running-requests=10`; H200 `1g.18gb` provides only `1/7` of full
  H200 SMs and `1/8` of its memory bandwidth, so it must not be tuned as a full
  H200 GPU.
- Do not include proxy, Tree-Gated, custom two-policy or telemetry overlays.

Status: submitted, then failed without a Portal score or detailed log.

## Decision 010 - Isolate H200 Memory Sizing As Attempt 03 Candidate

Reason:

- Attempt 02 with `mem-fraction-static=0.86` was already submitted before the
  H200-specific `0.88` edit was proposed.
- Submission artifacts must remain immutable so each Portal score maps to one
  exact checksum.
- `0.88` is still useful as a one-variable comparison because it preserves
  about `2.2 GB` non-static headroom on the 18GB MiG frame buffer.

Decision:

- Restore Attempt 02 to its submitted checksum.
- Store the `0.88` variant as
  `portal_submissions/attempt_03_sglang_h200_mem088_comparison.yml`.
- Do not conflate its future score with Attempt 02.

Status: later submitted and failed without a Portal score or detailed log; the
memory-only comparison is invalid because both direct launchers failed.

## Decision 011 - Reject Unscaled FP8 KV For Portal

Reason:

- E4M3 failed repeated GPQA stability and worst-case accuracy-drop gates.
- E5M2 kept local penalty proxy `1.0`, but changed `10/100` predictions between
  repeats and reduced local ERS by `0.007994` versus auto KV.
- Fixed-memory KV capacity doubled, but no additional request reached the
  scoring-critical TTFT `<=1500 ms` bucket.
- Pinned SGLang `0.5.14` lacks a Qwen3.5 `load_kv_cache_scales` method and uses
  uncalibrated scale `1.0` for E4M3.

Decision:

- Do not create or submit an unscaled FP8 Attempt 04.
- Keep auto KV for Attempt 02 and the memory-only Attempt 03 comparison.
- Revisit FP8 only with a custom calibrated Qwen3.5 implementation and repeated
  GPQA plus trace A/B.

Status: rejected for current Portal queue.

## Decision 012 - Tree-Gated Proxy Passes Local GPQA Quality Only

Reason:

- Direct and Tree-Gated paths stayed in the `0.37-0.38` accuracy range across
  sequential and concurrent repeated tests.
- Worst direct-minus-Tree accuracy drop was `0.01`, and the local official-style
  penalty remained `1.0`.
- The proxy processed all `400` Tree requests without error and real concurrent
  queueing occurred.
- Brain1 assigned the same `0.879565` priority to every GPQA request, so ranked
  policy behavior was not exercised.

Decision:

- Mark the proxy/admission path locally quality-safe.
- Do not infer that every differentiated Tree-Gated cohort is quality-safe.
- Do not create a Portal candidate until a full production-trace A/B overcomes
  the observed `25-132ms` proxy queue delay and prior negative admission ERS.

Status: quality accepted; performance promotion withheld.

## Decision 013 - Native Tree-Gated V2 Is Shadow Infrastructure Only

Reason:

- The scheduler-outcome dataset removes Brain1's circular RAM/VRAM labels and
  produces differentiated scores before the TTFT deadline.
- Actionable-window ROC AUC reached `0.900176` and whole observed-queue native
  score-plus-top-K cost was only `23.867us p99`.
- Four simple features account for all model gain, so a direct heuristic may
  be equally effective and easier to reason about.
- Gentle rejuvenation was near-neutral; stronger aging variants harmed
  precision/locality. Physical KV/Mamba-state migration was not implemented.

Decision:

- Permit the native model/scorer and gentle locality policy in offline or
  engine-native shadow mode.
- Do not add an HTTP proxy, Compose change, active queue mutation or Portal
  candidate from this result.
- Before active authority, compare model versus explicit four-feature
  heuristic, add a cache-local cohort boundary, measure live feature
  extraction and prove repeated causal ERS improvement.
- Context is considered preserved only when a valid attention KV plus Mamba
  state handle survives promotion or recomputation is paid and measured.

Status: research infrastructure accepted; serving promotion withheld.

## Decision 014 - Reject Exact-Cache-Node Tree/Heuristic Scheduling

Reason:

- The direct four-feature heuristic beat Tree-Gated on actionable offline AUC
  (`0.916530` versus `0.900176`).
- The live hook safely observed `263` exact non-root cache-node cohorts without
  exposing prompt/token content or changing native LPM.
- Tree and heuristic both selected the existing native cohort head in every
  observation, so active action support was zero.
- Evaluator p99 (`333.063us`) and same-session ERS delta (`-0.006411`) failed
  the predefined performance gates.
- Live AUC was based on only two score-positive outcomes and cannot override
  the zero-action result.

Decision:

- Do not activate or submit Tree-Gated v2, its four-feature heuristic, or the
  exact-node observer.
- Preserve the artifacts as research/diagnostic infrastructure only.
- Do not spend another GPU run optimizing this observer unless held-out replay
  first shows nonzero alternative selections.
- A future learned scheduler needs new pre-decision features and a broader
  locality-safe cohort before another live shadow or active A/B.

Status: exact-cache-node active branch rejected; serving baseline unchanged.

## Decision 015 - Close Static Cache-Cohort Ranking Branch

Reason:

- Ready 006l could not construct any broader cache-local cohort from direct
  parent, fixed-hop ancestor or absolute-depth ancestry: all same-ancestor
  groups still referred to one current node/branch.
- Removing native position inside exact nodes created many alternative
  selections but lost every validation-positive native head (`0` wins,
  `6` losses).
- The best policy reduced observed TTFT score component by `0.002374` despite
  preserving matched-prefix ratio.
- This analysis required no GPU run and used no forbidden policy feature.

Decision:

- Close parent/ancestor and static no-position cache ranking for the current
  trace/telemetry contract.
- Do not create a live 006l candidate, Compose overlay or Portal attempt.
- Keep native LPM and Ready 006e unchanged.
- Reopen learned scheduling only with new temporal/causal evidence that proves
  positive capture offline. Otherwise focus on engine resource budgets.

Status: branch rejected offline; no serving or submission change.

## Decision 016 - Preserve BTC Entrypoint And Warm SGLang Before Readiness

Reason:

- Portal Attempts 02 and 03 both used a direct SGLang entrypoint and failed
  without a score or diagnostic log, while the exact vLLM baseline entrypoint
  produced a valid score.
- A compatibility image preserving
  `python3 -m vllm.entrypoints.openai.api_server` completed the full local trace
  `120/120` and exposed one native SGLang runtime process.
- Moving five synthetic warmup requests before server readiness improved ERS
  from `0.492667` cold to `0.515174` and `0.519746` on repeats, while restoring
  TPOT p95 from `25.862 ms` to `18.397/16.916 ms`.
- Increasing local memory fraction `0.82 -> 0.86` raised peak VRAM
  `14836 -> 15791 MiB` but left ERS effectively unchanged (`0.519746` versus
  `0.519207`).

Decision:

- Use the exact-entrypoint SGLang custom image as Attempt 04 primary after it
  is published publicly and pull-verified.
- Use H200 memory fraction `0.86`; do not use `0.90/0.92` in this attempt.
- Keep auto KV and the existing hybrid Mamba/attention policy; do not add FP8,
  Tree-Gated, proxy or two-policy overlays.
- Keep official vLLM with text-only/`max-num-seqs=4` as Attempt 05 fallback.
- Never submit the namespace placeholder template.

Status: superseded by Decision 017 after Portal injected an unsupported flag.

## Decision 017 - Retire Attempt 04 And No-Op Portal Trust Flag

Reason:

- Portal appended `--trust-remote-code`, which was absent from the public
  sample; the strict v1 shim rejected it and exited `2`.
- Forwarding the flag in local v2 activated SGLang's multimodal startup path
  and OOMed during image warmup at memory fraction `0.86`.
- The fixed Qwen3.5 model already loads and completes the full trace through
  SGLang's built-in path without remote model code.

Decision:

- Keep strict validation but allow `--trust-remote-code` as an optional no-op.
- Retire Attempt 04/v1 and reject v2 passthrough.
- Create Attempt 06 only after v3 full-trace and public digest verification.
- Preserve Attempt 04 file and digest unchanged as failure evidence.

Status: superseded by Decision 018 after Portal injected a second unsupported
flag into Attempt 06.

## Decision 018 - Forward-Compatible Portal Boundary For Attempt 07

Reason:

- Attempt 06 accepted `--trust-remote-code`, then Portal appended another
  unpublished vLLM flag: `--disable-uvicorn-access-log`.
- Patching one discovered flag per submission wastes the 5/day quota and does
  not establish a stable launcher contract.
- The model/runtime policy must remain fixed while Portal metadata flags are
  allowed at the external vLLM-compatible boundary.

Decision:

- Retire Attempt 06/v3 without modifying its file or image.
- Keep all required BTC value flags and `--enable-prefix-caching` mandatory,
  unique and validated.
- Accept the two observed Portal flags as no-ops and ignore future unknown
  `--long-options`; never log their values. Continue rejecting positional args.
- Publish v4 only after unit tests, constrained full trace, anonymous pull and
  public-image future-flag dry-run pass.
- Submit only immutable Attempt 07 digest
  `sha256:34ed1a98452a16ef9c4176634d55b86d7df630ce290d7512d7f7006762523295`.

Status: superseded by Decision 019 after Attempt 07 timed out waiting for Pod
readiness despite passing all pre-submission gates.

## Decision 019 - Classify Attempt 07 As Readiness Timeout, Not Policy Failure

Reason:

- Portal returned `context deadline exceeded` while waiting for the contestant
  Pod to become Ready; it did not return an argument rejection, exit code or
  container log.
- The same v4 image subsequently reached `/v1/models` locally in `67s` using
  context `262144`, 3 CPU, 8GB RAM, existing CUDA graphs and startup warmup.
- The custom layer is only `98.3KB`; almost all of the `13.20GB` local image is
  inherited from SGLang. Cold pull time is therefore a material suspect.

Decision:

- Retire Attempt 07 as a submitted artifact and preserve its exact digest.
- Do not change model length, scheduler or warmup based on this timeout alone.
- Treat cold image pull / Pod scheduling and an H200-specific engine stall as
  separate hypotheses; request Pod events/logs from BTC if available.
- The next submission must isolate one variable: either retry the identical
  digest to test infrastructure/cache variance, or use a verified smaller
  SGLang runtime image. Do not mix this diagnosis with latency tuning.

Status: diagnosis narrowed; exact Portal root cause remains unproven because
the timeout response contains no Pod events or container logs.

## Decision 020 - Hold Attempt 08 And Wait For Official vLLM Attempt 05

Reason:

- Attempts 02/03 direct SGLang failed, while Attempt 07 was the first custom
  SGLang artifact to reach engine startup and then timed out on Pod readiness.
- Attempt 05 uses BTC's official vLLM image and exact external entrypoint, so
  its result is the cleanest available test of the supported runtime contract.
- Another identical SGLang retry would consume quota before the vLLM control
  result resolves the more important branch decision.

Decision:

- Do not create or submit Attempt 08 yet.
- Treat official vLLM as the operational primary pending Attempt 05's result.
- Keep SGLang as a research branch; do not claim Portal enforces vLLM-only,
  because Attempts 04/06 executed custom compatibility code behind the fixed
  vLLM module path.
- Resume submission work only after recording Attempt 05 status and logs.

Status: superseded by Decision 021 after Attempt 05 scored successfully.

## Decision 021 - Promote Attempt 05 As Operational H200 Baseline

Reason:

- Official vLLM completed `120/120` Portal requests with no failures.
- Accuracy drop is `0` and penalty is `1`, so model/runtime quality is safe.
- TBT median is `13ms`, below the `20ms` floor, while TTFT p50/p95 are
  `24025/25721ms` and only `4/120` requests pass SLO.
- Final score `50.13` is therefore decode-safe but almost entirely missing the
  TTFT half of the score.

Decision:

- Make Attempt 05 the team's operational H200 baseline.
- Keep the official vLLM image and exact BTC entrypoint for the next submission.
- Spend TBT headroom on TTFT/prefill/admission improvements; do not optimize
  decode below `13ms` as the primary objective.
- Keep auto KV because quality passed and unscaled FP8 KV remains rejected.
- Do not resume SGLang submissions until the vLLM branch has been tuned against
  this real H200 anchor.

Status: baseline established; next work is a focused vLLM TTFT candidate.

## Decision 022 - Reject Historical Tree-Gated Model Before Portal

Reason:

- The old Ready Round 002 proxy ran outside the vLLM container and therefore
  borrowed host CPU/RAM beyond the BTC `3 CPU/8GB` contract.
- A new official-vLLM-derived image corrected this by running proxy and backend
  in one container while preserving the exact BTC module path.
- Constrained `seq8 + FP8 E4M3` A/B showed direct vLLM beating both proxy
  variants on TTFT and throughput.
- The 60-stump model emitted only two priorities: sequences `0..39` were all
  high, and `40..119` were all low. This partition was already identical to
  arrival order, so Tree dispatch was mathematically the same as FCFS.

Decision:

- Do not publish or submit the historical Tree-Gated image.
- CPU speed on H200 cannot rescue a policy with zero useful reorder actions.
- Preserve the corrected same-cgroup image and tests as a future research base.
- Prefer direct official vLLM `seq8 + FP8 E4M3` for the next Portal discussion;
  it reached local TTFT p50 `8937.661ms` and TPOT p95 `15.176ms` without proxy
  overhead, subject to the already documented FP8 quality risk.
- A new Tree must train against TTFT-SLO admission and prove nonzero reorder
  support before another submission candidate is created.

Status: historical Tree branch rejected; no Attempt 08 file or image published.

## Decision 023 - Promote Direct vLLM Seq8 + FP8 KV As Attempt 08

Reason:

- Attempt 05 proves the official vLLM image and exact BTC entrypoint deploy
  successfully, complete `120/120` requests and retain penalty `1`.
- The causal same-cgroup comparison showed direct `seq8 + FP8 E4M3` beating
  both FCFS and historical Tree proxies; Tree added no useful reorder action.
- Direct local trace completed `120/120` with TTFT p50/p95
  `8937.661/14169.899ms`, TPOT p95 `15.176ms` and `566.629 tok/s`.
- Two new exact-vLLM GPQA runs scored `0.35/0.35`, parsed `100/100` each and
  changed `0/100` predictions between repeats. Against official baseline
  `0.40`, the local delta is `0.05` and penalty proxy remains `1.0`.

Decision:

- Create Attempt 08 with the official image and only two primary changes from
  Attempt 05: `max-num-seqs 4 -> 8` and `kv-cache-dtype auto -> fp8_e4m3`.
- Do not include Tree-Gated, a proxy, a custom image or a framework switch.
- Preserve baseline context `262144`, H200 memory utilization `0.95`, exact
  entrypoint, prefix cache and chunked prefill.
- Accept one Portal trial with an explicit five-question local quality margin.
- Keep Attempt 05 auto KV as the immutable rollback.

Status: immutable Compose validated, `1056` bytes, SHA-256
`aa22a7887cb2f34b122ef91dd80ca499fac2eab5df352971f4a2f4c68a407cd3`;
Portal result pending.

## Decision 024 - Hold Attempt 09 Compose; Provisional Seq11

Reason:

- Seq10 improved TTFT on all `120/120` paired requests versus the seq8 control
  midpoint while keeping the first run entirely below TPOT `20ms`.
- Seq11 improved TTFT on all `120/120` paired requests versus the seq10
  midpoint, with TTFT p50/p95 `6822.573/10083.576ms` and maximum TPOT
  `19.772ms` in the controlled memory-`0.74` bracket.
- Seq12 crossed the TPOT floor and lost ERS; interactivity was negative;
  `xxhash` is absent from the official image.
- A clean seq11 repeat is still missing because desktop applications consumed
  enough VRAM to reject later engine startups before model load.
- Attempt 08 H200 TBT and Accuracy Gate are needed to know whether increasing
  concurrency above eight is safe on the official evaluator.

Decision:

- Keep seq11 as the provisional Attempt 09 policy.
- Do not create or submit an Attempt 09 Compose yet.
- Wait for Attempt 08 H200 metrics, then repeat seq11 and run exact-context
  startup before promotion.
- If Attempt 08 TBT is near `20ms` or penalty is below `1`, fall back to seq10,
  seq8, or auto KV according to the observed failure mode.

Status: local branch selected; promotion gates incomplete.

## Decision 025 - Reject Seq8+ On H200; Keep FP8

Reason:

- Attempt 08 completed `120/120` with accuracy drop `0` and penalty `1`, which
  officially validates FP8 E4M3 quality for this model/evaluator.
- Seq8 reduced TTFT p50/p95 to `17507/19627ms`, but TBT median increased to
  `21ms`, above the `20ms` full-score floor.
- Passed SLO fell from `4/120` to `3/120`; final score fell from `50.13` to
  `45.29` despite better TTFT medians.
- If seq8 already exceeds H200 decode headroom, local seq10/11 gains cannot be
  promoted to the same MiG without contradicting official evidence.

Decision:

- Retire Attempt 08 as a performance candidate; preserve it as FP8 quality and
  H200 concurrency evidence.
- Supersede Decision 024's seq11 branch.
- Keep FP8 E4M3 and recalibrate `max-num-seqs` below eight.
- Attempt 09 begins with seq4 FP8 to isolate KV dtype, then considers seq5-7.
- Do not create a Portal Compose until the lower-concurrency sweep completes.

Status: H200 branch reset; Attempt 05 remains best score.

## Decision 026 - Select Plain Seq6 + FP8 For Attempt 09

Reason:

- Detailed Attempt 01 metrics show the default scheduler reached TTFT p50
  `737ms` and `84/120` passed SLO, but TBT median `59ms` exceeded the ceiling
  and score collapsed to `14.98`. TBT is therefore the first guardrail.
- Attempt 05's seq4 point has TBT `13ms` and score `50.13`, while Attempt 08's
  seq8+FP8 point has TBT `21ms` and score `45.29`. The target lies below seq8.
- The existing same-session seq4/6/8 FP8 bracket gives seq6 local TTFT p50/p95
  `10961.745/17544.772ms` and TPOT p50/p95 `9.816/11.986ms`.
- One-point H200 calibration projects seq6 TBT near `17.94ms`; interpolated
  seq7 is near `19.47ms` and has insufficient uncertainty margin.
- A new A/B/A rejected `max-num-batched-tokens=16384`: versus the seq6 control
  midpoint, it improved TPOT p50/p95 `7.3%/7.0%`, but worsened TPOT p99 `5.7%`,
  TTFT p50/p95 `15.7%/8.5%`, throughput `3.2%` and VRAM `11.2%`.

Decision:

- Attempt 09 uses official vLLM, FP8 E4M3 KV and `max-num-seqs=6` without an
  explicit max-num-batched-tokens override.
- Keep the BTC context `262144`, memory `0.95`, exact entrypoint, prefix cache,
  chunked prefill, text-only model and balanced performance mode.
- Accept only if Portal reports TBT `<=20ms`, passed SLO `>=4/120`, penalty
  `1`, no failures and final score above the Attempt 05 baseline `50.13`.
- Keep Attempt 05 as rollback until H200 measures this middle point.

Status: Compose validated at `1056` bytes, SHA-256
`aa291e8597efff14bd02a3cd60a60e5c9b70d1216f6d94aa2cc32fe620a72261`;
ready for upload.

## Decision 027 - Reclassify Max Model Length As Tunable

Reason:

- The official sample marks launcher, model path/name, host and port with
  `#Don't change this to vllm-server`.
- The `--max-model-len=262144` line has no such warning, and no published rule
  listed in this repository forbids changing max model length.
- The measured public trace maximum is about `27398` prompt tokens. Including
  200 requested output tokens gives `27598`, below candidate context `32768`
  with roughly `5170` tokens of margin.
- A lower context may reduce reservation/profile/graph overhead and improve
  usable cache capacity, but the actual gain is not yet measured.

Decision:

- Reclassify `262144` from protected contract field to baseline/rollback value.
- Approve `32768` for local research and same-session A/B.
- Require exact fixed-tokenizer plus exact-chat-template scan, full trace,
  GPQA, startup and no-truncation gates before Portal promotion.
- Preserve a `262144` Compose rollback and recheck BTC rules at submission time.

Status: policy updated; `32768` experiment not yet executed or promoted.

## Decision 028 - R000 Keeps Context 32768 And Rejects 28672

Reason:

- Exact vLLM-image tokenizer/chat-template audit measured maximum prompt plus
  requested output at `27598` tokens across all 120 requests.
- Context `32768` leaves `5170` tokens of margin.
- Context `28672` also fits but leaves only `1074`, making it sensitive to any
  template/runtime discrepancy without proving a performance gain.

Decision:

- Carry `32768` into R001 against baseline `262144`.
- Do not test/promote `28672` unless future evidence shows `32768` is still a
  material bottleneck and the exact contract remains unchanged.
- Keep `262144` as Portal rollback until R001 latency/startup and quality pass.

Status: token-fit gate complete; runtime A/B pending.

## Decision 029 - Keep Context 32768 As Enabler, Not Standalone Submission

Reason:

- R001 completed a 262144 A / 32768 / 262144 B full-trace bracket.
- Context 32768 won paired TTFT `109/120` and TPOT `120/120`, but mean relative
  gains were only `0.496%` and `0.331%`.
- ERS remained `0.500000`, peak VRAM remained `14186MiB`, and startup controls
  varied by the same five seconds seen in the candidate.

Decision:

- Use 32768 as the Score 100 local baseline context for dependent rounds.
- Do not spend a Portal submission on context reduction alone.
- Keep 262144 as rollback until an integrated H200 candidate validates 32768.

Status: accepted as enabling configuration; no official artifact changed.

## Decision 030 - Keep Wave20 Only As FP8-Weight Control

Reason:

- R002 seq20 completed reliably and improved paired TTFT by `58.609%`.
- TPOT regressed on all `120/120` requests; p95/p99 reached
  `27.641/30.504ms`, and local ERS fell to `0.401866`.
- Five waves overlapped following arrivals by more than five seconds, proving
  decode/preload throughput is insufficient despite adequate KV capacity.

Decision:

- Do not integrate or submit BF16-weight wave20.
- Preserve it as the exact R003 control for FP8 weight quantization.
- Keep seq6 context32768 as integrated/decode champion until a wave20 variant
  restores TPOT below 20ms.

Status: negative boundary accepted; R003 opened.

## Decision 031 - Promote Online FP8 Wave20 Into The Research Integration

Reason:

- Online FP8 beat BF16 wave20 on TTFT and TPOT for every paired request.
- ERS increased to `0.573075`; 60 requests reached TTFT <=1.5s and 114 reached
  TPOT <=20ms.
- Waves 2-5 now drain before the next wave and later waves exceed 1000 tok/s.
- Local GPQA repeats are stable at `0.34/0.34`, retaining penalty proxy `1.0`.

Risk:

- FP8 changes 11 predictions and eight correctness outcomes versus BF16;
  quality is two points lower with only four-question no-penalty margin.
- First-wave TPOT p95 remains `21.868ms`, and first-wave backlog remains 3.5s.

Decision:

- Promote context32768 + wave20 + online FP8 linear weights + FP8 KV as I002.
- Do not call it Portal-ready until first-wave TPOT/prefix and exact startup
  packaging are improved and H200 quality is measured.
- Open R004 on graph behavior and R005 on first-wave singleflight opportunity.

Status: provisional integrated champion selected.

## Decision 032 - Keep Exact Full CUDA Graph Size 20

Reason:

- Default vLLM captured full decode graphs only through size 16 even though
  the integrated scheduler runs one 20-request wave.
- Adding size 20 preserved every default graph size, increased graph memory by
  only about `0.01GiB`, and produced a resolved FULL graph with largest size
  20.
- In a full A/B/A bracket, candidate ERS was `0.567559` versus control
  midpoint `0.562347`.
- Candidate beat the per-request control midpoint on TTFT `97/120` and TPOT
  `119/120`, with mean improvements `3.479%` and `1.449%`.
- The candidate remained reliable at `120/120`, but first-wave backlog and
  eight first-wave TPOT values above `20ms` remain.

Decision:

- Promote the explicit size-20 graph profile to I003.
- Treat this as a small causal optimization, not a complete wave20 solution.
- Do not add arbitrary graph sizes or tune graph memory further without a
  measured miss/shape need.
- Open R005 on online prefix singleflight opportunity and R006 on retained
  hybrid state.

Status: accepted into the research integration; not yet Portal-ready.

## Decision 033 - Reject Custom Prefix Hold/Release; Keep Native Cache

Reason:

- Native vLLM served `79.536%` of all prompt tokens from local cache and
  computed only `495,102/2,419,342` prompt tokens.
- R006 resolved the runtime cache/hash block to `1072` tokens. The exact
  tokenizer trie at this granularity also requires `495,102` compute tokens,
  so native vLLM already reaches the executable lower bound.
- A runtime-hash scheduler held 19 wave-1 followers until the leader reached
  the shared prefix and released them after about `204ms`.
- The corrected AsyncScheduler implementation produced exactly the same local
  compute and cache-hit token counts as native.
- It lost TTFT on `81/120` requests, TPOT on `68/120`, reduced throughput
  `0.47%`, and reduced ERS `0.002802` versus the nearest control.
- The earlier synchronous implementation was a confounded failure because a
  custom class inheriting base Scheduler bypassed vLLM's default async class.

Decision:

- Keep native vLLM prefix caching in I003.
- Reject server-side follower hold/release and do not package the overlay.
- Preserve the code and event log as evidence for future scheduler work.
- Treat the earlier 16-token trie only as a diagnostic. Runtime block hashes,
  Mamba alignment and cache semantics are authoritative.
- Move to R006 and measure attention plus Mamba state residency directly.

Status: R005 closed negative; integrated champion unchanged.

## Decision 034 - Keep Align Residency And Promote BF16 Mamba SSM Cache

Reason:

- Align mode's conservative wave-6 active footprint is only `545/1424` cache
  blocks (`38.27%`); no memory tiering is needed.
- BF16 temporal SSM state reduces the hybrid block from `1072` to `560` tokens
  and increases logical blocks from `1424` to `2714` at nearly equal cache
  tensor bytes.
- A/B/A ERS improved from control midpoint `0.565262` to `0.586594`.
- Candidate won TTFT `99/120` and TPOT `112/120`; measured prompt compute fell
  `7.788%`, queue time `24.774%`, and decode time `8.315%`.
- The local 100-question GPQA proxy was exactly unchanged at `34/100` in two
  repeats, with `0/100` control/candidate prediction flips.

Decision:

- Keep `mamba_cache_mode=align`; reject CPU/NVMe offload and all-mode.
- Promote `--mamba-ssm-cache-dtype=bfloat16` into I004.
- Preserve FP32 SSM as rollback for any hidden-quality regression.
- Require another quality gate before combining BF16 state with any additional
  lossy quantization.
- Continue to R007 from I004.

Status: R006/R006b complete; I004 is the local integrated champion.

## Decision 035 - Prepare I004 As Attempt 10 H200 Calibration

Reason:

- I004 is the current local integrated champion at ERS `0.586594`, success
  `120/120`, TTFT p50 `1289ms`, and TPOT p95 `18.597ms`.
- Online FP8 weights and BF16 Mamba SSM state materially change decode/cache
  behavior, so Attempt 08's seq8-only TBT result cannot reject this seq20
  bundle without measurement.
- Exact context `32768` passed token-bound, full-trace, startup and quality
  gates with `5170` tokens of public-trace margin.
- R007 dual-budget, R008 backend overrides and R009 warmup are not measured and
  must not contaminate the calibration.

Decision:

- Create immutable candidate `attempt_10_vllm_i004_fp8_hybrid.yml`.
- Pin the exact official vLLM image manifest and protected BTC entrypoint.
- Use H200 memory `0.95`; keep backend selection native `auto`.
- Do not mark the attempt submitted until the user confirms Portal upload.
- Preserve Attempt 09 as the context/concurrency rollback.

Status: Compose prepared; Portal result pending.

## Decision 036 - Reject R007 Lifecycle Dual Token Budget

Reason:

- Fixed prefill budget 4096 improved headline TTFT/TPOT and throughput but
  reduced ERS by `0.001911` versus fixed 2048.
- The follow-up custom AsyncScheduler used 4096 only in prefill-only mode and
  2048 during mixed prefill/decode. Runtime startup passed in `102s`, and 20
  events proved the expected mode transitions across all five timed waves.
- Full A/B/A candidate ERS was `0.580253` versus control midpoint `0.582919`.
- Candidate lost paired TTFT `103/120` and TPOT `81/120`; local prompt compute
  rose `0.98%`, prefill-time sum `8.20%`, and peak VRAM about `316MiB`.
- Reliability was clean (`120/120`, zero preemptions), so this is a causal
  performance rejection rather than an implementation or resource failure.

Decision:

- Keep I004's native AsyncScheduler and fixed resolved budget 2048.
- Do not package the R007 overlay into Attempt 10 or the integrated candidate.
- Preserve source, tests, transition logs and paired report as negative
  evidence. Reopen only if a future scheduler can prove reduced prompt compute
  before a full latency run.
- Continue to R008 from I004, not from the rejected R007 candidate.

Status: R007 complete negative; I004 remains champion.

## Decision 037 - Keep Automatic Hybrid Backends After R008

Reason:

- On RTX 5080, GDN `auto` and forced `triton` both resolve to Triton/FLA.
- Forced GDN `flashinfer` reaches Ready but explicitly falls back to
  Triton/FLA, so it cannot provide local performance evidence for H200.
- The only real local change, forcing six full-attention layers from
  FlashInfer to `TRITON_ATTN`, reduced ERS `0.577521 -> 0.526927`.
- It lost all `120/120` paired TTFT and TPOT requests, raised TPOT p95 to
  `23.707ms`, and added timed JIT for `kernel_unified_attention`.
- The final control startup failed because desktop processes reduced free
  VRAM, not because of candidate behavior. The complete paired A/B screen is
  still decisive because every request regressed by a large margin.

Decision:

- Keep `attention-backend=auto` and `gdn-prefill-backend=auto` in I004.
- Do not force `TRITON_ATTN` and do not combine it with R009 warmup.
- Let vLLM select a supported Hopper GDN path on BTC H200; a Blackwell fallback
  is not evidence to force either backend in the Portal artifact.
- Preserve startup/resolver logs and paired screen; no quality run is needed
  for the rejected performance candidate.

Status: R008 complete local negative; I004 remains champion and R009 is next.

## Decision 038 - Promote Synthetic Shape Warmup As Local I005

Reason:

- A trace-independent `1.587s` endpoint warmup moved all five observed Triton
  JIT compilations before timed traffic; new timed JIT count was zero.
- Primary A/B/A ERS was `0.617657` versus control midpoint `0.581198` (`+6.27%`).
- Candidate won paired TTFT `96/120` and TPOT `92/120`; TTFT p50/p95 reached
  `1020/2454ms` and TPOT p95/p99 `17.364/19.688ms`.
- The warmup's `3220` prompt tokens were all local compute. After subtraction,
  trace compute/cache-hit totals exactly match controls, so no trace prefix or
  output was precomputed.
- A supplemental recovered repeat remained ERS-positive and again produced
  zero timed JIT, although its control B had a time gap after tool interruption.

Decision:

- Promote I004 plus synthetic shape warmup as local research champion I005.
- Keep Attempt 10 immutable and do not call I005 Portal-ready yet.
- R010 must gate service readiness until warmup completes, preserve the exact
  vLLM-compatible contract, and stay well below the 600-second startup limit.
- Do not add trace prompts, common trace prefixes, answers or request-specific
  metadata to the warmup.
- Quality rerun is not required for this runtime-only change; packaging and
  full-trace repeats remain mandatory after integration.

Status: R009 completed positive; R010 packaging/integration opened.

## Decision 039 - Promote Readiness-Gated Package As Local I006

Reason:

- R010 preserves the protected `python3 -m
  vllm.entrypoints.openai.api_server` path while loading one backend/model on
  internal port 8001 and exposing port 8000 only after warmup.
- Packaged warmup A/B/A completed `120/120` for all runs and raised ERS from
  control midpoint `0.589324` to `0.619885` (`+5.19%`).
- Candidate TTFT p50/p95 reached `1008/2414ms`, TPOT p95/p99
  `17.117/19.269ms`, and `93/120` requests reached TTFT <=1.5s.
- All five observed JIT warnings occurred before public readiness; timed
  traffic added zero warnings.
- A separate no-warmup direct/relay/direct bracket showed no measurable relay
  regression: relay ERS `0.594723` versus midpoint `0.592591`.
- The image tests passed `6/6`; Compose validation, public anonymous manifest
  inspection and the 600-second startup margin passed.

Risk:

- RTX 5080 warmup gains may not transfer to Hopper kernels or the Portal
  benchmark lifecycle.
- The custom image adds a supervisor and TCP relay even though the inference
  engine remains native vLLM.
- The hidden H200 run is still required to validate image pull, model mount,
  warmup readiness and score.

Decision:

- Promote I005 plus readiness gating as local packaged champion I006.
- Freeze public digest
  `sha256:7495ee2e07097e0bf3eaca762f3172a00131c35d2f3c93d51fc55dee20fe1c78`.
- Prepare immutable `attempt_11_vllm_i006_readiness_warmup.yml`; do not mark it
  submitted until the user confirms upload.
- Keep Attempt 10/I004 as the clean no-warmup H200 calibration and rollback.
- Do not use trace-derived warmup, a second model, request-aware relay logic or
  manual KV/Mamba-state transfer.

Status: R010 complete locally; I006 packaged; H200 Portal calibration pending.

## Decision 040 - Close Seq4/6/8 After Attempt 09

Reason:

- Attempt 09 seq6+FP8 completed `120/120`, TBT median `18ms`, penalty `1` and
  zero failures, so the runtime and decode prediction were valid.
- Its H200 TBT almost exactly matched the local projection `17.94ms`.
- TTFT remained `21496/24498ms` p50/p95, only `3/120` passed SLO, and final
  score `49.98` stayed below Attempt 05's `50.13`.
- Seq8 had lower TTFT but TBT `21ms` and score `45.29`; seq4 had safer TBT
  `13ms` but TTFT `24025ms`. Concurrency alone cannot cross the current score
  plateau.
- Accuracy drop displayed `2`, but penalty remained `1`; quality did not cause
  the score loss.
- Applying the seq6 TBT factor naively to I004 local median gives about
  `21.3ms`, so Attempt 10 has explicit TBT risk. It remains useful because its
  online FP8 weight path can scale differently from Attempt 09's BF16 weights.

Decision:

- Reject Attempt 09 as the operational winner and do not resubmit it.
- Close the KV-only seq4/6/8 concurrency branch.
- At this decision point, select clean I004/Attempt 10 because it changes
  decode throughput and hybrid cache granularity, not only admission. Treat it
  as architecture calibration, not a guaranteed score improvement.
- At this decision point, gate I006/Attempt 11 on Attempt 10 proving the shared
  core starts, completes `120/120`, and retains penalty `1` without
  catastrophic TBT/OOM.

Historical status at decision time: Attempt 09 recorded; Attempt 10 was next.
Superseded by Decision 041 after Attempts 10/11 were graded.

## Decision 041 - Keep Attempt 05 As Portal Champion After Attempts 10/11

Reason:

- Attempt 10 I004 completed `120/120`, penalty `1`, score `44.09`, TTFT
  `4522/5422ms` and TBT `21ms`.
- Attempt 11 I006 completed `120/120`, penalty `1`, score `46.03`, TTFT
  `4306/5334ms` and TBT `20ms`.
- The readiness warmup therefore transferred positively and remained legal:
  Portal reported `warmup_count=0`.
- Neither architecture beat Attempt 05 score `50.13`; Attempt 10/11 exchange a
  large TTFT gain for too much per-token pressure.

Decision:

- Keep Attempt 05 as the operational Portal champion.
- Keep I006 as the validated custom-package research base, not the final
  submission winner.
- Require a measured decode-kernel/state gain before spending more H200
  attempts on high-admission descendants.

Status: Attempts 10/11 measured; H200 packaging question closed positive and
performance breakthrough remains open.

## Decision 042 - Close R011 Concurrency-Only Descent

Reason:

- Seq18/graph18 improved local ERS by `0.006906` over its seq20 midpoint and
  won paired TPOT `118/120`, but missed the calibrated TPOT p50 target by
  `0.205ms` and reduced throughput to `814.132 tok/s`.
- Seq17 improved ERS only `0.000208` over its seq18 midpoint, split paired
  TTFT `60/60`, and still missed the target.
- Seq17 throughput `807.932 tok/s` leaves insufficient margin above the
  `800 tok/s` wave-drain guard to justify seq16.

Decision:

- Reject seq17 and do not continue to seq16.
- Keep seq18/graph18 as a conditional decode-guard fallback only.
- Start R012 hybrid hotspot profiling; do not use concurrency reduction as the
  main breakthrough mechanism.

Status: R011 complete; R012 next.

## Decision 043 - Adopt R012-R025 Final Research Roadmap

Reason:

- Portal evidence shows TTFT and TPOT must be improved together rather than
  moved along the existing concurrency frontier.
- The model has 18 recurrent linear-attention layers and six full-attention
  layers; generic Transformer-only KV or attention tuning is incomplete.
- Existing negative rounds rule out blind scheduler, backend and queue
  overlays without a new state representation or measured hotspot.

Decision:

- Execute profiling before custom kernels.
- Prioritize fused residual/RMSNorm/FP8, Mamba/GDN decode and long-prefill
  kernels, then mixed FP8 recurrent state.
- Build HybridRadixState before reopening singleflight or two-policy
  scheduling, so those policies gain actionable hybrid-state information.
- Integrate only A/B/A winners, perform final ablations, freeze a performance
  artifact plus rollback, and let H200 results determine the final roadmap.

Status: complete active roadmap is
`docs/planning/score_100_research_roadmap.md`; final distilled roadmap remains
deferred until R025.

## Decision 044 - Retain R013 FP8 Tied Head As Provisional Kernel Candidate

Reason:

- R012 measured the tied BF16 vocabulary projection at `28.6%` of pure decode.
- Output-only FP8 leaves embedding lookup unchanged and has a native fallback.
- Exact BF16 top-k4 repair restored all measured hidden-state top-1 results.
- Full trace improved ERS `0.514707 -> 0.532424` versus stable control mean and
  reduced TPOT p95 `24.214 -> 23.158ms`.
- Repeated GPQA changed accuracy `35/100 -> 33/100`; penalty proxy remains 1.

Risk:

- The extra FP8 copy costs `485MiB` and `6.63%` local KV capacity.
- Candidate evidence is not yet a packaged A/B/A or H200 result.

Decision:

- Keep top-k4, not raw FP8, as the R013 candidate.
- Do not replace I006 or publish a Portal artifact yet.
- Require package A/B/A, resource repeat and feature-flag rollback before
  integration promotion.

Status: R013 complete; provisional kernel candidate retained.

## Decision 045 - Reject R014-R017 GDN/Norm/State Changes

Reason:

- R014 norm/quant fusion replaced zero graph patterns.
- R015 launch geometry improved only `1.0015x` at batch 20.
- FP8 and INT8 recurrent state halve storage but are about `4-6%` slower than
  direct BF16 and fail strict repeated-state quality.
- R016 chunk32 is slower, chunk128 unsupported, and local alternate backends
  either fail package/runtime support or do not transfer to H200.
- Partial R017 state quantization cannot produce speedup when every quantized
  layer is slower on the measured kernel.

Decision:

- Keep native BF16 recurrent state and GDN decode geometry.
- Keep FLA chunk64 locally and `gdn-prefill-backend=auto` for target
  portability.
- Skip R017 integration and preserve all negative harnesses for regression.
- Add none of these flags or kernels to a submission image.

Status: R014-R017 closed; no integrated champion change.

## Decision 046 - Open R018 HybridRadixState Before Scheduler Work

Reason:

- R012 measured `96.8%` cache hit yet `219.5` residual tokens recomputed per
  cache-heavy branch due hybrid alignment.
- R005 proved HTTP singleflight changes no local-compute tokens without a new
  state-sharing primitive.
- R020 two-policy scheduling remains non-actionable until exact reusable
  hybrid-state depth and ownership exist.

Decision:

- Open R018 as a standalone snapshot correctness and copy-cost study.
- Snapshot contracts must include full-attention KV, Mamba/GDN state,
  position, dtype, scale and ownership metadata.
- Do not patch the engine until 128/256/512 spacing proves saved recompute
  exceeds GPU copy and VRAM cost with continuation equivalence.

Status: R018 next; R019/R020 dependency-blocked.

## Decision 047 - Reject R018 And Skip R019/R020

Reason:

- The exact model/cache contract requires `561152` raw BF16 state bytes per
  linear-attention layer and `10100736` bytes across all 18 layers.
- Ideal block16 can reduce measured prompt compute only `456542 -> 429646`,
  or `5.89%`.
- Linear R012 calibration bounds the largest affected wave at `180.0ms`, only
  `4.18%` of Attempt 11 TTFT p50, below the `10%` gate.
- Endpoint-only snapshots fit, but all-boundary snapshots consume `108.91%`
  of the measured pool at block560 and `2911.53%` at block16.
- GPU copy timing was unavailable, but it is downstream of the already-failed
  benefit gate and cannot increase the compute ceiling.

Decision:

- Reject HybridRadixState continuation and allocator overlays for this trace.
- Do not reopen R005-style singleflight as R019.
- Skip R020 two-policy scheduling because no new causal state feature exists.
- Preserve native BF16 align-mode endpoints and move to R021a packaged R013
  qualification.

Status: R018 complete negative; R019/R020 skipped; R021a next.

## Decision 048 - Promote Packaged Top-K4 As I007 And Keep Seq16

Reason:

- R021a used the exact same packaged image for feature off/on/off and all
  three runs completed `120/120` under the RTX 5080 3-CPU/8GB contract.
- Top-k4 ERS `0.626978` beat control midpoint `0.623107`; TTFT p50/p95 and
  TPOT p95 improved `2.65%`, `3.66%` and `3.80%`, respectively.
- Candidate throughput rose `813.584 -> 819.557 tok/s`; peak VRAM increased
  only `95MiB` versus the control midpoint.
- Repeated GPQA was deterministic at candidate `35/100` versus control
  `34/100`, with penalty proxy `1` throughout.
- R021b seq17 beat its same-session seq16 repeat by only `0.000747` ERS while
  worsening TPOT p99 by `0.863ms`; cross-session drift was over eight times
  larger than the ERS edge.
- The user reported an H200 score increase from roughly `46` to `51` after
  moving the I006-family active set to seq16. Metrics were pending at this
  decision point and were later completed in Decision 054.

Decision:

- Promote I006 plus output-only E4M3 tied head and exact BF16 top-k4 repair as
  local integrated champion I007.
- Keep `max-num-seqs=16`; reject seq15 and keep seq17 on watchlist only.
- Preserve feature-off rollback in the same image and retain immutable I006 as
  the H200-validated architecture rollback.
- Do not claim an official I007 score until its public digest and H200 result
  exist. Record the full metric panel for the user-reported score-51 run before
  attributing its gain.

Status: R021a/R021b complete; local I007 selected and publicly frozen as
Attempt 12 at digest `sha256:d6d626e8...b7611`; H200 gate remains.

## Decision 049 - Promote Attempt 12 I007 And Reopen One TTFT Margin Probe

Reason:

- Attempt 12 graded the exact frozen I007 seq16 artifact at score `51.37`,
  TBT `15ms`, hidden accuracy drop `0`, penalty `1` and zero failures.
- Versus fully recorded Attempt 11 seq20, score improves `5.34`, TTFT p50
  improves `158ms`, and TBT improves `5ms`.
- TTFT p95 worsens `5334 -> 7130ms` and only `16/120` pass SLO. Decode and
  numerical quality are no longer the primary residual; TTFT tail is.
- R021b seq17 added three fast-TTFT requests and improved TTFT p95 locally,
  but its `0.000747` ERS edge was below session drift and cost `0.863ms` TPOT
  p99. A proper bracket is required before target-hardware use.

Decision:

- Promote I007 seq16 as the immutable local and H200 Portal champion.
- Preserve top-k4, all precision settings, warmup, context and native hybrid
  backends; do not stack another lossy or kernel change.
- Open only R021c seq16/17/16 as a same-image admission bracket.
- Require `120/120`, TPOT p99 below `18.5ms`, more than `0.2%` ERS gain above
  the control midpoint, at least three extra requests under `1.5s`, improved
  TTFT p95, no fallback and throughput above `800 tok/s`.
- If any gate fails, close seq17 and keep Attempt 12 unchanged.

Status: superseded by Decisions 050 and 054; Attempt 12 remains a quality-safe
watchlist, while I006 seq16 is now the best complete metric panel.

## Decision 050 - Restore I006 Seq16 As H200 Score Champion

Reason:

- The later I006 seq16 rerun scored exactly `51.59`, above Attempt 12 I007 by
  `0.22` points (`0.43%`).
- Its current Compose SHA is `e28f0a...`; the immutable original Attempt 11
  seq20 artifact remains separately archived at `01fe58...` and score `46.03`.
- I006 seq16 and I007 seq16 differ only by image layer plus the two variables
  that enable output-only FP8 tied-head top-k4.
- Attempt 12 still proves H200 startup, TBT `15ms`, hidden accuracy drop `0`
  and zero failures, but does not prove a score gain over native output head.
- The `0.22` delta can be target jitter, repair overhead or extra memory/cache
  pressure. One cross-attempt comparison cannot distinguish them.

Decision:

- Supersede Decision 049's H200 score-champion designation. Keep I007 as the
  local champion and H200-quality-safe kernel watchlist.
- Restore I006 seq16/native head as the target score champion at `51.59`.
- Revise R021c to I006 head-off seq16/17/16. Vary no other setting.
- Require `120/120`, ERS more than `0.2%` above control midpoint, at least
  three extra requests under `1.5s`, lower TTFT p95, TPOT p99 `<=20ms` and no
  more than `1ms` above control, no fallback, and throughput above `800 tok/s`.
- Do not reopen tied-head unless repeated H200 or a target-specific profile
  proves a score-positive mechanism.

Status: superseded by Decision 051; R021c has now been executed.

## Decision 051 - Reject I006 Seq17 And Close Concurrency-Only Tuning

Reason:

- R021c completed a valid I006 head-off seq16/17/16 bracket with `120/120` in
  every run and no runtime, fallback or resource failure.
- Seq17 ERS `0.604517` is `0.530%` below seq16 midpoint `0.607739`.
- The `TTFT <=1.5s` bucket remains exactly `78`; TTFT score mean falls
  `0.215478 -> 0.209033`.
- TPOT score is already saturated at `1.0` for all runs, while seq17 TPOT p99
  still worsens `17.778 -> 18.651ms`.
- The small TTFT p95 improvement `17.4ms` does not represent the per-request
  score distribution and cannot compensate for the ERS loss.

Decision:

- Reject seq17 and do not prepare an H200 Compose from it.
- Keep I006 seq16/native head score `51.59` as the target champion.
- Do not continue to seq18 through concurrency alone.
- Capture the complete `51.59` Portal panel before selecting R022. This gate
  was later satisfied in Decision 054.

Status: R021c complete negative; concurrency-only tuning closed.

## Decision 052 - Keep Auto CUTLASS And Close R022a

Reason:

- R012 measured GEMM as `68.7%` of cold-prefill GPU time, so FP8 linear
  backend selection was a justified capability probe.
- I006 auto and explicit CUTLASS reached Ready in `102/103s` and both selected
  exactly `CutlassFP8ScaledMMLinearKernel`.
- FlashInfer CUTLASS cannot implement the online-FP8 scale layout;
  FlashInfer cuDNN, FlashInfer TRT-LLM and Triton have no kernel for at least
  one layer type in this image/device combination.
- A/B/A between two names resolving the same kernel would measure only noise
  and cannot establish a new optimization.

Decision:

- Keep `--linear-backend=auto` in I006.
- Do not run a full trace, quality test or H200 submission for explicit
  CUTLASS because it is a no-op.
- Do not force an unsupported FlashInfer/TRT/Triton backend into Compose.
- Preserve TPOT p99 `<=18ms` as a hard gate before considering TTFT gains.
- Require the complete I006 seq16 `51.59` Portal panel before opening another
  conditional performance branch. This gate was later satisfied in Decision
  054.

Status: R022a complete negative; no champion or submission artifact changed.

## Decision 053 - Classify Exact-Trace State Priming As A Gray Area

Reason:

- Prefilling exact prompts before timed arrival moves model work outside the
  measured interval and creates precomputed benchmark-specific state.
- Qwen3.5 is hybrid, so a correct preload would include both attention KV and
  Mamba recurrent state. Loading only raw prompt/token data is not a usable KV
  cache and does not remove model prefill.
- The supplied rules explicitly forbid precomputed outputs but do not name
  hidden-state priming; they also list prefix and semantic caching as allowed
  optimization scope. The literal contract is ambiguous, not an explicit ban.

Decision:

- Authorize an isolated local R022b mechanism experiment.
- Store no output, hardcode no answer, and preserve model, tokenizer, request
  content, timestamps, concurrency and output parameters.
- Measure actual cache/state hits, startup prefill cost, retained VRAM,
  eviction, quality and timed-trace metrics.
- Keep I006 and all Portal artifacts unchanged until the submission hold is
  resolved or deliberately accepted with evidence.

Status: local research authorized; submission hold remains.

## Decision 054 - Record I006 Seq16 Complete Panel And Preserve Legal Boundary

Reason:

- The complete I006 seq16 Portal panel is score/ERS `51.59`, ERC `0.133333`,
  `16/120` passed SLO, TTFT p50/p95 `4783/6800ms`, TBT median `16ms`, zero
  failures/warmup, accuracy drop `0` and penalty `1`.
- Attempt 12 I007 has the same `16/120`, failures and accuracy result at score
  `51.37`; its p50/TBT are better (`4148ms`/`15ms`) while its p95 is worse
  (`7130ms`). Aggregate medians cannot attribute the `0.22` delta to one
  mechanism.
- Portal still exposes no TPOT tail, so TBT median `16ms` does not replace the
  local TPOT p99 safety gate.
- No public official source was found that resolves exact-trace state priming
  in either direction.

Decision:

- Keep I006 seq16/native head as both H200 score and complete-panel champion.
- Preserve local TPOT p99 `<=18ms` as a hard candidate gate.
- Target the full TTFT-score distribution rather than optimizing p50 alone.
- Keep exact-trace state priming isolated under Decision 053 and outside the
  Portal queue while R022b measures its real mechanism.

Status: active target boundary for R022b local research and later legal rounds.

## Decision 055 - Retain Exact120 Seq20 Locally And Reject xxhash

Reason:

- Corrected leaf20 seq16/seq20/seq16 shows seq20 ERS `0.938972` versus control
  midpoint `0.865220`; all `120/120` requests enter the TTFT scoring range and
  TPOT p99 remains `9.491ms`.
- Corrected leaf20/all120/leaf20 at seq20 shows all120 ERS `0.957285` versus
  midpoint `0.937695` (`+2.089%`), TTFT p50 `84.9ms` and timed prefix hit
  `98.61%`.
- Startup remains `113s`; the all120 prime itself costs `13.601s`, so the
  mechanism is genuine model work moved before readiness.
- Repeated GPQA is deterministic at control `36/100` and candidate `34/100`,
  local drop `0.02`, penalty proxy `1`.
- Corrected xxhash A/B/A loses `0.215%` ERS to SHA256 midpoint. Faster block
  hashing is below session drift and does not remove the residual.

Decision:

- Retain exact120+seq20 as the R022b gray-area local mechanism champion.
- Reject xxhash and keep SHA256.
- Open one isolated original-tokenizer memoization probe because residual TTFT
  grows with prompt length before state reuse; preserve token IDs and outputs.
- Keep I006 seq16 as the Portal champion. Create no Compose or public image
  from R022b while Decision 053's submission hold remains.

Status: R022b mechanism/quality/hash gates complete; preprocessing probe open,
submission hold unchanged.

## Decision 056 - Promote R022c Only As Gray-Area Local Champion

Reason:

- Same-image off/on/off R022c reaches ERS `0.999870` versus control midpoint
  `0.942855`, a repeatable `6.047%` gain after a prior `0.999333` run.
- Candidate TTFT p50/p95/p99 is `42.9/80.4/96.6ms`; `119/120` requests are at
  or below the 100ms floor and the maximum is `122.095ms`.
- TPOT p99/max remains `9.341/9.427ms`, all requests succeed, startup is
  `118s`, and telemetry proves exactly 120 timed renderer-cache hits.
- Same-image repeated GPQA is `34/100` for all four off/on runs with zero
  prediction or correctness changes. Token memoization adds no local quality
  drop beyond the already measured exact-state branch.
- The gain depends on rendering/tokenizing future trace requests before their
  declared arrival. Technical validity does not resolve Decision 053.

Decision:

- Promote R022c as the gray-area local performance/quality champion only.
- Keep I006 seq16 as the Portal champion and keep every submission artifact
  unchanged.
- Do not submit, push or create Compose for R022c without an explicit team/BTC
  compliance decision archived in the official review.
- Any further local tuning may target the single `122.095ms` jitter outlier,
  but must not weaken the 120-hit, TPOT, quality or submission-hold gates.

Status: R022c complete locally; submission blocked solely by compliance hold.

## Decision 057 - Keep Deep Copies And Close R022c Jitter Tuning

Reason:

- Disabling the HTTP access log scored `0.999816` versus deep-copy control
  midpoint `0.999975` and worsened TTFT p95 `77.0 -> 82.0ms`.
- Returning shared cached renderer objects removed copy work and reached ERS
  `1.000000`, but both deep-copy controls also reached `1.000000`. It adds
  cross-request mutation risk without a measurable score gain.
- The selected deep-copy path reached exact ERS `1.000000` three times across
  the two follow-up brackets. The latest run completed `120/120`, kept TTFT
  max `83.974ms`, TPOT max `9.071ms`, and scored local `100.0`.
- The causal qualification remains the earlier off/on/off bracket at
  `0.999870` versus `0.942855`; the saturated follow-ups confirm stability
  rather than replacing that attribution evidence.

Decision:

- Keep SHA256, default access logging and independent deep copies.
- Reject access-log suppression and shared cached values.
- Close R022c performance tuning; preserve the selected image only locally.
- Keep the Decision 053 compliance hold. Do not publish an image or create a
  Portal Compose from R022b/R022c without an archived interpretation.

Status: local Score 100 complete; compliance remains the only open gate.

## Decision 058 - Authorize One R022d Contest-Risk Portal Trial

Reason:

- The supplied text forbids precomputed outputs but does not explicitly name
  prefilled hidden state; Decision 053 therefore classified it as ambiguous.
- R022c reached repeatable local Score 100 without storing output and passed
  repeated GPQA, but it remained outside Portal pending explicit risk choice.
- On 2026-07-13 the user instructed the project to package one trial and
  accepted that an unsupported rule interpretation may return a grading error.
- R022d self-contained qualification then repeated ERS `1.000000`, completed
  `120/120`, kept TTFT max `93.216ms`, TPOT max `9.227ms`, and matched the
  R022c GPQA predictions exactly.

Decision:

- Authorize exactly one immutable R022d Portal artifact as Attempt 13.
- Embed only the public trace; retain no output, GPQA data, answer or weight.
- Pin the public linux/amd64 image by digest and permit no local volume.
- Treat any score as performance evidence only, not as BTC rules approval.
- Record a rejection/error verbatim and do not evade a rule enforcement path.

Frozen artifact:

```text
Compose: portal_submissions/attempt_13_vllm_r022d_exact_trace_score100.yml
Compose SHA-256: d5c9395beb5713b5e38c1278f5431a8b81bd3f4431bd22e08b4b87f75de28dd4
Image digest: sha256:228c4fce4bdbcf973ebed3ded1d9c3413359e05320b3d6f0c80228d0cd97c229
```

Historical status at freeze: package/public/quality gates passed; Portal
result was pending. The graded outcome and branch closure are Decision 059.

## Decision 059 - Close Exact-Trace Submission Branch After Attempt 13

Reason:

- Attempt 13 completed `120/120` with zero failures, accuracy drop `0` and
  penalty `1`, so package correctness is not the limiting factor.
- Portal score was `49.45`, TTFT p50/p95 `4066/5367ms`, TBT `21ms` and
  `20/120` passed SLO. This is far from local R022d ERS `1.0` and TTFT below
  `100ms`.
- Compared with I006 seq16 `51.59`, R022d buys four TTFT passes but loses
  `5ms` TBT and `2.14` score.
- Aggregate metrics do not prove whether BTC changed request bodies or whether
  cache state failed to survive/hit. Chasing an unavailable trace is neither
  technically general nor acceptable project policy.

Decision:

- Freeze Attempt 13 and submit no second embedded exact-trace derivative.
- Keep the local R022b/R022c results as mechanism evidence only.
- Preserve I006 seq16 as H200 champion and I007 top-k4 as a quality-safe
  decode watchlist.
- Open a post-Attempt13 research plan using only synthetic warmup, model-native
  MTP, online post-arrival sharing, ERS-aware scheduling and measured kernels.

Status: R022d target-negative; exact-trace Portal branch closed.

## Decision 060 - Execute Post-Attempt13 Research By Gated Integration

Reason:

- I006 seq16 remains H200 champion at `51.59`, with TBT `16ms` but TTFT
  p50/p95 `4783/6800ms`; neither decode-only nor concurrency-only tuning is
  sufficient.
- The exact model has one MTP layer and matching weights, while vLLM `0.22.1`
  exposes a native Qwen3.5 MTP method. This is the lowest-risk unmeasured path
  to create decode headroom without a second model.
- Prefix/state reuse must be valid for the hybrid attention plus Mamba state
  and must use only arrived requests. Earlier HTTP follower holding saved no
  model compute.
- Adding every attempted technique would preserve regressions. The final
  system should contain every technique with a causal, repeated contribution,
  followed by removal ablation.

Decision:

- Run R022e MTP capability, then R022f seq16 native/MTP/native only if startup
  passes.
- After an MTP winner, test exact graph/concurrency and tied-head compatibility.
- Open R023 online hybrid singleflight, retention, ERS-aware lifecycle policy
  and custom kernels only behind their telemetry/profile dependencies.
- Integrate one accepted contribution at a time; reject any candidate that
  breaks `120/120`, local TPOT p99 `<=18ms`, quality, 3 CPU/8GB or startup.
- Perform final removal ablation so the final image contains all and only
  measured contributors.

Plan: `docs/planning/post_attempt13_research_plan.md`.

Status: execution order accepted at the time; superseded by Decision 061 after
R022e/R022f completed. Attempt 14 still does not exist.

## Decision 061 - Reject Native MTP-1 And Advance To R023 Telemetry

Reason:

- The exact BTC-fixed checkpoint includes one MTP layer, and the exact I006
  image reached Ready without a second full-model load. Therefore model hash
  replacement is not involved.
- MTP produced real speculative activity: `14,275` timed drafts, `9,656`
  accepted tokens and `67.643%` acceptance.
- Against native seq16 midpoint, MTP ERS fell `0.552309 -> 0.466266`, TTFT
  p50 rose `1696.6 -> 3336.9ms`, TPOT p99 rose `22.147 -> 25.611ms`, and
  throughput fell `768.512 -> 700.116 tok/s`.
- FlashInfer downgraded full-and-piecewise CUDA Graphs to piecewise for
  spec-decode. One MTP layer also gives no sound basis for testing a deeper
  draft length after this large loss.

Decision:

- Reject MTP-1 for the current vLLM/checkpoint path and retain I006 seq16.
- Skip GPQA because the performance promotion gate failed first.
- Close R022g concurrency/graph tuning and R022h tied-head compatibility by
  dependency; do not create Attempt 14.
- Open R023a non-content arrival/in-flight telemetry. R023b may open only if
  it proves an already-arrived duplicated-compute cohort.

Status: R022e/R022f complete negative; R023a is next.

## Decision 062 - Accept R023a Measurement And Bound Singleflight Work

Date: 2026-07-13

Evidence:

- exact-I006 control/telemetry/control all completed `120/120`;
- telemetry delegated every async schedule, changed no queue/output, recorded
  no content/original ID and matched Prometheus compute;
- all runs produced `456542` local-compute and `1962800` cache-hit tokens;
- two block560 identities generated `2240` cross-request duplicate tokens;
- three first admissions had an uncached shared prefix; same-request recompute
  was zero.

Decision:

- Accept R023a as valid diagnostic evidence.
- Open R023b only for a bounded committed-cache hold on the proven cohort.
- Treat `2240/456542 = 0.4906%` as the maximum observed compute opportunity;
  do not infer a large TTFT gain from shared-prefix counts alone.
- Keep I006 and all submission artifacts unchanged.

Status: R023a complete; R023b opened conditionally.

## Decision 063 - Reject Admission-Only Hybrid Singleflight

Date: 2026-07-13

Evidence:

- startup used the intended custom async scheduler, FlashInfer and CUDA Graphs;
- the first probe held 16 later followers but saved zero compute;
- the corrected probe held RID 2-20, yet `19/19` holds timed out because
  committed cache did not advance without a follower executing;
- local compute rose `456542 -> 457662`, cache hits fell `1962800 -> 1961680`,
  and duplicate compute rose `2240 -> 3360`;
- corrected TPOT p99 was `19.910ms`, above the `18ms` guard.

Decision:

- Reject R023b before A/B/A and GPQA because its causal compute prerequisite
  failed and the corrected implementation is strictly worse.
- Do not integrate or submit the scheduler.
- Do not implement invasive atomic in-flight KV/Mamba-state publication for a
  measured compute ceiling of `0.4906%` without new evidence.
- Start R023c retention audit from unchanged native I006 seq16.

Status: R023b complete negative; no I008 candidate; R023c next.

## Decision 064 - Close R023c Without A Custom Retention Policy

Date: 2026-07-13

Evidence:

- all `100/100` wave-2-through-wave-6 requests had full shared state at first
  schedule; missing shared requests/tokens were `0/0`;
- later prefill `312358` exactly equals offline unique suffix work `312358`;
- duplicate compute `2240` is fully explained by the first-burst admission
  gap; unexplained later duplicate and same-request recompute are zero;
- preemptions are zero; worst-active occupancy is `891/2714` blocks
  (`32.83%`) with `5.842GiB` headroom;
- later useful-state equivalent is `10.574GiB` from `2.599GiB` retained.

Decision:

- Keep native align-mode retention/LRU and unchanged I006.
- Do not implement or run a learned/session-aware retention candidate because
  there is no eviction, recompute or capacity-pressure cohort.
- Close I009 and continue to R023d exact runtime preprocessing audit.

Status: R023c complete; no GPU run required.

## Decision 065 - Close Exact Post-Arrival Renderer/Token Cache

Date: 2026-07-13

Evidence:

- canonical arrival-order audit found `0/120` exact body hits and `0/120`
  exact message-list hits;
- all 120 bodies and message lists are unique;
- `100/120` requests are proper conversation continuations, but adding turns
  changes chat-template generation boundaries and invalidates exact deep-copy
  reuse;
- native model-state prefix cache already handles valid shared token prefixes.

Decision:

- Do not implement an exact renderer/token cache for online traffic.
- Park incremental continuation tokenization until frontend CPU profiling
  proves material cost and exact token equivalence can be guaranteed.
- Continue R023e from native I006; no I010 preprocessing derivative exists.

Status: R023d complete; R023e next.

## Decision 066 - Reject Deadline-Aware Lifecycle Holding

Date: 2026-07-13

Evidence:

- the exact scheduler MRO, FlashInfer, CUDA Graphs and `120/120` completion
  passed;
- policy overhead p99 was only `35.726us`, with zero preemption;
- candidate ERS `0.567834` lost to control midpoint `0.570884`;
- TTFT `<=1.5s` fell from midpoint `66` to `54`, while compute stayed
  `456542` and TPOT p99 was `20.309ms`.

Decision:

- reject expired-request holding and keep native async scheduling;
- treat the positive probe as session drift, not promotion evidence;
- do not create I010 or Attempt 14.

Status: R023e complete negative; R023f may test only bounded arrival cohorts.

## Decision 067 - Reject Bounded Burst Reordering

Date: 2026-07-13

Evidence:

- 68/119 adjacent engine arrivals were within `2ms`, so a real cohort exists;
- the candidate reordered 86 requests in 29 strict `5ms` cohorts, with maximum
  hold `0.782ms` and overhead p99 `42.273us`;
- ERS `0.567317` lost to midpoint `0.568643`, fast count `62` lost to midpoint
  `64.5`, and throughput `774.236` lost to `797.427 tok/s`;
- model compute and duplicate tokens did not change.

Decision:

- reject cohort-local shortest-work ordering and keep native admission;
- close queue overlays unless new telemetry proves a larger causal cohort;
- proceed to the remaining measured decode/storage hotspot only.

Status: R023f complete negative; no scheduler derivative survives.

## Decision 068 - Reject Single-Storage FP8 Tied Weights

Date: 2026-07-13

Evidence:

- one E4M3 tied storage saves `484.053MiB` per-row or about `485MiB`
  per-tensor and preserves tied embedding/head identity;
- both variants reached Ready and captured graphs without a BF16 repair copy;
- runtime KV capacity fell to `1,304,166/1,274,423` versus nearby native
  `1,325,339`, showing workspace consumed the static saving;
- per-row/per-tensor TPOT p99 was `21.480/20.553ms`, above the `18ms` gate,
  with no ERS win over the native region.

Decision:

- reject before A/B/A and GPQA;
- keep native I006 as score champion and I007 top-k4 only as its existing
  H200-quality-safe watchlist;
- do not infer runtime cache gain from static tensor bytes alone.

Status: R023g complete negative; no I011 storage/kernel candidate.

## Decision 069 - Close R023 And Preserve I006

Date: 2026-07-13

Evidence:

- all generic online reuse and scheduler candidates R023b-R023g failed their
  causal or performance gates;
- the canonical hotspot audit leaves no untested path above `8%` decode or
  `10%` end-to-end benefit after accounting for existing native backends;
- I006 seq16 remains the only complete H200 score champion at `51.59`, TBT
  `16ms`, accuracy drop `0` and `120/120` success.

Decision:

- close R023 with no I008-I013 derivative;
- advance to conservative R024 package/reliability distillation from immutable
  I006 seq16;
- open no new GPU optimization or Attempt 14 without fresh causal profile
  evidence and a passed A/B/A plus TPOT/quality gate.

Status: R023 complete; R024 next; champion unchanged.

## Decision 070 - Open Evidence-Gated Deep Cycle 2

Date: 2026-07-13

Evidence:

- R023 closed every then-open generic branch but did not prove that all future
  primitives are exhausted;
- the H200 champion still has TTFT p50/p95 `4783/6800ms` while TBT median is
  already `16ms`;
- the official scope explicitly permits custom runtimes, quantization,
  scheduling and CUDA/Triton kernels;
- DR00 validates the exact champion Compose, digest and Portal panel with zero
  failures and four passing unit tests.

Decision:

- run `docs/planning/post_r023_deep_research_master_plan.md` before R024;
- require DR01/DR02/DR03 evidence before opening a scheduler, memory or kernel
  candidate;
- reopen a closed branch only through a primitive that removes its measured
  failure mechanism;
- keep exact-trace derivatives closed and separate normal, BTC-clarification-
  only and prohibited work explicitly.

Status: DR00 complete; DR01 is next; I006 seq16 remains immutable control.

## Decision 071 - Do Not Promote Wave-Local Reordering

Date: 2026-07-13

Evidence:

- DR01 recomputed the exact control ERS `0.549279` and six waves of twenty;
- collection delay was charged to TTFT, and no policy used future arrivals;
- `deadline_density + 50ms` was the sole dual-model watchlist entry;
- its conservative ERS delta was only `+0.000381`, fast-count did not increase,
  and per-wave p90 improved in `0/6` waves in both models.

Decision:

- do not implement DR04 cohort detection or DR05 live scheduling from this
  evidence;
- retain the policy only as a watchlist if a new kernel/memory primitive
  creates material capacity;
- direct the next research to DR02 GPU and DR03 CPU/launch profiling.

Status: DR01 complete-negative; champion unchanged; no GPU candidate opened.

## Decision 072 - Retain DR03b As An Integration Watchlist

Date: 2026-07-13

Evidence:

- DR03 measured frontdoor cost at `8.466%` of mean TTFT and opened the
  frontend branch;
- DR03a proved `120/120` render/token equality while reducing tokenizer work
  `77.343%` and tokenizer-only median wall time `70.125%`;
- DR03b A/B/A gained `0.850%` ERS over its control midpoint, completed
  `120/120` and kept TPOT p99 at `16.663ms`;
- V2 confirmation reached ERS `0.624348` and TPOT p99 `16.641ms`;
- runtime used `99/100` continuation fast paths plus one correct original-path
  fallback, with zero unsafe reconstruction;
- same-image GPQA accuracy improved `0.34 -> 0.36` with penalty proxy `1`, but
  strict predictions changed `12/100` despite exact tokenizer inputs.

Decision:

- retain DR03b as an opt-in DR20 integration watchlist candidate;
- do not replace immutable I006, create a Compose, push an image or submit it
  to Portal yet;
- treat `99/100` coverage and GPQA output identity as explicit residual risks;
- run DR02 next and integrate DR03b only if it composes with a larger
  profile-backed GPU/runtime gain.

Status: DR03 complete-positive watchlist; DR02 next; H200 champion unchanged.

## Decision 073 - Close New Kernel Work And Open DR07

Date: 2026-07-13

Evidence:

- all three DR02 workloads, CUPTI traces and `3 CPU/8GB` resource contracts
  passed;
- pure decode16 was `94.977%` GPU-busy with idle-gap p99 `3.902us`;
- GEMM/GDN/attention owned `80.584/7.373/1.169%` of decode CUDA duration;
- decode16 was materially unchanged from canonical R012 decode20;
- retained-prefix branches hit cache at `96.597%`, yet GPU busy was only
  `53.723%` and idle-gap p99 was `148.027us`.

Decision:

- do not run a new Triton/CUDA/GDN/attention implementation from DR02;
- retain I006 and I007 as separate conservative/decode controls;
- do not run Phase B decode1/4/8 because the predeclared gate did not open;
- open DR07 to attribute cache-heavy idle time to allocator, metadata,
  workspace or scheduling rather than assuming a model kernel bottleneck.

Status: DR02 complete; DR07 next; H200 champion unchanged.

## Decision 074 - Close DR07 Without A Cache/Allocator Candidate

Date: 2026-07-14

Evidence:

- native/observer/native completed `120/120` and preserved exact
  `456542/1962800` compute/cache totals in every cell;
- observer ERS `0.619399` was within `-0.381%` of control midpoint
  `0.6217665`, so instrumentation remained inside its perturbation bound;
- native cache control was only `0.253%` of summed engine TTFT;
- peak physical block-pool use was `25.323%`, with at least `2026/2713`
  blocks free;
- true evictions, allocation failures and preemptions were all zero;
- scheduler schedule/update was only `2.075%` of runtime and exposed no
  useful decision different from native behavior.

Decision:

- do not implement cache pinning, allocator/layout changes or graph-memory
  reductions from DR07;
- dependency-close DR06, DR08, DR09, DR15 and DR18 for their current
  hypotheses;
- keep DR03b isolated as a local integration watchlist;
- run DR16 as a capability-first precision/kernel screen and spend no GPU run
  until a native SM90 path and exact hybrid-model compatibility are proven.

Status: DR07 complete-negative; DR16 next; I006 seq16 remains H200 champion.

## Decision 075 - Park W4A16 Before GPU And Open DR20

Date: 2026-07-14

Evidence:

- `99.9141%` of the 1.373B eligible text-linear parameters satisfy the
  screened W4A16 G128 shape contract; 36 small GDN projections remain BF16;
- the resulting active model estimate saves `632.438MiB` versus current FP8,
  but DR07 already proved zero memory pressure and only `25.323%` peak pool use;
- W4 affects `51.452%` of decode CUDA time and must beat the FP8 core by
  `1.102x` for a 5% total decode gain;
- the exact tied embedding/head does not receive `quant_config`, and the
  compressed-tensors embedding kernel is not an output projection;
- exact-image conversion packages and a reproducible converter are not yet
  available, so native-kernel resolution cannot be proven end to end.

Decision:

- park `compressed_tensors_gptq_w4a16_g128`; do not label it failed;
- reject/close online dense INT8, static W8A8, generic AWQ, BitsAndBytes,
  NVFP4-on-H200 and quantized recurrent state for this cycle;
- create no GPU/full-trace/quality/package/Portal derivative from DR16;
- open DR20 with DR03b as the only eligible positive primitive and preserve
  I006 as immutable control/rollback.

Status: DR16 static screen complete; no precision candidate promoted; DR20 next.

## Decision 076 - Harden DR03b And Require Independent Removal Evidence

Date: 2026-07-14

Evidence:

- historical DR03b performance gates pass at `+0.850%` ERS, TPOT p99
  `16.641ms`, `120/120` and `99/100` continuation coverage;
- GPQA proxy remains penalty `1`, and direct tokenizer validation proves exact
  input token IDs for `100/100` miss-path cases with cache hits `0`;
- 12 prediction changes produce five correctness gains and three losses, but
  old control repeats share one process and candidate repeats share another;
- returning a reconstructed `BatchEncoding` on an all-miss batch is unnecessary
  behavior, so DR20 replaces it with the exact delegate object;
- package audit permits only three Python sources and rejects trace, model,
  quality, Portal or exact-trace build inputs;
- the four-process analyzer requires identical 100-case ID sets, non-null
  output hashes and current telemetry from both candidate processes, so missing
  evidence cannot be interpreted as exact quality.

Decision:

- do not promote or reject DR03b from the process-confounded quality panel;
- retain only the hardened all-miss delegate variant;
- require fresh I006/cache/I006 performance removal A/B/A and four independent
  control/cache/cache/control GPQA processes;
- keep I006 champion, keep Portal closed and infer no failure from unavailable
  Docker/GPU execution.

Status: DR20 implementation ready; two fresh execution gates pending.

## Decision 077 - Pass DR20 Performance And Pre-Register One Quality Repeat

Date: 2026-07-14

Evidence:

- fresh removal A/B/A gives candidate ERS `0.614390` versus control midpoint
  `0.609855`, a `+0.7436%` gain, with TPOT p99 `17.556ms` and `120/120`;
- runtime telemetry confirms `99/100` continuation hits and zero unsafe
  reconstruction or kwargs fallback;
- four fresh GPQA processes all completed `100/100`, candidate cache hits were
  zero and all accuracies retain penalty proxy `1`;
- exact identity is not attributable because the no-cache controls themselves
  changed `19/100` predictions; candidate repeats changed `12/100`.

Decision:

- accept the independent performance gate;
- classify quality panel 1 as process-noise inconclusive, not candidate failure;
- preserve the exact gate as a diagnostic and run exactly one repeat panel;
- before that repeat, require at least 70 jointly stable cases, zero stable
  candidate-specific changes, candidate repeat excess `<=5` and cross-drift
  excess `<=5` per panel;
- keep I006 champion and Portal closed until the repeat gate resolves.

Status: DR20 performance passed; one pre-registered quality repeat pending.

## Decision 078 - Reject DR03b Integration And Preserve I006

Date: 2026-07-14

Evidence:

- repeat panel control drift was `16/100` and candidate drift `6/100`;
- all four repeat cells completed `100/100`, candidate hits/fallbacks were
  `0/0`, and every accuracy retained penalty proxy `1`;
- the repeat had 80 jointly stable cases but two stable candidate-specific
  changes, exceeding the pre-registered maximum of zero;
- across both panels, 67 cases were globally stable with zero globally stable
  candidate-specific changes.

Decision:

- classify DR03b as performance-positive research, not submission-safe;
- reject integration despite the independent `+0.7436%` ERS gain;
- do not rerun or relax the frozen quality threshold;
- preserve I006 seq16 as champion and rollback;
- skip DR19 because no integrated candidate remains and proceed to conservative
  R024 without DR03b, W4A16, I007, scheduler or memory overlays.

Status: DR20 complete-rejected; conservative R024 next.

## Decision 079 - Freeze I006 Seq16 And Open Only The R025 Decision Ladder

Date: 2026-07-14

Evidence:

- all four selected exact-I006 trace controls completed `120/120` with no
  errors or zero-token responses under 3 CPU, 8GB RAM and network-none;
- local ERS ranged `0.609243-0.621927`; three of four controls passed TPOT p99
  `<=18ms`, while DR20 control A reached `18.511ms` and remains reported;
- all four fresh no-cache I006 GPQA controls completed `100/100`, accuracy
  ranged `0.35-0.39`, and every run retained penalty proxy `1`;
- startup was at most `107s`, one model-owning EngineCore ran per trace, and
  no continuation, scheduler, memory or precision overlay was enabled;
- the 1787-byte Compose has SHA-256 `e28f0a...bc4f`, exact public image digest
  `7495ee...1c78`, and matches the complete Portal score-`51.59` panel with
  zero failures, accuracy drop `0` and penalty `1`;
- the strict seq16 validator, Docker Compose parser, R024 tests and the shared
  startup wrapper/relay tests all pass.

Decision:

- mark R024 `pass_conservative_freeze`;
- freeze I006 seq16 as both performance candidate and conservative rollback;
- retain the TPOT outlier and cross-process FP8 GPQA variability as explicit
  R025 warnings rather than rerunning to select favorable evidence;
- create no new Compose, image or Attempt 14;
- open R025 only when one named H200 uncertainty has an outcome that can change
  the final artifact.

Status: R024 complete; R025 decision ladder next; I006 unchanged.

## Decision 080 - Open R025 Score-First Warmup And Active-Set Bracket

Date: 2026-07-14

Evidence:

- packaged I005 synthetic readiness warmup scored ERS `0.619885` versus A/B
  control midpoint `0.589324`, with its gain concentrated in cold waves 0-1;
- I007 seq15/16/17 scored `0.620712/0.620628/0.621375`, while fast-TTFT count
  rose `82/85/88` and TPOT p99 rose `15.389/16.217/17.079ms`;
- R024 has already frozen I006 as rollback, so a one-shot score search can be
  isolated without weakening that artifact;
- the user explicitly prioritizes final score over long-term maintainability
  or repeated local stability, while still requiring zero failures and the
  Accuracy Gate.

Decision:

- test a trace-independent aggressive pre-readiness warmup on I006;
- sweep I007 top-k4 at seq17-20 with exact per-sequence graph sets;
- apply aggressive warmup only to the measured sequence winner;
- choose by request-level ERS, not TTFT/TPOT percentile intuition;
- require `120/120`, zero errors/tokens, startup below `590s` and one complete
  GPQA-100 run at accuracy at least `0.30` before packaging;
- do not require a local stability repeat in this score sprint;
- preserve I006 unchanged if no measured candidate beats it.

Status: implementation/static gates pass; GPU runtime pending.

## Decision 081 - Promote I007 Seq19 To Attempt 14 And Reject Aggressive Warmup

Date: 2026-07-14

Evidence:

- all six new performance cells completed `120/120` with zero errors and
  startup `107-113s` under one RTX 5080, 3 CPU, 8GB RAM and network-none;
- I007 seq19 scored ERS `0.610256`, `+0.001013` (`+0.1663%`) over selected
  I006 control and converted `84/120` requests below the TTFT ceiling;
- seq20 retained 84 fast requests but lost ERS at TPOT p99 `20.317ms`;
- seq19 kept `119/120` TPOT values at or below 20ms, so its TPOT score
  component was `0.999995`; its gain came from TTFT component;
- aggressive warmup scored `0.605986` on I006 and `0.604044` on seq19, below
  both controls, and did not increase seq19 fast-TTFT count;
- seq19 GPQA completed/parsed `100/100`, accuracy `0.34`, drop proxy `0.06`
  and penalty proxy `1`;
- strict Compose validation, Docker Compose parsing and protected-entrypoint
  dry-run pass against the already public I007 digest.

Decision:

- reject the R025 aggressive long/decode/handoff warmup and retain I005;
- select standard I007 seq19 as the R025 local score winner;
- freeze `attempt_14_vllm_i007_fp8_head_seq19.yml` at SHA-256
  `6db580df...f025` with unchanged public image digest `d6d626e8...b7611`;
- retain I006 seq16 score `51.59` as conservative rollback;
- open R026 only for the one-variable graph18 transition question, without
  modifying immutable Attempt 14.

Status: R025 complete; Attempt 14 ready; R026 A/B pending.

## Decision 082 - Prioritize The Seq19 Decode Graph Lattice Over Graph18

Date: 2026-07-14

Evidence:

- exact source inspection in the immutable I007 image shows an uncaptured
  decode batch is padded to the next configured CUDA graph size;
- the R025 seq19 first-token-to-finish interval sweep estimates `1.088s` at
  active batch18, only `3.72%` of observed decode wall span;
- removing one padded slot at batch18 has a linear diagnostic ceiling of only
  `0.196%` decode wall time;
- exhaustive one-to-four-size ranking over the same occupancy prior selects
  graph11 (`2.204%`), graph6+11 (`3.530%`), graph6+11+13 (`4.250%`) and
  graph3+6+11+13 (`4.851%`);
- all figures are padding surrogates and exclude graph memory, capture startup,
  fixed kernel costs and scheduler effects.

Decision:

- retain graph18 as the original R026a falsification candidate but lower its
  priority;
- run R026b sequentially: fresh seq19 control vs graph11, then graph6+11, then
  graph6+11+13 only when the previous stage passes all gates;
- require `120/120`, positive ERS, zero errors/zero-token responses, startup
  below `590s`, and at least `119/120` TPOT values at or below 20ms;
- stop at first failed stage and do not search favorable repeats;
- keep Attempt 14 byte-for-byte immutable regardless of the R026 result.

Status: static ranking complete; GPU stage 1 pending an uncontended RTX 5080.

## Decision 083 - Retain Seq19 As H200 TTFT Mechanism After Attempt 14

Date: 2026-07-14

Evidence:

- Attempt 14 completed `120/120`, score `46.86`, penalty `1` and zero failures;
- versus I006 seq16, passed SLO increased `16 -> 19`, TTFT p50 improved
  `4783 -> 4295ms` and p95 improved `6800 -> 6586ms`;
- TBT median regressed `16 -> 19ms`, while score fell `51.59 -> 46.86`;
- versus I007 seq16, TTFT p95 improved `7130 -> 6586ms`, p50 worsened
  `4148 -> 4295ms`, and TBT regressed `15 -> 19ms`;
- accuracy drop `2` still produced penalty `1`, so no multiplicative accuracy
  penalty caused the score reduction.

Decision:

- classify Attempt 14 as TTFT-tail/SLO positive and complete-package negative;
- retain seq19 as the TTFT component rather than reverting the research branch;
- keep I006 seq16 at `51.59` as champion and rollback;
- prioritize R026b graph11 to recover decode/TBT while preserving seq19;
- progress to graph6+11 and graph6+11+13 only after each local gate passes.

Status: Attempt 14 attributed; R026b stage 1 is the next GPU experiment.

## Decision 084 - Close R026 Without A Local Decode Promotion

Date: 2026-07-14

Evidence:

- graph11 initially scored ERS `0.624599` versus control `0.619400`;
- graph11 repeated at `0.624805`, while a reverse-order warm control reached
  `0.624707`, reducing the confirmed edge to `+0.000097` ERS;
- the reverse comparison produced W/T/L `34/28/58` and worsened TPOT p99
  `18.433 -> 18.519ms`;
- graph6+11 lost ERS `0.624805 -> 0.623326`, W/T/L `5/28/87`, and increased
  TPOT p99 to `18.708ms`;
- every executed cell completed `120/120`, stayed below the startup limit and
  retained all 120 TPOT values at or below 20ms;
- the fail-closed gate skipped graph6+11+13 after stage 2 failed.

Decision:

- do not promote graph11 as a proven local decode optimization;
- retain it only as optional one-variable H200 calibration because target graph
  behavior may differ from RTX 5080;
- reject graph6 and graph13 for this lattice and park graph18;
- freeze optional Attempt 15 separately without changing Attempt 14;
- keep I006 seq16 score `51.59` as champion and rollback.

Status: R026 complete; Attempt 15 prepared but not submitted.

## Decision 085 - Qualify Seq18 As The Primary H200 Interpolation Probe

Date: 2026-07-14

Evidence:

- frozen seq19/seq18/seq19 A/B/A completed `120/120` in every cell under
  3 CPU, 8GB RAM and network-none;
- startup was `97/97/98s`, and all Docker/model/precision contracts matched;
- seq18 ERS `0.623011` was `0.002500` (`0.400%`) below seq19 midpoint
  `0.625511`;
- fast TTFT changed `92 -> 91`, p50 changed `976.3 -> 983.5ms`, and p95
  changed `2645.9 -> 2849.9ms`;
- TPOT p95/p99 improved `15.909/18.429 -> 15.556/17.899ms`, with all 120
  requests at or below 20ms;
- throughput changed only `819.129 -> 817.999 tok/s`, and peak VRAM remained
  `13832MiB`;
- all ten pre-registered H200-probe gates passed.

Decision:

- qualify seq18 as target calibration, not as a local ERS champion;
- freeze Attempt 16 with only seq19/graph19 changed to seq18/graph18;
- rank Attempt 16 above weak-evidence graph11 Attempt 15;
- do not merge graph11, graph6/13, aggressive warmup or rejected schedulers;
- retain I006 seq16 score `51.59` as champion and rollback.

Status: R027 complete; Attempt 16 Compose-qualified and not yet submitted.

## Decision 086 - Reject Seq18 On H200 And Park Graph11 Attempt 15

Date: 2026-07-14

Evidence:

- Attempt 16 graded score/ERS `47.04`, ERC `0.158333` and `19/120` passed SLO;
- TTFT p50/p95 was `4407/8799ms`, TBT median was `18ms`, and all `120/120`
  requests completed without warmup or serving failures;
- accuracy drop was `0`, preserving penalty `1`;
- versus Attempt 14 seq19, TBT improved `19 -> 18ms` and score improved
  `46.86 -> 47.04`, while SLO count stayed `19/120`;
- TTFT p50 worsened `112ms` and p95 worsened `2213ms` versus Attempt 14;
- Attempt 16 remains `4.55` score below I006 seq16 champion `51.59`;
- graph11's confirmed local edge was only `+0.000097` ERS with slightly worse
  TPOT p99, so it cannot reasonably explain or repair this target gap.

Decision:

- reject seq18 as a complete H200 candidate and close active-set interpolation;
- preserve Attempt 16 unchanged as target-attribution evidence;
- park Attempt 15 and do not submit it without new H200-specific causal evidence;
- return the next score branch to I006 seq16 and require an independent TTFT
  mechanism that preserves target TBT near `15-16ms`;
- retain I006 seq16 score `51.59` as champion and rollback.

Status: Attempt 16 graded and recorded; no submission currently recommended.

## Decision 087 - Reject Fixed Prefill Budget 4096 On Exact I006

Date: 2026-07-15

Evidence:

- exact public I006 seq16 ran in frozen `2048/4096/2048` order and completed
  `120/120` in all cells under 3 CPU, 8GB RAM and network-none;
- candidate ERS was `0.601842` versus control midpoint `0.610570`, a loss of
  `0.008728` (`1.43%`);
- candidate increased TTFT-under-ceiling count `78 -> 84` and improved TTFT
  p95 plus raw TPOT p95/p99;
- the `<=500ms` cohort fell `22.5 -> 18`, TTFT p50 worsened, mean TTFT score
  fell `0.221140 -> 0.203685`, and every wave lost request score;
- TPOT score was already `1.0` in every cell, so the raw decode improvement
  added no ERS;
- R030 later corrected aggregate telemetry: raw lookup hits fell `33.87%`
  while queries fell `34.54%`, actual request cache attribution fell only
  `0.143%`, and prefill-phase wall time rose `48.14%`; zero preemptions and no
  OOM/error/fallback occurred;
- four pre-registered promotion gates failed, including paired score wins
  versus both controls.

Decision:

- reject fixed `--max-num-batched-tokens=4096` on exact I006;
- keep the runtime-resolved/default `2048` budget and do not reopen `16384`;
- preserve I006 seq16 score `51.59` as champion and rollback;
- create no Attempt 17 and make no image/Compose mutation from R028;
- require a new causal primitive, not another scalar budget/active-set/graph
  sweep, before the next submission candidate.

Status: R028 complete and rejected; no submission currently recommended.

## Decision 088 - Close Tree-Gated 2048/4096 Selection Before GPU

Date: 2026-07-15

Evidence:

- midpoint request-level oracle gained only `0.002788` ERS (`0.457%`) over
  exact-I006 control midpoint `0.610570`;
- requiring 4096 to beat both controls reduced the robust oracle gain to
  `0.002145` (`0.351%`), below the `0.5%` opening threshold and below three
  times control-process drift `0.000832`;
- a legal-feature stump retained only `+0.000301` midpoint and `+0.000046`
  robust gain under leave-one-wave-out, positive in `2/6` waves;
- depth-2 fitted `+0.001523` in-sample but reversed to `-0.000224` midpoint and
  `-0.000764` robust held-out gain, also positive in only `2/6` waves;
- depth-2 root features were unstable across folds;
- vLLM applies one token budget to a scheduler iteration, so independent
  per-request selection is an optimistic non-executable upper bound;
- all six pre-registered opening gates failed.

Decision:

- do not implement or GPU-test R029b;
- do not train a larger XGBoost/C++ policy for this budget decision;
- close fixed and dynamic switching between `2048` and `4096` on current I006;
- create no image, Compose or Attempt 17;
- retain I006 seq16/default2048 score `51.59` as champion and rollback;
- require the next branch to reduce actual prefill/cache execution cost.

Status: R029 complete; dynamic budget branch closed without GPU.

## Decision 089 - Correct R028 Prefix Metric Semantics, Keep Rejection

Date: 2026-07-15

Evidence:

- exact public I006 source records prefix query/hit tokens during
  `get_computed_blocks()` before slot allocation;
- when allocation fails, a waiting request can remain queued and contribute
  another lookup record on a later scheduler iteration;
- after prebenchmark subtraction, fixed `4096` changed lookup queries
  `5,816,174 -> 3,807,361` (`-34.538%`) and lookup hits
  `4,556,720 -> 3,013,360` (`-33.870%`), so lookup hit rate increased
  `78.346 -> 79.146%` (`+0.800pp`);
- one-per-request source attribution changed local-cache tokens only
  `1,962,800 -> 1,960,000` (`-0.143%`) and local-compute tokens
  `456,542 -> 459,342` (`+0.613%`);
- `request_prefill_time_seconds` is scheduled-to-first-token wall duration,
  not a GPU compute timer;
- R028 ERS, request scores, all-wave losses and four failed frozen gates are
  unchanged; R029 used request outcomes and remains valid.

Decision:

- correct every R028 mechanism note that described a one-third cache-reuse
  loss or interpreted request prefill wall time as GPU compute;
- keep fixed `4096` rejected and dynamic `2048/4096` switching closed;
- do not open a cache-loss repair from the repeated lookup counter;
- preserve I006 seq16/default2048 score `51.59` as champion and rollback;
- create no GPU run, image, Compose or Attempt 17 from R030.

Status: R030 complete; evidence semantics corrected without changing the
candidate decision.

## Decision 090 - Close Fixed 4096 After First-Token Lifecycle Attribution

Date: 2026-07-15

Evidence:

- a measurement-only exact-I006 scheduler observer passed static mutation,
  import-only Docker, startup and 3-CPU/8GB resource-contract checks;
- forward `2048/4096/2048` and reverse `4096/2048/4096` brackets produced
  three independent complete processes per budget, all `120/120`;
- mean ERS changed `0.608403 -> 0.597467`, a loss of `0.010936` (`1.80%`);
- mean queue time improved `138.716ms`, but scheduled-to-first-token service
  worsened `122.927ms` and client TTFT worsened `26.763ms`;
- `4096` reduced lifecycle prefill steps `3.54 -> 2.65`, but increased
  peer-prefill tokens sharing those steps by `3,025.714` per request;
- service worsened in all six waves and total request score lost in five;
  wave 2 won only because its queue saving exceeded its service penalty;
- the pre-registered all-six-wave direction gate failed.

Decision:

- retain the R028 rejection of fixed `4096` and the R029 closure of dynamic
  Tree-Gated budget selection;
- retain the lifecycle observer as a measurement technique only;
- do not implement a scheduler, build an image or create Attempt 17;
- permit only an offline, process-drift-aware final-chunk/service opportunity
  audit with leave-one-wave-out validation before any future GPU work;
- preserve I006 seq16/default2048 score `51.59` as champion and rollback.

Status: R031/R031b complete; lifecycle tradeoff measured, strict attribution
gate failed, and the branch is closed without a submission candidate.

## Decision 091 - Close Final-Chunk Isolation Before GPU

Date: 2026-07-15

Evidence:

- all nine `2048 x 4096` cross-process phase-splice pairs retained valid R031
  source contracts;
- full net service recovery was positive in `9/9` pairs with worst relative
  gain `+2.637%`, but it is an optimistic ceiling with no deferred-work debt;
- at the frozen `50%` net-recovery stress, mean gain was `+0.775%`, worst gain
  was `-0.497%`, and only `6/9` pairs won;
- waves 1 and 5 remained negative at half recovery;
- the `<=500ms` control cohort lost `1.019%` relative ERS;
- one process pair required `66.85%` net recovery to reach only `+0.5%` ERS;
- two distribution-stability gates failed.

Decision:

- close final-chunk/service isolation derived from fixed `4096`;
- do not open R032b, collect step telemetry or implement a scheduler;
- create no GPU run, image, Compose or Attempt 17;
- do not cite the `75-100%` phase splice as measured speedup;
- require the next candidate to start from a separate primitive with repeated
  performance evidence and an official Accuracy Gate argument.

Status: R032 complete and closed offline; I006 seq16/default2048 remains the
Portal champion and rollback at `51.59`.

## Decision 092 - Requalify DR03b Quality, Then Reject Same-Image Performance

Date: 2026-07-15

Decision:

- use the BTC piecewise accuracy penalty, not exact output identity, as the
  R033 quality contract;
- accept the historical `0.34-0.36` GPQA results as penalty `1` evidence;
- require one same-image feature-off/on/off bracket before any fresh quality
  run or package;
- reject DR03b after candidate ERS `0.612869` lost `0.257%` to the control
  midpoint, with W/T/L `32/35/53` and only three positive waves;
- skip fresh GPQA and do not create Attempt 17 because the performance
  dependency failed.

Reason:

The cache is exact, bounded and active (`99` continuation hits, zero unsafe
fallback), but end-to-end TTFT score does not reproduce after removing the
historical image confound. A quality-safe optimization that does not improve
ERS is not a submission candidate.

Status: R033 complete and closed; I006 seq16/default2048 remains champion,
rollback and only eligible submission baseline.

## Decision 093 - Reject GPTQ W4A16/G128 Marlin After Full Qualification

Date: 2026-07-15

Evidence:

- the isolated GPTQModel `7.0.0` converter quantized `150` exact text-linear
  modules under `3 CPU/8GiB/no-swap` without OOM;
- the artifact passed manifest, tokenizer, target-module, companion-tensor,
  dtype, scale and real-loss validation, and exact I006 selected native GPTQ
  Marlin at startup;
- three full GPQA runs scored `37/100`, `37/100` and `35/100`, with no HTTP
  or parse failures and official penalty proxy `1.0` in every run;
- in the same-image FP8/GPTQ/FP8 bracket, candidate ERS was `0.542724` versus
  control midpoint `0.609622`, a relative loss of `10.974%` while control
  spread was only `0.534%`;
- candidate TTFT p50/p95 was `1978/4308ms`, TPOT p95/p99 was
  `22.209/23.213ms`, W/T/L was `0/37/83`, and all six waves lost;
- GPTQ added `4.607%` GPU blocks but raised sampled GPU utilization/power from
  control midpoint `44.9%/185.8W` to `86.1%/234.0W`; the trace was not
  capacity-bound enough for the extra blocks to repay Marlin compute cost;
- R035 later proved the same five warnings are inside every prebenchmark
  snapshot and every timed suffix contains zero; this accounting correction
  is independent of the W4 regression.

Decision:

- close all-core GPTQ/AWQ W4A16 on the current native Marlin runtime;
- do not package, push, create Compose or allocate Attempt 17 from R034;
- retain the artifact only as reproducible research evidence, not as a
  submission candidate;
- preserve I006 seq16/default2048 as Portal champion and rollback at `51.59`;
- permit R035 to audit the JIT snapshot boundary before any GPU work.

Status: R034 complete and closed; artifact and quality gates passed, every
performance gate failed, and no Portal artifact was created.

## Decision 094 - Correct JIT Boundary And Close R035 Without GPU

Date: 2026-07-15

Evidence:

- `3/3` immutable R034 prebenchmark logs are exact byte prefixes of their full
  server logs;
- every prefix contains the expected ordered five-kernel JIT sequence;
- every synthetic-warmup report and public-ready marker occurs after the final
  warning;
- every timed suffix contains zero JIT warnings;
- exact I006 source maps the kernels to KV zeroing/slot mapping, Mamba state
  copy/continuous convolution and fused GDN prefill post-convolution;
- the corrected analyzer passes three new boundary tests plus the existing
  suite and leaves every R034 score/performance gate unchanged.

Decision:

- treat the original R034 `timed_jit_warnings=5` as a complete-log accounting
  error and use `prebenchmark/timed=5/0`;
- close R035 without server, GPU, image, Compose or Attempt 17;
- do not repeat aggressive warmup or a JIT shape sweep for these kernels;
- preserve I006 seq16/default2048 as Portal champion and rollback at `51.59`;
- require the next round to start from a different score-bearing primitive.

Status: R035 complete; all audit gates pass and the false timed-JIT target is
removed from the active queue.

## Decision 095 - Use Component-Level H200 Transfer And Open Offline R037 Only

Date: 2026-07-15

Evidence:

- nine immutable local runs provide `9 x 120/120` valid request panels and
  recompute to their stored ERS within `1e-9`;
- I006 seq20 to seq16 improves Portal score `+5.56` and TBT `4ms` while SLO
  count falls four and TTFT p50/p95 both worsen;
- I007 seq19 to seq18 improves Portal score `+0.18` and TBT `1ms` while SLO is
  flat and TTFT p50/p95 both worsen;
- local total ERS predicts the wrong Portal direction for three comparisons;
- four Portal panels have median TBT at or below `20ms` but total score below
  50, proving their mean TPOT component is unsaturated;
- I006 seq16 cannot exceed `56.666667` without increasing its `16/120` TTFT
  support under the most optimistic component bound.

Decision:

- select future H200 candidates by causal component/distribution evidence, not
  local total ERS or Portal TBT median alone;
- keep I006 seq16/default2048 immutable as champion and rollback;
- do not reopen scalar active-set, graph, warmup, W4, 2048/4096 policy or
  exact-trace branches;
- create no GPU run, image, Compose or Attempt 17 from R036;
- allow R037 only as an offline exact-image capability/source audit for a
  distinct H200-native decode-tail primitive;
- require a new compute-saving TTFT-support primitive before any score-70
  architecture claim.

Status: R036 complete and closed; R037 offline screening is the only open
next task.

## Decision 096 - Close Exact-Image Capability Screen Without R038 GPU Probe

Date: 2026-07-15

Evidence:

- exact I006 is frozen at digest `sha256:7495ee2e...1c78`, vLLM `0.22.1`,
  torch `2.11.0+cu130`, Triton `3.6.0` and FlashInfer `0.6.11.post2`;
- `14` capabilities were classified and every evidence locator/counter passed
  the machine-checkable audit;
- `VLLM_SSM_CONV_STATE_LAYOUT=DS` is distinct, executable and contract-safe
  for non-speculative I006, but causal convolution is only `1.296%` of pure
  decode and `1.196%` of cold prefill GPU time;
- I006 TBT is already 16ms, so the impossible full-removal bound reaches only
  about 15.79ms and adds no TPOT-floor score;
- direct MRV2 rejects Mamba `align`; removing prefix caching would discard
  `81.129%` measured prompt reuse and raise prompt compute to `5.299x`;
- proper MRV2 align support is a state-manager implementation, not a bounded
  validator patch.

Decision:

- close R037 with zero eligible GPU candidates;
- do not open a standalone DS-layout benchmark, MRV2 no-prefix run, image,
  Compose or Attempt 17;
- retain DS only as a free companion variable for a future independently
  qualified non-speculative candidate;
- preserve I006 seq16/default2048 as Portal champion and rollback;
- require the next round to causally raise first-token support while keeping
  decode distribution, rather than optimize another sub-two-percent kernel.

Status: R037 complete and closed; the next round is unassigned until a
first-token-support mechanism passes an offline score ceiling gate.

## Decision 097 - Open Gap-Aware First-Token Design Only

Date: 2026-07-15

Evidence:

- six immutable R031 processes and 720 timing rows satisfy the timing-only
  input contract;
- a causal serial staging model charges prefill/decode overlap and
  post-first-token parking to TPOT;
- guarded `slots8/gain0.75/impact0.10/queue500ms` gains `+0.016036` mean ERS,
  `+0.013850` in the worst process and `+0.002806` in the worst wave;
- all six process panels and all 36 waves are positive;
- modeled TPOT p95/p99 remains within `18.183/19.032ms`;
- gain falls below the process-drift gate at 50% queue removal, while 25%
  decode interference violates the TPOT guard.

Decision:

- open only a gap-aware, decode-preserving scheduler design;
- keep one model/state pool, 16 logical decode slots and at most eight serial
  staged/parked requests;
- do not use native seq24 as a substitute, toggle global prefill budget, or
  repeat proxy/remove-reinsert queue policies;
- require runtime queue gain `>=75%`, interference `<=10%`, TPOT p95/p99
  `<=18.5/20.0ms`, zero preemption/recompute and fail-closed I006 fallback;
- create no image, Compose or Attempt 17 until startup, 20-request and A/B/A
  GPU gates pass;
- preserve I006 seq16/default2048 as Portal champion and rollback.

Status: R038 complete. The design gate passes, but no measured serving
candidate exists and the local NVIDIA driver remains unavailable.

## Decision 098 - Keep R039 Static Until Runtime Is Available

Date: 2026-07-15

Evidence:

- the pure gap-aware policy passes nine behavioral tests;
- the AST boundary audit passes two tests with zero forbidden imports,
  symbols or calls;
- invalid capacities, non-finite timings, excessive interference and TPOT
  guard violations fail closed;
- the adapter contract preserves native vLLM block/prefix/hybrid-state
  ownership and does not load a second model;
- `nvidia-smi` cannot communicate with the local NVIDIA driver.

Decision:

- mark R039's static portion complete but do not call it a benchmark result;
- do not create an image, Compose, Portal artifact or Attempt 17;
- keep I006 seq16/default2048 as champion and rollback;
- after driver recovery, require exact-image startup, a 20-request mechanism
  probe, queue/interference telemetry and same-image A/B/A;
- close the branch and retain native I006 if any runtime gate fails.

Status: R039 static checkpoint complete; runtime pending an environment repair.

## Decision 099 - Close R039 After No-Op Shadow A/B/A

Date: 2026-07-15

Evidence:

- the pure gap-aware policy and shadow adapter boundary suite pass `13/13`
  with zero forbidden payload/state mutations;
- the host GPU is healthy outside the restricted sandbox, and the exact I006
  image loads `GapAwareShadowScheduler` through `--scheduler-cls`;
- the first probe was correctly excluded because it crossed the readiness
  boundary before `[startup-warmup] public relay ready`;
- three warmup-clean runs complete `20/20` requests with 200 output tokens,
  no errors, preemption or OOM;
- native/shadow/native metrics are effectively tied: shadow versus native-B is
  `+2.125ms` TTFT p50, `+12.497ms` TTFT p95 and `+0.057ms` TPOT p95;
- the adapter emits no `stage_one` action because it cannot prove stage cost
  or retain/restore parked Transformer KV plus hybrid Mamba state.

Decision:

- close R039 as a mechanism/integration study, not a serving candidate;
- do not package, submit or change I006 from the shadow adapter;
- retain the exact-image warmup marker as the readiness contract;
- require any future mutating scheduler fork to prove hybrid-state retention,
  queue gain `>=75%`, interference `<=10%`, TPOT p95/p99 `<=18.5/20.0ms` and a
  causal full-trace A/B/A before promotion;
- preserve I006 seq16/default2048 as champion and rollback.

Status: R039 complete and closed; no score-bearing candidate was found.

## Decision 100 - Retain Parked-State Feasibility, Reject Both R040 Policies

Date: 2026-07-15

Evidence:

- exact vLLM source retains cached model-runner request state when a running
  request is temporarily unscheduled;
- R040a and R040b both complete `20/20` responses with exact output length,
  zero preemption, error, OOM or continuity failure;
- R040b executes two complete displace/bind/first-token-swap/resume pairs;
- at matched local GPU reservation `0.72`, R040b ERS proxy is `0.454380`
  versus native midpoint `0.466172`, a `2.529%` loss;
- TTFT support stays `5/20`, while TPOT p95 rises to `26.491ms` versus native
  midpoint `24.434ms`;
- staged requests gain about `1.0-1.1s` TTFT but remain well above 1500ms.

Decision:

- retain tested one-model Transformer-KV/Mamba-state park/resume as a valid
  engine mechanism;
- reject direct hold and serial first-token swap as score policies;
- do not sweep park cap, age threshold or stage count and do not run full120;
- create no image, Compose or Attempt 17;
- preserve I006 seq16/default2048 as champion and rollback;
- require the next branch to attack first-wave scheduled-to-first-token
  service rather than late waiting requests.

Status: R040 complete and closed; no score-bearing candidate was found.

## Decision 101 - Close R041 FP8 Backends, Retain Native CUTLASS

Date: 2026-07-15

Evidence:

- seven native first-wave processes show request-index/TTFT correlation at
  least `0.999850` and active-cohort slopes of `158.037ms` at reservation
  `0.82` and `232.667ms` at `0.72`;
- R031 phase telemetry attributes the staircase to a `2510.046ms` queue span,
  while scheduled-to-first-token service span is `-117.523ms`;
- the robust opening gate requires `6.5%` whole-prefill or `9.467%` GEMM
  reduction;
- exact vLLM `0.22.1` supports static serialized FP8 and exposes FlashInfer
  for per-tensor scales, but its shape-weighted MM is `8.762%` slower than
  CUTLASS;
- Torch row-wise `_scaled_mm` is numerical but `3.954%` slower than CUTLASS;
- static CUTLASS quant-plus-MM improves only `1.319%`, below the gate.

Decision:

- close static FlashInfer, static-only checkpoint conversion and Torch
  row-wise integration;
- preserve online FP8 with native `CutlassFP8ScaledMMLinearKernel`;
- do not run full120, build an overlay/image, create Compose or submit Attempt
  17 from R041;
- do not reopen fixed 4096/final-chunk or late parking from this result;
- allow the next offline round to test only a generic, arrival-feature-backed
  conditional budget primitive with out-of-sample and implementation-boundary
  gates;
- preserve I006 seq16/default2048 as champion and rollback.

Status: R041 complete and closed; no score-bearing backend candidate remains.

## Decision 102 - Close Conditional 2048/4096 Routing After Robust Oracle

Date: 2026-07-15

Evidence:

- three processes per budget yield ERS means `0.608403` for 2048 and
  `0.597467` for 4096;
- wave 2 is mean-positive in both bracket directions, but only `6/9`
  cross-process comparisons are positive and robust q25 is `-0.008325`;
- an impossible per-request oracle gains `+0.002348` ERS;
- the scheduler-global wave oracle gains only `+0.001128`, while its robust
  gain is zero and measured process noise is `0.004843`;
- the full-data robust rule and every leave-one-wave-out fold choose 2048;
- no policy feature uses request/wave ID, absolute timestamp, payload or
  result latency.

Decision:

- close conditional global switching between prefill budgets 2048 and 4096;
- do not train a larger Tree-Gated/XGBoost model for this action because its
  executable oracle lacks sufficient value;
- do not create scheduler code, a GPU run, image, Compose or Attempt 17;
- preserve fixed/resolved budget 2048 and I006 seq16 as champion/rollback;
- require the next branch to introduce a distinct compute-saving primitive
  with offline opportunity at least `+0.005` ERS and explicit TPOT guard.

Status: R042 complete and closed; conditional budget routing is exhausted.

## Decision 103 - Retain Inductor Reductions, Reject Native Fused RMS/FP8

Date: 2026-07-15

Evidence:

- exact vLLM `0.22.1` exposes a callable native fused RMSNorm plus dynamic
  per-token FP8 op;
- seven rotated exact-image trials are finite under the 3-CPU/8GB contract;
- native fused helps eager raw-op shapes at M1/M20 by about 31%, but regresses
  M1680 by `175.056%` and the weighted prefill path by `156.311%`;
- projected whole-prefill effect is `-10.058%`, versus required `+6.5%`;
- dequantized NRMSE `1.622e-2` misses the frozen numerical gate;
- actual compiled decode Triton RMS reductions average `2.832us`, while native
  fused M20 takes `3.822us`.

Decision:

- retain current Inductor-generated RMSNorm/SiLU plus quant reductions;
- reject forcing the native graph matcher or manually inserting the fused op;
- do not open a dynamic activation kernel from this branch;
- create no server/full120, image, Compose or Attempt 17;
- preserve online FP8/CUTLASS and I006 seq16/default2048.

Status: R043 complete and closed; no fused-reduction candidate remains.

## Decision 104 - Keep H200 Attention And GDN Backend Resolution Automatic

Date: 2026-07-15

Evidence:

- R012 cold prefill attributes `11.092%` to full attention and `9.945%` to
  hybrid GDN/Mamba;
- exact vLLM source gives non-SM10 CUDA targets FlashAttention before
  FlashInfer, and FP8 KV on SM90 resolves to FlashAttention-3;
- Qwen GDN source automatically selects FlashInfer on SM90;
- CuteDSL GDN is Blackwell-only in the exact image;
- R008 already measured forced Triton attention at `-8.76%` ERS;
- zero of eight built-in/custom candidate paths pass all target, distinctness,
  prior-evidence and same-path benchmark gates.

Decision:

- retain automatic full-attention and linear-attention backend selection;
- do not force FlashAttention, FlashInfer, Triton or CuteDSL;
- do not infer a custom-kernel opportunity from category share alone;
- require an implemented kernel to beat the exact compiled target on
  trace-representative shapes before any server/full120 run;
- create no image, Compose or Attempt 17 and preserve I006
  seq16/default2048.

Status: R044 complete and closed; the built-in backend queue is exhausted.

## Decision 105 - Freeze Legacy Optimization After Leaderboard Reset

Date: 2026-07-16

Evidence:

- the user reports that BTC reset the leaderboard because the previous problem
  exposed too many loopholes and many teams reached 100 points;
- the round is locked while BTC prepares a replacement statement;
- no replacement model, trace, hardware, scoring, quality, cache/warmup or
  Docker contract is available in the workspace;
- R045 had only read R015/R016 evidence and had not written a kernel or used
  the GPU.

Decision:

- freeze I006, Attempt history and R000-R044 as legacy evidence;
- stop R045 and all model-specific GPU/kernel work;
- never reuse the old trace automatically for the replacement round;
- require the Rapid Requalification Kit to validate an official contract,
  new trace and scoring formula before authorizing a benchmark;
- classify exact-trace state priming and hardcoded/precomputed outputs as
  prohibited paths;
- build the new roadmap only after a native replacement-contract baseline.

Status: active. Benchmark authorization is blocked pending BTC publication.

## Decision 106 - Treat Part 2 As Session-Reuse And Decode Requalification

Date: 2026-07-16

Evidence:

- the new public file contains 420 metadata records, 70 conversations and six
  turns per conversation;
- 90 requests are marked warmup and 330 are graded;
- every turn is about 4000 input tokens and 200 output tokens with 3000 ms
  think time;
- graded output volume is 2.75x Phase 1, while incremental input volume is
  only 54.5% of Phase 1;
- append-only replay without session reuse is about 4.784 million prompt
  tokens, versus about 1.320 million incremental tokens with ideal reuse;
- `docs/official/de_bai.md` still declares the old 120-request workload.

Decision:

- keep final contract/GPU authorization blocked;
- treat Transformer KV plus Mamba state continuity across turns as the first
  Part 2 architecture gate;
- prioritize FP8 weights/KV, BF16 Mamba SSM state, active cap 8-16, graph
  sizes 1-12, 4096-turn prefill and output-head retesting;
- retire six-wave and batch-20 assumptions as defaults;
- deprioritize Tree-Gated static routing because request shape is nearly
  uniform.

Status: active provisional direction pending official driver semantics.

## Decision 107 - Promote Part 2 FP8-KV Async B16384 Attempt 02

Date: 2026-07-17

Evidence:

- Portal Attempt 01 scored `56.53`, TTFT p50/p95 `73/116ms`, TBT median `4ms`
  and zero failures, exposing decode score as the remaining gap;
- FP8 E4M3 KV repeated above the BF16-KV control and passed paired local GPQA
  at `21/100` versus `21/100` with zero control-correct regressions;
- FP8 KV plus async scheduling repeated above 67 local ERS;
- b16384 scored `67.879/68.740` in two cold full traces with zero failures,
  versus the former b8192 incumbent mean `67.261`;
- seq10, throughput, E5M2, FP16 Mamba state, graph/compile overrides and higher
  local memory reservation failed their promotion gates.

Decision:

- promote seq8/interactivity/b16384 with FP8 E4M3 attention KV, BF16 Mamba
  state and async scheduling;
- keep the official vLLM 0.22.1 image, fixed entrypoint/model fields, context
  32768 and H200 memory `.95`;
- create Part 2 Attempt 02 without custom image, warmup or trace-specific code;
- preserve Attempt 01 as the target-H200 rollback until Attempt 02 is graded.

Status: package qualified and ready for Portal upload.

## 2026-07-17 - Part 2 Turn-Semantics Hedge

- Read `in_tokens_est=3997-4000` as complete per-request context for the
  primary candidate; a literal append-only reading grows to about 25k tokens.
- Retain prefix-enabled Attempt 02 because exact continuation replay obtains
  about 71.20% cache-hit tokens and collapses when caching is disabled.
- Promote a separate no-prefix Attempt 03 for fixed/rolling 4k prompts. Its
  full token-identical A/B scored `72.068` versus `68.458`, with zero failures.
- Do not overwrite either artifact; the Portal run resolves which hidden
  semantics transfers to H200.

## 2026-07-17 - Close Fixed-4k No-Prefix Scalar Tuning

- Keep Attempt 03 at seq8, b16384, chunked prefill and interactivity.
- Reject b20480 and throughput; both lose against the fixed control bracket.
- Reject no-chunk/b32768 as neutral and seq6 because its opening gain reverses
  on a cold repeat.
- Skip b24576 by the registered early-stop and create no new Portal artifact.

## 2026-07-17 - Reject Part 2 GPU N-Gram Speculative Decode

- Exact vLLM 0.22.1 startup resolved `ngram_gpu`, async scheduling and the
  target FP8 stack without eager fallback.
- FlashInfer downgraded the control's full-and-piecewise graph mode to
  piecewise for speculative decode.
- The diagnostic accepted `67/5820` draft tokens (`1.15%`) and accepted the
  first position on only `40/2910` steps (`1.37%`).
- Close `k3` because later positions cannot repair first-position misses.
- Close the `n1-3` fallback because the measured candidate already generated
  drafts at every eligible step; coverage was not the bottleneck.
- Keep Attempt 03 unchanged and create no speculative Portal artifact.

## 2026-07-18 - Resolve Part 2 Cache Policy From Portal

- Attempt 02 (prefix cache + Mamba `align`) scored `58.17`, TTFT p50/p95
  `58/82ms`, TBT median `4ms`.
- Attempt 03 (no prefix + Mamba `none`) scored `47.91`, TTFT p50/p95
  `98/201ms`, TBT median `4ms`.
- Retain the Attempt 02 cache/state policy and close the fixed-4k no-prefix
  hedge for the hidden driver.
- Do not treat the shared Portal `config_hash` as a full runtime fingerprint;
  the two repository artifacts differ in cache flags.
- Keep Attempt 01 as the zero-failure rollback because Attempt 02/03 reported
  `8/5` failures and unexpected `warmup_count=0`.

## 2026-07-18 - Retain Output-Only FP8 As A Quality-Blocked Ceiling

- Matched append-only full traces scored `56.729` native and `58.857` tensor
  FP8 without exact repair, a `+2.128` point gain with zero local failures.
- Tensor no-repair matched only `80/96` first tokens and `0/96` complete
  sequences against the deterministic native control.
- Row-wise top-k32 improved equivalence to `92/96` first tokens and `29/96`
  sequences but did not beat the latency branch or pass a conservative gate.
- Create no custom image or Compose from subround 002f. Reopen only with a
  better shortlist/residual method followed by paired GPQA.

## 2026-07-18 - Promote 002k Non-lossy Fused Greedy Head For Packaging

Evidence:

- Triton tile sweep passed exact token checks for M1, M2, M4 and M8;
- direct native-delegate audit matched `20/20` live batches, including M8;
- full 70-conversation replay improved ERS `56.7287 -> 58.3989` with
  `330/330` graded successes;
- TPOT p50/p95 improved `2.584/4.204 -> 2.477/3.649ms`, while TTFT p50
  improved `110.524 -> 106.503ms` and TTFT p95 moved `326.756 -> 336.066ms`.

Decision:

- retain native fallback for logprobs, grammar, penalties, speculative decode
  and any active/unknown logits processor;
- bake the overlay into a clean vLLM image and run a pull/startup smoke test;
- run the selected quality/GPQA procedure before choosing a Portal submission;
- keep the native b16384 artifact as rollback until the packaged candidate is
  independently verified.

## 2026-07-18 - Qualify Packaged 002k For An H200 Probe

Evidence:

- the clean image uses the exact vLLM `0.22.1` base digest and adds only the
  two audited overlay modules;
- independent startup captures PIECEWISE 16 and FULL 8 CUDA graphs without a
  source bind mount;
- strict-greedy serving activates the fused path, while logprob requests use
  native fallback; both smoke tests complete without failures;
- the packaged 100-case GPQA proxy retains `21/100` accuracy and changes zero
  correctness decisions versus the incumbent FP8-KV process.

Decision:

- promote image digest
  `sha256:10f819c0e765bb579b89caccac4de5f17984c2fd5342335130832dc7ad93f6f5`
  as the locally qualified R002k candidate;
- keep Attempt 02 as rollback because R002k's H200/MiG transfer is unmeasured;
- do not interpret four wrong-to-wrong cross-process prediction changes as a
  kernel mismatch; same-hidden-state native-delegate agreement remains the
  correctness criterion;
- require Docker Hub publication and a new immutable Compose before Portal
  submission rather than modifying an earlier attempt.

Publication result:

- Docker Hub returned the same audited digest
  `sha256:10f819c0e765bb579b89caccac4de5f17984c2fd5342335130832dc7ad93f6f5`;
- Part 2 Attempt 04 differs from Attempt 02 only in `image`, parses with Docker
  Compose, and is frozen at SHA-256
  `edeb74cfb5a5f5777e8c315e3f4a4b17294a2d96e8022b35d07ae09d51db0698`.

## 2026-07-18 - Close 002l Selective Final RMSNorm Standalone Candidate

Evidence:

- exact BF16 sampled rows and full output SHA match R002k;
- weighted isolated prefill improvement is `21.917%`, but the absolute saving
  is only `0.149ms`;
- full B/A/B scores are `55.8737 / 53.5857 / 52.7145` for candidate/control/
  candidate;
- candidate mean delta is `+0.7084` points and all six turn means are positive,
  but the second candidate loses and TPOT p95 rises to `4.568ms`;
- native logprob fallback passes `6/6` requests.

Decision:

- close R002l without creating any submission artifact from that branch;
- keep the phase-safe source primitive for a future larger prefill bundle;
- do not stack a sub-millisecond optimization onto R002k based on a noisy mean;
- keep Part 2 Attempt 04 immutable and wait for its H200 result.

## 2026-07-18 - Promote Attempt 04 As Part 2 H200 Incumbent

Evidence:

- the submitted digest is exactly the R002k package and predates R002l;
- Attempt 04 scores `60.75` versus Attempt 02 at `58.17`;
- TTFT p50/p95 improves `58/82 -> 44/80ms`;
- failures improve `8 -> 6`, while median TBT remains `4ms`;
- local same-hidden-state argmax and GPQA proxy gates already pass.

Decision:

- promote Attempt 04/R002k as the H200 performance incumbent;
- retain Attempt 02 as native-image rollback and Attempt 01 as zero-failure
  rollback;
- preserve R002k in future performance candidates unless running an explicit
  attribution probe;
- do not reinterpret Portal `accuracy_drop=0` as final GPQA evidence;
- keep R002l closed standalone despite this positive R002k transfer.

## 2026-07-18 - Close 002m Exact Decode-Head V2

Evidence:

- all candidates matched `240/240` native token IDs;
- direct-owned output gained only `0.011%` weighted GPU and `0.430%` wall time;
- even the unsafe reusable-output floor gained only `0.489%` GPU time;
- transposed BF16 lost `1.299%` GPU time while consuming `256 MiB` extra VRAM;
- manual GEMV lost about `154%` weighted GPU time because it reread the tied
  head independently for each active row.

Decision:

- stop before server integration and full trace;
- create no image, Compose or Portal attempt;
- do not implement an event-aware output ring because its theoretical ceiling
  is already below the opening gate;
- preserve R002k and move future custom work to a measured in-model hotspot.

## 2026-07-18 - Adopt Attempt 04 Exact-A/A Variance Envelope

Evidence:

- the immutable Compose and image scored `60.75` and `58.94` on two H200 runs;
- TTFT p50 moved `44 -> 55ms`, while p95 stayed `80 -> 79ms`;
- TBT median remained `4ms`, and failures moved `6 -> 7`;
- both runs report the same config hash, total count, warmup count and online
  accuracy fields.

Decision:

- retain `60.75` as the leaderboard best and use `59.845` as the two-run
  expected-value anchor;
- record `1.81` ERS, `11ms` TTFT p50 and one failure as the minimum observed
  same-artifact variance envelope, not a confidence interval;
- do not attribute future one-off score movements inside this envelope;
- keep R002k as incumbent because both observations remain above the single
  native-image Attempt 02 score, while acknowledging that its mean gain is
  `+1.675`, not a deterministic `+2.58`;
- spend future Portal attempts only on locally repeated candidates with a
  projected effect clearly above two points or a new tail/failure mechanism.

## 2026-07-18 - Retain R002n In-Projection FP8 As Research-Only

Evidence:

- Clean Kineto profiles identify ShortConv projections as `19.499%` of decode
  child-kernel time; the source path is active and the opportunity gate is
  met.
- `out_proj` is slower in isolation. The all-projection variant improves the
  profile but loses the local GPQA proxy (`18/100` versus `21/100`) and full
  replay ERS (`58.2706` versus the R002k anchor `58.3989`).
- In-only FP8 retains proxy accuracy (`22/100` twice, zero
  reference-correct-to-candidate-wrong transitions) and scores `58.9590` and
  `59.4062` in two full replays. The immediate control/candidate delta is
  `+2.1084` points, while TPOT p50 improves to about `2.41ms`.
- The control itself moves to `57.2978`, candidate runs each miss the 400ms
  TTFT SLO once, and only `268/330` first tokens match the paired control.
  Local `.95` startup is unavailable because display VRAM leaves only
  `13.89/15.44 GiB` free.

Decision:

- Keep R002k and its published digest as the only submission-ready artifact.
- Do not build an image or create a Portal Compose for R002n yet.
- Retain `in_proj` as the sole next candidate for a clean-target transfer
  test; close `out_proj` and all-projection FP8.
- Require another paired replay plus the existing quality/token-drift gates
  before any promotion. Do not interpret the positive local ERS as a stable
  H200 estimate.

The attempted reversed-order repeat is excluded: the display-contaminated
local GPU OOMed during a 20.8k-token CUDA-Graph shape after six requests.
This reinforces the requirement that the next attribution run use a clean GPU
or explicitly lower-memory diagnostic conditions.

## 2026-07-18 - Close 002o Phase-Aware ShortConv

Evidence:

- shared checkpoint loading and native-BF16/FP8-decode storage separation pass
  for all ten ShortConv layers;
- real prefill/decode routing and both required CUDA Graph modes pass;
- all-layer decode profiling improves non-output-head child-kernel time by
  `6.72%`, while the two-layer quality-neutral mask improves only `0.845%`;
- the all-layer quality proxy remains `21/100` twice with zero correctness
  regressions;
- exact candidate replays score `53.2023` and `52.9304` versus control
  `55.4200`, with worse request-level tails despite zero HTTP failures.

Decision:

- close both all-layer and selective phase-aware variants;
- build no image and create no Compose or Portal attempt;
- retain the shared-load implementation only as a source reference;
- keep R002k as the submission incumbent;
- require end-to-end evidence before promoting profiler-only kernel savings.

## 2026-07-19 - Close 002p Static FP8 Activation Scaling

Evidence:

- dynamic activation quantization is only `4.6195%` of clean decode
  child-kernel time;
- force-static removes all 44 dynamic producers but improves total batch-8
  child-kernel time by only `1.2296%` in a low-drift A/B/A bracket;
- row and Python scheduler-state routing cannot replace an already captured
  dynamic CUDA Graph;
- prefill-only calibration scores `20/100` and `18/100`, while decode-aware
  calibration scores `19/100` and `18/100`, versus R002k `21/100`.

Decision:

- retain native dynamic per-token activation quantization;
- stop before full replay because the registered quality gate failed;
- create no image, Compose or Portal attempt;
- do not spend more runs on selective static subsets whose whole-step ceiling
  falls below the opening gate;
- move future FP8 work to graph-native dynamic fusion/backend changes and keep
  R002k immutable.

## 2026-07-19 - Close R002q Graph-Native Fusion

Evidence:

- `fuse_norm_quant` is a real LFM2.5 rewrite, but only five patterns match and
  the complete decode A/B/A gain is `0.597%`;
- `fuse_act_quant` matches zero partitions;
- `fuse_attn_quant` matches zero partitions and changes the graph policy from
  `FULL_AND_PIECEWISE` to `FULL`;
- the norm candidate retains 44 FP8 GEMMs and passes startup/request safety,
  but the complete-step gain is below the registered 1% promotion gate.

Decision:

- close R002q before quality, replay, image and Portal work;
- do not carry any fusion flag into R002k or a submission;
- retain the source fact that five norm patterns exist for a future runtime
  version comparison;
- open the next research branch on FP8 linear backend/GEMM dispatch, which is
  the largest measured decode family.

## 2026-07-19 - Close R002r FP8 Linear Backend Screen

Evidence:

- auto and explicit CUTLASS select the same kernel and pass all startup checks;
- FlashInfer cannot implement dynamic per-token activation scales and is not
  available on target SM90; Marlin is blocked on native-FP8 GPUs;
- Torch is the only startup-valid portable alternate, but its complete
  batch-8 gain is `-0.0396%` with only `0.1555%` control drift;
- Torch's local raw GEMM saving is canceled by the channel-wise fallback's
  FP32 scale/dequant/output-conversion work.

Decision:

- retain automatic/CUTLASS dispatch in R002k;
- close backend-name tuning for the current FP8 representation;
- do not force Marlin, add `--linear-backend=torch`, or convert solely to
  unlock a local-only FlashInfer path;
- stop before quality/replay and create no image, Compose or Portal attempt;
- require a representation change or a kernel that natively consumes dynamic
  per-token activation scales for the next linear-path experiment.

## 2026-07-19 - Close R002s Online Block FP8

Evidence:

- `fp8_per_block` is a valid online representation and all LFM2.5 shapes meet
  the exact block-kernel alignment constraints;
- H200 SM90 has a source-proven FlashInfer/DeepGEMM route, while local SM120
  correctly falls back to CUTLASS;
- local complete decode loses `4.2785%` with `0.1632%` control drift;
- repeated quality falls from R002k `21/100` to `18/100`, including four
  stable reference-correct losses;
- the diagnostic full replay completes all requests but loses `9.0407` ERS
  points and builds a peak-active backlog of 11.

Decision:

- retain `fp8_per_tensor`, FP8 E4M3 KV, BF16 aligned Mamba state and R002k;
- do not create an H200 risk-trial Compose from source portability alone;
- do not use the local CUTLASS loss as an H200 speed estimate, but treat the
  quality loss as representation-level blocking evidence;
- reopen only with a concrete quality-preserving calibration/scale method and
  a target-kernel measurement path;
- create no image, Compose or Portal artifact for R002s.

## 2026-07-19 - Close R002t Small-M FP8 Kernel

Evidence:

- Torch's raw GEMM advantage is only `0.026708ms/step`, an absolute
  zero-overhead ceiling of `1.2119%`;
- more than `83.07%` of measured fallback overhead must disappear to clear the
  registered 1% whole-step gate;
- direct BF16 scaled-MM is bit-exact after row-wise scale expansion but loses
  `35.51-132.98%` end to end;
- the custom Triton epilogue is bit-exact, but at M=8 it costs `0.320015ms`
  against a `0.004669ms` budget and leaves the complete path `36.34%` slower;
- production consumer fusion invalidates the apparently favorable isolated
  Torch-epilogue comparison.

Decision:

- retain automatic/CUTLASS online per-tensor FP8 and immutable R002k;
- close both direct row-wise scaling and separately launched custom epilogues;
- skip startup, quality, replay, image, Compose and Portal work;
- require one native fused GEMM primitive, not another standalone epilogue,
  before reopening this representation;
- move the next research branch away from FP8 linear backend dispatch.

## 2026-07-19 - Close R002u BF16 ShortConv Kernel

Evidence:

- twenty native BF16 ShortConv projections are a real `19.7198%` batch-8
  decode family and need only `5.0710%` internal gain for +1% complete decode;
- a single-weight Triton screen appears `11.31-20.50%` faster, but repeats only
  one cache-resident matrix and is not a valid model-layer benchmark;
- the registered ten-layer sequence cycles exactly 320 MiB, and its M8 native
  control matches the production projection anchor within `4.98%`;
- best M8 Triton gains only `2.256%` versus that immediate control, remains
  slower than the production anchor and projects `-0.516%` complete-step gain;
- numerical output passes, so performance is the sole rejection reason.

Decision:

- keep native BF16 ShortConv projections in R002k;
- close standalone Torch/Triton GEMM replacement before live integration;
- require cyclic distinct-weight benchmarks whenever a model family exceeds
  cache capacity;
- reopen ShortConv only for projection+consumer fusion or direct H200 evidence
  that changes the measured memory-traffic ceiling;
- create no quality run, replay, image, Compose or Portal artifact.

## 2026-07-19 - Promote R002v as H200 Target Probe

Evidence:

- two host-overlay candidates beat a matched control by `+3.4976/+1.1956`;
- two allocator-qualified packaged candidates beat packaged control by
  `+3.0660/+3.6943`, with `330/330` graded successes each;
- candidate TPOT improves on at least `284/330` paired requests;
- expandable allocator segments eliminate the repeatable long-prefill OOM;
- packaged GPQA remains `21/100` with aggregate drop zero, but one
  reference-correct case flips to wrong.

Decision:

- publish R002v under a new immutable digest and create Part 2 Attempt 05;
- preserve R002k Attempt 04 as the custom-image rollback and Attempt 02 as the
  native-image rollback;
- bake and declare `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
- treat Attempt 05 as an H200 performance/quality probe, not an incumbent,
  until its Portal panel and final selected quality result are available;
- do not combine R002v with rejected phase-aware or output-projection variants.

## 2026-07-19 - Repeat Attempt 05 For Variance Attribution

Attempt 05 first Portal panel:

- ERS `60.19`, TTFT p50/p95 `50/90ms`, TBT median `4ms`;
- `6/420` failed requests, `warmup_count=0`, accuracy drop `0`;
- only `+0.345` over the R002k exact-artifact mean `59.845`, within its
  observed `1.81`-point A/A range.

Decision:

- submit the byte-identical repeat artifact once;
- change no image, flags, allocator setting or model contract;
- treat the next panel as variance measurement, not a new optimization;
- retain R002k Attempt 04 as rollback until the repeat and selected quality
  result are known.

## 2026-07-19 - Close R002v After Exact-Digest Pair

Evidence:

- Attempt 05 and its byte-identical repeat scored `60.19` and `59.34`;
- the target pair mean/range is `59.765/0.85`, versus R002k's exact-pair
  `59.845/1.81`;
- TBT median stays `4ms`, accuracy drop stays `0`, and failures move only
  `6 -> 5` on the repeat;
- R002v TTFT means (`55.5/93.5ms`) are worse than R002k (`49.5/79.5ms`).

Decision:

- close the isolated ShortConv `in_proj` FP8 overlay as a neutral target
  probe, not a promoted optimization;
- keep R002k as the H200 incumbent and immutable rollback;
- retain `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as a packaging
  reliability fix, independent of the rejected precision claim;
- do not spend another Portal submission on the same digest;
- require the next branch to attack a distinct decode primitive or fused
  projection/consumer path and to report the per-request score components,
  failure cohort and TPOT tail, not just median TBT.

## 2026-07-19 - Close R002w ShortConv Decode Epilogue Fusion

Evidence:

- exact runtime-layout output and recurrent state are bit-exact through 16
  updates at M1/M2/M4/M8;
- ten M8 epilogues improve `0.285780 -> 0.091605ms`, and repeated live median
  decode throughput improves `2.86%`;
- all ten layers capture and replay under FULL_AND_PIECEWISE with only
  `320 KiB` persistent staging and native prefill/mixed fallback;
- candidate/control/candidate full ERS is
  `50.5693/50.6229/50.1385`; candidate mean delta is `-0.2689` and neither
  candidate run is positive.

Decision:

- close R002w at the repeated full-replay gate;
- keep R002k as H200 incumbent and create no public image, Compose or Portal
  attempt from R002w;
- skip GPQA by dependency because the optimization is BF16 bit-exact locally
  but already performance-negative end to end;
- retain the kernel only as a component for a future larger
  projection/state/projection fusion with a new complete-step proof;
- do not infer submission value from isolated microkernel or steady-decode
  throughput alone.

## 2026-07-19 - Close R002x Production Graph-Step Attribution

Evidence:

- twelve valid M1/M2/M4/M8 panels each contain `64/64` pure-decode steps,
  zero failures and matched FULL_AND_PIECEWISE capture;
- exact full-replay telemetry shows `98.744%` pure-decode active steps and
  `99.401%` eligible decode rows, rejecting mixed-fallback attribution;
- weighted complete GPU gain is `3.501%`, but M4 gain (`0.851%`) equals
  control drift and M8 regresses `1.687%`;
- direct M8 ShortConv saving is `0.039981ms/step`, `4.86x` below the eager
  estimate, and R002w's prior complete replay remains `-0.2689` ERS.

Decision:

- enforce the pre-registered M8-positive gate and close R002x;
- do not implement projection/state/projection fusion from this evidence;
- create no GPQA run, image, Compose or Portal attempt;
- keep R002k Attempt 04 as H200 incumbent and R002w source as research-only;
- require complete production graph-step attribution before future eager
  kernel gains can open integration;
- move the next score-focused profile to cached-turn prefill/first-token work,
  not another standalone ShortConv decode scalar.

## 2026-07-19 - Complete R002y Cached-Turn Prefill Attribution

Evidence:

- all six turn shapes and cache progression are valid after all 90 primer
  requests, with weighted control drift `1.418%`;
- newly scheduled prefill stays near `4.2k` tokens, yet complete GPU service
  grows `60.801 -> 114.685ms`;
- attention grows `4.825 -> 54.742ms` and explains `92.64%` of the turn-depth
  growth; FP8 and BF16 GEMM costs remain approximately flat;
- the local dominant kernel is FlashInfer, but exact source proves H200 SM90
  auto-selects packaged FA3 for FP8 KV;
- TRT-LLM attention requires SM100, and the declared prefill/decode split flag
  has no runtime call site in this image.

Decision:

- accept the causal attribution but do not promote a local kernel;
- keep R002k, its decode policy and Attempt 04 immutable;
- create no quality run, image, Compose or Portal attempt from R002y;
- open only H200-specific FA3 prefill qualification, with target execution and
  decode-isolation proofs required before implementation;
- retain the aligned second-step `2.296ms` ideal ceiling as secondary research,
  not a serving claim.

## 2026-07-19 - Complete Part 2 R003 Hybrid Attention Window

Evidence:

- an import-verified `8192` sliding window reaches exactly the six LFM2.5
  full-attention layers and leaves all ten ShortConv layers and R002k decode
  hooks unchanged;
- the append-only full bracket completes `420/420` in all four runs;
  candidate mean ERS is `61.1708` versus control mean `56.5450`, with TTFT
  p50/p95 `95.35/269.89ms` versus `114.26/315.56ms` and TPOT p95
  `3.661ms` versus `4.159ms`;
- the score gain grows by turn depth and the independent 4k boundary check is
  slightly negative (`67.9362` versus `68.3485`), so the effect is tied to
  retained long history;
- the short GPQA proxy has no candidate-specific drop, but it does not test
  long-context reasoning and the candidate allocates a larger KV pool.

Decision:

- record R003/8192 as the current local research winner for the inferred
  closed-loop workload;
- keep R002k as the submission incumbent and do not publish or submit R003;
- require a window-size sweet-spot screen, long-context quality evidence and
  an explicit rules/driver interpretation before promotion;
- treat the result as a composite bounded-attention plus cache-capacity gain,
  not as a portable H200 FA3 kernel claim.

## 2026-07-19 - Complete R003a Attention-Window Frontier

Evidence:

- repeated `4096` full replay scores `64.5121/64.5886`, mean `64.5504`,
  versus repeated unbounded control mean `56.5450`;
- `4096` lowers TTFT p50/p95 by `29.78/81.27ms`, lowers TPOT p95 by
  `0.818ms`, completes `330/330` graded requests in both runs and has only a
  `0.0766`-point score range;
- its gain grows from `+1.180` at turn 0 to `+13.514` at turn 5, consistent
  with bounded retained-history attention;
- paired retrieval loses `2/14` control-pass cases at `4096`, `1/14` at
  `6144`, and none at `8192/12288`; short GPQA never activates a window;
- `12288` scores only `57.0213`, while `16384` fails the profile saving gate.

Decision:

- keep `4096` as the local score-ceiling research branch, explicitly not
  quality-safe;
- retain `8192` as the semantics-balanced research branch;
- close `6144`, `12288` and `16384` from carry-forward;
- keep R002k as the Portal incumbent and create no image/Compose until BTC
  long-context construction, rules acceptability and H200 behavior are
  resolved.

## 2026-07-19 - Close R003b Selective Global Placement

Evidence:

- `global14`, `global2`, `global8` and `global2_14` all pass the isolated
  performance gate, with turn-5 savings from `32.443` to `37.932ms`;
- every candidate completes all 14 retrieval requests but loses the same two
  control-pass cases at actual prompt lengths `5084` and `7109` tokens;
- no placement gains a retrieval case, so the global-layer repair hypothesis
  has no quality support.

Decision:

- close R003b before full replay;
- keep R002k as the submission incumbent;
- do not package a selective-global overlay;
- require a retention pattern that preserves selected old information in each
  affected layer before reopening bounded-attention score work.

## Decision 016 - Retain R003d Safe8192 As Research Only

Reason:

- A tiered horizon map with `4096` on full-attention layers 2 and 5 and
  `8192` on layers 8, 10, 12 and 14 passed the paired retrieval gate twice.
- Two full replays averaged `61.8876` versus `56.5450` for two unbounded
  controls, with lower TTFT p50/p95 and TPOT p95 and zero graded failures.
- More aggressive maps lost control-pass retrieval cases, while the short GPQA
  proxy is too short to validate long-context semantics.
- The score gain also includes increased KV capacity at the same memory
  fraction, so it cannot be described as an isolated attention-kernel gain.

Decision:

- Keep `safe8192` as the balanced local research reference.
- Close `mid6144` and `front3` and close further scalar horizon tuning.
- Do not merge the map into R002k or create a Docker/Compose/Portal artifact.
- Reopen only with request-local prefix-plus-recent retention or a target-
  qualified attention implementation, plus a real long-context quality gate.

Status: accepted as research evidence; serving promotion withheld.

## Decision 017 - Close Native Triton Prefix-Range Retention

Evidence:

- a no-prefix Triton 4096-window control saves `39.512ms` at turn 5 versus
  the unbounded control;
- native prefixes `64/128/256` instead lose `72.365/73.128/72.324ms` at turn
  5 versus unbounded, despite exact metadata delivery and matched CUDA Graph;
- prefix size has negligible effect, indicating that the non-empty disjoint
  range path, not the number of retained prefix tokens, owns the regression.

Decision:

- close R003e before retrieval and full replay;
- retain `safe8192` as the balanced research reference and R002k as Portal
  incumbent;
- do not sweep more native prefix sizes or package this overlay;
- reopen only with a split-range kernel or equivalent path that avoids work
  over the discarded middle span and first passes the same profile gate.
