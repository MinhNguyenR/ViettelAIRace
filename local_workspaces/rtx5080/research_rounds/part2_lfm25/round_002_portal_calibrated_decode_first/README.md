# Round 002 - Portal-Calibrated Decode First

Status: core round and subrounds 002a-002z complete; subround 002aa is active
and waiting for an idle-GPU qualification window. Portal Attempt 02 confirms
the prefix-enabled append-only branch; Attempt 03 rejects the fixed-size/
no-prefix hedge on H200. The output-only FP8 head is latency-positive but
blocked by token drift.

## Goal

Reduce TPOT under the Part 2 closed-loop workload while spending only the TTFT
headroom that increases total ERS. The current local random-arrival proxy
transfers TTFT but understates Portal decode interference.

## Score-Slope Rule

At TTFT `73ms` and TPOT `4ms`, a median-only score proxy is `57.37`. Around
that point, reducing TPOT by `1ms` is worth approximately the same score as
increasing TTFT by `34ms`. Candidate selection therefore uses total per-request
ERS, not either latency metric alone.

## Evidence Boundary

The public trace contains arrival, turn, token-count and warmup metadata but no
prompt body. The local driver reproduces the closed-loop lifecycle and exact
token counts with deterministic synthetic token IDs. It measures scheduler and
GPU behavior; it does not claim to reproduce BTC tokenizer cost or hidden
prefix content.

## Final Selection

```text
weights:                  fp8_per_tensor
attention KV:             fp8_e4m3
Mamba state:              bfloat16, align
scheduler:                async, interactivity
max sequences:            8
max batched tokens:       16384
prefix/chunked prefill:    enabled
```

The full evidence and rejected branches are recorded in `results.md`.

Subround 002k adds a strict Triton fused BF16 greedy output head. It passed
same-hidden-state argmax audits and improved the full local replay from ERS
`56.7287` to `58.3989` with zero graded failures. Clean image startup,
strict-greedy activation, native logprob fallback and the 100-case GPQA proxy
all pass. Portal Attempt 04 scored ERS `60.75`, TTFT p50/p95 `44/80ms`, median
TBT `4ms` and six failures. This beats Attempt 02 by `+2.58` points and makes
R002k the H200 incumbent; Attempt 02 is the native-image rollback.

An exact-digest repeat scored `58.94`, TTFT `55/79ms`, TBT `4ms` and seven
failures. R002k's two-run H200 mean is `59.845` and observed range is `1.81`;
the implementation remains incumbent, while the range becomes the minimum
Portal attribution envelope.

Subround 002l then moved final LFM2 RMSNorm after sampled-row selection only
for prefills of at least 2048 rows. The path is bit-exact and saves `21.917%`
of that isolated operation, but only `0.149ms` absolute. Its B/A/B full-trace
mean is `+0.7084` points over the middle R002k control, while the second run
loses and violates the TPOT guard. It is closed as a standalone submission;
R002k and Attempt 04 remain unchanged.

Subround 002m re-opened only the remaining exact output-head overhead. A
nine-sample/100-iteration confirmation found direct-owned output effectively
neutral (`+0.011%` GPU), a no-clone lower bound only `+0.489%`, transposed BF16
storage negative despite `256 MiB` extra VRAM, and manual GEMV about `154%`
slower when weighted across batches. No candidate opened live testing; R002k
remains the complete output-head implementation.

Subround 002n profiled steady decode child kernels and found that the ten
ShortConv layers leave their two projection matrices on BF16 despite the
active FP8 linear quantization config. An opt-in FP8 replacement was screened
per projection. `out_proj` was negative; replacing both projections reduced
profiled kernel time but lost the local quality proxy and full-replay ERS.
`in_proj` alone retained GPQA proxy accuracy and produced two positive local
full replays (`58.9590` and `59.4062`), but it also changed generated tokens,
missed the 400ms TTFT SLO once per run and has not been qualified at clean
`.95` memory utilization. R002v later isolates, qualifies and packages this
single retained candidate; the broader R002n variants remain closed.

Subround 002o isolated the R002n idea by keeping native BF16 ShortConv
projection for every iteration containing prefill and selecting a separately
quantized FP8 copy only for decode-only iterations. Shared checkpoint loading,
native/FP8 storage separation, PIECEWISE/FULL CUDA Graph capture and a real
prefill-to-decode transition all passed. The all-layer path reduced profiled
non-output-head child-kernel time by `6.72%` and matched the R002k quality proxy
at `21/100`, but exact full replays scored `53.2023` and `52.9304` versus the
immediate R002k control at `55.4200`. A two-layer mask saved only `0.845%` in
the profiler. Both variants are closed; R002k remains incumbent.

Subround 002p measured the 44 existing FP8 linear calls. Fused dynamic
activation quantization is `4.62%` of clean decode child-kernel time. Static
scaling removed those producers but improved total batch-8 child-kernel time
by only `1.2296%`. Python phase routing could not alter an already captured
CUDA Graph, and all-graph static candidates failed repeated quality
(`20/18` and `19/18` versus R002k `21/100`). The branch is closed before full
replay or packaging; R002k remains incumbent.

Subround 002q audited vLLM's graph-native FP8 fusion controls on the exact
LFM2.5 image. `fuse_norm_quant` replaced five graph patterns and reduced
dynamic quant producers from `44` to `22` per decode step, but the complete
batch-8 A/B/A gain was only `0.597%`. `fuse_act_quant` matched zero patterns;
norm+act added nothing; `fuse_attn_quant` matched zero patterns and changed
the graph to FULL. The branch is closed before quality/replay and R002k stays
the incumbent.

Subround 002r screened every exact-image FP8 linear backend. Auto already
selects CUTLASS. FlashInfer cannot consume the dynamic per-token activation
scale and is also unavailable on H200 SM90; Marlin is blocked on native-FP8
GPUs; Triton/DeepGEMM have no matching per-tensor-online class. The only
portable alternate, Torch channel-wise fallback, measured `2.2047690ms/step`
against a `2.2038956ms` CUTLASS mean (`-0.0396%` gain) with `0.1555%` control
drift. It is closed before quality/replay; R002k remains unchanged.

Subround 002s changed the online representation to static 128x128 block-FP8
weights and dynamic 1x128 activation groups. Exact source proves an H200 SM90
FlashInfer/DeepGEMM path, while local SM120 falls back to CUTLASS. The local
fallback lost `4.2785%` in a low-drift decode profile, repeated quality fell
from R002k `21/100` to `18/100`, and a diagnostic full replay scored `49.3582`
versus `58.3989`. It is closed without packaging; target-specific dispatch
does not override the failed numerical gate.

Subround 002t then kept R002k's exact per-tensor representation and tested
whether Torch's `0.026708ms/step` raw-GEMM advantage could fund a custom
small-M output path. Direct BF16 `_scaled_mm` required expanded row-wise
weight scales and lost `35.51-132.98%` end to end. A bit-exact Triton
dequantize/cast epilogue consumed `0.320015ms` per 44 M=8 calls, `68.53x` its
live +1% budget, and made the complete path `36.34%` slower than CUTLASS. It
is closed before integration; R002k remains unchanged.

Subround 002u preserved BF16 and screened the remaining 20 ShortConv
projection calls. A single-weight loop falsely projected `2.23-4.04%`
complete-step gains because the matrix stayed L2-hot. The registered
ten-layer confirmation cycled 320 MiB of weights: M8 native matched the live
profile within `4.98%`, while best Triton gained only `2.256%` versus that
control and remained slower than the production anchor. It is closed before
integration.

Subround 002v returned to the only positive R002n factor: online per-tensor
FP8 for all ten ShortConv `in_proj` matrices while retaining BF16 `out_proj`.
Two host-overlay runs beat control by `+3.4976/+1.1956` ERS. After an initial
fragmentation OOM, expandable CUDA allocator segments produced two clean
packaged `420/420` replays at `52.6084/53.2367` versus packaged control
`49.5424`, a mean gain of `+3.3802`. GPQA remains `21/100` with zero aggregate
drop but one correct-to-wrong and one wrong-to-correct movement. The H200
Attempt 05 pair scores `60.19/59.34`, with mean `59.765`, versus R002k's
exact-pair mean `59.845`; TBT remains `4ms`. The overlay is neutral on target;
R002k remains rollback and incumbent.

Subround 002w then fused the decode-only BF16 `B*x`, width-three recurrent
state update and `C*conv` staging path for all ten ShortConv layers. The exact
runtime-layout kernel is bit-exact over 16 histories, reduces the isolated M8
ten-layer epilogue from `0.285780ms` to `0.091605ms` and improves repeated
batch-8 steady decode median by `2.86%`. FULL_AND_PIECEWISE graph capture and
native prefill/mixed fallbacks pass. Complete candidate/control/candidate
replays score `50.5693/50.6229/50.1385`; candidate mean delta is `-0.2689`.
The branch is closed without publication or submission and R002k remains the
incumbent.

Subround 002x then measured R002w at the complete production CUDA Graph step
instead of projecting eager microbenchmarks. Exact scheduler telemetry shows
`98.744%` of active steps are pure decode, so fallback is not the cause of the
full-replay loss. The fusion removes about 30 kernels per step and saves only
`0.039981ms` directly at M8, `4.86x` less than the eager estimate. M4 gain is
equal to control drift and M8 complete GPU time regresses `1.687%`; the
registered gate fails. No larger ShortConv fusion or serving artifact opens.

Subround 002y moved to cached-turn prefill/first-token attribution. Across a
clean A/Kineto/B bracket, cache reuse rises to `83.16%` and newly scheduled
prefill remains near `4.2k` tokens, yet complete GPU time rises
`60.801 -> 114.685ms`. Attention accounts for `92.64%` of that growth. The
local kernel is FlashInfer, while exact source proves H200 auto-selects
packaged FA3 for FP8 KV. This closes local kernel promotion, preserves R002k,
and opens only target-specific FA3 qualification.

Subround 002z completed that target-specific boundary as a source-only audit.
The exact image already enables FA3 and its AOT schedule on H200 SM90 with FP8
KV; no safe scalar exposes a better split policy, and TRT-LLM attention needs
SM100. It is closed without GPU, image or Portal work.

Subround 002aa attacks a different R002y finding: align-mode cached turns run
an extra `2-14` token full-model step to preserve the final ShortConv block
boundary. A sparse dual-checkpoint overlay now stores the boundary and final
state in one forward, retains about `1.59M` local cache tokens and changes
cached turns from two model steps to one. Pad32 failed full replay. No-pad won
two strictly clean matched pairs by `+0.2348/+1.5078` points and scored
`22/100` twice versus R002k `21/100`, with no reference-correct answer lost.
The baked package completes `420/420` after removing an accidental profiler
dependency. R002aa is a local H200 probe; R002k is still the incumbent.
Its immutable registry digest is pinned in Part 2 Attempt 06; no target result
has been attributed yet.

## Files

- `round_plan.md`: ordered gates and promotion rules.
- `configs/`: exact local vLLM argument sets.
- `scripts/run_closed_loop.py`: metadata-driven 70-conversation client.
- `scripts/run_closed_loop.sh`: reproducible Docker client wrapper.
- `results.md`: measured answers, rejected branches and final decision.
- `tests/`: deterministic driver/scorer tests.
- `portal_submissions/part2_attempt_02_lfm25_fp8pt_kvfp8_async_interactive_seq8_b16384.yml`:
  immutable upload artifact in the repository root submission folder.
- `portal_submissions/part2_attempt_06_lfm25_r002aa_shortconv_sparse_nopad_seq8_b16384.yml`:
  one-variable R002aa H200 probe, pinned to the qualified registry digest.
- `subrounds/subround_002a_turn_semantics_bracket/`: completed causal
  append-only versus fixed-size cache-policy bracket.
- `subrounds/subround_002b_fixed4k_no_prefix_tuning/`: active tuning of the
  promoted fixed-size/no-prefix branch.
- `subrounds/subround_002d_fp8_tied_lm_head/`: output-only tied-head FP8
  implementation and fixed-4k qualification.
- `subrounds/subround_002e_prefill_admission/`: scheduler admission flag
  qualification; relaxed reservation lost and concurrent partial prefill is
  unsupported in vLLM 0.22.1.
- `subrounds/subround_002f_append_only_head_integration/`: matched append-only
  integration; no-repair FP8 gains `+2.128` points but fails output-equivalence.
- `subrounds/subround_002g_hadamard_fp8_head/`: pure Hadamard output-head
  transforms; quality remained insufficient.
- `subrounds/subround_002h_signed_hadamard_fp8_head/`: deterministic signed
  Hadamard transforms; closed for weak equivalence.
- `subrounds/subround_002i_residual_fp8_head/`: low-rank residual repair;
  closed after weak full-sequence agreement.
- `subrounds/subround_002j_candidate_coverage/`: tensor top-k512/1024 coverage;
  closed without a quality-safe promotion.
- `subrounds/subround_002k_fused_bf16_greedy_head/`: non-lossy Triton fused
  greedy head; local full replay ERS `58.3989`, packaged and quality-qualified.
- `subrounds/subround_002l_selective_final_rmsnorm/`: exact prefill-only final
  RMSNorm deferral; source and correctness gates pass, end-to-end repeatability
  and TPOT guards fail, so no package is created.
- `subrounds/subround_002m_decode_head_v2/`: exact output ownership, transpose
  and GEMV follow-up; all candidates fail the microbenchmark opening gate.
- `subrounds/subround_002n_decode_hotspot_profile/`: Kineto batch-1/2/4/8
  profiling and ShortConv projection FP8 ablation; `in_proj` is retained for
  clean-target transfer testing, with no package created.
- `subrounds/subround_002o_phase_aware_short_conv/`: dual BF16-prefill/
  FP8-decode ShortConv projection path; implementation and profile gates pass,
  but repeated full replay loses more than two ERS points, so it is closed.
- `subrounds/subround_002p_fp8_activation_scale_audit/`: exact FP8 activation
  quantization attribution, static-scale microbench/integration and repeated
  quality rejection; no full replay or package was created.
- `subrounds/subround_002q_graph_native_fp8_fusion_audit/`: exact-image source,
  startup and graph-native fusion screen; norm fusion matched five patterns
  but gained only `0.597%`, while activation/attention paths were rejected.
- `subrounds/subround_002r_fp8_linear_backend_screen/`: exact registry,
  startup and A/B/A backend screen; CUTLASS remains fastest/most compatible
  for the active dynamic per-token activation layout.
- `subrounds/subround_002s_online_block_fp8_target_transfer/`: online block-FP8
  SM90 transfer audit; source/startup pass, but local profile, repeated quality
  and full replay reject the candidate before packaging.
- `subrounds/subround_002t_small_m_fp8_kernel_feasibility/`: exact per-tensor
  FP8 small-M primitive and Triton-epilogue audit; numerical gates pass but
  direct scaling and custom epilogue both lose decisively to CUTLASS.
- `subrounds/subround_002u_bf16_shortconv_kernel_screen/`: completed exact
  BF16 primitive screen; hot-cache gains disappear under the 320 MiB
  ten-layer sequence and no candidate clears the complete-step gate.
- `subrounds/subround_002v_shortconv_in_fp8_target_qualification/`: completed
  isolated ShortConv `in_proj` FP8 qualification, allocator stabilization,
  repeated packaged replay, quality proxy and immutable Attempt 05 package.
- `subrounds/subround_002w_shortconv_decode_epilogue_fusion/`: completed
  graph-safe decode epilogue/state fusion; micro/live decode gates pass,
  repeated full ERS fails, and no submission artifact exists.
- `subrounds/subround_002x_production_graph_step_attribution/`: complete
  M1/M2/M4/M8 CUDA Event and Kineto A/B/A attribution with exact full-replay
  scheduler weights; M8 reverses and closes standalone ShortConv fusion.
- `subrounds/subround_002y_cached_turn_prefill_attribution/`: complete six-turn
  cached-prefill CUDA Event/Kineto attribution. Attention explains `92.64%` of
  the TTFT GPU growth; the SM120/SM90 backend mismatch blocks local promotion.
- `subrounds/subround_002z_h200_fa3_boundary/`: complete exact-image H200 FA3
  source boundary; automatic dispatch already activates the safe mechanisms,
  so the branch closes without a candidate.
- `subrounds/subround_002aa_lfm2_shortconv_state_checkpoint/`: locally
  qualified sparse ShortConv no-pad package; target validation remains pending
  and R002k stays rollback.
