# Subround 002x - Production Graph-Step Attribution

Status: complete; opening gate failed at M8 and the standalone ShortConv
fusion branch is closed.

## Question

R002w is bit-exact and removes about `0.194ms` from an eager ten-layer M8
ShortConv epilogue sequence. It also gains `2.86%` median steady decode
throughput, yet its full candidate mean loses `0.2689` ERS. R002x determines
where that apparent gain disappears before any larger kernel is attempted.

## Frozen Comparison

- Control: R002k image
  `sha256:10f819c0e765bb579b89caccac4de5f17984c2fd5342335130832dc7ad93f6f5`.
- Candidate: local R002w image
  `sha256:07739966bfd124acf2dc4d7baabaf98cf97a5d33dbd713ba05637602fb795a1e`.
- Model/runtime/flags: exact Part 2 R002k contract, including FP8 model/KV,
  BF16 aligned Mamba state, seq8, b16384 and FULL_AND_PIECEWISE graphs.
- Local envelope: RTX 5080, 3 CPU cores, 8 GiB RAM, profile-only memory
  fraction `.80`; candidate and both controls use the same value.
- No model, precision, scheduler or request-policy change is permitted.

## Required Answer

For M1/M2/M4/M8, report independently:

- CUDA Event duration around complete `execute_model + sample_tokens`;
- summed Kineto child-kernel duration and launch count;
- native causal-convolution versus R002w fused-kernel contribution;
- CPU runner duration and wall throughput;
- control A/B drift and candidate delta;
- batch-weighted opportunity using the completed public replay.

The result must distinguish a real graph-step gain from eager-baseline bias,
profiler synchronization bias and full-workload queue variance.

## Promotion Boundary

R002x does not itself create a serving candidate. A larger
projection/state/projection fusion may open only if:

- candidate complete GPU step gains at least `1%` in the weighted aggregate;
- gain direction is positive at M4 and M8;
- candidate gain exceeds control A/B drift;
- CPU duration, graph mode and request completion do not regress materially.

Otherwise close the ShortConv fusion branch and keep R002k unchanged.

## Measured Answer

All twelve valid M1/M2/M4/M8 panels complete `64/64` pure-decode steps with
zero failures. Exact full-replay scheduler telemetry shows that `98.744%` of
active steps are pure decode and `99.401%` of decode rows are eligible for the
R002w path, so mixed-path fallback is not the missing gain.

Scheduler-weighted complete GPU time improves `3.501%`, but the result is
concentrated at M1/M2. M4 gains only `0.851%`, equal to its `0.851%` control
drift, and M8 regresses `1.687%`. Direct M8 ShortConv saving is only
`0.039981ms/step`, `4.86x` smaller than the eager projection. The registered
M8-positive gate fails.

No larger fusion, image, Compose, quality run or Portal attempt is authorized.
Read `results.md` for the complete attribution and artifact paths.
