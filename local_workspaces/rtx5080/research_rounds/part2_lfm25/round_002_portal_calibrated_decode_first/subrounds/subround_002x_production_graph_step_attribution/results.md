# R002x Results - Production Graph-Step Attribution

Status: complete; registered opening gate failed and the standalone ShortConv
fusion branch is closed.

## Frozen Bracket

Three clean processes used the same Part 2 R002k flags and profile-only
`--gpu-memory-utilization=.80`:

```text
control A: control_a-control-20260719-171300
candidate: candidate_a-candidate-20260719-171442
control B: control_b-control-20260719-171604
```

The controls use R002k digest
`sha256:10f819c0e765bb579b89caccac4de5f17984c2fd5342335130832dc7ad93f6f5`.
The candidate uses local R002w digest
`sha256:07739966bfd124acf2dc4d7baabaf98cf97a5d33dbd713ba05637602fb795a1e`.
Every M1/M2/M4/M8 panel contains `64/64` uncontaminated pure-decode steps,
zero request failures and the expected FULL_AND_PIECEWISE graph mode. The
candidate records ten fused dispatches; both controls record zero.

## Complete Graph-Step Result

CUDA Events surround the complete `execute_model + sample_tokens` boundary.
Kineto child-kernel time is analyzed separately and no panel synchronizes per
step.

| Batch | Control GPU ms | Candidate GPU ms | GPU gain | Control drift | Kernel gain | Kernels/step C -> B | Direct ShortConv saving | CPU gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 2.377273 | 2.294168 | +3.496% | 2.281% | +2.203% | 228.125 -> 198.125 | 0.027714 ms | +7.186% |
| M2 | 2.405373 | 2.282689 | +5.100% | 1.445% | +3.289% | 228.250 -> 198.125 | 0.041019 ms | +5.527% |
| M4 | 2.411910 | 2.391376 | +0.851% | 0.851% | +2.276% | 228.250 -> 198.250 | 0.040602 ms | -1.748% |
| M8 | 2.489390 | 2.531386 | -1.687% | 2.633% | +2.138% | 227.250 -> 197.250 | 0.039981 ms | -9.490% |

The candidate consistently removes about 30 child kernels per decode step.
At M8 the direct replacement is:

```text
remove native causal convolution:       0.016840 ms/step
remove native elementwise multiply:     0.030400 ms/step
remove one staging copy per layer:       0.006940 ms/step
add R002w fused kernels:                -0.014199 ms/step
direct measured saving:                  0.039981 ms/step
fraction of control child-kernel time:   1.705%
```

The R002w eager sequence predicted `0.194175ms` saving at M8. Production graph
attribution measures only `0.039981ms`, so eager timing overstated the usable
gain by `4.86x`. CUDA Graph already amortizes most of the launch overhead that
the eager decomposition charged independently.

## Workload Occupancy

A separate full 420-request R002k replay instrumented the model runner's
actual `num_scheduled_tokens` rows. It completed all `330/330` graded requests
and was used for occupancy only, not as a candidate score comparison.

```text
active scheduler steps:                 48,263
pure-decode steps:                      47,657 (98.744%)
mixed steps:                               293
prefill-only steps:                        313
eligible pure-decode rows:              83,109 / 83,610 (99.401%)
pure-decode batch histogram:
  M1 28,171   M2 10,084   M3 4,889   M4 3,046
  M5    979   M6    392   M7    96   M8     0
```

This rejects the hypothesis that R002w loses because mixed-prefill fallback
usually bypasses the fused path. Nearly all decode work is eligible. Exact
scheduler weighting, with interpolation for M3/M5/M6/M7, gives:

```text
complete GPU mean:       2.390551 -> 2.306848 ms, +3.501%
control GPU drift:                               1.045%
child-kernel time:       2.245125 -> 2.188992 ms, +2.500%
kernel count:          228.165351 -> 198.132490,  -13.163%
direct fused saving:                          0.033087 ms
direct kernel fraction:                           1.474%
```

The weighted signal is dominated by M1/M2. M4 gain is equal to control drift
within `0.0001` percentage point, and M8 reverses direction. The full replay
already measured R002w candidate mean `-0.2689` ERS versus its control.

## Gate Decision

Registered gate:

```text
weighted GPU gain >= 1%:              pass
M4 complete GPU gain positive:        pass, but equals drift
M8 complete GPU gain positive:        FAIL
weighted gain exceeds control drift:  pass
CPU regression <= 2% weighted:        pass
all panels complete without failure:  pass
overall:                              FAIL
```

R002x therefore closes the standalone ShortConv epilogue branch. Do not build
a larger projection/state/projection fusion from this evidence, publish an
image, create Compose, run GPQA or spend a Portal attempt. Keep R002k as the
H200 incumbent. The reusable result is methodological: future kernels must be
measured at the complete production CUDA Graph boundary with the full model
working set before an eager microbenchmark can open integration.

## Artifacts

- `outputs/bracket.json`: exact scheduler-weighted A/B/A comparison.
- `outputs/profiles/`: 12 valid panel traces plus one documented invalid
  pre-profile startup.
- `outputs/telemetry/control-u0p80-20260719-172401/`: full-replay scheduler
  telemetry and request summary.
- `scripts/analyze_panel.py`: Kineto family attribution.
- `scripts/compare_bracket.py`: registered gate and scheduler weighting.
- `failed_actions.md`: invalid `.84` desktop-memory startup.
