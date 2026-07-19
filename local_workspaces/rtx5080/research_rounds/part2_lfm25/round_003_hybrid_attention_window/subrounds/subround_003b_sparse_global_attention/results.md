# Subround 003b Results - Selective Global Attention

Status: complete local research; no full replay opened.

## Fixed Control

The unbounded control is the registered R003 control profile on the exact R002k
image. Its six-turn boundary GPU times are:

```text
turn 0: 60.782 ms
turn 1: 73.763 ms
turn 2: 83.607 ms
turn 3: 93.569 ms
turn 4: 103.034 ms
turn 5: 114.735 ms
```

The calibrated retrieval control answers `4/14` key cases. This is a paired
regression probe, not an absolute accuracy benchmark.

## Profile Screen

| Candidate | Global layers | Turn 0 ms | Turn 5 ms | Turn-5 saving ms | Turn-0 delta | KV tokens | Profile gate |
|---|---|---:|---:|---:|---:|---:|---|
| `global14` | `14` | 61.473 | 77.894 | 36.840 | +1.14% | 1,819,020 | pass |
| `global2` | `2` | 60.221 | 80.429 | 34.306 | -0.92% | 1,827,977 | pass |
| `global8` | `8` | 60.453 | 76.802 | 37.932 | -0.54% | 1,819,020 | pass |
| `global2_14` | `2,14` | 59.175 | 82.291 | 32.443 | -2.64% | 1,976,697 | pass |

All candidates retained the expected cache progression and completed six of
six panels. `global8` is the fastest profile, but the profile result alone was
never sufficient for promotion.

## Retrieval Screen

| Candidate | Successful | Candidate key cases | Lost control-pass cases | Gained cases | Quality gate |
|---|---:|---:|---:|---:|---|
| `global14` | 14/14 | 2/14 | 2 | 0 | fail |
| `global2` | 14/14 | 2/14 | 2 | 0 | fail |
| `global8` | 14/14 | 2/14 | 2 | 0 | fail |
| `global2_14` | 14/14 | 2/14 | 2 | 0 | fail |

The lost cases are identical across placements:

- actual prompt length `5084`, key `KDF3C13FB89`;
- actual prompt length `7109`, key `KCD0433D6CD`.

The control outputs the expected key on both cases; every selective candidate
fails the key check. Therefore the failure is caused by the windowed layers as
a group, not by the absence of a single global layer at a particular depth.

## Interpretation

Leaving one or two layers unbounded is insufficient because information needed
by the two retrieval cases must survive through multiple bounded attention
updates. A global layer can re-read the full history, but it cannot reconstruct
information already discarded by earlier windowed layers. The result also
rules out choosing the first, middle or final full-attention layer as a cheap
quality repair.

The latency mechanism remains real: every placement saves at least `32.443 ms`
at turn 5 while leaving turn 0 within the 5% gate. The quality/performance
frontier therefore needs a different retention topology, such as preserving a
small set of old-token/sink positions in every attention layer, rather than
adding global layers after truncation.

## Decision

- Do not run full replay for R003b.
- Do not build a candidate image or Compose.
- Keep `global8` only as a measured profile reference, not a configuration.
- Keep R002k as the submission incumbent.
- Carry forward the negative result to prevent repeated global-layer placement
  experiments.

Machine-readable summaries:

```text
analysis/profile_screen.json
analysis/retrieval_screen.json
```
