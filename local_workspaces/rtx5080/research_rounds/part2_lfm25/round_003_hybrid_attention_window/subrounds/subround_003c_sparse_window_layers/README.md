# Subround 003c - Sparse Windowed Layers

Status: complete as a local research screen; `early2` is retained as the
balanced sparse-placement branch, not as a submission artifact.

## Question

R003b showed that windowing four or five full-attention layers at `4096` loses
the same two long-context retrieval cases, even when one or two layers remain
global. This subround asks whether windowing only two or three layers can keep
the quality signal while still removing enough deep-history attention work to
matter for score.

This uses the already verified R003b routing overlay. A candidate specifies the
global complement; every full-attention layer outside that complement receives
the `4096` window. ShortConv layers remain untouched.

## Fixed Contract

- Exact `localhost/part2-lfm25-r002k:fused-bf16-greedy` image and model mount.
- vLLM `0.22.1`, RTX 5080, 3 CPU cores, 8 GiB RAM, GPU utilization `0.80`.
- Full-attention layers `[2, 5, 8, 10, 12, 14]`.
- R002k precision, prefix cache, b16384, scheduler, CUDA Graph policy and
  fused greedy head are unchanged.
- Only `R003B_GLOBAL_LAYERS` changes between candidates.

## Candidate Matrix

| Label | Global layers | Windowed layers | Rationale |
|---|---|---|---|
| `late2` | `2,5,8,10` | `12,14` | cheapest late-layer window |
| `late3` | `2,5,8` | `10,12,14` | contiguous late semantic layers |
| `spread3` | `5,10,12` | `2,8,14` | depth-spread sparse window |
| `early3` | `10,12,14` | `2,5,8` | early-layer contrast |
| `early2` | `8,10,12,14` | `2,5` | quality-boundary refinement |
| `first1` | `5,8,10,12,14` | `2` | one-layer minimum |

## Gates

Each candidate gets one isolated six-turn profile and the same calibrated
14-case paired retrieval process. The profile opening gate is turn-0
regression at most 5%, turn-5 saving at least 8 ms, exact cache progression
and zero failures. The quality gate requires zero lost control-pass retrieval
cases and 14/14 successful requests.

Only a candidate passing both gates may open a full replay. A profile-only win
or a quality regression is retained as research evidence and not packaged.

## Reproduction

- `scripts/run_profile.sh LABEL GLOBAL_LAYERS [PORT]`
- `scripts/run_retrieval.sh LABEL GLOBAL_LAYERS [PORT]`
- `scripts/run_full_replay.sh LABEL GLOBAL_LAYERS [PORT]`
- `scripts/run_quality.sh LABEL GLOBAL_LAYERS [PORT]`
- `scripts/analyze_screen.sh CONTROL OUTPUT RETRIEVAL QUALITY FULL_REPLAY CANDIDATE...`:
  builds the canonical screen summary from all measurement summaries.
- `analysis/screen_summary.json` is the cross-signal summary; the component
  summaries are `profile_screen.json`, `retrieval_screen.json`,
  `quality_summary.json` and `full_replay_comparison.json`.
- Raw run directories remain ignored; the committed JSON files point back to
  their source run paths.

## Decision

`early2` (global [8,10,12,14], windowed [2,5]) is the only sparse-placement
candidate that preserved every pass in the fresh 14-case control probe across
two repeats. It also completed both 420-request replays without failures and
scored `58.0922` on average versus `56.5450` for the matched unbounded
control. The gain is local and modest (+1.5472 points), with TTFT p50/p95
improving by `8.81/28.14ms` and TPOT p95 improving by `0.065ms`.

The GPQA proxy was `21/100` for `early2` versus a repeated control of
`22/100` with a `21..22/100` control range. This is neutral within run
noise and does not validate long-context semantics. The branch remains
research-only because the stronger `4096` window signal is not
retrieval-safe and the BTC request construction/rules boundary is unresolved.
