# Subround 003a - Attention Window Sweet Spot

Status: complete; `4096` is the score-max branch and `8192` remains the
semantics-balanced branch. Neither is promoted to a Portal submission.

## Question

What is the least aggressive full-attention window that preserves most of
R003's long-history performance gain without creating an unacceptable
long-context recall loss?

The registered R003 result uses window `8192`. The primary bracket compares
`4096`, `8192`, `16384` and the unbounded control. Adaptive midpoint probes at
`6144` and `12288` are allowed only after that bracket identifies a real
performance/quality boundary. No candidate changes the ten ShortConv layers,
precision policy, scheduler, CUDA Graph mode or fused greedy head.

## Method

1. Run two isolated six-turn timing processes for each new window on graded
   conversation `15` under the exact R003 resource envelope.
2. Compare complete GPU time, attention saturation and cache progression with
   the registered control and `8192` bracket.
3. Run a deterministic long-distance retrieval probe. GPQA is retained only
   as a short-prompt regression gate because its maximum input is `847` tokens
   and therefore cannot activate any candidate window.
4. Run a 420-request qualification replay for candidates that pass both gates.
   A gate-failing candidate may receive a diagnostic replay only when needed
   to measure the score ceiling, and cannot be promoted from that replay.

The primary bracket selected `8192` as the only initial candidate passing both
performance and retrieval gates. The midpoint refinement then established a
frontier rather than one universal winner:

- `4096` repeats at `64.5121/64.5886` local points, but loses two retrieval
  cases that the unbounded control answers;
- `6144` scores `61.7945` once and loses one retrieval case;
- `8192` repeats at a `61.1708` mean with no control-pass retrieval losses;
- `12288` preserves the same retrieval cases but scores only `57.0213` once;
- `16384` fails the profile saving gate and does not proceed to full replay.

## Promotion Rule

Prefer the largest window that:

- keeps turn-0 complete GPU regression at or below `5%`;
- saves at least `15ms` at turn 5;
- retains at least `70%` of the registered `8192` turn-5 saving, unless the
  long-context probe shows a materially safer quality boundary;
- keeps cache progression identical and introduces no failures;
- does not worsen the long-distance retrieval pass rate by more than one case
  against the unbounded control.

No result from this subround is a Portal candidate until the BTC conversation
construction and rules interpretation are resolved.

## Decision

Use `4096` only as a score-ceiling research branch. Use `8192` when a bounded
window is needed for further semantics-aware experiments. Keep the exact R002k
artifact as the submission incumbent because every bounded window changes the
original long-context attention function and BTC has not explicitly approved
that change.

The compact evidence is in `analysis/window_profile_sweep.json`,
`analysis/long_context_retrieval.json`, `analysis/full_replay_frontier.json`
and `analysis/full_replay_comparison.json`.
