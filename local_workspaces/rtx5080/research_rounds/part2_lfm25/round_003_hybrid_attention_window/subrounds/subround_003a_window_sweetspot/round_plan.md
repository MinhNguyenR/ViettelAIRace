# Subround 003a Plan

Status: complete; performance winner `4096`, balanced branch `8192`, no Portal
promotion.

## Fixed Control

- Image: `localhost/part2-lfm25-r002k:fused-bf16-greedy`.
- Resource envelope: one RTX 5080, 3 CPU cores, 8 GiB RAM.
- Runtime: the exact R002k/R003 command with `gpu-memory-utilization=0.80`,
  `max-num-seqs=8` and `max-num-batched-tokens=16384`.
- Workload: graded conversation `15`, six sequential turns, append-only
  deterministic assistant history.
- Existing evidence: unbounded control once and `8192` twice. New windows are
  each measured twice in independent server processes.

## Phase A - Profile Sweep

Status: complete.

Run `4096` twice and `16384` twice. Then, only if they bracket the registered
`8192` result, run `6144` and `12288` twice as adaptive midpoints. Record per
turn:

- complete boundary GPU milliseconds;
- control-relative and `8192`-relative saving;
- computed/scheduled token progression;
- startup KV capacity and graph/backend verification.

Reject a window on partial patching, cache-progression mismatch, turn-0
regression above `5%` or turn-5 saving below `15ms`.

Primary result: `4096` passes performance, `16384` fails the `15ms` and `70%`
retention gates, and `8192` passes. This justifies the two midpoint probes.
Both midpoint probes complete: `6144` and `12288` pass the profile gate. One
contaminated `6144` process is excluded and documented.

## Phase B - Long-Context Retrieval

Status: complete.

Use deterministic synthetic prompts with a random code near the beginning and
the query at the end. Test multiple effective distances around all three
boundaries. Keep generation, sampling and prompt construction identical across
control and candidates. Record exact-match, parse failure and output hash.

This is a risk probe, not a substitute for the official accuracy gate. It is
needed because the short GPQA proxy never activates the window.

Primary result: the unbounded control recalls four of fourteen passkeys.
Window `4096` loses two of those four, `8192` loses none, and `16384` loses
none. Window `6144` loses one; `12288` loses none.

## Phase C - Full Replay Qualification

Status: complete.

Run one 420-request candidate replay first. Compare it against the two
registered R003 controls and the two `8192` runs. Repeat only if the candidate
can change the decision after accounting for local run drift.

Completed full replays for `12288`, `6144`, and `4096`; repeated `4096`
because its first score changed the decision. The repeat differs by only
`0.0766` point. A second `6144` full replay is not justified because it is
quality-weaker than `8192` and far behind repeated `4096` on score.

## Phase D - Decision

Status: complete.

Choose one of four outcomes:

1. `16384` replaces `8192` as the safer research candidate.
2. `8192` remains the best measured balance.
3. `4096` becomes a score-max branch but remains quality/rules blocked.
4. All bounded windows stay research-only and R002k remains the sole
   submission incumbent.

Update `results.md`, `failed_actions.md`, the parent round and shared ledgers
after the decision. Preserve negative results.

Outcome: options 2 and 3 coexist as separate branches. `8192` is the measured
balance; `4096` is the score-max but quality/rules-blocked branch. R002k remains
the submission incumbent.
