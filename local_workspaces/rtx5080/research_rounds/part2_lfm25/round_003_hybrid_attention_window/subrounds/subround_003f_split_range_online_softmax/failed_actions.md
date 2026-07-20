# R003f Failed And Excluded Actions

Only valid, dependency-complete runs may enter the registered comparison.

## Wrong Overlay Mount

Run: `full-replay-p128w4096-candidate-u0p80-20260720-120227`.

The first replay wrapper derived the overlay path from an already nested
round directory and mounted no R003f code. Startup marker assertions stopped
the run before a client summary was accepted. The path calculation was fixed;
this run is excluded.

## Forced 2D Decode

Run: `full-replay-p128w4096-fixed-candidate-u0p80-20260720-120425`.

The first working split-range integration disabled native segmented 3D decode.
It completed without request failures but scored `54.9403`; TPOT p50 rose to
`3.021ms`. The compact virtual tile list was then partitioned across native
segments and reduced with exact online-softmax state. The 2D design is
rejected, while its result remains a valid diagnostic.

## Lost Exec Handle And Container Collision

Runs:

- `full-replay-p64w4096-3d-repeat-candidate-u0p80-20260720-124105`
- `full-replay-p64w4096-3d-repeat-clean-candidate-u0p80-20260720-124624`

The first client continued in an execution namespace after its tool handle was
lost. An operator-side process check incorrectly concluded that it had ended,
and its server was removed. A replacement reused the same container name and
port. The stale client later connected to that replacement, then its cleanup
trap deleted the replacement after two successful warmup requests.

Both summaries are invalid: one mixes server lifecycles and has 255 graded
connection failures; the other has 330 graded connection failures. Neither is
performance or kernel-crash evidence.

The replay harness now assigns a timestamped container name and refuses an
occupied port before startup. A clean P64 stability replay on port 8074 then
completed 420/420 with zero graded failures and score `60.3961`.

## Quality Equivalence Failure

P64 short-GPQA repeats are `20/100`; controls are `21/100` and `22/100`.
Because every prompt is below the recent window, this is not a context-loss
effect. Treat it as implementation/numerical risk and do not promote the fork
until full-overlap outputs are equivalent or a stronger official-quality
bracket proves the difference harmless.

## Native Full-Overlap Fallback

An optional branch sent requests whose maximum sequence length was at most
`4096` tokens back through native unbounded attention. It was intended to
remove the short-prompt numerical risk before another quality run.

The matched profile rejected the branch before replay:

| Path | t0 | t5 |
|---|---:|---:|
| P64 native full-overlap fallback | `88.759ms` | `111.134ms` |
| P64 current compact control | `63.715ms` | `78.746ms` |
| Regression | `+25.045ms` | `+32.388ms` |

The fallback run had higher desktop GPU activity, but the regression at both
profiled attention layers is too large to justify a replay. The branch is
removed; no quality or full-trace result is attributed to it.
