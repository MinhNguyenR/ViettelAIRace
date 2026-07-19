# Round 003 - Hybrid Full-Attention Window

Status: complete local research candidate; not promoted to a submission.

## Question

Can the six LFM2.5 full-attention layers use a bounded causal window while the
ten ShortConv layers, hybrid cache lifecycle, R002k fused head and decode
policy remain unchanged?

R002y found that a cached turn still schedules only about 4.2k new tokens,
while attention cost grows from `4.825ms` to `54.742ms` as retained history
approaches 25k tokens. That makes bounded attention a direct hypothesis for
the observed first-token tail.

## Candidate

The candidate injects a trace-independent `8192`-token
`per_layer_sliding_window` into exactly the six full-attention layers
`[2, 5, 8, 10, 12, 14]`. It does not inspect request IDs, prompt contents,
turn numbers, arrival times or endpoint paths. ShortConv layers
`[0, 1, 3, 4, 6, 7, 9, 11, 13, 15]` are untouched.

The R002k runtime policy remains fixed: FP8 per-tensor weights, FP8 E4M3 KV,
BF16 aligned ShortConv state, prefix caching, b16384, async/interactivity
scheduling and the fused BF16 greedy head.

## Result

On the inferred append-only closed-loop replay, two candidate runs average
`61.1708` ERS points versus two matched controls at `56.5450` (`+4.6258`
points). TTFT p50/p95 improve from `114.26/315.56ms` to
`95.35/269.89ms`; TPOT p95 improves from `4.159ms` to `3.661ms`. All four
full runs complete `420/420` requests.

The gain grows with turn depth: `-0.124, +1.328, +3.082, +6.274, +7.821,
+9.373` score points for turns 0 through 5. An independent 4k-prompt check is
neutral/slightly negative, so the technique is specifically a long-history
optimization rather than a generic 4k prefill shortcut.

## Risk Boundary

The window changes model semantics for contexts beyond 8192 tokens. The short
GPQA proxy passes, but it cannot validate long-context reasoning. The public
trace exposes per-turn lengths and the contract labels closed-loop semantics
as inferred; BTC has not supplied the exact request-body construction. The
candidate therefore remains research-only until these two uncertainties are
resolved.

## Files

- `round_plan.md`: gates, result and follow-up 003a.
- `results.md`: complete measurements and interpretation.
- `overlay/`: kill-switchable R003 hook.
- `scripts/run_prefill_process.sh`: isolated turn-depth profile.
- `scripts/run_quality_process.sh`: 100-case GPQA proxy.
- `scripts/run_full_replay.sh`: full/bounded append-only or independent replay.
- `scripts/analyze_full_replay.py`: score-component and per-turn analysis.
- `analysis/full_replay_comparison.json`: four-run full bracket.
- `failed_actions.md`: invalid probes and unresolved boundaries.
