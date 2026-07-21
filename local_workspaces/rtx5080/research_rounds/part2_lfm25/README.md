# Part 2 LFM2.5 Research

This tree contains only experiments for the revised LFM2.5-1.2B-Instruct
competition contract. Qwen3.5 results remain historical evidence and are not
numeric baselines for this tree.

Read:

1. `round_001_one_shot_requalification/README.md`
2. `round_001_one_shot_requalification/round_plan.md`
3. `round_001_one_shot_requalification/results.md`

Every candidate must run with three CPU cores and 8 GiB RAM. Local RTX 5080
results are directional evidence; final flags must preserve H200 Hopper paths.

Current target state: R002k is the H200 incumbent. R002v Attempt 05 and its
byte-identical repeat scored `60.19/59.34` (mean `59.765`) versus R002k's
exact-pair mean `59.845`, so the isolated ShortConv `in_proj` FP8 overlay is
archived and not promoted. R002aa is the next immutable target probe: sparse
ShortConv boundary checkpointing won two clean no-pad replay pairs, passed
quality/package gates and is pinned in Part 2 Attempt 06. It does not replace
R002k before an H200 result.
