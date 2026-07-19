# R002x Round Plan

Status: complete; all phases executed and the registered promotion gate failed.

## Phase A - Profiler Semantics

Status: complete.

1. Reuse the exact vLLM `GPUModelRunner` boundary.
2. Preallocate CUDA Event pairs before each measured panel.
3. Record one start event before `execute_model` and one end event after
   `sample_tokens`; synchronize only after the complete panel.
4. Run Kineto concurrently for child-kernel attribution without per-step
   synchronization.
5. Record CPU wall duration separately so GPU and scheduler effects are not
   conflated.

## Phase B - Multi-Batch Harness

Status: complete. Each of the twelve valid panels contains `64/64` clean
pure-decode steps and zero request failures.

Run M1/M2/M4/M8 sequentially inside one clean server. Each panel must contain
64 pure-decode steps at exactly the target batch, use unique prompts and
produce a separate trace/metadata artifact. Reject contaminated panels.

## Phase C - Independent Bracket

Status: complete. Control A, candidate and control B ran in separate clean
processes with matched `.80` profile memory utilization.

Run three clean processes in this order:

```text
control A: R002k
candidate: R002w
control B: R002k
```

Every process uses 3 CPU cores, 8 GiB RAM and identical serving flags. The
candidate must show all ten R002w dispatch markers; controls must show none.

## Phase D - Attribution

Status: complete. Exact model-runner scheduler telemetry replaced the initial
client-side occupancy estimate; the client estimate remains only a crosscheck.

For every batch, compare control mean, control drift and candidate on:

- complete GPU event mean/median/p95;
- child-kernel sum and count per step;
- ShortConv native/fused family duration;
- CPU mean/median/p95;
- response throughput and failures.

Estimate trace batch weights from the completed public replay's decode
intervals and report the sensitivity to uniform weighting. Do not call an
estimated distribution exact scheduler telemetry.

## Phase E - Decision

Status: complete. Weighted GPU gain passes, but M4 is indistinguishable from
control drift and M8 regresses `1.687%`. The overall gate fails.

Open a larger fusion only if the registered promotion boundary passes.
Otherwise document the failed metric and close without code, image, Compose,
quality run or Portal attempt.

Decision: close the standalone ShortConv fusion branch, preserve R002k and
create no serving artifact from R002x.
