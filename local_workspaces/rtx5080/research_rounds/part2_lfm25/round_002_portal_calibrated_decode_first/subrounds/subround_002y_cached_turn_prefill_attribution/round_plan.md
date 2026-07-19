# R002y Round Plan

Status: complete. Phases A-F are closed.

## Phase A - Source And Trace Contract

Status: complete.

Audit the public six-turn token shapes, append-only driver semantics,
`SchedulerOutput` request metadata and the exact R002k runner boundary. Record
what is measured directly and what remains a hidden-driver assumption.

## Phase B - First-Token Profiler

Status: complete.

Patch only `GPUModelRunner.execute_model` and `sample_tokens`. Arm one isolated
request, discover its request ID and initial computed-token count, record CUDA
Events across every prefill chunk, and stop immediately after the first sample.
Synchronize only after the panel. Kineto is optional per command.

## Phase C - Independent Controls

Status: complete after excluding first-touch and incomplete warmup brackets.

Run:

```text
control A: four timing repeats for turns 0-5
attribution: one Kineto trace for turns 0-5
control B: four timing repeats for turns 0-5
```

Every process starts clean with the same image, flags, `.80` profile memory,
3 CPU cores and 8 GiB RAM. Reject panels containing another request.
Before arming panels, run all 15 public primer conversations and 90 unmeasured
requests so every published append-only shape is outside the timing bracket.

## Phase D - Attribution

Status: complete. Weighted control drift is `1.418%`; attention explains
`92.64%` of turn-0 to turn-5 complete-GPU growth.

Report control A/B drift, per-turn GPU service, prefill chunks, cache progress,
client first-token latency, top child-kernel families and ideal removable
latency. Weight turns equally because the public graded trace has 55 requests
at each turn index.

## Phase E - Decision

Status: complete. The attribution gate passes, but the serving implementation
gate remains closed because local SM120 uses FlashInfer while H200 SM90 uses
FA3. Only target-specific backend qualification opens.

Apply the registered opening gate before implementation. A family name alone
is insufficient: source ownership, numerical behavior, H200 portability and
decode isolation must all be explicit. Update the shared map, experiment log,
decision log, status checkpoint and technique watchlist after the answer.

## Phase F - Git Checkpoint

Status: complete after static checks, byte-identical comparison regeneration
and the scoped source/documentation checkpoint prepared below.

Run static tests and artifact validation. Stage only source, compact summaries
and documentation. Exclude model files, caches, raw Kineto traces and generated
Python bytecode. Commit to `MinhNguyenR/ViettelAIRace` only after the research
checkpoint is internally consistent.
