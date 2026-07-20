# Open Questions

These must be resolved before final optimization/submission decisions.

- What exact HuggingFace commit/hash will BTC use for `Qwen/Qwen3.5-2B`?
- Does BTC mount model weights at `/model`, or must the submitted image include/download weights before serving?
- Is network disabled only during serving, or also during container startup?
- Does BTC allow custom Docker images larger than the baseline image?
- Will the Accuracy Gate use the same served endpoint and decoding parameters?
- Are extra healthcheck endpoints allowed, or only OpenAI-compatible endpoints?
- Are logs size-limited?
- Does the MiG H200 expose FP8 path fully for vLLM/TensorRT-LLM in the BTC environment?
- Does Portal honor the Compose healthcheck/readiness transition before sending
  timed requests, or probe port 8000 independently? Attempt 11 gates both by
  not binding the public port until warmup completes.
- Does BTC report internal startup warmup traffic in `warmup_count`, or count
  only benchmark-client requests sent to public port 8000?
- Are the five RTX 5080 pre-readiness JIT kernels also compiled before public
  readiness on H200? R035 proves local timed JIT is already zero, so this is a
  startup portability question rather than an active local score branch.
- Can BTC expose request-level TPOT/TBT distribution or at least p95/p99?
  R036 proves median `<=20ms` can hide a score-bearing TPOT tail, so the median
  alone cannot identify the H200 decode loss.
- Will BTC explicitly permit contestants to prefill the exact public trace and
  retain prompt-derived KV/Mamba state before timed arrivals? Until written
  permission is archived, this project treats the technique as prohibited.
- On the exact H200 MIG image, does online block FP8 select the expected
  FlashInfer/DeepGEMM wrapper and beat per-tensor CUTLASS end to end? R002s
  proves source eligibility but is closed on quality, so this remains an
  unmeasured architecture fact rather than a pending submission action.
- For Part 2, does BTC send each approximately 4k turn as an independent
  prompt, or resend the complete multi-turn history? R003 wins only the latter
  inferred protocol; the public trace does not resolve the API body.
- For Part 2, is a trace-independent attention horizon below the model's
  declared context considered a serving optimization, or a disallowed model
  behavior change? R003 remains out of the Portal queue until this is clear.
- On H200 SM90, does the same per-layer window retain the expected FA3 path and
  the larger KV-capacity benefit without changing decode graph behavior?
- Can the R003f vLLM Triton fork compile and retain its compact 2D/3D tile
  behavior under BTC CUDA 13.x on H200 SM90, and can its full-overlap path match
  native outputs closely enough to remove the observed short-GPQA risk?

## Resolved

- The I006 seq16 `51.59` panel is complete: ERC `0.133333`, `16/120` passed
  SLO, TTFT p50/p95 `4783/6800ms`, TBT median `16ms`, failed/warmup `0/0`,
  accuracy drop `0`, and penalty `1`.
- R037 completed the exact-image static capability half. No distinct bounded
  H200 decode primitive passed the score gate. H200 GDN FlashInfer and CUTLASS
  paths are already auto-active; DS conv-state layout is executable but below
  ceiling; MRV2 is incompatible with retained Mamba align prefix caching.
- R038 resolved the offline first-token opportunity question. A causal serial
  lane is score-positive across all six processes and 36 waves only under a
  guarded `>=75%` queue reduction and `<=10%` decode-interference contract.
  Whether exact vLLM can realize that contract remains a GPU implementation
  question, not an unresolved scoring question.
