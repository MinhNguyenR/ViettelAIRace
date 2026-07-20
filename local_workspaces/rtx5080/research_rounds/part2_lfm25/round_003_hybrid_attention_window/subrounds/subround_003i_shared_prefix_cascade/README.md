# Subround 003i - Exact Shared-Prefix Cascade

Status: complete and rejected. Neither forced FlashInfer nor native
FlashAttention cascade is promoted.

## Question

Can vLLM 0.22.1's exact cascade path help a real common prefix shared by at
least eight requests while retaining every non-shared token? The first branch
re-enabled FlashInfer's disabled predicate. The follow-up preserves R002k's
native FlashAttention backend and only enables vLLM's global cascade switch.

This is different from R003h. R003h discarded the middle of one long
sequence. R003i follows the native vLLM metadata design: level one computes
attention over the shared prefix once for all queries, level two computes each
request's private suffix, and FlashInfer merges the two online-softmax states.
No token range is intentionally dropped.

## Hypothesis

The vLLM source already builds exact two-level metadata when
`common_prefix_len > 0`. FlashInfer disables its admission predicate, while
FlashAttention has a hardware-aware heuristic but the server-wide cascade flag
is disabled by default. A long prefix and a decode cohort of at least eight may
reduce repeated prefix bandwidth. This is mathematically exact attention, but
the changed reduction order is not bitwise identical and still requires a
quality gate.

## Scope And Controls

- Use the exact local cu130/vLLM 0.22.1 image and the existing R002k fused head.
- Keep the model, tokenizer, output length, request driver and resource
  envelope unchanged.
- First use a synthetic `fixed_prefix` diagnostic to make the mechanism
  observable; this is not evidence that the public BTC trace shares a prefix.
- Compare cascade off/on with the same BF16 KV configuration first, because
  the shipped source explicitly warns that its cascade wrapper does not yet
  support a KV dtype different from the query/model dtype.
- Only test FP8 KV after the BF16 path is numerically and operationally safe.
- Do not use a request ID, prompt body, output, or trace position to decide
  activation. The only activation signal is vLLM's online `common_prefix_len`.

## Promotion Gates

1. The server starts and records the actual common-prefix length and batch.
2. Cascade output is finite. Record token-level divergence against the control
   and do not claim bitwise identity.
3. The candidate improves score or attention service time after matched repeats
   and does not worsen TPOT or failures.
4. The public append-only trace shows a nontrivial online common prefix before
   any consideration of packaging.
5. FP8 KV is rejected unless the image's wrapper accepts it without a silent
   dtype or numerical fallback.

## Expected Outcomes

- **Positive:** retain an exact shared-prefix candidate and test it on the
  public trace only if its live prefix cohort is large enough.
- **Neutral:** document that the public proxy does not expose the required
  cross-request locality; keep R002k's native FlashAttention path unchanged.
- **Negative:** close the path if the disabled source contract reflects a real
  runtime failure or if cascade overhead exceeds the native decode path.

## Current Finding

Forced FlashInfer is not promotable. At batch 8 it added decode overhead. The
full public proxy scored `51.1766` with zero cascade activations versus
`52.6311` for the matched cascade-off control using the same backend.

Native FlashAttention is also rejected locally. At batch 8, global cascade
reduced the direct-wave score from `41.1370` to `35.6212`; at batch 20 it
reduced the score from `38.2097` to `19.8181`. Its upstream heuristic uses the
runtime SM count, so the H200 `1g.18gb` profile would make activation more
likely, not safer. Without a positive local bracket or H200 A/B evidence, this
is too speculative for a submission flag.
