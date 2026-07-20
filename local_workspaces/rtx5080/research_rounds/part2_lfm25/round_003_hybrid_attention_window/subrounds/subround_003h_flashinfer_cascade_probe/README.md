# Subround 003h - FlashInfer Cascade Prefix/Recent Probe

Status: complete and rejected as a submission candidate.

## Question

Can FlashInfer's native `MultiLevelCascadeAttentionWrapper` represent a
prefix-plus-recent decode attention without forcing vLLM onto the Triton
backend?

This is deliberately a primitive probe. It uses one attention call with
page-aligned segments and compares the cascade result with a dense reference.
The serving overlay tested the sparse set `prefix OR recent`; it deliberately
omits the middle of a long sequence. That is a bounded-attention approximation,
not an exact implementation of the original full-attention model.

## Constraints

- Use the exact local cu130/vLLM 0.22.1 image.
- Run with the local RTX 5080 and the same 3 CPU / 8 GB memory envelope.
- Keep query dtype BF16 and test BF16 and FP8 KV separately.
- Do not change the model, tokenizer, trace, request order or submission image.

## Promotion Gates

1. Cascade output is finite and matches a dense prefix+recent reference within
   a documented tolerance for BF16 and FP8 KV.
2. The wrapper accepts the actual FlashInfer cache dtype and query shape.
3. Cascade latency is lower than full native attention for the long decode
   case after planning overhead is amortized.
4. Any vLLM integration is restricted to decode-only, exact page boundaries,
   and falls back to native FlashInfer for all other cases.
5. No integration is promoted without a replay and quality comparison.

The primitive gates passed, but the serving promotion gate failed: the sparse
overlay did not improve the matched full replay and changes long-context
semantics. The exact next question is shared-prefix cascade, where the middle
tokens are retained rather than discarded.

## Known Risks

- FlashInfer cascade has page-level, not token-level, segment boundaries.
- The vLLM 0.22.1 builder disables its own cascade path for a KV dtype that
  differs from the model dtype and has a TODO-disabled cascade selector.
- Applying this to prefill is not automatically correct: intermediate query
  positions need causal access to keys between the two selected ranges.
- A two-level merge adds work and may lose to native FlashInfer on short
  sequences.
- A prefix-plus-recent layout is only safe as an intentional sparse/windowed
  model change. It must not be described as exact full attention.
