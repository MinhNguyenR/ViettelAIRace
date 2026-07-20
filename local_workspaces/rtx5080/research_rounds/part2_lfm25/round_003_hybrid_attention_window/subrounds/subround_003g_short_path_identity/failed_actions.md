# R003g Failed Or Excluded Actions

## Rejected Candidate: Short Native Triton Branch

- Intended effect: use the copied upstream Triton wrapper with
  `split_prefix_tokens=0` whenever the recent window covers the whole key
  sequence.
- Why it was rejected: primitive outputs were bit-identical to native Triton,
  and the forced Triton-zero GPQA run stayed at `20/100`, while FlashInfer
  controls reached `21..22/100`.
- Consequence: changing only the split prefix cannot recover the quality
  difference. Do not promote this overlay, build an image, or submit it.

## Excluded Profile Gate

The planned A/B/A profile for the short branch was not run. Once the quality
attribution showed that both the candidate and control were the same Triton
backend, an additional timing bracket could not alter the promotion decision.

## Backend Limitation Found During Audit

- vLLM's FlashInfer builder currently disables cascade attention when the KV
  cache dtype differs from the model dtype.
- Its native cascade path is also disabled by a TODO in vLLM 0.22.1.
- FlashAttention's cascade helper explicitly rejects sliding-window mode.

These are constraints for the next FlashInfer/FA3-native subround, not reasons
to silently reuse R003g.
