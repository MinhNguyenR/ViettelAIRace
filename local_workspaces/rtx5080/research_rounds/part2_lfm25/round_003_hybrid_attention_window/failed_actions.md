# Round 003 Failed Or Rejected Actions

- Do not apply a model-wide CLI sliding-window value without proving it reaches
  only the six LFM2 attention layers.
- Do not infer long-context quality from short GPQA prompts alone.
- Do not use trace identity, turn number, prompt content or separate quality
  routing to choose the window.
- Do not promote a local FlashInfer speedup as H200 FA3 performance without a
  backend-independent reduction in attended token count.
- Do not sweep 4096/8192/16384 before the registered 8192 cache-lifecycle gate.
- Do not create an image, Compose or Portal attempt while the rules-risk gate
  remains unresolved.

## Completed Invalid Or Limiting Probes

- The first short smoke used conversation `0`, which is a primer conversation.
  Both control and candidate had a fully primed prefix and returned unusually
  short later-turn timings. Those observations are excluded from the result;
  graded conversation `15` was used for the clean profile instead.
- The first control GPQA run scored `21/100` and its repeat scored `22/100`.
  The candidate also scored `22/100` and matched the control repeat's parsed
  answers and hashes. The one-point difference is baseline runtime variance,
  not evidence of a candidate quality gain.
- The exact public trace does not contain prompt bodies or the API's history
  construction. The append-only full replay is a documented inference from
  the multi-turn schema, not a BTC-confirmed request contract.
- The independent 4k boundary run did not improve with the window (`67.9362`
  versus control `68.3485` on 90 graded requests). Do not generalize the
  append-only result to independent 4k traffic.
- Candidate startup allocates a larger KV pool than control at the same GPU
  memory fraction. Full-replay gain therefore includes both bounded attention
  work and increased cache capacity; it must not be advertised as a pure
  attention-kernel speedup.

## Promotion Blockers

- No H200 execution has verified the exact FA3/window interaction.
- A short GPQA set cannot certify semantic preservation for 25k-token
  conversations.
- The official rules permit serving optimizations but do not explicitly say
  whether changing the effective attention horizon is valid. Keep the
  candidate out of the Portal queue until this is reviewed.
