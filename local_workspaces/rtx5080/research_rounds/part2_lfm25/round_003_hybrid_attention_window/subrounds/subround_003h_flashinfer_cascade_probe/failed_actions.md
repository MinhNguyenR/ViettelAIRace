# R003h Failed Or Excluded Actions

## 1. Sparse prefix-plus-recent promotion

- **Attempt:** Use FlashInfer's two-level wrapper with the first `128` tokens
  and the last `4096` tokens of a long decode sequence.
- **Why it was attractive:** The primitive skipped the expensive middle KV
  pages and was much faster at synthetic batch/length settings.
- **Failure:** The middle range is discarded, so this is not exact full
  attention. The matched full replay scored `60.6272` versus `64.7741` for the
  cascade-off control and had a higher TPOT median.
- **Decision:** rejected for submission; retained only as a sparse-attention
  research reference.

## 2. Short quality run as a semantic proof

- **Attempt:** Use the 100-case GPQA proxy as the main acceptance gate.
- **Why excluded:** The prompts were shorter than the cascade activation
  threshold, so the overlay did not activate. The `22/100` result only checks
  the inactive fallback path.
- **Decision:** no long-context semantic claim can be made from this run.

## 3. Partial-page and prefill activation

- **Attempt:** Generalize the custom cascade to arbitrary prefill/mixed batches
  and partial pages.
- **Why excluded:** FlashInfer's two-level causal merge contract is not a safe
  drop-in for intermediate prefill queries, and page-level boundaries require
  careful masking. The overlay therefore falls back to native attention.
- **Decision:** do not broaden this branch; investigate exact shared-prefix
  cascade separately.
