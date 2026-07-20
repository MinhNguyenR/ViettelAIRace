# Subround 003g - Short-Path Native Identity

Status: closed as a negative result; the proposed short-path specialization is
not a quality fix.

## Question

R003f improves long-context retrieval and reaches the recent-only attention
profile floor, but its short GPQA proxy scores `20/100` versus controls at
`21..22/100`. Every GPQA prompt is shorter than the 4096-token window, so the
custom split-range specialization should be mathematically inactive.

Can short and fully overlapping batches use the exact native Triton
specialization while long batches retain compact prefix-plus-recent work,
without changing the attention backend, CUDA Graph contract or decode path?

The answer from this subround is no, for the intended quality problem. The
short-path dispatch is numerically identical to the existing Triton control,
and the GPQA result follows the Triton backend rather than the split-range
branch.

## Corrected Target Contract

- BTC host: Ubuntu 24.04, NVIDIA driver 590.x, CUDA 13.x compatible.
- Local host: driver 595.71.05, CUDA capability reported as 13.2.
- Exact image: PyTorch 2.11.0+cu130, CUDA runtime 13.0, vLLM 0.22.1.
- Hardware remains different: local RTX 5080 SM120 versus BTC H200 SM90.

CUDA 12 is no longer a blocker. SM90 compilation, register pressure and
performance remain unverified until target execution is available.

## Candidate (rejected)

When every sequence in a call fits inside its configured recent window, call
the copied upstream wrapper with `split_prefix_tokens=0`. This selects the
fork's byte-equivalent native branch at Triton compile time. Otherwise retain
the R003f compact split-range specialization.

The candidate is opt-in through `R003G_SHORT_NATIVE=1`; setting it to zero is
the same-source R003f control. No scheduler, cache, model, tokenizer, weight or
request metadata changes are allowed.

## Gates And Outcome

1. Direct native-versus-zero-specialization output must be bit-identical for
   BF16 and FP8 paged KV in both 2D and segmented 3D modes.
2. Full-overlap dispatch must choose zero, while long-context dispatch must
   retain the configured prefix.
3. Matched GPQA isolation must distinguish the split arithmetic from the
   backend selection.
4. Retrieval and full replay are not opened after the attribution gate fails.

Observed:

- Native Triton versus copied `split_prefix_tokens=0` was bit-identical for
  BF16/FP8, 2D/3D numerical cases.
- Triton `split=0` scored `20/100`, exactly the same as the R003f repeated
  run; all 100 fingerprints matched that repeat.
- FlashInfer controls scored `21/100` and `22/100` on the same local subset.
- Therefore the short-path branch cannot recover the FlashInfer quality
  signal. The likely cause is backend numerical behavior, not split-range
  arithmetic.

The old profile gate was intentionally not run: proving that a no-op branch
has a different latency would not change this attribution, and the candidate
cannot be promoted without a backend-native implementation.
