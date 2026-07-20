# R003g Round Plan

Status: complete; candidate rejected and backend-attribution evidence retained.

## Phase A - Contract And Source

Confirm the current BTC CUDA contract, exact image runtime and native Triton
call signature. Diff the R003f fork against the installed vLLM source and
reject any unexplained modification.

Answer: BTC now specifies driver 590.x/CUDA 13.x. The exact image is cu130.
The fork differs from native only in the documented split-range specialization
and wrapper argument. `max_seqlen_k` and `window_size` are host scalars at the
backend call boundary.

## Phase B - Numerical Identity

Implement a pure dispatch helper and the zero-specialization short path. Test
native versus copied-zero and copied-split outputs for BF16/FP8, prefill and
segmented decode. Record exact equality, maximum absolute error and token-level
risk rather than relying only on dense-reference tolerance.

## Phase C - Performance Bracket

The planned profile bracket was superseded by the attribution result. The
candidate's short branch calls the copied Triton wrapper with zero prefix
tokens, so it cannot restore the FlashInfer path. A profile of this branch
would not answer a remaining decision question.

## Phase C - Quality Attribution

Run a 100-case GPQA isolation with Triton forced and prefix tokens set to zero.
Compare it with R003f's split run and the FlashInfer controls. The Triton-zero
run reproduced the R003f quality level and fingerprint set. This closes the
short-path candidate before retrieval or full replay.

## Phase D - Target Decision

Create no image or Compose. Retain the identity findings and investigate a
FlashInfer/FA3-native prefix-plus-recent implementation in a separate
subround. The BTC contract is CUDA 13.x on H200 SM90; the local image is cu130,
but this branch did not validate an SM90 kernel build.
