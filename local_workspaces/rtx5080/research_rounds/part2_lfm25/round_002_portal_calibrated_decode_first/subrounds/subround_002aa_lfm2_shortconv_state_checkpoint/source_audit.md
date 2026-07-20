# R002aa Source Audit

Status: initial audit complete against vLLM `0.22.1` in the R002k image.

## Scheduler Cause

`vllm/v1/core/sched/scheduler.py::_mamba_block_aligned_split` forces a prefill
that crosses `last_cache_position` to stop exactly at that position. For the
public append-only proxy, a prompt of 8199 tokens with 4000 cached tokens is
therefore scheduled as `4192 + 7`, even though the token budget is 16384.

The condition is enabled only for hybrid models using
`mamba-cache-mode=align`.

## State Semantics

Align mode keeps only a small number of non-null ShortConv state blocks. At
each model step, `preprocess_mamba` copies the previous running state to the
new final block. The scheduler split is required so one forward ends at the
full-block cache boundary; a later forward can then keep a separate running
state after the partial tail.

Removing the scheduler condition alone would overwrite the only useful state
with the state after the partial tail and make the full-block prefix entry
incorrect.

## Existing APC Capability

The packaged causal-conv Triton kernel already accepts:

- a per-request table of state block IDs;
- first and last scheduled block indices;
- an initial state index and prior computed-token count.

It writes the final state and intermediate block-aligned states during the
same forward. Mamba mixers wire these fields for `mamba-cache-mode=all`, but
LFM2's `ShortConv.forward_cuda` does not. `Lfm2ForCausalLM` therefore rejects
all-mode explicitly.

R002aa first completes this existing all-mode wiring. This is a functional
proof, not the final memory design.

The first successful proof also exposed a second default: all-mode uses LFM2's
`2048` token ShortConv chunk size and raises the attention block size to 2048.
That explains a turn-1 hit of only 2048 tokens. A user-specified Mamba block
size of 16 is required for the matched semantic/timing proof; it does not by
itself remove all-mode's per-block state cost.

## Memory Boundary

LFM2.5 has ten ShortConv layers with a BF16 state shape of
`2048 x (3 - 1)`, or 8192 bytes per layer checkpoint. All-mode stores one
checkpoint per 16-token block; align mode reserves only the small running
state set. All-mode can therefore reduce usable prefix-cache capacity sharply
even when it eliminates the extra model step.

The intended final design is sparse: only the crossed cache boundary and the
partial-tail running state are required for this append-only workload.

## Sparse Proof

The sparse overlay leaves vLLM in align mode, reserves one additional state
block only when a request crosses its final full block, and exposes a compact
three-entry state table to the existing APC kernel:

1. prior running state;
2. crossed full-block checkpoint;
3. final partial-tail state.

The scheduler bypasses `_mamba_block_aligned_split` only when the proposed
step starts from a nonzero cached prefix, completes the prompt and crosses
that final boundary. Cold prefill, decode and all other align cases use native
behavior.

At memory fraction `.78`, sparse reports `1,592,154` cache tokens, matching
align capacity. All-mode block16 reports about `530k`; it remains proof-only.

## Token-Shape Finding

Kineto shows checkpoint storage is not the new bottleneck. Ten sparse
causal-conv kernels total only about `0.156ms`. The larger variation is in
GEMM and normalization when the one-step shape changes from the aligned
`4192 + tail` control to `4199-4213` tokens.

vLLM's runner already supports padded model batches. R002aa therefore exposes
an experimental prefill padding multiple for sparse requests above 4096
tokens. No-pad, pad16 and pad32 must be compared in one process on an idle GPU;
cross-process timings are too noisy for this decision.
