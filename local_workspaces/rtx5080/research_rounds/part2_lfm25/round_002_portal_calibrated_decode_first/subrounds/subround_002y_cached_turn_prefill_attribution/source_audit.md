# R002y Source Audit - Attention Portability

Status: complete against `localhost/part2-lfm25-r002k:fused-bf16-greedy`.

## Backend Priority

`vllm/platforms/cuda.py::_get_backend_priorities` places FlashAttention before
FlashInfer for non-MLA devices outside capability major 10. Backend validation
then removes unsupported choices.

For the local RTX 5080:

```text
device capability: 12.0
selected full-attention backend: FlashInfer
get_flash_attn_version(): 2
```

`FlashAttentionBackend.supports_kv_cache_dtype` permits FP8 only when the
selected implementation is FA3 and the device is capability family 90. SM120
therefore rejects FlashAttention under R002k's FP8 E4M3 KV contract and uses
FlashInfer.

For the target H200:

```text
device capability: 9.0
default FlashAttention implementation: FA3
FP8 E4M3 KV support: yes, specifically on capability family 90
backend priority: FlashAttention, then FlashInfer
```

The image successfully imports `_vllm_fa2_C`, `_vllm_fa3_C`, and the FA4
interface, so FA3 is packaged rather than an optional missing dependency.
Provided the BTC H200 runtime exposes SM90 as specified, automatic selection is
expected to use FA3 for the six full-attention layers.

## Rejected Shortcut

`AttentionConfig.use_trtllm_attention` cannot provide an alternate H200
prefill kernel. `supports_trtllm_attention()` requires capability family 100
and NVIDIA artifact access. Forcing it on H200 only falls back with a warning.

`AttentionConfig.use_prefill_decode_attention` is declared but has no usage
outside its config definition in this vLLM image. It cannot be treated as a
working phase-isolation control.

## Consequence

R002y's local FlashInfer profile identifies the full-attention algorithmic
slope, not the exact H200 kernel cost. A next round may qualify FA3-specific
prefill scheduling or an explicit prefill-only override, but it must prove:

1. the target process actually selects FA3 with FP8 KV;
2. decode output and TBT remain on the R002k path;
3. the change targets long cached-prefill queries, not a local SM120 fallback;
4. a Portal or H200-equivalent bracket beats the `1.81`-point incumbent range.
