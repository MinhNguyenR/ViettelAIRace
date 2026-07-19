# R002z Failed Or Rejected Actions

- Do not force FlashInfer on H200: it replaces the expected FA3 target path.
- Do not force TRT-LLM attention: this image gates it to SM100.
- Do not use batch-invariant attention as a prefill-only knob: it changes the
  decode path and disables the default FA3 AOT behavior.
- Do not sweep `flash_attn_max_num_splits_for_cuda_graph`: R002k long prefill is
  outside the captured graph range, so the knob does not own that execution.
- Do not treat `use_prefill_decode_attention` as implemented: no runtime call
  site exists in the exact image.
- Do not tune FA3 `num_splits` from RTX 5080 results: SM120 cannot execute the
  target FP8-KV FA3 path.
