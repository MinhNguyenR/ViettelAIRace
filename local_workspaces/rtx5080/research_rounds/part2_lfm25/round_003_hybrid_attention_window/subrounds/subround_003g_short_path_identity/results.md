# R003g Results

Status: complete; negative result with decisive backend attribution.

## Runtime And Source Contract

- BTC target in `docs/official/de_bai.md`: H200 MIG 18GB, 3 CPU cores, 8GB
  RAM, NVIDIA driver 590.x and CUDA 13.x compatible.
- Local host: RTX 5080, driver 595.71.05, CUDA 13.2 reported by the driver.
- Exact local image: PyTorch `2.11.0+cu130`, CUDA runtime `13.0`, vLLM
  `0.22.1`.
- The R003f overlay differs from installed vLLM only in the documented
  split-range helper/branch and wrapper argument.

## Numerical Identity

Command:

```text
bash scripts/run_primitive_numerics.sh
```

For BF16 and FP8, 2D and segmented 3D full-overlap cases:

- native Triton versus copied wrapper with `split_prefix_tokens=0`:
  `torch.equal=true`, maximum absolute error `0.0`;
- native Triton versus copied split-prefix wrapper: also bit-identical in the
  tested full-overlap cases.

This means the short-path dispatch does not change arithmetic in the tested
case.

## GPQA Backend Isolation

The isolated process was:

```text
bash scripts/run_quality.sh triton-zero \
  2:4096,5:4096,8:4096,10:4096,12:4096,14:4096 0 8075
```

Run directory:
`analysis/quality-triton-zero-20260720-141241`

| Run | Backend | Prefix split | Correct | Parsed | Latency p50/p95 (ms) |
|---|---|---:|---:|---:|---:|
| Triton-zero | TRITON_ATTN | 0 | 20/100 | 88 | 17.613 / 33.472 |
| R003f P64 repeat | TRITON_ATTN | 64 | 20/100 | 88 | previously recorded |
| FlashInfer control | FLASHINFER | native | 21/100 | control | control |
| FlashInfer control repeat | FLASHINFER | native | 22/100 | control | control |

The Triton-zero and R003f repeat have the same correct-answer ID set. Their
full reduced fingerprints (`id`, correctness, parsed answer, output hash and
status) are identical for all 100 records. The first R003f run differs in only
four parsed answers because of normal generation variance, but has the same
correct ID set.

## Decision

Reject R003g as a promotion candidate. It cannot explain or repair the GPQA
gap because it remains on `TRITON_ATTN`. The next useful experiment must keep
the native FlashInfer/FA3 numerical path and add any prefix-plus-recent work
through that backend's own LSE/cascade primitives.
