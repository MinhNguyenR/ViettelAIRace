# R002aa Packaging

Status: local package and immutable registry artifact qualified; H200 Portal
execution is pending.

## Image

```text
tag:    localhost/part2-lfm25-r002aa:shortconv-sparse-nopad
digest: sha256:9e5586a637dfedec185a0665bb10284d00284be13cea0ce09f9238dfc7739765
size:   9,223,634,914 bytes
base:   localhost/part2-lfm25-r002k:fused-bf16-greedy
remote: minhnguyenrobert/viettel-ai-race-vllm@sha256:9e5586a637dfedec185a0665bb10284d00284be13cea0ce09f9238dfc7739765
```

The image bakes only the R002aa serving overlay on top of R002k. It enables
`R002AA_SPARSE_DUAL=1` and `R002AA_PREFILL_PAD_MULTIPLE=0`. R002y's
profiling module is deliberately absent from the image.

## Build

```bash
packaging/scripts/build_image.sh
```

Override `R002AA_IMAGE` only when a distinct immutable tag is required.

## Qualification

```bash
packaging/scripts/run_packaged_replay.sh candidate LABEL PORT
```

The runner enforces 3 CPU cores, 8 GiB RAM, 32,768 context, seq8, batched
tokens 16,384, FP8 per-tensor weights, FP8 E4M3 attention KV, BF16 Mamba state,
async scheduling and no prefill padding.

The rebuilt package completed `420/420` requests and `330/330` graded
requests with no server failure. That run was a functional gate only because
desktop GPU utilization was high. Performance promotion comes from the two
clean host-overlay pairs recorded in the parent `results.md`.

## Submission Boundary

Do not push or submit the earlier digest `sha256:2bf2386a...`; it contains the
research profiler and crashes on the first request without profiler variables.
The final digest is pinned in
`portal_submissions/part2_attempt_06_lfm25_r002aa_shortconv_sparse_nopad_seq8_b16384.yml`.
That 1,216-byte Compose passes `docker compose config` and differs from
Attempt 04 only in `services.model.image`. Its SHA-256 is
`e319dd9958ead7e4c571fe89c0d8de72375f6a28fe9b3f13764295e9902bb485`.
R002k remains the rollback and incumbent until the target result arrives.
