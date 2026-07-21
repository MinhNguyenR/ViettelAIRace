# Part 2 Attempt 06 Validation

Status: immutable artifact ready; H200 Portal result pending.

## Artifact Identity

```text
Compose: portal_submissions/part2_attempt_06_lfm25_r002aa_shortconv_sparse_nopad_seq8_b16384.yml
Compose SHA-256: e319dd9958ead7e4c571fe89c0d8de72375f6a28fe9b3f13764295e9902bb485
Compose size: 1216 bytes
Image: minhnguyenrobert/viettel-ai-race-vllm@sha256:9e5586a637dfedec185a0665bb10284d00284be13cea0ce09f9238dfc7739765
Base R002k digest: sha256:10f819c0e765bb579b89caccac4de5f17984c2fd5342335130832dc7ad93f6f5
Compose parse: pass
Attempt 04 diff: services.model.image only
```

## Local Qualification

```text
clean paired score deltas:    +0.2348 / +1.5078
clean paired mean delta:      +0.8713
mean paired TTFT delta:       -6.4606 ms
mean paired TPOT delta:       +0.0395 ms
graded successes:             330/330 in every clean run
GPQA candidate/reference:     22/100 twice / 21/100
reference-correct losses:     0 / 0
packaged functional replay:   420/420 requests
focused tests:                6/6
```

The package functional replay ran under a contaminated desktop GPU sample and
is used only to prove startup and request completion. Performance attribution
uses the two clean host-overlay pairs whose every registered preflight sample
was at most 10 percent.

## Frozen Delta

Attempt 06 inherits every server argument and environment entry from Attempt
04. Only the image digest changes. The image adds the no-pad sparse ShortConv
boundary checkpoint overlay on top of R002k. It deliberately excludes R002y's
profiler and rejects cold, incomplete or allocation-failed checkpoint state by
falling back to the native align path.

## Result Interpretation

- A target score above the R002k exact-pair envelope with no quality or failure
  regression opens an exact-digest repeat.
- A result inside the `58.94..60.75` R002k range is operational transfer but
  not attributable promotion evidence.
- A lower score, increased TBT or startup failure closes R002aa and keeps
  Attempt 04.

Do not edit this Compose after submission. Record the Portal panel in this
folder and in the common experiment/decision logs.
