# Subround 003c Failed Actions

Record candidate startup failures, contaminated processes and gate failures
without overwriting earlier evidence.

## 2026-07-20 - Sparse Placement Quality Rejections

- late2, late3 and spread3 completed all 14 requests but lost three
  control-pass retrieval cases each.
- early3 and first1 completed all 14 requests but lost two and three
  control-pass cases respectively.
- The losses cluster at the 5084- and 7109-token probes. A global complement
  at other layers does not repair information discarded by an earlier
  windowed layer.

These candidates are rejected before full replay. Their profile wins remain
useful mechanism evidence only.

## 2026-07-20 - Relative Output Directory Harness Error

The first control-repeat retrieval launch passed a relative result directory
through a Docker bind mount. Docker interpreted the path as an invalid volume
name and the container never served a request. The run was discarded and
repeated with an absolute path; the successful control result is in
retrieval-control_repeat-control-20260720-001520.

This is a harness failure, not a model result.

## 2026-07-20 - Quality Proxy Is Not a Long-Context Gate

early2 scored 21/100 versus control runs of 21/100 and 22/100. The proxy
prompts are below the 4096 window, so the result cannot certify the
long-context retention change. Keep the candidate research-only.
