# R003e Failed Or Invalid Actions

- The first startup reached a healthy Triton server but stopped before the
  client because R003e's `sitecustomize` shadowed the R002y profiler hook. It
  is excluded. The profiler is now composed directly in the R003e runner
  patcher.
- The first runs labeled `p64`, `p256` and `p0` did not forward R003e variables
  through the parent Docker harness. Startup logs proved that all three used
  the default `prefix_tokens=128`; they are excluded from the matrix.
- All shared harnesses now forward the three R003e variables and fail before
  benchmarking unless startup logs prove the exact map, prefix size and
  metadata state. A zero-prefix control also fails if a non-empty range marker
  appears.
- Do not interpret `p0w4096` as a semantic repair. It is a backend-only
  diagnostic and retains the information-loss boundary of plain window 4096.
- Do not open retrieval or full replay for a non-empty native mm-prefix
  candidate: all three valid sizes fail the pre-registered profile gate.
