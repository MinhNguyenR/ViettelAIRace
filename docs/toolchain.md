# Toolchain

## Required Local Tools

- Python 3.11; current workspace pins benchmark tooling to Python 3.11 for ML ecosystem compatibility.
- Docker Engine.
- Docker Compose v2.
- NVIDIA driver.
- NVIDIA Container Toolkit.

## Serving Runtime Candidates

Runtime chưa khóa:

- vLLM: baseline candidate vì có OpenAI-compatible server và metrics tương đối thuận tiện.
- SGLang: candidate cho scheduling/prefix/cache experiments.
- TensorRT-LLM: candidate sau baseline, đặc biệt nếu model và H200 environment phù hợp.

## Metrics / Profiling

- `nvidia-smi`: quick health and memory checks.
- DCGM Exporter: production-style GPU metrics.
- Prometheus: scrape metrics.
- Grafana: dashboard.
- Nsight Systems: timeline profiling.
- Nsight Compute: kernel-level profiling.

## H200 Readiness Notes

- H200 preparation should be config-driven, not hardcoded.
- Do not assume number of GPUs, NVLink, MIG, CPU limit, or container privileges.
- Pin image versions only after BTC publishes official constraints.
- Keep Phase 1 submission aligned with Docker Compose.
