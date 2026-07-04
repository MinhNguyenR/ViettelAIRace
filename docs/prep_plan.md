# Preparation Plan

## Sprint 0 - Lab Setup

Goal: workspace chạy được dù chưa có đề chính thức.

- Python CLI.
- Docker benchmark image.
- Docker Compose serving placeholder.
- Monitoring Compose.
- Mock workload generator.
- Dry benchmark.
- Static report.

Exit criteria:

```bash
make doctor
make generate-dry-workload
make dry-benchmark
make dry-report
make test
```

## Sprint 1 - Baseline

Chỉ chạy nguyên bản, không tối ưu.

Candidate initial baseline:

```text
GPU -> vLLM -> OpenAI-compatible API -> benchmark runner -> report
```

Không dùng TensorRT-LLM trước khi có số baseline.

## Benchmark Harness v1

Implemented target:

- Automatic experiment folders.
- Matrix runner over concurrency, prompt class, decode class, request pattern.
- Per-case workload, raw metrics, and HTML report.
- Experiment-level `summary.json`, `manifest.json`, and `reports/summary.html`.
- Optional `nvidia-smi` GPU sampler.
- Submission skeleton validator.

Dry command:

```bash
scripts/run_dry_experiment.sh
```

## Sprint 2 - Metrics

Thu metric tối thiểu:

- TTFT
- latency
- output tokens
- success/error
- GPU utilization qua DCGM
- serving metrics nếu runtime expose Prometheus

## Sprint 3 - Bottleneck

Phân loại bottleneck:

- prefill-bound
- decode-bound
- memory-bound
- scheduler-bound
- CPU/network-bound
- quality-gated

## Sprint 4 - Optimization

Chỉ thử candidate sau khi có baseline có thể tái lập.

Mỗi experiment phải có:

- config
- command
- raw metrics
- report
- conclusion
- rollback note
