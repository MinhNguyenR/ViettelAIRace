# Status Checkpoint

Date: 2026-07-04

## Completed

- Workspace skeleton created.
- Python 3.11 benchmark CLI created and installed into `.venv`.
- Mock workload generator created.
- Dry benchmark runner created.
- Static HTML report generator created.
- Docker benchmark image built: `llm-race-benchmark:local`.
- Docker Compose placeholder created for vLLM serving.
- Docker Compose monitoring stack created for DCGM, Prometheus, Grafana.
- Research Draft v1 saved as source-of-truth.
- H200 placeholder profile saved.
- Benchmark Harness v1 created.
- Phase 1 submission skeleton created.

## Verified

- Unit tests pass.
- Dry pipeline works without model/GPU.
- Compose configs render with explicit profiles.
- Benchmark Docker image builds.
- Benchmark Docker entrypoint runs.
- Host NVIDIA driver works outside sandbox.
- Docker GPU passthrough works with `--gpus all`.
- `.venv/bin/llm-race doctor` passes in host mode.
- `docker run --rm --gpus all llm-race-benchmark:local doctor` passes in container mode.
- `scripts/run_dry_experiment.sh` creates experiment folders without model/GPU.
- `scripts/validate_submission.sh submissions/phase1` validates compose structure.

## Known Blockers

- `nvidia-smi` fails inside Codex sandbox because `/dev/nvidia*` is not mounted there.
- GPU commands that need device access should be run outside sandbox or through Docker with `--gpus all`.
- Official BTC problem statement, model, dataset, scoring metric, and runtime constraints are not available yet.
- Serving image was not pulled or run.
- No model was downloaded.

## Next Recommended Task

Keep using these GPU checks when needed:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
```

Next engineering step:

1. Define serving contract tests for `/v1/models`, `/v1/chat/completions`, `/metrics`, and healthcheck.
2. After BTC publishes the official model or allows a local mock model, run first real vLLM baseline through Benchmark Harness v1.
3. Do not start TensorRT-LLM, quantization, or scheduler changes before that baseline exists.
