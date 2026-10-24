# Viettel AI Race 2026 - LLM Serving Lab

Workspace chuẩn bị cho **Bài 3 - LLM Inference Optimization Challenge**.

Mục tiêu hiện tại: dựng lab trước khi có đề chi tiết, chưa tải model, chưa tối ưu runtime. Repo này ưu tiên workflow:

```text
Docker environment -> mock workload -> baseline -> metrics -> bottleneck -> optimization -> submission
```

## Language Choice

Ngôn ngữ chính: **Python 3.11**.

Lý do:

- Phù hợp để viết benchmark runner, workload generator, metric aggregation, report và automation.
- Dễ nối với OpenAI-compatible serving endpoint của vLLM/SGLang/TensorRT-LLM.
- Runtime serving vẫn nằm trong Docker, nên sau này đổi engine không làm vỡ benchmark workflow.

## Current Phase

```text
Phase: pre-problem / lab setup
Model: not downloaded
Serving engine: not locked
Target submission: Docker Compose for Phase 1
Target hardware: local GPU first, H200-ready profile later
```

## Quick Commands

Run local environment checks:

```bash
python3.11 -m llm_race_lab.cli doctor
```

Generate a mock workload without any model:

```bash
PYTHONPATH=src python3.11 -m llm_race_lab.cli generate-workload \
  --output workloads/mock_v0.jsonl \
  --requests 64
```

Run a dry benchmark and generate report:

```bash
PYTHONPATH=src python3.11 -m llm_race_lab.cli run-benchmark \
  --workload workloads/mock_v0.jsonl \
  --output metrics/dry_run_results.jsonl \
  --dry-run

PYTHONPATH=src python3.11 -m llm_race_lab.cli report \
  --input metrics/dry_run_results.jsonl \
  --output reports/dry_run_report.html
```

Run Benchmark Harness v1 dry experiment:

```bash
scripts/run_dry_experiment.sh
```

This creates:

```text
experiments/000X-mock-v1-dry/
  config.json
  command.txt
  workloads/
  metrics/
  reports/
  summary.json
  manifest.json
  conclusion.md
```

Validate Phase 1 submission skeleton:

```bash
scripts/validate_submission.sh submissions/phase1
```

Docker Compose files are prepared under `docker/`. Do not start serving until `MODEL_ID` is set.

## Directory Map

- `docs/`: research draft, assumptions, toolchain, decision log.
- `configs/`: local/mock/H200 config profiles.
- `src/llm_race_lab/`: benchmark CLI and utilities.
- `docker/`: benchmark container, serving compose, monitoring compose.
- `monitoring/`: Prometheus/Grafana bootstrap config.
- `workloads/`: mock and official workload files.
- `metrics/`: raw benchmark outputs.
- `reports/`: generated summaries.
- `experiments/`: experiment notes and logs.
- `submissions/`: final competition artifacts only.
  - `submissions/phase1/`: Docker Compose submission skeleton.

