.PHONY: doctor venv install generate-dry-workload dry-benchmark dry-report dry-experiment test docker-build-benchmark validate-submission

PYTHON ?= python3.11
PYTHONPATH ?= src

doctor:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m llm_race_lab.cli doctor

venv:
	$(PYTHON) -m venv .venv

install:
	. .venv/bin/activate && pip install -e .

generate-dry-workload:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m llm_race_lab.cli generate-workload --output workloads/mock_v0.jsonl --requests 64

dry-benchmark:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m llm_race_lab.cli run-benchmark --workload workloads/mock_v0.jsonl --output metrics/dry_run_results.jsonl --dry-run

dry-report:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m llm_race_lab.cli report --input metrics/dry_run_results.jsonl --output reports/dry_run_report.html

dry-experiment:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m llm_race_lab.cli run-experiment --name mock-v1-dry --requests-per-case 4 --concurrency 1,4 --prompt-classes short,medium --decode-classes short,medium --patterns continuous,burst --dry-run

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests

docker-build-benchmark:
	docker build -f docker/Dockerfile.benchmark -t llm-race-benchmark:local .

validate-submission:
	scripts/validate_submission.sh submissions/phase1
