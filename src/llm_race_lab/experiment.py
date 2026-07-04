from __future__ import annotations

import json
import re
import shlex
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_race_lab.benchmark import run_benchmark
from llm_race_lab.gpu import NvidiaSmiSampler
from llm_race_lab.metrics import summarize
from llm_race_lab.report import write_experiment_report, write_report
from llm_race_lab.workload import generate_records, write_jsonl


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    concurrency: int
    prompt_class: str
    decode_class: str
    pattern: str


def parse_csv(value: str, cast=str) -> list:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [cast(item) for item in items]


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "experiment"


def next_experiment_dir(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for child in root.iterdir():
        if child.is_dir() and re.match(r"^\d{4}-", child.name):
            existing.append(int(child.name[:4]))
    next_id = max(existing, default=0) + 1
    return root / f"{next_id:04d}-{slugify(name)}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_cases(
    concurrencies: list[int],
    prompt_classes: list[str],
    decode_classes: list[str],
    patterns: list[str],
) -> list[ExperimentCase]:
    cases: list[ExperimentCase] = []
    for concurrency in concurrencies:
        for prompt_class in prompt_classes:
            for decode_class in decode_classes:
                for pattern in patterns:
                    case_id = f"c{concurrency}-{prompt_class}-{decode_class}-{pattern}"
                    cases.append(
                        ExperimentCase(
                            case_id=case_id,
                            concurrency=concurrency,
                            prompt_class=prompt_class,
                            decode_class=decode_class,
                            pattern=pattern,
                        )
                    )
    return cases


def run_experiment(
    *,
    name: str,
    root: Path,
    requests_per_case: int,
    concurrencies: list[int],
    prompt_classes: list[str],
    decode_classes: list[str],
    patterns: list[str],
    seed: int,
    api_base: str,
    model: str,
    timeout: float,
    stream: bool,
    dry_run: bool,
    time_scale: float,
    collect_gpu: bool,
    gpu_interval_s: float,
    command: list[str],
) -> Path:
    experiment_dir = next_experiment_dir(root, name)
    workloads_dir = experiment_dir / "workloads"
    metrics_dir = experiment_dir / "metrics"
    reports_dir = experiment_dir / "reports"

    config = {
        "name": name,
        "requests_per_case": requests_per_case,
        "concurrencies": concurrencies,
        "prompt_classes": prompt_classes,
        "decode_classes": decode_classes,
        "patterns": patterns,
        "seed": seed,
        "api_base": api_base,
        "model": model,
        "timeout": timeout,
        "stream": stream,
        "dry_run": dry_run,
        "time_scale": time_scale,
        "collect_gpu": collect_gpu,
        "gpu_interval_s": gpu_interval_s,
    }
    cases = build_cases(concurrencies, prompt_classes, decode_classes, patterns)

    experiment_dir.mkdir(parents=True, exist_ok=False)
    _write_json(experiment_dir / "config.json", config)
    (experiment_dir / "command.txt").write_text(" ".join(shlex.quote(x) for x in command) + "\n", encoding="utf-8")
    (experiment_dir / "conclusion.md").write_text(
        "# Conclusion\n\n"
        "## Result\n\nTBD\n\n"
        "## Bottleneck\n\nTBD\n\n"
        "## Trade-off\n\nTBD\n\n"
        "## Next Decision\n\nTBD\n",
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "experiment_dir": str(experiment_dir),
        "started_at_epoch": round(time.time(), 6),
        "config": config,
        "cases": [asdict(case) for case in cases],
        "case_results": [],
    }

    gpu_path = experiment_dir / "gpu_samples.csv"
    sampler = NvidiaSmiSampler(gpu_path, gpu_interval_s) if collect_gpu else None
    if sampler:
        sampler.start()

    try:
        for idx, case in enumerate(cases):
            case_seed = seed + idx * 997
            workload_path = workloads_dir / f"{case.case_id}.jsonl"
            metrics_path = metrics_dir / f"{case.case_id}.jsonl"
            report_path = reports_dir / f"{case.case_id}.html"
            records = generate_records(
                requests_per_case,
                case_seed,
                prompt_class=case.prompt_class,
                decode_class=case.decode_class,
                pattern=case.pattern,
                case_id=case.case_id,
            )
            write_jsonl(workload_path, records)
            run_meta = run_benchmark(
                workload_path=workload_path,
                output_path=metrics_path,
                api_base=api_base,
                model=model,
                concurrency=case.concurrency,
                limit=0,
                timeout=timeout,
                stream=stream,
                dry_run=dry_run,
                time_scale=time_scale,
            )
            write_report(metrics_path, report_path)
            case_records = _read_jsonl(metrics_path)
            manifest["case_results"].append(
                {
                    "case": asdict(case),
                    "run": run_meta,
                    "summary": summarize(case_records),
                    "workload_path": str(workload_path),
                    "metrics_path": str(metrics_path),
                    "report_path": str(report_path),
                }
            )
            _write_json(experiment_dir / "manifest.json", manifest)
    finally:
        if sampler:
            sampler.stop()

    manifest["finished_at_epoch"] = round(time.time(), 6)
    manifest["duration_s"] = round(manifest["finished_at_epoch"] - manifest["started_at_epoch"], 6)
    if sampler and sampler.error:
        manifest["gpu_sampler_error"] = sampler.error
    elif sampler:
        manifest["gpu_samples_path"] = str(gpu_path)

    summary = {
        "experiment_dir": str(experiment_dir),
        "config": config,
        "duration_s": manifest["duration_s"],
        "case_results": manifest["case_results"],
        "gpu_samples_path": str(gpu_path) if collect_gpu else None,
        "gpu_sampler_error": manifest.get("gpu_sampler_error"),
    }
    _write_json(experiment_dir / "manifest.json", manifest)
    _write_json(experiment_dir / "summary.json", summary)
    write_experiment_report(experiment_dir / "summary.json", reports_dir / "summary.html")
    print(f"Experiment complete: {experiment_dir}")
    return experiment_dir
