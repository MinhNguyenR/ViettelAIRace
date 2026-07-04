from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_race_lab.benchmark import run_benchmark
from llm_race_lab.doctor import run_doctor
from llm_race_lab.experiment import parse_csv, run_experiment
from llm_race_lab.report import write_report
from llm_race_lab.workload import generate_workload


def main() -> None:
    parser = argparse.ArgumentParser(prog="llm-race")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Check local Docker/Python/NVIDIA readiness.")

    workload = sub.add_parser("generate-workload", help="Generate mock JSONL workload.")
    workload.add_argument("--output", type=Path, required=True)
    workload.add_argument("--requests", type=int, default=128)
    workload.add_argument("--seed", type=int, default=20260704)
    workload.add_argument("--prompt-class", choices=["short", "medium", "long", "mixed"], default="mixed")
    workload.add_argument("--decode-class", choices=["short", "medium", "long", "mixed"], default="mixed")
    workload.add_argument("--pattern", choices=["continuous", "burst", "idle", "mixed"], default="mixed")

    bench = sub.add_parser("run-benchmark", help="Run benchmark against OpenAI-compatible API.")
    bench.add_argument("--workload", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--api-base", default="http://localhost:8000/v1")
    bench.add_argument("--model", default="btc-model")
    bench.add_argument("--concurrency", type=int, default=8)
    bench.add_argument("--limit", type=int, default=0)
    bench.add_argument("--timeout", type=float, default=120.0)
    bench.add_argument("--dry-run", action="store_true")
    bench.add_argument("--no-stream", action="store_true")
    bench.add_argument("--time-scale", type=float, default=1.0)

    exp = sub.add_parser("run-experiment", help="Run benchmark matrix into an experiment folder.")
    exp.add_argument("--name", default="mock-v1")
    exp.add_argument("--root", type=Path, default=Path("experiments"))
    exp.add_argument("--requests-per-case", type=int, default=8)
    exp.add_argument("--concurrency", default="1,4,8,16,32,64")
    exp.add_argument("--prompt-classes", default="short,medium,long")
    exp.add_argument("--decode-classes", default="short,medium,long")
    exp.add_argument("--patterns", default="continuous,burst")
    exp.add_argument("--seed", type=int, default=20260704)
    exp.add_argument("--api-base", default="http://localhost:8000/v1")
    exp.add_argument("--model", default="btc-model")
    exp.add_argument("--timeout", type=float, default=120.0)
    exp.add_argument("--dry-run", action="store_true")
    exp.add_argument("--no-stream", action="store_true")
    exp.add_argument("--time-scale", type=float, default=0.0)
    exp.add_argument("--collect-gpu", action="store_true")
    exp.add_argument("--gpu-interval", type=float, default=1.0)

    report = sub.add_parser("report", help="Generate static HTML report from result JSONL.")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "doctor":
        raise SystemExit(run_doctor())
    if args.command == "generate-workload":
        generate_workload(
            args.output,
            args.requests,
            args.seed,
            prompt_class=args.prompt_class,
            decode_class=args.decode_class,
            pattern=args.pattern,
        )
        return
    if args.command == "run-benchmark":
        run_benchmark(
            workload_path=args.workload,
            output_path=args.output,
            api_base=args.api_base,
            model=args.model,
            concurrency=args.concurrency,
            limit=args.limit,
            timeout=args.timeout,
            stream=not args.no_stream,
            dry_run=args.dry_run,
            time_scale=args.time_scale,
        )
        return
    if args.command == "run-experiment":
        run_experiment(
            name=args.name,
            root=args.root,
            requests_per_case=args.requests_per_case,
            concurrencies=parse_csv(args.concurrency, int),
            prompt_classes=parse_csv(args.prompt_classes, str),
            decode_classes=parse_csv(args.decode_classes, str),
            patterns=parse_csv(args.patterns, str),
            seed=args.seed,
            api_base=args.api_base,
            model=args.model,
            timeout=args.timeout,
            stream=not args.no_stream,
            dry_run=args.dry_run,
            time_scale=args.time_scale,
            collect_gpu=args.collect_gpu,
            gpu_interval_s=args.gpu_interval,
            command=sys.argv,
        )
        return
    if args.command == "report":
        write_report(args.input, args.output)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
