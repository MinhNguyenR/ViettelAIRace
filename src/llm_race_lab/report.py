from __future__ import annotations

import html
import json
from pathlib import Path

from llm_race_lab.metrics import group_counts, summarize


def _load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _table(mapping: dict) -> str:
    rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(_fmt(value))}</td></tr>"
        for key, value in mapping.items()
    )
    return f"<table>{rows}</table>"


def _counts_table(title: str, counts: dict[str, int]) -> str:
    rows = "\n".join(
        f"<tr><td>{html.escape(key)}</td><td>{value}</td></tr>" for key, value in counts.items()
    )
    return f"<h3>{html.escape(title)}</h3><table><tr><th>value</th><th>count</th></tr>{rows}</table>"


def _style() -> str:
    return """
    body { font-family: system-ui, sans-serif; margin: 32px; color: #111827; line-height: 1.45; }
    table { border-collapse: collapse; width: 100%; max-width: 1180px; margin: 12px 0 24px; }
    th, td { border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; vertical-align: top; }
    th { background: #f3f4f6; }
    code { background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
    .ok { color: #047857; font-weight: 700; }
    .warn { color: #b45309; font-weight: 700; }
    .small { color: #4b5563; font-size: 13px; }
  """


def write_report(input_path: Path, output_path: Path) -> None:
    records = _load(input_path)
    summary = summarize(records)
    errors = [r for r in records if r.get("status") != "ok"][:20]
    error_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(r.get('request_id')))}</td>"
        f"<td>{html.escape(str(r.get('error')))}</td>"
        "</tr>"
        for r in errors
    )
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LLM Race Benchmark Report</title>
  <style>{_style()}</style>
</head>
<body>
  <h1>LLM Race Benchmark Report</h1>
  <p>Source: <code>{html.escape(str(input_path))}</code></p>
  <h2>Summary</h2>
  {_table(summary)}
  <h2>Breakdown</h2>
  <div class="grid">
    <div>{_counts_table("Prompt Class", group_counts(records, "prompt_class"))}</div>
    <div>{_counts_table("Decode Class", group_counts(records, "decode_class"))}</div>
    <div>{_counts_table("Scenario", group_counts(records, "scenario"))}</div>
    <div>{_counts_table("Status", group_counts(records, "status"))}</div>
  </div>
  <h2>First Errors</h2>
  <table><tr><th>request_id</th><th>error</th></tr>{error_rows}</table>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body, encoding="utf-8")
    print(f"Wrote report to {output_path}")


def write_experiment_report(summary_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = summary.get("config", {})
    case_results = summary.get("case_results", [])
    rows = []
    for item in case_results:
        case = item.get("case", {})
        metrics = item.get("summary", {})
        rows.append(
            "<tr>"
            f"<td>{html.escape(case.get('case_id', ''))}</td>"
            f"<td>{html.escape(str(case.get('concurrency', '')))}</td>"
            f"<td>{html.escape(case.get('prompt_class', ''))}</td>"
            f"<td>{html.escape(case.get('decode_class', ''))}</td>"
            f"<td>{html.escape(case.get('pattern', ''))}</td>"
            f"<td>{html.escape(_fmt(metrics.get('success_rate')))}</td>"
            f"<td>{html.escape(_fmt(metrics.get('latency_p95_ms')))}</td>"
            f"<td>{html.escape(_fmt(metrics.get('ttft_p95_ms')))}</td>"
            f"<td>{html.escape(_fmt(metrics.get('wall_tokens_per_s')))}</td>"
            f"<td>{html.escape(_fmt(metrics.get('requests_per_s')))}</td>"
            f"<td><a href='{html.escape(Path(item.get('report_path', '')).name)}'>case report</a></td>"
            "</tr>"
        )
    gpu_status = (
        f"<p class='warn'>GPU sampler warning: {html.escape(summary['gpu_sampler_error'])}</p>"
        if summary.get("gpu_sampler_error")
        else f"<p class='ok'>GPU samples: {html.escape(str(summary.get('gpu_samples_path') or 'disabled'))}</p>"
    )
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LLM Race Experiment Report</title>
  <style>{_style()}</style>
</head>
<body>
  <h1>LLM Race Experiment Report</h1>
  <p>Experiment: <code>{html.escape(str(summary.get('experiment_dir')))}</code></p>
  <p class="small">Duration: {html.escape(_fmt(summary.get('duration_s')))} seconds</p>
  {gpu_status}
  <h2>Config</h2>
  {_table(config)}
  <h2>Cases</h2>
  <table>
    <tr>
      <th>case</th><th>conc</th><th>prompt</th><th>decode</th><th>pattern</th>
      <th>success</th><th>p95 latency ms</th><th>p95 TTFT ms</th>
      <th>wall tok/s</th><th>req/s</th><th>report</th>
    </tr>
    {''.join(rows)}
  </table>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body, encoding="utf-8")
    print(f"Wrote experiment report to {output_path}")
