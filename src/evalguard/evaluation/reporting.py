from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd
import plotly.express as px
from jinja2 import Template

from ..config import TelemetryConfig
from ..logging import get_logger
from ..pipelines import PipelineRunResult
from ..utils import ensure_dir, git_sha, hash_text, save_csv, save_json, save_jsonl, timestamp_token
from . import EvaluationReport

LOGGER = get_logger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>LLM Evaluation Report</title>
  <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>
  <style>
    body { font-family: Arial, sans-serif; padding: 1.5rem; background-color: #fafafa; }
    h1, h2 { color: #222; }
    .metric { display: inline-block; margin-right: 1.5rem; padding: 0.5rem 1rem; border-radius: 6px; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.08); }
    table { border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; background: #fff; }
    th, td { padding: 0.5rem; border: 1px solid #ddd; text-align: left; }
    .status-pass { color: #0b875b; font-weight: bold; }
    .status-fail { color: #d7263d; font-weight: bold; }
  </style>
</head>
<body>
  <h1>LLM Evaluation Report</h1>
  <p>Generated at {{ generated_at }} | Git SHA: {{ git_sha or "n/a" }}</p>

  <h2>Aggregate Metrics</h2>
  <div class=\"metric\">Faithfulness: {{ aggregate.faithfulness_mean | round(3) }}</div>
  <div class=\"metric\">Answer Relevancy: {{ aggregate.answer_relevancy_mean | round(3) }}</div>
  <div class=\"metric\">Context Precision: {{ aggregate.context_precision_mean | round(3) }}</div>
  <div class=\"metric\">Context Recall: {{ aggregate.context_recall_mean | round(3) }}</div>
  <div class=\"metric\">Toxicity (max): {{ aggregate.toxicity_max | round(3) }}</div>

  <h2>Per-Example Metrics</h2>
  {{ table_html | safe }}

  <h2>Metric Distributions</h2>
  <div id=\"metric-chart\"></div>

  <h2>Adversarial Outcomes</h2>
  {{ adversarial_html | safe }}

  <script>
    const data = {{ chart_data | safe }};
    Plotly.newPlot('metric-chart', data, { title: 'Metric distributions by model', barmode: 'group' });
  </script>
</body>
</html>
"""


def persist_run_artifacts(
    out_dir: Path,
    run_results: List[PipelineRunResult],
    reports: List[EvaluationReport],
    adversarial: List[Dict[str, Any]],
    telemetry_config: Optional[TelemetryConfig] = None,
) -> Dict[str, Path]:
    ensure_dir(out_dir)
    per_example_records = [result.to_record() for result in run_results]
    save_json(out_dir / "per_example.json", per_example_records)
    save_csv(out_dir / "per_example.csv", per_example_records)

    aggregate: Dict[str, Any] = {}
    for report in reports:
        aggregate.update({f"{report.name}_{k}": v for k, v in report.aggregate.items()})

    tox_stats = {
        "toxicity_mean": sum(res.guardrail.toxicity for res in run_results)
        / max(len(run_results), 1),
        "toxicity_max": max((res.guardrail.toxicity for res in run_results), default=0.0),
    }
    guardrail_pass_rate = (
        sum(1 for res in run_results if res.guardrail.passed) / max(len(run_results), 1)
    )
    aggregate.update(tox_stats)
    aggregate["guardrail_pass_rate"] = round(guardrail_pass_rate, 4)
    telemetry_cfg = telemetry_config or TelemetryConfig()
    telemetry_rows: List[Dict[str, Any]] = []
    if telemetry_cfg.enabled:
        for result in run_results:
            for attempt in result.telemetry:
                record = attempt.to_dict()
                record.update(
                    question=_maybe_redact(result.question, telemetry_cfg.redact_prompts),
                    answer=_maybe_redact(result.answer, telemetry_cfg.redact_responses),
                    provider=result.provider,
                )
                telemetry_rows.append(record)
        if telemetry_rows:
            total_cost = sum(row.get("cost_usd", 0.0) for row in telemetry_rows)
            avg_latency = sum(row.get("latency_ms", 0.0) for row in telemetry_rows) / max(
                len(telemetry_rows), 1
            )
            aggregate.update(
                {
                    "telemetry_total_cost": round(total_cost, 6),
                    "telemetry_avg_latency_ms": round(avg_latency, 3),
                    "telemetry_request_count": len(telemetry_rows),
                }
            )
            if telemetry_cfg.persist_requests_jsonl:
                save_jsonl(out_dir / "requests.jsonl", telemetry_rows)
            if telemetry_cfg.persist_requests_csv:
                save_csv(out_dir / "requests.csv", telemetry_rows)

    # Fill unprefixed aliases so existing templates can render without errors.
    metric_aliases = {
        "faithfulness_mean": [
            "ragas_faithfulness_mean",
            "trulens_faithfulness_mean",
        ],
        "answer_relevancy_mean": ["ragas_answer_relevancy_mean"],
        "context_precision_mean": ["ragas_context_precision_mean"],
        "context_recall_mean": ["ragas_context_recall_mean"],
    }
    for alias, candidates in metric_aliases.items():
        for candidate in candidates:
            if candidate in aggregate:
                aggregate.setdefault(alias, aggregate[candidate])
                break
    save_json(out_dir / "aggregate.json", aggregate)

    if adversarial:
        save_json(out_dir / "adversarial.json", adversarial)

    metadata = {
        "generated_at": timestamp_token(),
        "git_sha": git_sha(),
        "num_examples": len(run_results),
        "models": sorted({result.model for result in run_results}),
    }
    save_json(out_dir / "metadata.json", metadata)
    return {"per_example": out_dir / "per_example.json", "aggregate": out_dir / "aggregate.json"}


def _build_chart_data(per_example: pd.DataFrame) -> str:
    if per_example.empty:
        return json.dumps([])
    melted = per_example.melt(
        id_vars=["model", "question"],
        value_vars=[col for col in per_example.columns if col not in {"model", "question"}],
    )
    fig = px.bar(
        melted, x="value", y="question", color="model", facet_col="variable", orientation="h"
    )
    return cast(str, fig.to_json())


def generate_html_report(
    aggregate_path: Path,
    per_example_path: Path,
    adversarial_path: Optional[Path],
    output_html: Path,
) -> None:
    aggregate = json.loads(aggregate_path.read_text())
    per_example = pd.read_json(per_example_path)
    adversarial_html = "<p>No adversarial runs.</p>"
    if adversarial_path and adversarial_path.exists():
        adv_df = pd.read_json(adversarial_path)
        adversarial_html = adv_df.to_html(index=False)

    table_html = per_example.to_html(index=False)
    chart_data = _build_chart_data(per_example)

    html = Template(HTML_TEMPLATE).render(
        aggregate=aggregate,
        table_html=table_html,
        adversarial_html=adversarial_html,
        chart_data=chart_data,
        generated_at=timestamp_token(),
        git_sha=git_sha(),
    )
    output_html.write_text(html, encoding="utf-8")
    LOGGER.info("HTML report written to %s", output_html)


def _maybe_redact(text: str, should_redact: bool) -> str:
    if not text:
        return ""
    if not should_redact:
        return text
    return f"[redacted:{hash_text(text)}]"
