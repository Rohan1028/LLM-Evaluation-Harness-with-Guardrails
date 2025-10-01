from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
from jinja2 import Template

from ..logging import get_logger
from ..pipelines import PipelineRunResult
from ..utils import ensure_dir, git_sha, save_csv, save_json, timestamp_token
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
) -> Dict[str, Path]:
    ensure_dir(out_dir)
    per_example_records = [result.to_record() for result in run_results]
    save_json(out_dir / "per_example.json", per_example_records)
    save_csv(out_dir / "per_example.csv", per_example_records)

    aggregate: Dict[str, Any] = {}
    for report in reports:
        aggregate.update({f"{report.name}_{k}": v for k, v in report.aggregate.items()})

    tox_stats = {
        "toxicity_mean": sum(res.guardrail.toxicity for res in run_results) / max(len(run_results), 1),
        "toxicity_max": max((res.guardrail.toxicity for res in run_results), default=0.0),
    }
    aggregate.update(tox_stats)
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
    melted = per_example.melt(id_vars=["model", "question"], value_vars=[col for col in per_example.columns if col not in {"model", "question"}])
    fig = px.bar(melted, x="value", y="question", color="model", facet_col="variable", orientation="h")
    return fig.to_json()


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
