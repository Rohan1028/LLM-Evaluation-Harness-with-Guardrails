from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from jinja2 import Template

from ..logging import get_logger
from ..utils import ensure_dir, save_json

LOGGER = get_logger(__name__)

LOWER_IS_BETTER_KEYWORDS = ("toxicity", "latency", "cost", "queued", "retry")

COMPARISON_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Guardrail Comparison</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 1.5rem; background-color: #fafafa; }
    table { border-collapse: collapse; width: 100%; margin-top: 1.5rem; background: #fff; }
    th, td { border: 1px solid #ddd; padding: 0.5rem; text-align: left; }
    .improved { color: #0b875b; font-weight: bold; }
    .regressed { color: #d7263d; font-weight: bold; }
    .neutral { color: #555; }
    .meta { margin: 0.25rem 0; }
  </style>
</head>
<body>
  <h1>Before vs After Metrics</h1>
  <p class="meta"><strong>Before:</strong> {{ before_label }}</p>
  <p class="meta"><strong>After:</strong> {{ after_label }}</p>
  <p class="meta"><strong>Improved:</strong> {{ improved }} | <strong>Regressed:</strong> {{ regressed }}</p>
  <table>
    <thead>
      <tr>
        <th>Metric</th>
        <th>Before</th>
        <th>After</th>
        <th>Δ</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
      {% for metric in metrics %}
      <tr>
        <td>{{ metric.name }}</td>
        <td>{{ metric.before | round(4) }}</td>
        <td>{{ metric.after | round(4) }}</td>
        <td>{{ metric.delta | round(4) }}</td>
        <td class="{{ metric.status_class }}">{{ metric.status }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</body>
</html>
"""


def generate_comparison_report(
    before: Path,
    after: Path,
    output_html: Path,
    summary_json: Path | None = None,
) -> Dict[str, Any]:
    before_data = json.loads(before.read_text())
    after_data = json.loads(after.read_text())
    metrics = _build_metric_rows(before_data, after_data)
    improved = sum(1 for row in metrics if row["status"] == "Improved")
    regressed = sum(1 for row in metrics if row["status"] == "Regressed")
    html = Template(COMPARISON_TEMPLATE).render(
        metrics=metrics,
        improved=improved,
        regressed=regressed,
        before_label=before.parent.name,
        after_label=after.parent.name,
    )
    ensure_dir(output_html.parent)
    output_html.write_text(html, encoding="utf-8")
    LOGGER.info("Comparison report written to %s", output_html)
    summary = {
        "before": str(before),
        "after": str(after),
        "metrics": metrics,
        "improved": improved,
        "regressed": regressed,
    }
    if summary_json:
        save_json(summary_json, summary)
    return summary


def _build_metric_rows(
    before: Dict[str, Any], after: Dict[str, Any]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    numeric_keys = sorted(
        key for key in before.keys() & after.keys() if _is_numeric(before[key]) and _is_numeric(after[key])
    )
    for key in numeric_keys:
        before_val = float(before[key])
        after_val = float(after[key])
        delta = after_val - before_val
        direction = _direction_for_metric(key)
        improved = (delta >= 0) if direction == "asc" else (delta <= 0)
        status = "Improved" if improved else ("Regressed" if delta != 0 else "No change")
        status_class = "improved" if improved else ("regressed" if delta != 0 else "neutral")
        rows.append(
            {
                "name": key,
                "before": before_val,
                "after": after_val,
                "delta": delta,
                "status": status,
                "status_class": status_class,
            }
        )
    return rows


def _direction_for_metric(metric: str) -> str:
    lowered = metric.lower()
    for keyword in LOWER_IS_BETTER_KEYWORDS:
        if keyword in lowered:
            return "desc"
    return "asc"


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float))
