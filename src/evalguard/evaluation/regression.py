from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from ..logging import get_logger
from ..utils import save_json

LOGGER = get_logger(__name__)


def compare_metrics(
    current_path: Path,
    baseline_path: Path,
    policies: Dict[str, float],
    update_baseline: bool = False,
) -> bool:
    current = json.loads(current_path.read_text())
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}

    regressions = {}
    for metric, threshold in policies.items():
        key = _map_metric(metric)
        current_value = current.get(key)
        if current_value is None:
            LOGGER.warning("Metric '%s' missing from current run", key)
            continue
        baseline_value = baseline.get(key, current_value)
        if metric == "toxicity":
            if current_value > threshold or current_value > baseline_value:
                regressions[metric] = {"current": current_value, "baseline": baseline_value}
        else:
            if current_value < threshold or current_value < baseline_value:
                regressions[metric] = {"current": current_value, "baseline": baseline_value}

    if update_baseline or not baseline_path.exists():
        save_json(baseline_path, current)

    if regressions:
        for metric, values in regressions.items():
            LOGGER.error("Regression detected for %s: %s", metric, values)
        return False
    LOGGER.info("No regressions detected against policy thresholds.")
    return True


def _map_metric(metric: str) -> str:
    mapping = {
        "faithfulness": "ragas_faithfulness_mean",
        "answer_relevancy": "ragas_answer_relevancy_mean",
        "context_precision": "ragas_context_precision_mean",
        "context_recall": "ragas_context_recall_mean",
        "coherence": "trulens_coherence_mean",
        "toxicity": "toxicity_max",
    }
    return mapping.get(metric, metric)
