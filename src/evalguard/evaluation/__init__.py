from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..pipelines import PipelineRunResult

__all__ = ["EvaluationReport", "PipelineRunResult"]


@dataclass
class EvaluationReport:
    name: str
    per_example: List[Dict[str, Any]]
    aggregate: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "per_example": self.per_example, "aggregate": self.aggregate}
