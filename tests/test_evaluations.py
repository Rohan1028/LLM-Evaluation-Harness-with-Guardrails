from evalguard.evaluation.ragas_runner import RagasRunner
from evalguard.evaluation.trulens_runner import TruLensRunner
from evalguard.pipelines import PipelineRunResult, PipelineMetadata
from evalguard.pipelines.guardrails import GuardrailResult, Citation
from evalguard.pipelines.rag_pipeline import RetrievedContext


def build_result(answer: str = "Answer [doc:0]") -> PipelineRunResult:
    guardrail = GuardrailResult(
        answer=answer,
        citations=[Citation("doc", 0)],
        toxicity=0.1,
        violations=[],
        needs_retry=False,
        retry_prompt=None,
    )
    context = RetrievedContext(doc_id="doc", chunk_id=0, text="Context chunk", score=1.0)
    meta = PipelineMetadata(
        model="mock",
        provider="mock",
        question="Question",
        retry_count=0,
        guardrail_passed=True,
        citations=[{"doc_id": "doc", "chunk_id": 0}],
        toxicity=0.1,
    )
    return PipelineRunResult(
        question="Question",
        answer=answer,
        ground_truth="Ground truth answer",
        model="mock",
        provider="mock",
        contexts=[context],
        guardrail=guardrail,
        metadata=meta,
    )


def test_ragas_runner() -> None:
    runner = RagasRunner()
    report = runner.evaluate([build_result()])
    assert "answer_relevancy_mean" in report.aggregate


def test_trulens_runner() -> None:
    runner = TruLensRunner()
    report = runner.evaluate([build_result()])
    assert "faithfulness_mean" in report.aggregate
