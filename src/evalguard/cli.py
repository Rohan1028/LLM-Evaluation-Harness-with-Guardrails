from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import typer
from dotenv import load_dotenv
from rich import print as rprint

from .config import Settings, load_settings
from .embeddings import build_embedder
from .evaluation import EvaluationReport
from .evaluation.ragas_runner import RagasRunner
from .evaluation.regression import compare_metrics
from .evaluation.reporting import generate_html_report, persist_run_artifacts
from .evaluation.safety import run_adversarial_suite
from .evaluation.trulens_runner import TruLensRunner
from .logging import configure_logging, get_logger
from .pipelines import RAGPipeline
from .providers import Provider, build_provider_from_spec
from .vectorstore import ChromaVectorStore, ingest_corpus
from .utils import load_jsonl, ensure_dir

load_dotenv()
app = typer.Typer(add_completion=False)
LOGGER = get_logger(__name__)


def _init_settings(config: Optional[Path]) -> Settings:
    settings = load_settings(config)
    configure_logging()
    return settings


def _build_vector_store(settings: Settings, collection: Optional[str] = None) -> ChromaVectorStore:
    collection_name = collection or settings.rag.collection
    return ChromaVectorStore(collection_name=collection_name, persist_directory=str(settings.persist_dir))


def _build_pipeline(provider: Provider, settings: Settings) -> RAGPipeline:
    embedder = build_embedder(prefer_fallback=False)
    vector_store = _build_vector_store(settings)
    return RAGPipeline(provider=provider, embedder=embedder, vector_store=vector_store, settings=settings)


@app.command()
def ingest(
    corpus: Path = typer.Option(Path("./data/corpus"), help="Path to corpus directory"),
    collection: str = typer.Option("demo", help="Name of the Chroma collection"),
    config: Optional[Path] = typer.Option(None, help="Path to configuration YAML"),
) -> None:
    """Embed and ingest a corpus into the vector store."""
    settings = _init_settings(config)
    embedder = build_embedder(prefer_fallback=False)
    vector_store = ChromaVectorStore(collection_name=collection, persist_directory=str(settings.persist_dir))
    stats = ingest_corpus(corpus_dir=corpus, collection_name=collection, embedder=embedder, vector_store=vector_store, rag_config=settings.rag)
    rprint(f"[green]Ingestion complete[/green]: {stats.documents} docs -> {stats.chunks} chunks")


@app.command()
def run(
    suite: str = typer.Option("demo", help="QA suite name (e.g., demo)"),
    models: List[str] = typer.Argument(["mock:deterministic"], help="Provider:model specifications"),
    config: Optional[Path] = typer.Option(None, help="Path to configuration YAML"),
    k: int = typer.Option(None, help="Override retrieval top-k"),
    out: Optional[Path] = typer.Option(None, help="Output directory for run artifacts"),
) -> None:
    """Execute a QA evaluation suite with guardrails and metrics."""
    settings = _init_settings(config)
    top_k = k or settings.rag.retriever_top_k
    data_path = settings.data_dir / "qa" / f"{suite}_qa.jsonl"
    if not data_path.exists():
        data_path = settings.data_dir / "qa" / "demo_qa.jsonl"
    dataset = load_jsonl(data_path)
    embedder = build_embedder()
    vector_store = _build_vector_store(settings)
    results = []
    provider_configs = settings.providers

    for spec in models:
        name, provider = build_provider_from_spec(spec, provider_configs)
        LOGGER.info("Running suite '%s' with provider %s (%s)", suite, name, provider.model)
        pipeline = RAGPipeline(provider=provider, embedder=embedder, vector_store=vector_store, settings=settings)
        pipeline.rag_config.retriever_top_k = top_k
        for row in dataset:
            result = pipeline.run(
                question=row["question"],
                ground_truth=row.get("ground_truth"),
                metadata=row.get("metadata"),
            )
            results.append(result)

    reports: List[EvaluationReport] = []
    ragas_report = RagasRunner().evaluate(results)
    trulens_report = TruLensRunner().evaluate(results)
    reports.extend([ragas_report, trulens_report])

    out_dir = out or (settings.reports_dir / f"run_{suite}")
    artifact_paths = persist_run_artifacts(out_dir=out_dir, run_results=results, reports=reports, adversarial=[])

    rprint(f"[green]Run complete[/green]. Artifacts stored in {out_dir}")
    rprint(json.dumps({report.name: report.aggregate for report in reports}, indent=2))
    rprint(artifact_paths)


@app.command()
def adversarial(
    suite: str = typer.Option("all", help="Suite name (jailbreaks|injections|safety|all)"),
    models: List[str] = typer.Argument(["mock:deterministic"], help="Provider:model spec"),
    config: Optional[Path] = typer.Option(None, help="Path to configuration YAML"),
    out: Optional[Path] = typer.Option(None, help="Output directory"),
) -> None:
    """Run adversarial prompt suites against providers."""
    settings = _init_settings(config)
    suites = ["jailbreaks", "injections", "safety"] if suite == "all" else [suite]
    suite_paths = [settings.data_dir / "adversarial" / f"{name}.yaml" for name in suites]
    suite_paths = [path for path in suite_paths if path.exists()]
    if not suite_paths:
        raise FileNotFoundError("No adversarial suites found.")

    results = []
    for spec in models:
        name, provider = build_provider_from_spec(spec, settings.providers)

        def factory() -> RAGPipeline:
            return _build_pipeline(provider, settings)

        run_results = run_adversarial_suite(suite_paths, factory)
        results.extend(run_results)
        LOGGER.info("Adversarial suite complete for %s", name)

    out_dir = out or (settings.reports_dir / "adversarial")
    ensure_dir(out_dir)
    path = out_dir / "adversarial.json"
    path.write_text(json.dumps([r.__dict__ for r in results], indent=2), encoding="utf-8")
    rprint(f"[green]Adversarial run complete[/green]. Results saved to {path}")


@app.command()
def report(
    input: Path = typer.Option(..., help="Path to aggregate JSON"),
    html: Path = typer.Option(..., help="Output HTML path"),
    per_example: Optional[Path] = typer.Option(None, help="Per-example JSON path"),
    adversarial: Optional[Path] = typer.Option(None, help="Adversarial JSON path"),
) -> None:
    """Render HTML dashboard from stored artifacts."""
    per_example_path = per_example or input.parent / "per_example.json"
    adversarial_path = adversarial or input.parent / "adversarial.json"
    generate_html_report(
        aggregate_path=input,
        per_example_path=per_example_path,
        adversarial_path=adversarial_path if adversarial_path.exists() else None,
        output_html=html,
    )
    rprint(f"[green]Report generated[/green] -> {html}")


@app.command()
def regress(
    current: Path = typer.Option(..., help="Current aggregate JSON"),
    baseline: Path = typer.Option(..., help="Baseline aggregate JSON"),
    config: Optional[Path] = typer.Option(None, help="Path to configuration YAML"),
    update_baseline: bool = typer.Option(False, help="Update baseline with current metrics"),
) -> None:
    """Compare current metrics to baseline thresholds and fail on regression."""
    settings = _init_settings(config)
    ok = compare_metrics(current_path=current, baseline_path=baseline, policies=settings.policies.to_dict(), update_baseline=update_baseline)
    if not ok:
        raise typer.Exit(code=1)
    rprint("[green]Regression check passed[/green]")


if __name__ == "__main__":
    app()
