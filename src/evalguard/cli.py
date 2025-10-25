from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Callable, List, Optional, TypeVar

import typer
from dotenv import load_dotenv
from rich import print as rprint

from .config import Settings, load_settings
from .embeddings import build_embedder
from .evaluation import EvaluationReport
from .evaluation.comparison import generate_comparison_report
from .evaluation.ragas_runner import RagasRunner
from .evaluation.regression import compare_metrics
from .evaluation.reporting import generate_html_report, persist_run_artifacts
from .evaluation.safety import run_adversarial_suite
from .evaluation.trulens_runner import TruLensRunner
from .logging import configure_logging, get_logger
from .pipelines import RAGPipeline
from .providers import Provider, build_provider_from_spec
from .utils import ensure_dir, load_jsonl
from .vectorstore import ChromaVectorStore, ingest_corpus

load_dotenv()
app = typer.Typer(add_completion=False)
LOGGER = get_logger(__name__)


CommandFunc = TypeVar("CommandFunc", bound=Callable[..., Any])
DEFAULT_QA_MODELS = ("mock:deterministic",)
DEFAULT_ADV_MODELS = ("mock:deterministic",)


def _parse_models(raw: Optional[str], default: tuple[str, ...]) -> List[str]:
    if raw is None:
        return list(default)
    tokens = [part.strip() for part in raw.replace(",", " ").split() if part.strip()]
    return tokens or list(default)


def typer_command(*args: Any, **kwargs: Any) -> Callable[[CommandFunc], CommandFunc]:
    command: Callable[[CommandFunc], CommandFunc] = app.command(*args, **kwargs)
    return command


def _init_settings(config: Optional[Path]) -> Settings:
    settings = load_settings(config)
    configure_logging()
    return settings


def _build_vector_store(settings: Settings, collection: Optional[str] = None) -> ChromaVectorStore:
    collection_name = collection or settings.rag.collection
    return ChromaVectorStore(
        collection_name=collection_name, persist_directory=str(settings.persist_dir)
    )


def _build_pipeline(provider: Provider, settings: Settings) -> RAGPipeline:
    embedder = build_embedder(prefer_fallback=False)
    vector_store = _build_vector_store(settings)
    return RAGPipeline(
        provider=provider, embedder=embedder, vector_store=vector_store, settings=settings
    )


@typer_command()
def ingest(
    corpus: Annotated[Path, typer.Option(help="Path to corpus directory")] = Path("./data/corpus"),
    collection: Annotated[str, typer.Option(help="Name of the Chroma collection")] = "demo",
    config: Annotated[Optional[Path], typer.Option(help="Path to configuration YAML")] = None,
) -> None:
    """Embed and ingest a corpus into the vector store."""
    settings = _init_settings(config)
    embedder = build_embedder(prefer_fallback=False)
    vector_store = ChromaVectorStore(
        collection_name=collection, persist_directory=str(settings.persist_dir)
    )
    stats = ingest_corpus(
        corpus_dir=corpus,
        collection_name=collection,
        embedder=embedder,
        vector_store=vector_store,
        rag_config=settings.rag,
    )
    rprint(f"[green]Ingestion complete[/green]: {stats.documents} docs -> {stats.chunks} chunks")


@typer_command()
def run(
    suite: Annotated[str, typer.Option(help="QA suite name (e.g., demo)")] = "demo",
    models: Annotated[
        Optional[str],
        typer.Option("--model", "-m", "--models", help="Provider:model specifications"),
    ] = None,
    config: Annotated[Optional[Path], typer.Option(help="Path to configuration YAML")] = None,
    k: Annotated[Optional[int], typer.Option(help="Override retrieval top-k")] = None,
    out: Annotated[Optional[Path], typer.Option(help="Output directory for run artifacts")] = None,
) -> None:
    """Execute a QA evaluation suite with guardrails and metrics."""
    settings = _init_settings(config)
    top_k = k or settings.rag.retriever_top_k
    selected_models = _parse_models(models, DEFAULT_QA_MODELS)
    data_path = settings.data_dir / "qa" / f"{suite}_qa.jsonl"
    if not data_path.exists():
        data_path = settings.data_dir / "qa" / "demo_qa.jsonl"
    dataset = load_jsonl(data_path)
    embedder = build_embedder()
    vector_store = _build_vector_store(settings)
    results = []
    provider_configs = settings.providers

    for spec in selected_models:
        name, provider = build_provider_from_spec(spec, provider_configs)
        LOGGER.info("Running suite '%s' with provider %s (%s)", suite, name, provider.model)
        pipeline = RAGPipeline(
            provider=provider, embedder=embedder, vector_store=vector_store, settings=settings
        )
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
    artifact_paths = persist_run_artifacts(
        out_dir=out_dir,
        run_results=results,
        reports=reports,
        adversarial=[],
        telemetry_config=settings.telemetry,
    )

    rprint(f"[green]Run complete[/green]. Artifacts stored in {out_dir}")
    rprint(json.dumps({report.name: report.aggregate for report in reports}, indent=2))
    rprint(artifact_paths)


@typer_command()
def adversarial(
    suite: Annotated[
        str, typer.Option(help="Suite name (jailbreaks|injections|safety|all)")
    ] = "all",
    models: Annotated[
        Optional[str], typer.Option("--model", "-m", "--models", help="Provider:model spec")
    ] = None,
    config: Annotated[Optional[Path], typer.Option(help="Path to configuration YAML")] = None,
    out: Annotated[Optional[Path], typer.Option(help="Output directory")] = None,
) -> None:
    """Run adversarial prompt suites against providers."""
    settings = _init_settings(config)
    suites = ["jailbreaks", "injections", "safety"] if suite == "all" else [suite]
    suite_paths = [settings.data_dir / "adversarial" / f"{name}.yaml" for name in suites]
    suite_paths = [path for path in suite_paths if path.exists()]
    selected_models = _parse_models(models, DEFAULT_ADV_MODELS)
    if not suite_paths:
        raise FileNotFoundError("No adversarial suites found.")

    results = []
    for spec in selected_models:
        name, provider = build_provider_from_spec(spec, settings.providers)
        provider_instance = provider

        def factory(provider_ref: Provider = provider_instance) -> RAGPipeline:
            return _build_pipeline(provider_ref, settings)

        run_results = run_adversarial_suite(suite_paths, factory)
        results.extend(run_results)
        LOGGER.info("Adversarial suite complete for %s", name)

    out_dir = out or (settings.reports_dir / "adversarial")
    ensure_dir(out_dir)
    path = out_dir / "adversarial.json"
    path.write_text(json.dumps([r.__dict__ for r in results], indent=2), encoding="utf-8")
    rprint(f"[green]Adversarial run complete[/green]. Results saved to {path}")


@typer_command()
def report(
    input_path: Annotated[
        Path,
        typer.Option(
            ...,
            "--input-path",
            "--input",
            help="Path to aggregate JSON",
        ),
    ],
    html_path: Annotated[
        Path,
        typer.Option(
            ...,
            "--html-path",
            "--html",
            help="Output HTML path",
        ),
    ],
    per_example: Annotated[Optional[Path], typer.Option(help="Per-example JSON path")] = None,
    adversarial: Annotated[Optional[Path], typer.Option(help="Adversarial JSON path")] = None,
) -> None:
    """Render HTML dashboard from stored artifacts."""
    per_example_path = per_example or input_path.parent / "per_example.json"
    adversarial_path = adversarial or input_path.parent / "adversarial.json"
    generate_html_report(
        aggregate_path=input_path,
        per_example_path=per_example_path,
        adversarial_path=adversarial_path if adversarial_path.exists() else None,
        output_html=html_path,
    )
    rprint(f"[green]Report generated[/green] -> {html_path}")


@typer_command()
def compare(
    before: Annotated[
        Path,
        typer.Option(..., help="Path to baseline aggregate JSON"),
    ],
    after: Annotated[
        Path,
        typer.Option(..., help="Path to current aggregate JSON"),
    ],
    html: Annotated[
        Path,
        typer.Option("--html", help="Output HTML path"),
    ] = Path("reports/comparison.html"),
    summary: Annotated[
        Optional[Path],
        typer.Option(help="Optional JSON summary output path"),
    ] = None,
) -> None:
    """Create a before/after dashboard comparing two evaluation runs."""
    generate_comparison_report(before=before, after=after, output_html=html, summary_json=summary)
    rprint(f"[green]Comparison written[/green] -> {html}")


@typer_command()
def vectorstore(
    action: Annotated[str, typer.Argument(help="stats|export|compact")],
    collection: Annotated[str, typer.Option(help="Collection name")] = "demo",
    export_path: Annotated[
        Optional[Path],
        typer.Option(help="Destination for exports (required for export action)"),
    ] = None,
    config: Annotated[Optional[Path], typer.Option(help="Path to configuration YAML")] = None,
) -> None:
    """Utility operations for the configured Chroma vector store."""
    settings = _init_settings(config)
    store = ChromaVectorStore(
        collection_name=collection, persist_directory=str(settings.persist_dir)
    )
    if action == "stats":
        count = store.count()
        rprint(f"[blue]Collection[/blue] {collection}: {count} vectors")
        return
    if action == "export":
        if not export_path:
            raise typer.BadParameter("export_path is required for export action")
        store.export(export_path)
        rprint(f"[green]Exported collection[/green] -> {export_path}")
        return
    if action == "compact":
        summary = store.compact()
        rprint(f"[green]Compaction summary[/green]: {summary}")
        return
    raise typer.BadParameter(f"Unknown action '{action}'. Expected stats|export|compact.")


@typer_command()
def regress(
    current: Annotated[Path, typer.Option(..., help="Current aggregate JSON")],
    baseline: Annotated[Path, typer.Option(..., help="Baseline aggregate JSON")],
    config: Annotated[Optional[Path], typer.Option(help="Path to configuration YAML")] = None,
    update_baseline: Annotated[
        bool, typer.Option(help="Update baseline with current metrics")
    ] = False,
) -> None:
    """Compare current metrics to baseline thresholds and fail on regression."""
    settings = _init_settings(config)
    ok = compare_metrics(
        current_path=current,
        baseline_path=baseline,
        policies=settings.policies.to_dict(),
        update_baseline=update_baseline,
    )
    if not ok:
        raise typer.Exit(code=1)
    rprint("[green]Regression check passed[/green]")


if __name__ == "__main__":
    app()
