from __future__ import annotations
import os
from pathlib import Path
import random
import numpy as np
import typer

from lexiclass.classifier import SVMDocumentClassifier
from lexiclass.index import DocumentIndex
from lexiclass.plugins import registry
from lexiclass.io import DocumentLoader, load_labels
from lexiclass.config import get_settings
from lexiclass.logging_utils import configure_logging
from lexiclass.evaluation import (
    load_predictions,
    load_ground_truth,
    evaluate_predictions,
    format_results_text,
    format_results_json,
    format_results_tsv,
)

app = typer.Typer(add_completion=False, no_args_is_help=True, help="LexiClass CLI")


@app.callback()
def _init(
    verbose: bool = typer.Option(False, "-v", help="Verbose logging (overrides env)"),
) -> None:
    # Configure logging via centralized utility. Env controls defaults; -v forces DEBUG.
    configure_logging(override_level="DEBUG" if verbose else None)
    # Seed PRNGs for reproducibility (env-driven seed)
    settings = get_settings()
    random.seed(settings.random_seed)
    try:
        np.random.seed(settings.random_seed)
    except Exception:  # noqa: BLE001 - numpy may not be present in minimal envs
        pass


@app.command()
def build_index(
    data_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True, help="Directory of .txt documents"),
    index_path: Path = typer.Argument(..., help="Output path prefix for index"),
    tokenizer: str = typer.Option('icu', help="Tokenizer plugin (e.g., icu)"),
    locale: str = typer.Option('en', help="Tokenizer locale"),
    features: str = typer.Option('bow', help="Feature extractor plugin (e.g., bow)"),
    token_cache_path: Path | None = typer.Option(None, help="Optional JSONL(.gz) token cache to avoid double tokenization"),
    auto_cache_tokens: bool = typer.Option(True, help="Automatically cache tokens to avoid re-tokenization"),
) -> None:
    """Build a document similarity index from a directory of .txt files.

    By default, tokens are automatically cached to {index_path}.tokens.jsonl.gz
    to avoid tokenizing documents twice (once for dictionary, once for indexing).
    Set --no-auto-cache-tokens to disable this behavior.
    """
    tokenizer_obj = registry.tokenizers[tokenizer](locale=locale)
    feature_extractor = registry.features[features]()
    index = DocumentIndex()

    # Create a proper factory function instead of passing a generator
    data_dir_str = str(data_dir)
    def document_stream_factory():
        return DocumentLoader.iter_documents_from_directory(data_dir_str)

    index.build_index(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer_obj,
        index_path=str(index_path),
        document_stream_factory=document_stream_factory,
        token_cache_path=str(token_cache_path) if token_cache_path else None,
        auto_cache_tokens=auto_cache_tokens,
    )
    with open(str(index_path) + '.extractor', 'wb') as f:
        import pickle
        pickle.dump(feature_extractor, f, protocol=2)
    typer.echo(f"Index built and saved to {index_path}")


@app.command()
def train(
    index_path: Path = typer.Argument(..., help="Path prefix of an existing index"),
    labels_file: Path = typer.Argument(..., exists=True, help="TSV file of doc_id<TAB>label"),
    model_path: Path = typer.Argument(..., help="Output model path (.pkl)"),
    verbose: bool = typer.Option(False, "-v", help="Deprecated: use global -v"),
) -> None:
    classifier = SVMDocumentClassifier()
    classifier.load_index(str(index_path))
    labels = load_labels(str(labels_file))
    classifier.train(labels)
    classifier.save_model(str(model_path), index_path=str(index_path))
    typer.echo(f"Model trained and saved to {model_path}")


@app.command()
def predict(
    model_path: Path = typer.Argument(..., exists=True, help="Path to trained model .pkl"),
    index_path: Path = typer.Argument(..., help="Index path prefix used for the model"),
    data_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True, help="Directory of .txt docs to predict"),
    output: Path | None = typer.Option(None, help="Optional output TSV file"),
    verbose: bool = typer.Option(False, "-v", help="Deprecated: use global -v"),
) -> None:
    classifier = SVMDocumentClassifier.load_model(str(model_path), index_path=str(index_path))
    docs = DocumentLoader.load_documents_from_directory(str(data_dir))
    preds = classifier.predict(docs)
    if output:
        parent_dir = os.path.dirname(output)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(output, 'w', encoding='utf-8') as f:
            for doc_id, (label, score) in preds.items():
                f.write(f"{doc_id}\t{label}\t{score:.6f}\n")
        typer.echo(f"Predictions written to {output}")
    else:
        for doc_id, (label, score) in list(preds.items())[:20]:
            typer.echo(f"{doc_id}\t{label}\t{score:.4f}")


@app.command()
def similar(
    index_path: Path = typer.Argument(..., help="Path prefix of an existing index"),
    doc_id: str = typer.Argument(..., help="Document ID contained in the index"),
    top_k: int = typer.Option(10, help="Top K results"),
    threshold: float = typer.Option(0.0, help="Minimum similarity threshold"),
    verbose: bool = typer.Option(False, "-v", help="Deprecated: use global -v"),
) -> None:
    index = DocumentIndex.load_index(str(index_path))
    results = index.query_by_id(doc_id, threshold=threshold)[:top_k]
    for rid, score in results:
        typer.echo(f"{rid}\t{score:.4f}")


@app.command()
def evaluate(
    predictions_file: Path = typer.Argument(..., exists=True, help="TSV file with predictions (doc_id<TAB>label<TAB>score)"),
    ground_truth_file: Path = typer.Argument(..., exists=True, help="TSV file with ground truth labels (doc_id<TAB>label)"),
    output: Path | None = typer.Option(None, help="Optional output file for results"),
    format: str = typer.Option("text", help="Output format: text, json, or tsv"),
    confusion_matrix: bool = typer.Option(False, "--confusion-matrix", help="Show confusion matrix in output"),
) -> None:
    """Evaluate predictions against ground truth labels.

    Calculate precision, recall, F1-score, and accuracy by comparing
    predictions with ground truth labels. Supports multiple output formats.

    Example:
        lexiclass evaluate preds.tsv labels.tsv
        lexiclass evaluate preds.tsv labels.tsv --output results.txt
        lexiclass evaluate preds.tsv labels.tsv --format json --output metrics.json
    """
    try:
        # Load data
        predictions = load_predictions(predictions_file)
        ground_truth = load_ground_truth(ground_truth_file)

        # Calculate metrics
        metrics = evaluate_predictions(predictions, ground_truth)

        # Format output
        if format == "json":
            output_str = format_results_json(metrics)
        elif format == "tsv":
            output_str = format_results_tsv(metrics)
        else:  # text (default)
            output_str = format_results_text(metrics, show_confusion_matrix=confusion_matrix)

        # Write or print output
        if output:
            parent_dir = os.path.dirname(output)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(output, 'w', encoding='utf-8') as f:
                f.write(output_str)
            typer.echo(f"Evaluation results written to {output}")
        else:
            typer.echo(output_str)

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(code=1)
