from __future__ import annotations
import os
from pathlib import Path
import random
import numpy as np
import typer

from lexiclass.index import DocumentIndex
from lexiclass.classifier import DocumentClassifier
from lexiclass.plugins import registry, PluginType
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

# Create sub-app for plugin commands
plugins_app = typer.Typer(help="Manage and inspect plugins")
app.add_typer(plugins_app, name="plugins")


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
    locale: str = typer.Option('en', help="Tokenizer locale (for ICU tokenizer)"),
    features: str = typer.Option('bow', help="Feature extractor plugin (e.g., bow, tfidf)"),
    token_cache_path: Path | None = typer.Option(None, help="Optional JSONL(.gz) token cache to avoid double tokenization"),
    auto_cache_tokens: bool = typer.Option(True, help="Automatically cache tokens to avoid re-tokenization"),
) -> None:
    """Build a document similarity index from a directory of .txt files.

    By default, tokens are automatically cached to {index_path}.tokens.jsonl.gz
    to avoid tokenizing documents twice (once for dictionary, once for indexing).
    Set --no-auto-cache-tokens to disable this behavior.

    Available tokenizers: icu (default)
    Available feature extractors: bow (default), tfidf

    Use 'lexiclass plugins list' to see all available plugins.
    """
    # Create plugin instances using new registry API
    # Track parameters used for metadata
    tokenizer_kwargs = {'locale': locale} if tokenizer == 'icu' else {}

    try:
        tokenizer_obj = registry.create(tokenizer, plugin_type=PluginType.TOKENIZER, **tokenizer_kwargs)
    except Exception as e:
        typer.echo(f"Error creating tokenizer '{tokenizer}': {e}", err=True)
        typer.echo(f"Available tokenizers: {', '.join(registry.list_plugins(PluginType.TOKENIZER))}", err=True)
        raise typer.Exit(code=1)

    # Track feature extractor parameters
    feature_extractor_kwargs = {}
    try:
        feature_extractor = registry.create(features, plugin_type=PluginType.FEATURE_EXTRACTOR, **feature_extractor_kwargs)
    except Exception as e:
        typer.echo(f"Error creating feature extractor '{features}': {e}", err=True)
        typer.echo(f"Available feature extractors: {', '.join(registry.list_plugins(PluginType.FEATURE_EXTRACTOR))}", err=True)
        raise typer.Exit(code=1)

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
        tokenizer_name=tokenizer,
        feature_extractor_name=features,
        tokenizer_params=tokenizer_kwargs,
        feature_extractor_params=feature_extractor_kwargs,
    )
    # Note: save_index() now handles saving tokenizer and feature_extractor
    typer.echo(f"Index built and saved to {index_path}")


@app.command()
def train(
    index_path: Path = typer.Argument(..., help="Path prefix of an existing index"),
    labels_file: Path = typer.Argument(..., exists=True, help="TSV file of doc_id<TAB>label"),
    model_path: Path = typer.Argument(..., help="Output model path (.pkl)"),
    classifier: str = typer.Option('svm', help="Classifier plugin (e.g., svm, xgboost, transformer)"),
) -> None:
    """Train a classifier on indexed documents.

    Available classifiers: svm (default), xgboost, transformer

    Use 'lexiclass plugins list --type classifier' to see all available classifiers.
    """
    try:
        # Load index and create classifier using the library API
        doc_classifier = DocumentClassifier.load_index(str(index_path))
        doc_classifier.set_classifier(classifier)

        # Train (handles all orchestration internally)
        typer.echo(f"Training {classifier} classifier...")
        doc_classifier.train(str(labels_file))

        # Save model
        doc_classifier.save(str(model_path), index_path=str(index_path))

        typer.echo(f"Model trained and saved to {model_path}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        if "not found" in str(e).lower():
            typer.echo(f"Available classifiers: {', '.join(registry.list_plugins(PluginType.CLASSIFIER))}", err=True)
        raise typer.Exit(code=1)


@app.command()
def predict(
    model_path: Path = typer.Argument(..., exists=True, help="Path to trained model .pkl"),
    index_path: Path = typer.Argument(..., help="Index path prefix used for the model"),
    data_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True, help="Directory of .txt docs to predict"),
    output: Path | None = typer.Option(None, help="Optional output TSV file"),
) -> None:
    """Make predictions on documents using a trained classifier.

    The classifier type is automatically detected from the model file.
    """
    try:
        # Load model using the library API
        doc_classifier = DocumentClassifier.load(str(model_path), index_path=str(index_path))

        typer.echo(f"Using {doc_classifier.classifier_name} classifier")

        # Predict (handles all orchestration internally)
        typer.echo(f"Making predictions...")
        predictions = doc_classifier.predict(str(data_dir))

        if not predictions:
            typer.echo("Warning: No predictions generated", err=True)
            raise typer.Exit(code=1)

        # Output predictions (CLI responsibility - formatting only)
        if output:
            parent_dir = os.path.dirname(output)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(output, 'w', encoding='utf-8') as f:
                for doc_id, (label, score) in predictions.items():
                    f.write(f"{doc_id}\t{label}\t{score:.6f}\n")
            typer.echo(f"Predictions written to {output} ({len(predictions)} documents)")
        else:
            # Show first 20 predictions
            for doc_id, (label, score) in list(predictions.items())[:20]:
                typer.echo(f"{doc_id}\t{label}\t{score:.4f}")
            if len(predictions) > 20:
                typer.echo(f"... ({len(predictions) - 20} more predictions)")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


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


# Plugin management commands

@plugins_app.command("list")
def list_plugins(
    plugin_type: str | None = typer.Option(None, "--type", help="Filter by plugin type (tokenizer, feature_extractor, classifier)"),
    available_only: bool = typer.Option(False, "--available-only", help="Only show plugins with installed dependencies"),
) -> None:
    """List all registered plugins.

    Examples:
        lexiclass plugins list
        lexiclass plugins list --type tokenizer
        lexiclass plugins list --available-only
    """
    # Convert string to PluginType enum if provided
    plugin_type_enum = None
    if plugin_type:
        try:
            plugin_type_enum = PluginType(plugin_type)
        except ValueError:
            typer.echo(f"Invalid plugin type: {plugin_type}", err=True)
            typer.echo(f"Valid types: tokenizer, feature_extractor, classifier", err=True)
            raise typer.Exit(code=1)

    # Get all plugins
    all_plugins = registry._plugins

    # Group by type
    by_type: dict[PluginType, list[str]] = {
        PluginType.TOKENIZER: [],
        PluginType.FEATURE_EXTRACTOR: [],
        PluginType.CLASSIFIER: [],
    }

    for name, registration in all_plugins.items():
        if plugin_type_enum and registration.metadata.plugin_type != plugin_type_enum:
            continue
        if available_only and not registration.is_available():
            continue
        by_type[registration.metadata.plugin_type].append(name)

    # Display grouped results
    if plugin_type_enum:
        # Single type - show as list
        plugins = by_type[plugin_type_enum]
        if not plugins:
            typer.echo(f"No {plugin_type_enum.value} plugins found")
            return

        for name in sorted(plugins):
            registration = registry.get(name)
            status = "✓" if registration.is_available() else "✗"
            typer.echo(f"  {status} {name:<20} {registration.metadata.display_name}")
    else:
        # All types - show grouped
        typer.echo("\nTokenizers:")
        if by_type[PluginType.TOKENIZER]:
            for name in sorted(by_type[PluginType.TOKENIZER]):
                registration = registry.get(name)
                status = "✓" if registration.is_available() else "✗"
                typer.echo(f"  {status} {name:<20} {registration.metadata.display_name}")
        else:
            typer.echo("  (none)")

        typer.echo("\nFeature Extractors:")
        if by_type[PluginType.FEATURE_EXTRACTOR]:
            for name in sorted(by_type[PluginType.FEATURE_EXTRACTOR]):
                registration = registry.get(name)
                status = "✓" if registration.is_available() else "✗"
                typer.echo(f"  {status} {name:<20} {registration.metadata.display_name}")
        else:
            typer.echo("  (none)")

        typer.echo("\nClassifiers:")
        if by_type[PluginType.CLASSIFIER]:
            for name in sorted(by_type[PluginType.CLASSIFIER]):
                registration = registry.get(name)
                status = "✓" if registration.is_available() else "✗"
                typer.echo(f"  {status} {name:<20} {registration.metadata.display_name}")
        else:
            typer.echo("  (none)")

        typer.echo()


@plugins_app.command("describe")
def describe_plugin(
    name: str = typer.Argument(..., help="Plugin name to describe"),
) -> None:
    """Show detailed information about a specific plugin.

    Examples:
        lexiclass plugins describe icu
        lexiclass plugins describe bow
        lexiclass plugins describe svm
    """
    try:
        description = registry.describe(name)
        typer.echo(f"\n{description}\n")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        typer.echo(f"\nAvailable plugins: {', '.join(registry.list_plugins())}", err=True)
        raise typer.Exit(code=1)
