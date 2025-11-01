from __future__ import annotations
import os
from pathlib import Path
import random
import numpy as np
import typer

from lexiclass.index import DocumentIndex
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
    try:
        kwargs = {'locale': locale} if tokenizer == 'icu' else {}
        tokenizer_obj = registry.create(tokenizer, plugin_type=PluginType.TOKENIZER, **kwargs)
    except Exception as e:
        typer.echo(f"Error creating tokenizer '{tokenizer}': {e}", err=True)
        typer.echo(f"Available tokenizers: {', '.join(registry.list_plugins(PluginType.TOKENIZER))}", err=True)
        raise typer.Exit(code=1)

    try:
        feature_extractor = registry.create(features, plugin_type=PluginType.FEATURE_EXTRACTOR)
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
    classifier: str = typer.Option('svm', help="Classifier plugin (e.g., svm, xgboost, transformer)"),
) -> None:
    """Train a classifier on indexed documents.

    Available classifiers: svm (default), xgboost, transformer

    Use 'lexiclass plugins list --type classifier' to see all available classifiers.
    """

    try:
        classifier_obj = registry.create(classifier, plugin_type=PluginType.CLASSIFIER)
    except Exception as e:
        typer.echo(f"Error creating classifier '{classifier}': {e}", err=True)
        typer.echo(f"Available classifiers: {', '.join(registry.list_plugins(PluginType.CLASSIFIER))}", err=True)
        raise typer.Exit(code=1)

    # Load index and extract features
    index = DocumentIndex.load_index(str(index_path))
    labels_dict = load_labels(str(labels_file))

    # Extract feature vectors for training documents
    import pickle
    from scipy import sparse
    import numpy as np

    doc_ids = list(labels_dict.keys())
    feature_vectors = []
    valid_labels = []

    for doc_id in doc_ids:
        if doc_id in index.doc2idx:
            idx = index.doc2idx[doc_id]
            vector = index.index.vector_by_id(idx)
            feature_vectors.append(vector)
            valid_labels.append(labels_dict[doc_id])

    if not feature_vectors:
        typer.echo("Error: No documents found in index matching the labels file", err=True)
        raise typer.Exit(code=1)

    # Stack vectors into matrix (handle both sparse and dense)
    # Check if vectors are sparse or dense
    if sparse.issparse(feature_vectors[0]):
        # Sparse features (e.g., TF-IDF, BoW)
        feature_matrix = sparse.vstack(feature_vectors)
    else:
        # Dense features (e.g., Sentence-BERT, FastText)
        feature_matrix = np.vstack(feature_vectors)

    # Train classifier
    typer.echo(f"Training {classifier} classifier on {len(valid_labels)} documents...")
    classifier_obj.train(feature_matrix, valid_labels)

    # Save classifier and index path
    with open(str(model_path), 'wb') as f:
        pickle.dump({
            'classifier': classifier_obj,
            'classifier_type': classifier,
            'index_path': str(index_path),
        }, f, protocol=2)

    typer.echo(f"Model trained and saved to {model_path}")


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
    import pickle
    from scipy import sparse

    # Try to load as plugin-based classifier first
    try:
        with open(str(model_path), 'rb') as f:
            model_data = pickle.load(f)

        # Check if it's a plugin-based classifier
        # if isinstance(model_data, dict) and 'classifier_type' in model_data:
        classifier_obj = model_data['classifier']
        classifier_type = model_data['classifier_type']
        stored_index_path = model_data.get('index_path', str(index_path))

        typer.echo(f"Using {classifier_type} classifier")

        # Load index
        index = DocumentIndex.load_index(stored_index_path)

        # Load documents
        docs = DocumentLoader.load_documents_from_directory(str(data_dir))

        # Extract features for prediction
        doc_ids = list(docs.keys())
        feature_vectors = []

        # Load feature extractor
        with open(stored_index_path + '.extractor', 'rb') as f:
            feature_extractor = pickle.load(f)

        # For each document, tokenize and extract features
        for doc_id in doc_ids:
            text = docs[doc_id]
            # Use the tokenizer from the index (we need to reconstruct it)
            # For now, we'll use the document index approach
            if doc_id in index.doc2idx:
                idx = index.doc2idx[doc_id]
                vector = index.index.vector_by_id(idx)
                feature_vectors.append(vector)
            else:
                # Document not in index, need to tokenize and extract features
                # This requires access to the tokenizer which we don't have saved
                # For simplicity, we'll skip documents not in the index
                typer.echo(f"Warning: Document {doc_id} not found in index, skipping", err=True)
                continue

        if not feature_vectors:
            typer.echo("Error: No documents found for prediction", err=True)
            raise typer.Exit(code=1)

        # Stack vectors into matrix (handle both sparse and dense)
        import numpy as np
        if sparse.issparse(feature_vectors[0]):
            # Sparse features (e.g., TF-IDF, BoW)
            feature_matrix = sparse.vstack(feature_vectors)
        else:
            # Dense features (e.g., Sentence-BERT, FastText)
            feature_matrix = np.vstack(feature_vectors)

        # Predict
        typer.echo(f"Making predictions on {len(doc_ids)} documents...")
        predictions, scores = classifier_obj.predict(feature_matrix)

        # Format results
        preds = {}
        for i, doc_id in enumerate(doc_ids):
            if i < len(predictions):
                preds[doc_id] = (predictions[i], float(scores[i]))

    except Exception as e:
        typer.echo(f"Error loading model: {e}", err=True)
        raise typer.Exit(code=1)

    # Output predictions
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
