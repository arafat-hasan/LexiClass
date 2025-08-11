from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterator, Tuple

import typer

from ..classifier import SVMDocumentClassifier
from ..datasets.wikipedia import iter_wikipedia_dataset_local
from ..features import FeatureExtractor
from ..index import DocumentIndex
from ..plugins import registry
from ..io import DocumentLoader, load_labels
from ..tokenization import ICUTokenizer
from ..config import get_settings
from ..logging_utils import configure_logging
import random
import numpy as np


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
    verbose: bool = typer.Option(False, "-v", help="Deprecated: use global -v"),
) -> None:
    tokenizer_obj = registry.tokenizers[tokenizer](locale=locale)
    feature_extractor = registry.features[features]()
    index = DocumentIndex()
    def stream_factory() -> Iterator[Tuple[str, str]]:
        return DocumentLoader.iter_documents_from_directory(str(data_dir))
    index.build_index(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer_obj,
        index_path=str(index_path),
        document_stream_factory=stream_factory,
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


@app.command("demo-wikipedia")
def demo_wikipedia(
    max_eval: int = typer.Option(5000, help="Number of streaming eval articles"),
    index_prefix: Path = typer.Option(Path("indexes/wikipedia"), help="Where to save the index"),
    model_path: Path = typer.Option(Path("wikipedia_classifier.pkl"), help="Where to save the model"),
    verbose: bool = typer.Option(False, "-v", help="Deprecated: use global -v"),
) -> None:
    logger = logging.getLogger(__name__)
    categories = ['science_technology', 'history', 'geography', 'biography']

    settings = get_settings()
    num_total = max_eval
    num_train = int(0.8 * num_total)
    num_test = num_total - num_train

    # Prepare a streaming train generator (no HTTP, local cache only)
    train_labels_data: Dict[str, str] = {}

    def train_stream_factory() -> Iterator[Tuple[str, str]]:
        def gen() -> Iterator[Tuple[str, str]]:
            for doc_id, text, category in iter_wikipedia_dataset_local(
                date=settings.wikipedia_date,
                language=settings.wikipedia_language,
                min_length=settings.wikipedia_min_length,
                subset_categories=categories,
                max_articles=num_train,
                skip_first=0,
                cache_dir=settings.hf_cache_dir,
                offline_env=settings.hf_offline,
            ):
                train_labels_data[doc_id] = category
                yield doc_id, text
        return gen()

    classifier = SVMDocumentClassifier()
    logger.info("Building document index from local cached training split (no network)...")
    classifier.build_index(index_path=str(index_prefix), document_stream_factory=train_stream_factory)

    logger.info("Training classifier...")
    classifier.train(train_labels_data)

    # Evaluate on the remaining split in streaming batches to avoid loading all into RAM
    logger.info("Evaluating classifier on test split (streaming batches)...")
    batch_size = 256
    total = 0
    total_correct = 0
    batch_docs: Dict[str, str] = {}
    batch_labels: Dict[str, str] = {}

    def flush_batch() -> None:
        nonlocal total, total_correct, batch_docs, batch_labels
        if not batch_docs:
            return
        preds = classifier.predict(batch_docs)
        total += len(batch_docs)
        for _doc_id, (_pred_label, _score) in preds.items():
            if batch_labels.get(_doc_id) == _pred_label:
                total_correct += 1
        batch_docs = {}
        batch_labels = {}

    for doc_id, text, category in iter_wikipedia_dataset_local(
        date=settings.wikipedia_date,
        language=settings.wikipedia_language,
        min_length=settings.wikipedia_min_length,
        subset_categories=categories,
        max_articles=num_test,
        skip_first=num_train,
        cache_dir=settings.hf_cache_dir,
        offline_env=settings.hf_offline,
    ):
        batch_docs[doc_id] = text
        batch_labels[doc_id] = category
        if len(batch_docs) >= batch_size:
            flush_batch()

    # Flush any remaining examples
    flush_batch()

    accuracy = (total_correct / total) if total > 0 else 0.0
    typer.echo(f"Accuracy: {accuracy*100:.2f}% ({total_correct}/{total})")

    # Save model and index
    classifier.save_model(str(model_path), index_path=str(index_prefix))
    typer.echo(f"Model saved to {model_path}")


