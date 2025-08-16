from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterator, Tuple

import typer

from lexiclass.classifier import SVMDocumentClassifier
from lexiclass.datasets.wikipedia import iter_wikipedia_dataset, iter_wikipedia_dataset_local
from lexiclass.features import FeatureExtractor
from lexiclass.index import DocumentIndex
from lexiclass.plugins import registry
from lexiclass.io import DocumentLoader, load_labels
from lexiclass.tokenization import ICUTokenizer
from lexiclass.config import get_settings
from lexiclass.logging_utils import configure_logging
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
    token_cache_path: Path | None = typer.Option(None, help="Optional JSONL(.gz) token cache to avoid double tokenization"),
) -> None:
    tokenizer_obj = registry.tokenizers[tokenizer](locale=locale)
    feature_extractor = registry.features[features]()
    index = DocumentIndex()
    
    index.build_index(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer_obj,
        index_path=str(index_path),
        document_stream_factory=DocumentLoader.iter_documents_from_directory(str(data_dir)),
        token_cache_path=str(token_cache_path) if token_cache_path else None,
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

