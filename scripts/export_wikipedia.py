#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

try:
    # Use the helpers provided by the package
    from lexiclass.datasets.wikipedia import (
        iter_wikipedia_dataset,
        iter_wikipedia_dataset_local,
    )
    from lexiclass.config import get_settings
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        "This script must be run in an environment where the 'lexiclass' package is installable.\n"
        "From the repo root, run: pip install -e .\n"
        f"Import failed with: {exc}"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a subset of Wikipedia to a directory of .txt files (one doc per file) "
            "and a labels.tsv file (doc_id<TAB>label)."
        )
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="Directory to write .txt files and labels.tsv",
    )
    parser.add_argument(
        "--num-articles",
        type=int,
        default=1000,
        help="Number of articles to export (default: 1000)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Wikipedia snapshot date, e.g. 20231101 (defaults from settings)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code, e.g. en (defaults from settings)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=None,
        help="Minimum character length for an article (defaults from settings)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of categories to keep (e.g. 'science_technology,history'). "
            "If omitted, keep all."
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use local HF cache only (no network)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face datasets cache dir",
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help=(
            "Skip the first N valid articles (useful to split into train/test across multiple runs)."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def export_wikipedia_corpus(
    *,
    output_directory: Path,
    num_articles: int,
    date: Optional[str],
    language: Optional[str],
    min_length: Optional[int],
    categories: Optional[str],
    offline: bool,
    cache_dir: Optional[Path],
    skip_first: int,
) -> None:
    logger = logging.getLogger("export_wikipedia")

    settings = get_settings()
    snapshot_date = date or settings.wikipedia_date
    lang = language or settings.wikipedia_language
    min_len = int(min_length or settings.wikipedia_min_length)
    subset_categories: Optional[List[str]] = None
    if categories:
        subset_categories = [c.strip() for c in categories.split(",") if c.strip()]

    output_directory.mkdir(parents=True, exist_ok=True)
    labels_path = output_directory / "labels.tsv"

    iterator_fn = iter_wikipedia_dataset_local if offline else iter_wikipedia_dataset

    logger.info(
        "Export params: date=%s lang=%s min_length=%d max=%d subset=%s offline=%s cache_dir=%s out=%s",
        snapshot_date,
        lang,
        min_len,
        num_articles,
        subset_categories,
        offline,
        str(cache_dir) if cache_dir else None,
        str(output_directory),
    )

    written = 0
    with labels_path.open("w", encoding="utf-8") as labels_file:
        if iterator_fn is iter_wikipedia_dataset_local:
            iterator = iterator_fn(
                date=snapshot_date,
                language=lang,
                min_length=min_len,
                subset_categories=subset_categories,
                max_articles=num_articles,
                skip_first=skip_first,
                cache_dir=str(cache_dir) if cache_dir else None,
                offline_env=True,
            )
        else:
            iterator = iterator_fn(
                date=snapshot_date,
                language=lang,
                min_length=min_len,
                subset_categories=subset_categories,
                max_articles=num_articles,
            )

        for document_id, text, label in iterator:
            (output_directory / f"{document_id}.txt").write_text(text, encoding="utf-8")
            labels_file.write(f"{document_id}\t{label}\n")
            written += 1

    logger.info("Exported %d documents to %s (labels at %s)", written, output_directory, labels_path)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    # Respect offline mode via env if requested
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    try:
        export_wikipedia_corpus(
            output_directory=args.out_dir,
            num_articles=args.num_articles,
            date=args.date,
            language=args.language,
            min_length=args.min_length,
            categories=args.categories,
            offline=args.offline,
            cache_dir=args.cache_dir,
            skip_first=args.skip_first,
        )
    finally:
        # Force garbage collection and cleanup before exit to prevent PyArrow threading issues
        import gc
        gc.collect()
        
        # Give PyArrow threads time to clean up
        import time
        time.sleep(0.1)


if __name__ == "__main__":
    main()


