#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from datasets import load_dataset
from typing import Optional, List

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Universal dataset downloader using Hugging Face Datasets."
    )
    parser.add_argument(
        "out_dir", type=Path, help="Directory to store exported data (e.g., ./data)"
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset name on Hugging Face (e.g. 'wikipedia', 'imdb', 'ag_news')"
    )
    parser.add_argument(
        "--subset", default=None, help="Optional dataset subset (e.g., '20231101.en' for Wikipedia)"
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split to download (default: train)"
    )
    parser.add_argument(
        "--num-records", type=int, default=None, help="Limit number of records to download"
    )
    parser.add_argument(
        "--min-length", type=int, default=None, help="Minimum text length (characters)"
    )
    parser.add_argument(
        "--categories", type=str, default=None,
        help="Comma-separated categories or labels to include (if supported by dataset)"
    )
    parser.add_argument(
        "--offline", action="store_true", help="Use local cache only (no internet access)"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="Custom Hugging Face cache directory"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level",
    )
    return parser.parse_args()


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

def export_dataset(
    dataset_name: str,
    subset: Optional[str],
    split: str,
    out_dir: Path,
    num_records: Optional[int],
    min_length: Optional[int],
    categories: Optional[List[str]],
    offline: bool,
    cache_dir: Optional[Path],
):
    import os

    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading dataset: {dataset_name} (subset={subset}, split={split})")

    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        logging.info("Running in offline mode (no network access).")

    # Load dataset in streaming mode for memory safety
    ds_iterable = load_dataset(
        dataset_name,
        subset,
        split=split,
        streaming=True,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    text_field_candidates = ["text", "content", "article", "document", "body"]
    label_field_candidates = ["label", "labels", "category", "topic"]

    text_dir = out_dir / "texts"
    text_dir.mkdir(exist_ok=True, parents=True)
    labels_path = out_dir / "labels.tsv"

    n_written = 0
    with open(labels_path, "w", encoding="utf-8") as label_file:
        label_file.write("id\tlabel\n")

        for idx, record in enumerate(ds_iterable):
            if num_records and idx >= num_records:
                break

            text = None
            label = None
            for key in text_field_candidates:
                if key in record:
                    text = record[key]
                    break
            for key in label_field_candidates:
                if key in record:
                    label = str(record[key])
                    break

            if text is None:
                continue

            if min_length and len(text) < min_length:
                continue

            if categories and label not in categories:
                continue

            file_path = text_dir / f"{idx:08d}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text.strip())

            label_file.write(f"{idx:08d}\t{label or 'N/A'}\n")
            n_written += 1

            if n_written % 100 == 0:
                logging.info(f"Processed {n_written} records...")

    logging.info(f"✅ Export complete. Total written: {n_written}")
    logging.info(f"Data saved in: {out_dir.absolute()}")


def main():
    args = parse_args()
    setup_logging(args.log_level)

    categories = args.categories.split(",") if args.categories else None

    export_dataset(
        dataset_name=args.dataset,
        subset=args.subset,
        split=args.split,
        out_dir=args.out_dir,
        num_records=args.num_records,
        min_length=args.min_length,
        categories=categories,
        offline=args.offline,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
