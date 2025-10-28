# Dataset Preparation Guide

This guide explains how to **download, filter, and export** datasets from [Hugging Face Datasets](https://huggingface.co/datasets) into a local directory, for use with **Lexiclass** or similar machine learning systems.

---

## Overview

The `download_dataset.py` script provides a standardized way to fetch text datasets, optionally filter and label them, and export them as plain text files with corresponding metadata.

It supports both **labeled datasets** (e.g., AG News, IMDB) and **unlabeled corpora** (e.g., Wikipedia).

---

## Usage

### Basic Syntax

```bash
python scripts/download_dataset.py <output_dir> \
  --dataset <dataset_name> \
  [--subset <subset_name>] \
  [--split <split_name>] \
  [--num-records <N>] \
  [--min-length <min_chars>] \
  [--categories <comma,separated,categories>]
```

### Arguments

| Argument        | Required | Description                                                                        |
| --------------- | -------- | ---------------------------------------------------------------------------------- |
| `output_dir`    | ✅        | Directory where the dataset will be saved.                                         |
| `--dataset`     | ✅        | Name of the Hugging Face dataset (e.g., `ag_news`, `imdb`, `wikimedia/wikipedia`). |
| `--subset`      | Optional | Subset or configuration of the dataset (e.g., `20231101.en` for Wikipedia).        |
| `--split`       | Optional | Split to use, such as `train`, `test`, or `validation`. Default: `train`.          |
| `--num-records` | Optional | Number of samples to export. Default: all available records.                       |
| `--min-length`  | Optional | Minimum text length (in characters) to include.                                    |
| `--categories`  | Optional | Comma-separated list of target categories (used to filter labeled datasets).       |

---

## Output Format

Each record is exported as a `.txt` file in the following directory layout:

```
<output_dir>/
├── metadata.csv
├── world/
│   ├── 000001.txt
│   ├── 000002.txt
│   └── ...
├── business/
│   ├── 000001.txt
│   ├── 000002.txt
│   └── ...
└── unlabeled/
    ├── 000001.txt
    └── ...
```

### `metadata.csv`

Contains metadata for all exported samples:

| id | category | text_path            | text_length |
| -- | -------- | -------------------- | ----------- |
| 1  | world    | world/000001.txt     | 832         |
| 2  | sports   | sports/000002.txt    | 942         |
| 3  | -        | unlabeled/000001.txt | 1021        |

---

## Example Commands

### 1. **Download IMDB Dataset**

```bash
python scripts/download_dataset.py ~/data/imdb \
  --dataset imdb \
  --split train \
  --num-records 5000
```

Exports labeled movie reviews (`pos`, `neg`).

---

### 2. **Download AG News Dataset**

```bash
python scripts/download_dataset.py ~/data/agnews \
  --dataset ag_news \
  --split train \
  --num-records 50000 \
  --min-length 50
```

Exports 4 labeled categories: *World, Sports, Business, Science/Tech*.

---

### 3. **Download Wikipedia Corpus**

```bash
python scripts/download_dataset.py ~/data/wiki \
  --dataset wikimedia/wikipedia \
  --subset 20231101.en \
  --split train \
  --num-records 50000 \
  --min-length 500
```

Exports plain text articles (no labels).

---

## Automatic Label Handling

The script automatically maps known dataset labels to human-readable categories.
For example:

| Dataset   | Raw Label | Mapped Category    |
| --------- | --------- | ------------------ |
| ag_news   | 0         | world              |
| ag_news   | 1         | sports             |
| ag_news   | 2         | business           |
| ag_news   | 3         | science_technology |
| imdb      | 0         | neg                |
| imdb      | 1         | pos                |
| wikipedia | -         | unlabeled          |

If a dataset does not contain labels (like Wikipedia), all records are placed in the `unlabeled/` folder.

---

## Implementation Notes

* Uses `datasets.load_dataset()` with streaming mode for large datasets.
* Automatically skips malformed or empty records.
* Uses Hugging Face caching for efficient re-runs.
* Compatible with both **offline** (cached) and **online** modes.
* Exports human-readable logs with timestamps and progress indicators.

---

## Common Issues

| Problem                                                       | Possible Cause                                 | Fix                                                                        |
| ------------------------------------------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------- |
| `ValueError: BuilderConfig ... doesn't have a 'offline' key.` | Old-style dataset script (e.g. `wikipedia.py`) | Use canonical dataset names from Hugging Face Hub (`wikimedia/wikipedia`). |
| `Export complete. Total written: 0`                           | Filters excluded all records                   | Check your `--categories` or `--min-length` filter.                        |
| Process aborted (SIGABRT)                                     | Large dataset + memory exhaustion              | Use fewer records (`--num-records 10000`) or enable streaming.             |

---

## Recommended Datasets

| Dataset      | Type                    | Hugging Face Name     | Labels                                              |
| ------------ | ----------------------- | --------------------- | --------------------------------------------------- |
| IMDB         | Sentiment               | `imdb`                | `pos`, `neg`                                        |
| AG News      | News Classification     | `ag_news`             | `world`, `sports`, `business`, `science_technology` |
| Wikipedia    | General Corpus          | `wikimedia/wikipedia` | None                                                |
| DBpedia 14   | Ontology Classification | `dbpedia_14`          | 14 categories                                       |
| Yelp Reviews | Sentiment               | `yelp_polarity`       | `pos`, `neg`                                        |

