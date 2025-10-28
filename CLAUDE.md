# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LexiClass is an extensible document classification toolkit using SVM with document similarity search. Built with protocol-based interfaces for easy plugin-style extensibility.

## Development Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or .venv/bin/activate.fish for Fish shell

# Install in editable mode
pip install -e .

# Optional: ICU tokenization support
pip install "PyICU>=2.11"

# Optional: Wikipedia dataset support
pip install "datasets>=2.12"
```

## Common Commands

### Running the CLI

```bash
# Show all available commands
lexiclass --help

# Use -v flag for verbose DEBUG logging (overrides LEXICLASS_LOG_LEVEL)
lexiclass -v <command>
```

### Building an index

```bash
lexiclass build-index /path/to/texts ./my_index \
  --tokenizer icu --locale en \
  --features bow \
  --token-cache-path ./cache.jsonl.gz  # optional, avoids double tokenization
```

### Training a classifier

```bash
# labels.tsv format: doc_id<TAB>label (one per line)
lexiclass train ./my_index ./labels.tsv ./model.pkl
```

### Making predictions

```bash
lexiclass predict ./model.pkl ./my_index /path/to/texts --output preds.tsv
```

### Evaluating predictions

```bash
# Basic evaluation with console output
lexiclass evaluate preds.tsv ground_truth_labels.tsv

# With confusion matrix
lexiclass evaluate preds.tsv ground_truth_labels.tsv --confusion-matrix

# Save results to file
lexiclass evaluate preds.tsv ground_truth_labels.tsv --output results.txt

# JSON format for programmatic use
lexiclass evaluate preds.tsv ground_truth_labels.tsv --format json --output metrics.json

# TSV format for spreadsheet analysis
lexiclass evaluate preds.tsv ground_truth_labels.tsv --format tsv --output metrics.tsv
```

### Similarity search

```bash
lexiclass similar ./my_index DOC_ID --top-k 5 --threshold 0.0
```

### Wikipedia demo

```bash
# Streams from Hugging Face, trains and evaluates
LEXICLASS_WIKIPEDIA_MIN_LENGTH=600 lexiclass demo-wikipedia --max-eval 2000
```

### Exporting datasets

```bash
# Export Wikipedia to directory structure
python scripts/export_wikipedia.py /path/to/output \
  --num-articles 50000 \
  --date 20231101 --language en --min-length 500 \
  --categories science_technology,history,geography,biography,sports,arts_culture,business_economics \
  --offline  # use local HF cache only

# Generic dataset downloader (AG News, IMDB, etc.)
python scripts/download_dataset.py ~/data/agnews \
  --dataset ag_news \
  --split train \
  --num-records 50000 \
  --min-length 50
```

## Architecture Overview

### Core Pipeline Flow

The system follows a pipeline: **Tokenization → Feature Extraction → Indexing → Classification/Similarity**

1. **Tokenization** (`tokenization.py`): ICUTokenizer uses locale-aware word boundaries via PyICU, with regex fallback
2. **Feature Extraction** (`features.py`): FeatureExtractor builds Gensim dictionary + sparse bag-of-words via streaming or in-memory
3. **Indexing** (`index.py`): DocumentIndex uses two-pass streaming build to create Gensim Similarity index without materializing full corpus
4. **Classification** (`classifier.py`): SVMDocumentClassifier orchestrates the pipeline, uses LinearSVC (binary/multi-class/multi-label via OvR)

### Protocol-Based Extensibility

Core abstractions defined in `interfaces.py` as Protocol classes:

- `TokenizerProtocol`: `.tokenize(text) -> list[str]`
- `FeatureExtractorProtocol`: `.fit()`, `.fit_streaming()`, `.transform()`, `.tokens_to_bow()`, `.num_features()`
- `DocumentIndexProtocol`: `.build_index()`, `.query_by_id()`, `.query_by_vector()`
- `ClassifierProtocol`: `.train()`, `.predict()`

This allows swapping implementations without modifying core code.

### Plugin System

The `plugins.py` module provides a simple registry for tokenizers and feature extractors:

```python
from lexiclass.plugins import registry

# Register custom tokenizer
registry.tokenizers["mytok"] = lambda locale='en': MyTokenizer(locale)

# Register custom feature extractor
registry.features["myfeat"] = lambda: MyFeatureExtractor()
```

CLI uses these via `--tokenizer` and `--features` flags.

### Key Implementation Details

#### Two-Pass Streaming Index Build

`DocumentIndex.build_index()` uses a two-pass approach (see `index.py`):

1. **Pass 1**: Stream through documents, tokenize, build dictionary via `fit_streaming()`, optionally cache tokens to JSONL.gz
2. **Pass 2**: Stream again (or use token cache), transform to sparse vectors, feed into `gensim.similarities.Similarity`

This avoids loading entire corpus into memory. Token cache prevents redundant tokenization.

#### Memory-Efficient Feature Extraction

`FeatureExtractor.fit_streaming()` (see `features.py`):

- Processes documents in batches (1000 docs)
- Uses `corpora.Dictionary.add_documents()` with pruning threshold (2M tokens)
- Adaptive batch sizing based on memory monitoring (`memory_utils.py`)
- Multi-threaded transformation using ThreadPoolExecutor

#### SVM Classification

`SVMDocumentClassifier` (see `classifier.py`) handles three cases:

- **Binary**: Uses `BinaryClassEncoder` + single `LinearSVC`
- **Multi-class**: Uses `MultiClassEncoder` + single `LinearSVC` (OvR internally)
- **Multi-label**: Uses `MultiLabelBinarizer` + `OneVsRestClassifier(LinearSVC)`

Encoder automatically selected based on label structure during `train()`.

#### Path Handling

All path parameters (index_path, token_cache_path, filepath) are converted to strings via `str()` to handle both string and `pathlib.Path` objects (see `classifier.py:49-52`, `73-75`).

### Configuration System

Configuration via environment variables (see `config.py`), optionally loaded from `.env` via `python-dotenv`:

- `LEXICLASS_LOG_LEVEL`: default `INFO`, override with `-v` flag
- `LEXICLASS_LOG_FORMAT`: `text` or `json`
- `LEXICLASS_LOG_FILE`: path to log file (default: stderr)
- `LEXICLASS_GENSIM_LOG_LEVEL`: default `WARNING`
- `LEXICLASS_SKLEARN_LOG_LEVEL`: default `WARNING`
- `LEXICLASS_RANDOM_SEED`: default `42`, affects SVM training
- `LEXICLASS_LOCALE`: default `en`
- Wikipedia-specific: `LEXICLASS_WIKIPEDIA_DATE`, `LEXICLASS_WIKIPEDIA_LANG`, `LEXICLASS_WIKIPEDIA_MIN_LENGTH`

Logging configured centrally via `logging_utils.configure_logging()`, called at CLI startup.

### File Structure

```
src/lexiclass/
├── __init__.py
├── classifier.py          # SVMDocumentClassifier orchestrates pipeline
├── features.py            # FeatureExtractor (Gensim dictionary + BoW)
├── index.py               # DocumentIndex (two-pass streaming Similarity)
├── tokenization.py        # ICUTokenizer (locale-aware)
├── interfaces.py          # Protocol definitions
├── plugins.py             # Registry for pluggable components
├── encoding.py            # Label encoders (binary, multi-class, multi-label)
├── io.py                  # DocumentLoader, load_labels
├── config.py              # Environment-based settings
├── logging_utils.py       # Centralized logging configuration
├── memory_utils.py        # Memory monitoring and adaptive batching
├── cli/
│   └── main.py            # Typer-based CLI commands
└── datasets/
    ├── __init__.py
    └── wikipedia.py       # Streaming Wikipedia helpers (HF datasets)
```

### Dataset Scripts

Two standalone scripts in `scripts/`:

- `export_wikipedia.py`: Exports Wikipedia articles to `.txt` files + `labels.tsv` with category-based categorization
- `download_dataset.py`: Generic HF datasets downloader (AG News, IMDB, DBpedia, etc.) to directory structure + `metadata.csv`

Both use streaming mode to handle large datasets efficiently. See `dataset_preparation.md` for details.

### Known Issues

- PyArrow threading errors may occur during HF dataset export (`PyGILState_Release` errors). The scripts have cleanup to minimize this, but it's a known HF datasets library issue.

## Evaluation Metrics

The `evaluate` command compares predictions with ground truth labels and calculates:

- **Accuracy**: Overall percentage of correct predictions
- **Precision**: Per-class and averaged (macro/weighted)
- **Recall**: Per-class and averaged (macro/weighted)
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of samples per class
- **Confusion Matrix**: Optional detailed breakdown of predictions

**Input files**:
- Predictions: `doc_id<TAB>label<TAB>score` (output from `predict` command)
- Ground truth: `doc_id<TAB>label` (standard labels file)

**Output formats**:
- `text`: Human-readable table (default, for console)
- `json`: JSON format (for programmatic use)
- `tsv`: Tab-separated values (for spreadsheet analysis)

**Edge cases handled**:
- Documents in ground truth but not in predictions: excluded with warning
- Documents in predictions but not in ground truth: excluded with warning
- Only the intersection of doc_ids is evaluated

## Labels File Format

TSV format: `doc_id<TAB>label` (one per line)

```text
doc123	science_technology
doc456	history
```

For multi-label: `doc_id<TAB>label1,label2,label3`

## Predictions File Format

TSV format: `doc_id<TAB>label<TAB>score` (output from `predict` command)

```text
00006371	3	5.824180
00000700	2	6.412939
00004566	2	11.013783
```

## Index Artifacts

When building an index at path `my_index`, the following files are created:

- `my_index`: Gensim Similarity index (Similarity.* shards)
- `my_index.doc2idx`: Pickled doc_id → index mapping
- `my_index.extractor`: Pickled FeatureExtractor (dictionary)
- `my_index.tokens.jsonl.gz`: Token cache (if auto_cache_tokens=True)

When saving a model, provide `index_path` to save index artifacts alongside model pickle.
