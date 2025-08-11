# LexiClass

Extensible document classification toolkit (SVM-based) with document similarity search. Built with separation of concerns and plugin-friendly interfaces so you can swap tokenizers, feature extractors, and indexes with minimal friction.

## Features

- Tokenization: ICU-based word boundaries with regex fallback
- Features: Gensim dictionary + sparse bag-of-words; streaming dictionary build
- Indexing: Two-pass, memory-friendly similarity index (Gensim Similarity)
- Classification: Linear SVM (binary, multi-class, multi-label via OvR)
- CLI: Typer CLI for building index, training, predicting, similarity search
- Datasets: Optional, streaming Wikipedia loader (Hugging Face `datasets`)
- Extensibility: Protocol interfaces and a simple plugin registry

## Prepare a development environment

Create and activate a virtual environment.

```bash
# Bash/Zsh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Fish
python3 -m venv .venv
source .venv/bin/activate.fish
python -m pip install --upgrade pip setuptools wheel
```

Install the project (editable recommended):

```bash
pip install -e .

# Optional ICU support for locale-aware tokenization
pip install "PyICU>=2.11"

# For the Wikipedia demo (Hugging Face datasets)
pip install "datasets>=2.12"
```

If editable install fails due to older build tools, either upgrade pip/setuptools or use:

```bash
pip install .
```

## Configuration via environment

LexiClass reads configuration from environment variables (optionally from a `.env` file if `python-dotenv` is installed). Key variables:

- `LEXICLASS_LOG_LEVEL` (default: `INFO`)
- `LEXICLASS_LOG_FORMAT` (`text` or `json`, default: `text`)
- `LEXICLASS_LOG_FILE` (default: unset → stderr)
- `LEXICLASS_GENSIM_LOG_LEVEL` (default: `WARNING`)
- `LEXICLASS_SKLEARN_LOG_LEVEL` (default: `WARNING`)
- `LEXICLASS_RANDOM_SEED` (default: `42`)
- `LEXICLASS_LOCALE` (default: `en`)
- `LEXICLASS_WIKIPEDIA_DATE` (default: `20231101`)
- `LEXICLASS_WIKIPEDIA_LANG` (default: `en`)
- `LEXICLASS_WIKIPEDIA_MIN_LENGTH` (default: `500`)

Example `.env` file:

```bash
LEXICLASS_LOG_FORMAT=json
LEXICLASS_LOG_LEVEL=DEBUG
LEXICLASS_LOG_FILE=./logs/lexiclass.log
LEXICLASS_RANDOM_SEED=1337
```

## CLI usage

View all commands:

```bash
lexiclass --help
```

- Build a similarity index from a directory of `.txt` files (one doc per file):

```bash
lexiclass -v build-index /path/to/texts ./my_index \
  --tokenizer icu --locale en \
  --features bow
```

- Train a classifier with labels (TSV `doc_id<TAB>label`):

```bash
lexiclass -v train ./my_index ./labels.tsv ./model.pkl
```

- Predict on a directory of `.txt` files:

```bash
lexiclass -v predict ./model.pkl ./my_index /path/to/texts --output preds.tsv
```

- Find similar documents by `doc_id`:

```bash
lexiclass -v similar ./my_index DOC_ID --top-k 5 --threshold 0.0
```

- Run the Wikipedia demo (streams from Hugging Face; requires `datasets`):

```bash
LEXICLASS_WIKIPEDIA_MIN_LENGTH=600 lexiclass -v demo-wikipedia --max-eval 2000
```

Labels file format example:

```text
doc123<TAB>science_technology
doc456<TAB>history
```

Notes:

- Index artifacts: `my_index`, `my_index.doc2idx`, and `my_index.extractor` are produced.
- `demo-wikipedia` builds a streaming index from Wikipedia (subset categories), trains on a split of streamed items, evaluates, and saves the model.

## Using as a library in your Python project

Install the package (in your project’s environment):

```bash
pip install -e /path/to/this/repo
```

Build an index, train, and predict programmatically:

```python
from lexiclass.classifier import SVMDocumentClassifier
from lexiclass.io import DocumentLoader, load_labels

# Build an index from a directory of .txt documents
clf = SVMDocumentClassifier()

def stream_factory():
    return DocumentLoader.iter_documents_from_directory("/path/to/texts")

clf.build_index(index_path="my_index", document_stream_factory=stream_factory)

# Train
labels = load_labels("labels.tsv")
clf.train(labels)
clf.save_model("model.pkl", index_path="my_index")

# Predict
docs = DocumentLoader.load_documents_from_directory("/path/to/texts")
preds = clf.predict(docs)  # dict: doc_id -> (label, score)
```

Similarity search (index must be built or loaded):

```python
from lexiclass.index import DocumentIndex
index = DocumentIndex.load_index("my_index")
print(index.query_by_id("doc123", threshold=0.1)[:5])
```

## Plugins and interfaces

- Tokenizers and feature extractors can be swapped at CLI time using the built-in registry:
  - `--tokenizer icu`, `--features bow`
- Programmatically, you can register your own factories:

```python
from lexiclass.plugins import registry

# Register a custom tokenizer
registry.tokenizers["mytok"] = lambda locale='en': MyTokenizer(locale)
# Register a custom feature extractor
registry.features["myfeat"] = lambda: MyFeatureExtractor()
```

Core interfaces are defined as Protocols in `lexiclass.interfaces`:

- `TokenizerProtocol`: `.tokenize(text) -> list[str]`
- `FeatureExtractorProtocol`: `.fit[_streaming]`, `.transform`, `.tokens_to_bow`, `.num_features`
- `DocumentIndexProtocol`: `.build_index(...)`, `.query_by_id`, `.query_by_vector`

This makes it easy to add new implementations without modifying the core code.

## Project structure

- `lexiclass.tokenization.ICUTokenizer`: locale-aware tokenizer
- `lexiclass.features.FeatureExtractor`: Gensim dictionary + sparse features
- `lexiclass.index.DocumentIndex`: streaming two-pass Similarity index
- `lexiclass.classifier.SVMDocumentClassifier`: training, prediction, similarity
- `lexiclass.datasets.wikipedia`: optional streaming dataset helpers
- `lexiclass.io.DocumentLoader`: disk IO; `load_labels` parses TSV labels
- `lexiclass.plugins`: simple registry for plugins
- `lexiclass.interfaces`: Protocols for pluggable components

## Tips and troubleshooting

- If `pip install -e .` fails due to older build tools, upgrade:

  ```bash
  python -m pip install --upgrade pip setuptools wheel
  ```

- For better tokenization across languages, install `PyICU` and use `--tokenizer icu --locale <code>`.
- For large corpora, prefer the streaming `document_stream_factory` approach to avoid loading all data into memory.

## License

MIT
