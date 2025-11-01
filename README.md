# LexiClass

**Production-ready, extensible document classification toolkit** with modern ML capabilities and plugin architecture. Choose from 11+ pre-built plugins or create your own.

**Version:** 0.3.0 | **License:** MIT

## Features

### Core Capabilities
- **11 Built-in Plugins**: 4 tokenizers, 4 feature extractors, 3 classifiers
- **Flexible Architecture**: Protocol-based plugin system for easy customization
- **Production Ready**: Comprehensive error handling, logging, and configuration
- **Memory Efficient**: Streaming support for large datasets with token caching
- **Multi-Label Support**: Handle both single-label and multi-label classification
- **GPU Acceleration**: Optional GPU support for transformer models
- **CLI & Library**: Use as command-line tool or Python library

### Plugin Ecosystem

**Tokenizers:**
- **ICU**: Fast, locale-aware tokenization (baseline)
- **spaCy**: Linguistic features with stop word filtering
- **SentencePiece**: Trainable subword tokenization for any language
- **Hugging Face**: Access to 1000+ pre-trained tokenizers

**Feature Extractors:**
- **Bag-of-Words (BoW)**: Simple word counts (baseline)
- **TF-IDF**: Term frequency-inverse document frequency weighting
- **FastText**: Subword embeddings with OOV handling
- **Sentence-BERT**: State-of-the-art transformer embeddings

**Classifiers:**
- **SVM**: Fast linear classifier (baseline)
- **XGBoost**: Gradient boosting with GPU support
- **Transformer**: Fine-tune BERT/RoBERTa for classification

### Performance

| Pipeline | Time (10K docs) | Accuracy | Use Case |
|----------|-----------------|----------|----------|
| **Fast** | ~21s | 88% | Prototyping |
| **Production** | ~81s | 92% | Balanced |
| **State-of-the-Art** | ~20min* | 95% | Research |

*With GPU. See [benchmarks](docs/BENCHMARKS.md) for details.

## Quick Start

### Installation

**Basic installation:**
```bash
# Clone or download the repository
git clone https://github.com/yourusername/LexiClass.git
cd LexiClass

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or .venv/bin/activate.fish for Fish shell

# Install LexiClass
pip install -e .
```

**With optional plugins:**
```bash
# For spaCy tokenizer
pip install spacy
python -m spacy download en_core_web_sm

# For XGBoost classifier
pip install xgboost

# For Sentence-BERT and Transformer plugins
pip install sentence-transformers transformers torch

# For SentencePiece tokenizer
pip install sentencepiece

# ICU tokenizer (locale-aware)
pip install PyICU
```

### 30-Second Example

**Using the CLI:**
```bash
# Discover available plugins
lexiclass plugins list

# Build an index from documents
lexiclass build-index ./my_docs ./my_index --tokenizer icu --features tfidf

# Train a classifier
lexiclass train ./my_index ./labels.tsv ./model.pkl

# Make predictions
lexiclass predict ./model.pkl ./my_index ./test_docs --output predictions.tsv

# Evaluate results
lexiclass evaluate predictions.tsv ground_truth.tsv --confusion-matrix
```

**Using the library:**
```python
from lexiclass.plugins import registry
from lexiclass.classifier import SVMDocumentClassifier
from lexiclass.io import DocumentLoader, load_labels

# Create custom pipeline
tokenizer = registry.create('spacy', model_name='en_core_web_sm')
features = registry.create('tfidf', normalize=True)
classifier = registry.create('xgboost', n_estimators=200)

# Build index and train
clf = SVMDocumentClassifier(
    tokenizer=tokenizer,
    feature_extractor=features
)

def doc_stream():
    return DocumentLoader.iter_documents_from_directory('./my_docs')

clf.build_index(
    index_path='./my_index',
    document_stream_factory=doc_stream
)

labels = load_labels('./labels.tsv')
clf.train(labels)
clf.save_model('./model.pkl', index_path='./my_index')

# Predict
test_docs = DocumentLoader.load_documents_from_directory('./test_docs')
predictions = clf.predict(test_docs)
```

## Documentation

### Comprehensive Guides

- **[User Guide](docs/USER_GUIDE.md)** - Complete guide to using LexiClass (CLI, library, integrations)
- **[Plugin Development Guide](docs/PLUGIN_DEVELOPMENT_GUIDE.md)** - Create custom plugins in 30 minutes
- **[Performance Benchmarks](docs/BENCHMARKS.md)** - Detailed performance comparisons and recommendations
- **[Migration Guide](docs/MIGRATION_GUIDE.md)** - Upgrade from v0.1.0 to v0.3.0

### Implementation Details

- **[Phase 1 Complete](PHASE1_COMPLETE.md)** - Plugin infrastructure
- **[Phase 2 Complete](PHASE2_COMPLETE.md)** - Modern ML alternatives
- **[Phase 3 Complete](PHASE3_COMPLETE.md)** - State-of-the-art transformers
- **[Phase 4 Complete](PHASE4_COMPLETE.md)** - Production ready
- **[Implementation Complete](IMPLEMENTATION_COMPLETE.md)** - Overall project summary

### Project Instructions

- **[CLAUDE.md](CLAUDE.md)** - Development setup and architecture overview

## Configuration

LexiClass reads configuration from environment variables (optionally from a `.env` file):

```bash
# Logging
LEXICLASS_LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
LEXICLASS_LOG_FORMAT=text         # text or json
LEXICLASS_LOG_FILE=./app.log      # Optional log file path

# Library-specific logging
LEXICLASS_GENSIM_LOG_LEVEL=WARNING
LEXICLASS_SKLEARN_LOG_LEVEL=WARNING

# Behavior
LEXICLASS_RANDOM_SEED=42          # For reproducibility
LEXICLASS_LOCALE=en               # Default locale for ICU tokenizer
```

Use `-v` flag with CLI commands to enable verbose DEBUG logging.

## CLI Reference

### Plugin Management

```bash
# List all available plugins
lexiclass plugins list

# List by type
lexiclass plugins list --type tokenizer
lexiclass plugins list --type feature_extractor
lexiclass plugins list --type classifier

# Show only available plugins (dependencies installed)
lexiclass plugins list --available-only

# Get detailed information about a plugin
lexiclass plugins describe tfidf
lexiclass plugins describe xgboost
```

### Building an Index

```bash
# Basic: Build index from directory of .txt files
lexiclass build-index ./my_docs ./my_index

# With custom plugins
lexiclass build-index ./my_docs ./my_index \
  --tokenizer spacy \
  --features tfidf

# With token caching (avoids re-tokenization)
lexiclass build-index ./my_docs ./my_index \
  --tokenizer spacy \
  --features tfidf \
  --token-cache-path ./cache.jsonl.gz
```

### Training and Prediction

```bash
# Train classifier (labels.tsv format: doc_id<TAB>label)
lexiclass train ./my_index ./labels.tsv ./model.pkl

# Make predictions
lexiclass predict ./model.pkl ./my_index ./test_docs --output predictions.tsv

# Evaluate predictions
lexiclass evaluate predictions.tsv ground_truth.tsv
lexiclass evaluate predictions.tsv ground_truth.tsv --confusion-matrix
lexiclass evaluate predictions.tsv ground_truth.tsv --format json --output metrics.json
```

### Similarity Search

```bash
# Find similar documents
lexiclass similar ./my_index DOC_ID --top-k 5 --threshold 0.0
```

**For more CLI examples, see the [User Guide](docs/USER_GUIDE.md).**

## Using as a Library

### Basic Usage

```python
from lexiclass.classifier import SVMDocumentClassifier
from lexiclass.io import DocumentLoader, load_labels

# Create classifier
clf = SVMDocumentClassifier()

# Build index from documents
def doc_stream():
    return DocumentLoader.iter_documents_from_directory("./my_docs")

clf.build_index(
    index_path="./my_index",
    document_stream_factory=doc_stream
)

# Train
labels = load_labels("./labels.tsv")
clf.train(labels)
clf.save_model("./model.pkl", index_path="./my_index")

# Predict
test_docs = DocumentLoader.load_documents_from_directory("./test_docs")
predictions = clf.predict(test_docs)  # dict: doc_id -> (label, score)
```

### Using Custom Plugins

```python
from lexiclass.plugins import registry
from lexiclass.classifier import SVMDocumentClassifier

# Create plugins
tokenizer = registry.create('spacy', model_name='en_core_web_sm')
features = registry.create('tfidf', normalize=True)

# Use with classifier
clf = SVMDocumentClassifier(
    tokenizer=tokenizer,
    feature_extractor=features
)
# ... rest of workflow
```

### Advanced: Direct Plugin Usage

```python
from lexiclass.plugins import registry

# Create XGBoost classifier directly
xgb = registry.create('xgboost', n_estimators=200, use_gpu=True)

# Train with feature matrix
xgb.train(feature_matrix, labels)

# Predict
predictions = xgb.predict(feature_matrix)
```

### Integration with External Services

LexiClass can be integrated into web services, background job processors, and streaming pipelines:

- **REST API**: Flask, FastAPI integration examples
- **Background Jobs**: Celery task examples
- **Streaming**: Kafka consumer examples
- **Batch Processing**: Large dataset handling

**See the [User Guide](docs/USER_GUIDE.md) for complete integration examples.**

## Creating Custom Plugins

LexiClass uses a protocol-based architecture that makes it easy to create custom plugins.

### Quick Example

```python
from lexiclass.plugins import registry, PluginType, PluginMetadata

# Define your plugin
class MyTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.lower().split()

# Register it
metadata = PluginMetadata(
    name="mytok",
    display_name="My Custom Tokenizer",
    description="Simple whitespace tokenizer",
    plugin_type=PluginType.TOKENIZER,
    dependencies=[],
    performance_tier="fast",
    quality_tier="basic"
)

registry.register(metadata, lambda: MyTokenizer())

# Use it
tokenizer = registry.create('mytok')
```

### Plugin Protocols

Core interfaces are defined as Protocols in `lexiclass.interfaces`:

- **TokenizerProtocol**: `.tokenize(text) -> list[str]`
- **FeatureExtractorProtocol**: `.fit()`, `.fit_streaming()`, `.transform()`, `.tokens_to_bow()`, `.num_features()`
- **ClassifierProtocol**: `.train()`, `.predict()`, `.save()`, `.load()`
- **DocumentIndexProtocol**: `.build_index()`, `.query_by_id()`, `.query_by_vector()`

**For a complete guide to creating custom plugins, see the [Plugin Development Guide](docs/PLUGIN_DEVELOPMENT_GUIDE.md).**

## Architecture

### Project Structure

```
src/lexiclass/
├── plugins/
│   ├── tokenizers/      # ICU, spaCy, SentencePiece, HuggingFace
│   ├── features/        # BoW, TF-IDF, FastText, Sentence-BERT
│   └── classifiers/     # SVM, XGBoost, Transformer
├── classifier.py        # SVMDocumentClassifier orchestration
├── index.py             # DocumentIndex (similarity search)
├── features.py          # Base feature extraction
├── tokenization.py      # Base tokenization
├── interfaces.py        # Protocol definitions
├── io.py                # Document loading and label parsing
├── config.py            # Environment-based configuration
├── exceptions.py        # Custom exceptions
└── cli/                 # Command-line interface
```

### Key Design Patterns

- **Protocol-based plugins**: Easy to extend without modifying core
- **Streaming support**: Memory-efficient for large datasets
- **Two-pass indexing**: Build indices without loading all data
- **Token caching**: Avoid re-tokenization between passes
- **Dependency checking**: Clear error messages for missing dependencies
- **Lazy loading**: Models loaded only when needed

## Use Cases

### By Dataset Size

| Size | Recommended Pipeline | Time | Accuracy |
|------|---------------------|------|----------|
| Small (<10K) | Any pipeline | Fast | 85-95% |
| Medium (10K-100K) | spaCy + TF-IDF + XGBoost | ~1-2min | 88-92% |
| Large (100K-1M) | ICU + TF-IDF + SVM | ~5-10min | 88-91% |
| Very Large (>1M) | ICU + BoW + SVM | ~20-30min | 85-89% |

### By Accuracy Requirements

- **Quick prototyping (85-88%)**: ICU + BoW + SVM
- **Production balanced (88-92%)**: spaCy + TF-IDF + XGBoost
- **High quality (90-93%)**: spaCy + FastText + XGBoost
- **State-of-the-art (93-95%)**: HuggingFace + Sentence-BERT + Transformer (GPU required)

**See [benchmarks](docs/BENCHMARKS.md) for detailed comparisons.**

## Contributing

Contributions are welcome! Areas for contribution:

- **New plugins**: Add tokenizers, feature extractors, or classifiers
- **Documentation**: Improve guides and examples
- **Testing**: Add unit tests and integration tests
- **Benchmarks**: Run benchmarks on new datasets
- **Bug fixes**: Fix issues and improve error handling

## Troubleshooting

**Installation issues:**
```bash
# Upgrade build tools if installation fails
python -m pip install --upgrade pip setuptools wheel
```

**Missing dependencies:**
```bash
# Check which plugins need dependencies
lexiclass plugins list

# Get installation instructions
lexiclass plugins describe <plugin_name>
```

**macOS OpenMP issue with XGBoost:**
```bash
brew install libomp
```

**Memory issues with large datasets:**
- Use streaming mode with `document_stream_factory`
- Enable token caching with `--token-cache-path`
- Use simpler plugins (ICU + BoW instead of spaCy + SBERT)

**For more troubleshooting help, see the [User Guide](docs/USER_GUIDE.md).**

## License

MIT License - see LICENSE file for details.

## Links

- **Documentation**: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)
- **Issues**: Please report bugs and feature requests via GitHub issues
- **Contributing**: See contributing section above

---

**LexiClass v0.3.0** - Production-ready document classification toolkit
