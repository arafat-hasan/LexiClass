# LexiClass Architecture

**Version:** 0.3.0
**Last Updated:** 2025-11-06

## Overview

LexiClass follows a **layered architecture** with **plugin-based extensibility**, combining several design patterns for flexibility and maintainability.

## Architectural Layers

```
┌─────────────────────────────────────────────────────┐
│              CLI Layer (cli/main.py)                │
│ - Argument parsing, I/O formatting, user interaction│
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│         High-Level API Layer (Facades)              │
│  - DocumentClassifier (classifier.py)               │
│  - DocumentIndex (index.py)                         │
│  - Simple, user-friendly interfaces                 │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│          Core Processing Layer                      │
│  - Tokenization, Feature Extraction                 │
│  - Classification, Similarity Search                │
│  - Evaluation, Encoding                             │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│           Plugin System (Strategy Pattern)          │
│  - Tokenizers (ICU, spaCy, etc.)                    │
│  - Feature Extractors (BoW, TF-IDF, etc.)           │
│  - Classifiers (SVM, XGBoost, Transformer)          │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│           Infrastructure Layer                      │
│  - I/O (DocumentLoader, label loading)              │
│  - Config, Logging, Memory Management               │
│  - Protocol definitions (interfaces.py)             │
└─────────────────────────────────────────────────────┘
```

## Design Patterns Used

### 1. **Facade Pattern** (Primary Pattern)

**Purpose:** Provide simple, high-level interfaces that hide complexity.

**Implementation:**
- `DocumentClassifier` - Facade for the entire classification workflow
- `DocumentIndex` - Facade for indexing and similarity search

**Benefits:**
- Library users get simple APIs
- CLI becomes a thin wrapper
- Complex orchestration is hidden

**Example:**
```python
# Before (complex):
tokenizer = ICUTokenizer()
features = TfidfFeatureExtractor()
index = DocumentIndex()
index.build_index(...)
svm = SVMClassifier()
# ... manual orchestration ...

# After (simple facade):
classifier = DocumentClassifier.from_plugins('icu', 'tfidf', 'svm')
classifier.train(documents, labels)
predictions = classifier.predict(test_documents)
```

### 2. **Strategy Pattern** (Plugin System)

**Purpose:** Allow algorithms to be selected and swapped at runtime.

**Implementation:**
- Plugin registry with factories
- Protocol-based interfaces (TokenizerProtocol, FeatureExtractorProtocol, etc.)
- Runtime plugin selection via registry

**Benefits:**
- Easy to add new algorithms
- No code changes needed to support new plugins
- Loose coupling

**Example:**
```python
# Swap tokenizers without changing code
tokenizer = registry.create('icu', locale='en')
tokenizer = registry.create('spacy', model='en_core_web_sm')

# Swap classifiers
classifier = registry.create('svm')
classifier = registry.create('xgboost', n_estimators=200)
```

### 3. **Factory Pattern** (Plugin Creation)

**Purpose:** Encapsulate object creation logic.

**Implementation:**
- `PluginRegistry` with factory functions
- `registry.create(name, **params)` - creates plugin instances

**Benefits:**
- Centralized creation logic
- Parameter validation
- Dependency checking

### 4. **Protocol Pattern** (Duck Typing with Type Safety)

**Purpose:** Define interfaces without inheritance.

**Implementation:**
- `TokenizerProtocol`
- `FeatureExtractorProtocol`
- `ClassifierProtocol`
- `DocumentIndexProtocol`

**Benefits:**
- Loose coupling
- No forced inheritance
- Type checking support
- Easy testing with mocks

### 5. **Builder Pattern** (Index Construction)

**Purpose:** Separate construction from representation, handle complex builds.

**Implementation:**
- `DocumentIndex.build_index()` with streaming, caching, two-pass processing
- Progressive configuration

**Benefits:**
- Handles memory efficiently
- Supports different build modes (streaming, in-memory)
- Clear construction process

### 6. **Template Method Pattern** (Implicit)

**Purpose:** Define skeleton of algorithm, let subclasses override steps.

**Implementation:**
- Feature extractor base behavior (fit, transform, tokens_to_bow)
- Classifier base behavior (train, predict)

**Benefits:**
- Consistent interfaces
- Reusable structure
- Extensibility

## Core Components

### 1. DocumentClassifier (classifier.py)

**Responsibility:** High-level API for document classification workflow.

**Methods:**
```python
class DocumentClassifier:
    def __init__(tokenizer, feature_extractor, classifier_plugin)

    @classmethod
    def from_plugins(cls, tokenizer_name, feature_name, classifier_name, **kwargs)

    def build_index(documents, index_path)
    def train(labels)
    def predict(documents)
    def evaluate(documents, ground_truth)

    def save(model_path, index_path)
    @classmethod
    def load(cls, model_path)
```

**Design Principles:**
- Facade over complex workflow
- Manages index, tokenizer, features, classifier together
- Handles all state (trained/untrained)
- Provides both convenience and control

### 2. DocumentIndex (index.py)

**Responsibility:** Manage document similarity index with metadata.

**Key Features:**
- Two-pass streaming build
- Stores tokenizer, feature_extractor, metadata
- Self-contained (all components saved together)
- Backward compatible loading

**Metadata Tracking:**
- Plugin names and parameters
- Build timestamp and version
- Document/feature counts

### 3. Plugin System (plugins/)

**Structure:**
```
plugins/
├── base.py              # PluginMetadata, PluginRegistration
├── registry.py          # PluginRegistry (global singleton)
├── tokenizers/          # Tokenizer implementations
├── features/            # Feature extractor implementations
└── classifiers/         # Classifier implementations
```

**Responsibility:**
- Discover and validate plugins
- Create instances with parameters
- Check dependencies
- Provide metadata

### 4. CLI (cli/main.py)

**Responsibility:** Command-line interface ONLY.

**What CLI SHOULD do:**
- Parse command-line arguments
- Format output for console
- Handle file I/O for CLI context
- Call library methods

**What CLI SHOULD NOT do:**
- Business logic
- Data transformations
- Algorithm implementation
- Complex orchestration

**Example (Good CLI Code):**
```python
@app.command()
def train(index_path, labels_file, model_path, classifier='svm'):
    """Train a classifier."""
    # Load data
    labels = load_labels(labels_file)

    # Use library API
    clf = DocumentClassifier.load_index(index_path)
    clf.set_classifier(classifier)
    clf.train(labels)
    clf.save(model_path)

    # CLI output
    typer.echo(f"Model saved to {model_path}")
```

## Data Flow

### Training Workflow

```
Documents (text files)
    ↓
DocumentLoader.load_documents_from_directory()
    ↓
DocumentClassifier.build_index()
    ↓ [Tokenizer → Feature Extractor]
DocumentIndex (saved with metadata)
    ↓
DocumentClassifier.train(labels)
    ↓ [Extract features → Classifier plugin]
Trained Model (saved with index reference)
```

### Prediction Workflow

```
New Documents
    ↓
DocumentClassifier.load(model_path)
    ↓ [Loads index, tokenizer, features, classifier]
DocumentClassifier.predict(documents)
    ↓ [Tokenize → Extract features → Classify]
Predictions (doc_id → label, score)
```

## Extensibility Points

### 1. Adding New Tokenizers

```python
# 1. Implement TokenizerProtocol
class MyTokenizer:
    def tokenize(self, text: str) -> List[str]:
        # Implementation
        pass

# 2. Register plugin
metadata = PluginMetadata(
    name="my_tokenizer",
    plugin_type=PluginType.TOKENIZER,
    # ... other metadata
)
registry.register(metadata, lambda **kwargs: MyTokenizer(**kwargs))
```

### 2. Adding New Feature Extractors

```python
# Implement FeatureExtractorProtocol
class MyFeatureExtractor:
    def fit(self, documents): ...
    def transform(self, documents): ...
    def tokens_to_bow(self, tokens): ...
    def num_features(self): ...

# Register
registry.register(metadata, factory)
```

### 3. Adding New Classifiers

```python
# Implement ClassifierProtocol
class MyClassifier:
    def train(self, feature_matrix, labels): ...
    def predict(self, feature_matrix): ...

# Register
registry.register(metadata, factory)
```

## Separation of Concerns

### Module Responsibilities

| Module | Responsibility | Depends On |
|--------|---------------|------------|
| `cli/main.py` | CLI interface only | All library modules |
| `classifier.py` | Classification workflow facade | index, plugins, io |
| `index.py` | Similarity index management | interfaces |
| `plugins/` | Algorithm implementations | interfaces |
| `io.py` | File I/O operations | None |
| `evaluation.py` | Metrics calculation | sklearn |
| `interfaces.py` | Protocol definitions | None |
| `config.py` | Configuration management | None |

### Dependency Rules

1. **CLI depends on everything, nothing depends on CLI**
2. **High-level modules don't depend on low-level details**
3. **Plugins depend only on interfaces**
4. **No circular dependencies**

## Error Handling Strategy

### 1. Custom Exceptions (exceptions.py)

```python
class PluginNotFoundError(Exception): ...
class PluginDependencyError(Exception): ...
class IndexNotFoundError(Exception): ...
class ClassifierNotTrainedError(Exception): ...
```

### 2. Validation at Boundaries

- CLI validates arguments before calling library
- Library validates inputs at public API boundaries
- Plugins validate parameters on creation

### 3. Logging Strategy

- DEBUG: Internal state, intermediate steps
- INFO: User-facing operations (building index, training)
- WARNING: Recoverable issues (missing optional files)
- ERROR: Failures that require attention

## Testing Strategy

### 1. Unit Tests

- Test each plugin independently
- Mock protocols for isolation
- Test error conditions

### 2. Integration Tests

- Test DocumentClassifier end-to-end
- Test CLI commands
- Test plugin combinations

### 3. Test Structure

```
tests/
├── unit/
│   ├── test_tokenizers.py
│   ├── test_features.py
│   └── test_classifiers.py
├── integration/
│   ├── test_classifier.py
│   ├── test_index.py
│   └── test_cli.py
└── fixtures/
    └── sample_documents/
```

## Performance Considerations

### 1. Memory Management

- Streaming document processing
- Batch processing for transformations
- Adaptive batch sizing based on memory
- Token caching to avoid recomputation

### 2. Parallel Processing

- ThreadPoolExecutor for feature extraction
- Multi-core tokenization (plugin-dependent)
- Batch prediction

### 3. Disk I/O Optimization

- Gzip compression for token cache
- Buffered writing (10MB buffers)
- Efficient pickle protocol (protocol=2)

## Future Extensions

### 1. Async Support

- Async document loading
- Async prediction API
- Batch processing with async/await

### 2. Distributed Processing

- Dask integration for large-scale
- Ray for distributed training
- Spark integration

### 3. Model Versioning

- Track model versions
- Migration tools
- Compatibility checking

### 4. Model Serving

- REST API wrapper
- gRPC service
- Model monitoring

## Configuration Management

### Environment Variables

```bash
LEXICLASS_LOG_LEVEL=INFO
LEXICLASS_LOG_FORMAT=text
LEXICLASS_RANDOM_SEED=42
LEXICLASS_LOCALE=en
```

### Settings Class

```python
from lexiclass.config import get_settings

settings = get_settings()
settings.random_seed  # 42
```

## Migration Guide (for Contributors)

### Adding Business Logic

**❌ Don't add logic to CLI:**
```python
# BAD - in cli/main.py
@app.command()
def train(...):
    # Load index, extract features, train - all in CLI
    index = DocumentIndex.load_index(path)
    features = []
    for doc_id in labels.keys():
        idx = index.doc2idx[doc_id]
        features.append(index.index.vector_by_id(idx))
    # ... more logic ...
```

**✅ Add logic to library modules:**
```python
# GOOD - in classifier.py
class DocumentClassifier:
    def train(self, labels):
        features = self._extract_features_for_labels(labels)
        self.classifier.train(features, labels)

# cli/main.py just calls it
@app.command()
def train(...):
    clf = DocumentClassifier.load_index(index_path)
    clf.train(load_labels(labels_file))
    clf.save(model_path)
```

## Summary

LexiClass architecture prioritizes:

1. **Simplicity** - Facades hide complexity
2. **Extensibility** - Plugin system for algorithms
3. **Separation of Concerns** - Clear module responsibilities
4. **Type Safety** - Protocol-based interfaces
5. **Performance** - Streaming, caching, parallelization
6. **Usability** - Both CLI and library APIs
7. **Reproducibility** - Metadata tracking

The architecture supports both rapid prototyping (simple API) and production use (full control over components).
