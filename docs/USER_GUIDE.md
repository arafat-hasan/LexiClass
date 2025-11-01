# LexiClass User Guide

**Version:** 0.3.0
**Complete Guide to Using LexiClass**

This comprehensive guide covers everything you need to know to use LexiClass effectively, from basic CLI commands to advanced library integration.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Plugin System Overview](#plugin-system-overview)
5. [CLI Usage](#cli-usage)
6. [Library Usage](#library-usage)
7. [Integration Guide for External Services](#integration-guide-for-external-services)
8. [Advanced Topics](#advanced-topics)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is LexiClass?

LexiClass is an **extensible document classification toolkit** that combines:
- **Traditional ML methods** (SVM, TF-IDF) for speed
- **Modern ML techniques** (XGBoost, FastText) for better accuracy
- **State-of-the-art transformers** (BERT, Sentence-BERT) for best quality

### Key Features

✅ **11 plugins** covering the complete ML pipeline
✅ **Plugin-based architecture** for easy extensibility
✅ **Memory-efficient** streaming support
✅ **GPU acceleration** for transformer models
✅ **CLI and library** interfaces
✅ **Production-ready** with comprehensive error handling

### Who Should Use LexiClass?

- **Researchers:** Access to SOTA transformer models
- **Data Scientists:** Rapid prototyping and experimentation
- **ML Engineers:** Production-ready classification pipelines
- **Developers:** Easy integration via library API
- **Organizations:** Scalable, well-documented solution

---

## Installation

### Basic Installation

```bash
# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install LexiClass
pip install -e .
```

### Optional Dependencies

Install plugins as needed:

```bash
# Phase 2 plugins (modern ML)
pip install xgboost spacy
python -m spacy download en_core_web_sm

# Phase 3 plugins (transformers)
pip install sentence-transformers transformers torch datasets sentencepiece

# GPU support (optional, for faster inference)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Verify Installation

```bash
# Check version
lexiclass --help

# List available plugins
lexiclass plugins list

# Check Python API
python -c "import lexiclass; print(lexiclass.__version__)"
```

---

## Quick Start

### 30-Second Quick Start (CLI)

```bash
# 1. Build an index from documents
lexiclass build-index ./my_documents ./my_index

# 2. Train a classifier
lexiclass train ./my_index ./labels.tsv ./model.pkl

# 3. Make predictions
lexiclass predict ./model.pkl ./my_index ./test_documents --output predictions.tsv

# 4. Evaluate results
lexiclass evaluate predictions.tsv ground_truth.tsv
```

### 2-Minute Quick Start (Library)

```python
from lexiclass.plugins import registry
from lexiclass.index import DocumentIndex
from lexiclass.classifier import SVMDocumentClassifier
from lexiclass.io import DocumentLoader, load_labels

# 1. Create plugins
tokenizer = registry.create('icu', locale='en')
features = registry.create('tfidf')

# 2. Build index
docs = DocumentLoader.load_documents_from_directory('./my_documents')
index = DocumentIndex()

def doc_stream():
    return DocumentLoader.iter_documents_from_directory('./my_documents')

index.build_index(
    tokenizer=tokenizer,
    feature_extractor=features,
    index_path='./my_index',
    document_stream_factory=doc_stream
)

# 3. Train classifier
classifier = SVMDocumentClassifier(
    tokenizer=tokenizer,
    feature_extractor=features,
    document_index=index
)

labels = load_labels('./labels.tsv')
classifier.train(labels)

# 4. Make predictions
test_docs = DocumentLoader.load_documents_from_directory('./test_documents')
predictions = classifier.predict(test_docs)

# 5. Save model
classifier.save_model('./model.pkl', index_path='./my_index')
```

---

## Plugin System Overview

### Available Plugins

LexiClass provides **11 plugins** across 3 categories:

#### Tokenizers (4)

| Plugin | Description | Speed | Quality | Use Case |
|--------|-------------|-------|---------|----------|
| **icu** | Locale-aware tokenizer | ⚡⚡⚡ | ⭐⭐⭐ | Default, fast |
| **spacy** | Linguistic tokenizer | ⚡⚡⚡ | ⭐⭐⭐⭐ | High quality |
| **sentencepiece** | Trainable subword | ⚡⚡ | ⭐⭐⭐⭐ | Neural models |
| **huggingface** | 1000+ pre-trained | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Transformers |

#### Feature Extractors (4)

| Plugin | Description | Speed | Quality | Output |
|--------|-------------|-------|---------|--------|
| **bow** | Bag-of-words | ⚡⚡⚡ | ⭐⭐ | Sparse |
| **tfidf** | TF-IDF weighting | ⚡⚡⚡ | ⭐⭐⭐ | Sparse |
| **fasttext** | Subword embeddings | ⚡⚡ | ⭐⭐⭐⭐ | Dense |
| **sbert** | Sentence-BERT | ⚡ | ⭐⭐⭐⭐⭐ | Dense |

#### Classifiers (3)

| Plugin | Description | Speed | Quality | Multi-label |
|--------|-------------|-------|---------|-------------|
| **svm** | Linear SVM | ⚡⚡⚡ | ⭐⭐⭐ | ✅ |
| **xgboost** | Gradient boosting | ⚡⚡ | ⭐⭐⭐⭐ | ✅ |
| **transformer** | BERT fine-tuning | ⚡ | ⭐⭐⭐⭐⭐ | ✅ |

### Plugin Discovery

```bash
# List all plugins
lexiclass plugins list

# Filter by type
lexiclass plugins list --type tokenizer
lexiclass plugins list --type feature_extractor
lexiclass plugins list --type classifier

# Show only available plugins
lexiclass plugins list --available-only

# Get detailed information
lexiclass plugins describe tfidf
lexiclass plugins describe sbert
```

---

## CLI Usage

### Complete CLI Reference

#### 1. Building an Index

**Basic usage:**
```bash
lexiclass build-index <data_dir> <index_path>
```

**With options:**
```bash
lexiclass build-index ./documents ./my_index \
  --tokenizer spacy \
  --locale en \
  --features tfidf \
  --token-cache-path ./tokens.jsonl.gz \
  --auto-cache-tokens
```

**Parameters:**
- `data_dir`: Directory containing .txt files
- `index_path`: Output path for index files
- `--tokenizer`: Tokenizer plugin (default: icu)
- `--locale`: Locale for tokenizer (default: en)
- `--features`: Feature extractor plugin (default: bow)
- `--token-cache-path`: Path to cache tokens
- `--auto-cache-tokens`: Automatically cache tokens (default: True)

**Example:**
```bash
# Build index with fast baseline
lexiclass build-index ~/data/agnews/texts ./agnews_index

# Build index with better quality
lexiclass build-index ~/data/agnews/texts ./agnews_index \
  --tokenizer spacy \
  --features tfidf
```

---

#### 2. Training a Classifier

**Basic usage:**
```bash
lexiclass train <index_path> <labels_file> <model_path>
```

**Example:**
```bash
# Train with default SVM
lexiclass train ./agnews_index ./agnews_labels.tsv ./agnews_model.pkl
```

**Labels file format (TSV):**
```
doc_001	sports
doc_002	business
doc_003	technology
...
```

**Multi-label format:**
```
doc_001	sports,health
doc_002	business,technology
doc_003	politics
...
```

---

#### 3. Making Predictions

**Basic usage:**
```bash
lexiclass predict <model_path> <index_path> <data_dir>
```

**With output:**
```bash
lexiclass predict ./agnews_model.pkl ./agnews_index ./test_data \
  --output predictions.tsv
```

**Output format (TSV):**
```
doc_001	sports	5.824180
doc_002	business	6.412939
doc_003	technology	11.013783
...
```

**Example:**
```bash
# Predict and save to file
lexiclass predict ./model.pkl ./index ./test_docs --output preds.tsv

# Predict and show first 20
lexiclass predict ./model.pkl ./index ./test_docs
```

---

#### 4. Evaluating Predictions

**Basic usage:**
```bash
lexiclass evaluate <predictions_file> <ground_truth_file>
```

**With options:**
```bash
lexiclass evaluate predictions.tsv ground_truth.tsv \
  --confusion-matrix \
  --output results.txt \
  --format text
```

**Parameters:**
- `--confusion-matrix`: Show confusion matrix
- `--output`: Save results to file
- `--format`: Output format (text, json, tsv)

**Example:**
```bash
# Basic evaluation (console output)
lexiclass evaluate preds.tsv truth.tsv

# With confusion matrix
lexiclass evaluate preds.tsv truth.tsv --confusion-matrix

# Save results
lexiclass evaluate preds.tsv truth.tsv --output results.txt

# JSON format for programmatic use
lexiclass evaluate preds.tsv truth.tsv --format json --output metrics.json
```

**Output:**
```
Accuracy: 0.9123

Per-class metrics:
                precision    recall  f1-score   support
     sports       0.93      0.91      0.92       500
   business       0.89      0.92      0.90       500
 technology       0.94      0.93      0.93       500
   politics       0.91      0.90      0.90       500

   accuracy                           0.91      2000
  macro avg       0.92      0.91      0.91      2000
weighted avg       0.92      0.91      0.91      2000
```

---

#### 5. Similarity Search

**Basic usage:**
```bash
lexiclass similar <index_path> <doc_id>
```

**With options:**
```bash
lexiclass similar ./my_index DOC_12345 \
  --top-k 10 \
  --threshold 0.5
```

**Example:**
```bash
# Find 5 most similar documents
lexiclass similar ./index doc_001 --top-k 5

# With similarity threshold
lexiclass similar ./index doc_001 --top-k 10 --threshold 0.3
```

**Output:**
```
doc_001	1.0000
doc_145	0.8923
doc_892	0.8654
doc_234	0.8432
doc_567	0.8201
```

---

#### 6. Plugin Management

**List plugins:**
```bash
# All plugins
lexiclass plugins list

# By type
lexiclass plugins list --type tokenizer

# Only available
lexiclass plugins list --available-only
```

**Describe plugin:**
```bash
lexiclass plugins describe tfidf
```

**Output:**
```
Plugin: TF-IDF (tfidf)
Type: feature_extractor
Status: ✓ Available
Description: Term Frequency-Inverse Document Frequency weighting
Performance: fast
Quality: good
Memory: medium
Streaming: Yes
Dependencies: gensim>=4.3, scipy>=1.8
Default params: {'normalize': True, 'smartirs': 'ntc'}
```

---

### CLI Workflow Examples

#### Example 1: Quick Baseline

```bash
# 1. Build index (fast)
lexiclass build-index ~/data/train ./index

# 2. Train model
lexiclass train ./index ./labels.tsv ./model.pkl

# 3. Predict
lexiclass predict ./model.pkl ./index ~/data/test --output preds.tsv

# 4. Evaluate
lexiclass evaluate preds.tsv ~/data/test_labels.tsv
```

**Time:** ~2-3 minutes for 10K documents
**Accuracy:** ~88-90%

---

#### Example 2: Production Quality

```bash
# 1. Build index with better plugins
lexiclass build-index ~/data/train ./index \
  --tokenizer spacy \
  --features tfidf

# 2. Train model
lexiclass train ./index ./labels.tsv ./model.pkl

# 3. Predict
lexiclass predict ./model.pkl ./index ~/data/test --output preds.tsv

# 4. Evaluate with details
lexiclass evaluate preds.tsv ~/data/test_labels.tsv \
  --confusion-matrix \
  --output results.txt
```

**Time:** ~5-10 minutes for 10K documents
**Accuracy:** ~91-93%

---

## Library Usage

### Basic Library Workflow

```python
from lexiclass.plugins import registry, PluginType
from lexiclass.index import DocumentIndex
from lexiclass.classifier import SVMDocumentClassifier
from lexiclass.io import DocumentLoader, load_labels

# 1. Create plugins
tokenizer = registry.create('icu', locale='en')
features = registry.create('tfidf')

# 2. Load documents
docs = DocumentLoader.load_documents_from_directory('./train')

# 3. Build index
index = DocumentIndex()
def doc_stream():
    return DocumentLoader.iter_documents_from_directory('./train')

index.build_index(
    tokenizer=tokenizer,
    feature_extractor=features,
    index_path='./index',
    document_stream_factory=doc_stream
)

# 4. Train classifier
labels = load_labels('./labels.tsv')
classifier = SVMDocumentClassifier(
    tokenizer=tokenizer,
    feature_extractor=features,
    document_index=index
)
classifier.train(labels)

# 5. Save model
classifier.save_model('./model.pkl', index_path='./index')

# 6. Load and predict
classifier = SVMDocumentClassifier.load_model('./model.pkl', index_path='./index')
test_docs = DocumentLoader.load_documents_from_directory('./test')
predictions = classifier.predict(test_docs)

# Results: {doc_id: (predicted_label, confidence_score)}
for doc_id, (label, score) in predictions.items():
    print(f"{doc_id}: {label} (confidence: {score:.4f})")
```

---

### Using Different Plugins

#### Example 1: XGBoost Classifier

```python
from lexiclass.plugins import registry
from scipy import sparse
import numpy as np

# Create XGBoost classifier
xgb = registry.create('xgboost', n_estimators=200, max_depth=8)

# Get feature matrix from index
doc_ids = list(labels.keys())
doc_vectors = []
for doc_id in doc_ids:
    if doc_id in index.doc2idx:
        idx = index.doc2idx[doc_id]
        vector = index.index.vector_by_id(idx)
        doc_vectors.append(vector)

feature_matrix = sparse.vstack(doc_vectors)
label_list = [labels[doc_id] for doc_id in doc_ids]

# Train
xgb.train(feature_matrix, label_list)

# Predict
predictions, scores = xgb.predict(test_matrix)
```

---

#### Example 2: Sentence-BERT Embeddings

```python
from lexiclass.plugins import registry

# Create Sentence-BERT
sbert = registry.create(
    'sbert',
    model_name='all-mpnet-base-v2',
    batch_size=64,
    normalize_embeddings=True
)

# Tokenize documents
documents = [tokenizer.tokenize(text) for text in texts]

# Fit (loads model)
sbert.fit(documents)

# Transform to embeddings
embeddings = sbert.transform(documents)
print(f"Shape: {embeddings.shape}")  # (num_docs, 768)

# Use with XGBoost
xgb = registry.create('xgboost')
xgb.train(embeddings, labels)
predictions, scores = xgb.predict(test_embeddings)
```

---

#### Example 3: Transformer Fine-Tuning

```python
from lexiclass.plugins import registry

# Create transformer classifier
transformer = registry.create(
    'transformer',
    model_name='distilbert-base-uncased',
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5
)

# Train with raw text (not tokenized!)
texts = ["Document text 1...", "Document text 2...", ...]
labels = ["category1", "category2", ...]

transformer.train(texts, labels)

# Predict
test_texts = ["Test doc 1...", "Test doc 2...", ...]
predictions, scores = transformer.predict(test_texts)
```

---

### Advanced Plugin Usage

#### Custom Tokenizer

```python
# Use spaCy with custom settings
spacy_tok = registry.create(
    'spacy',
    model_name='en_core_web_sm',
    lowercase=True,
    remove_punct=True,
    remove_stop=True
)

tokens = spacy_tok.tokenize("The quick brown fox jumps!")
# Result: ['quick', 'brown', 'fox', 'jump']  # No "The", "!", stop words/punct removed
```

---

#### Training SentencePiece

```python
# Create and train SentencePiece
sp = registry.create(
    'sentencepiece',
    vocab_size=8000,
    model_type='unigram'
)

# Train on your corpus
texts = ["Sentence 1", "Sentence 2", ...]
sp.train(texts, output_path='my_sp.model')

# Use for tokenization
tokens = sp.tokenize("New text here")
```

---

### Plugin Discovery and Metadata

```python
from lexiclass.plugins import registry, PluginType

# List available plugins
tokenizers = registry.list_plugins(PluginType.TOKENIZER)
print(f"Tokenizers: {tokenizers}")

# Get metadata
meta = registry.get_metadata('sbert')
print(f"Quality: {meta.quality_tier}")
print(f"Performance: {meta.performance_tier}")
print(f"Memory: {meta.memory_usage}")
print(f"Models: {meta.pretrained_models}")

# Check availability
registration = registry.get('transformer')
if not registration.is_available():
    missing = registration.get_missing_dependencies()
    print(f"Missing: {missing}")
    print(f"Install: pip install {' '.join(missing)}")
else:
    transformer = registration.create()
```

---

## Integration Guide for External Services

### Use Case 1: REST API Service

```python
from flask import Flask, request, jsonify
from lexiclass.classifier import SVMDocumentClassifier
import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Load model once at startup
classifier = None

@app.before_first_request
def load_model():
    global classifier
    logger.info("Loading LexiClass model...")
    classifier = SVMDocumentClassifier.load_model(
        './models/classifier.pkl',
        index_path='./models/index'
    )
    logger.info("Model loaded successfully")

@app.route('/classify', methods=['POST'])
def classify():
    """Classify a document.

    Request:
        {
            "text": "Document text here...",
            "doc_id": "optional_id"
        }

    Response:
        {
            "doc_id": "doc_123",
            "label": "predicted_category",
            "confidence": 0.9234
        }
    """
    try:
        data = request.json
        text = data.get('text')
        doc_id = data.get('doc_id', 'unknown')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Predict
        docs = {doc_id: text}
        predictions = classifier.predict(docs)

        # Extract result
        label, confidence = predictions[doc_id]

        return jsonify({
            'doc_id': doc_id,
            'label': label,
            'confidence': float(confidence),
            'status': 'success'
        })

    except Exception as e:
        logger.error(f"Classification error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/classify/batch', methods=['POST'])
def classify_batch():
    """Classify multiple documents.

    Request:
        {
            "documents": [
                {"doc_id": "doc1", "text": "Text 1"},
                {"doc_id": "doc2", "text": "Text 2"}
            ]
        }

    Response:
        {
            "results": [
                {"doc_id": "doc1", "label": "cat1", "confidence": 0.92},
                {"doc_id": "doc2", "label": "cat2", "confidence": 0.87}
            ]
        }
    """
    try:
        data = request.json
        documents = data.get('documents', [])

        if not documents:
            return jsonify({'error': 'No documents provided'}), 400

        # Prepare docs dict
        docs = {doc['doc_id']: doc['text'] for doc in documents}

        # Predict
        predictions = classifier.predict(docs)

        # Format results
        results = []
        for doc_id, (label, confidence) in predictions.items():
            results.append({
                'doc_id': doc_id,
                'label': label,
                'confidence': float(confidence)
            })

        return jsonify({
            'results': results,
            'count': len(results),
            'status': 'success'
        })

    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Usage:**
```bash
# Single document
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Stock market rises today", "doc_id": "doc_123"}'

# Batch
curl -X POST http://localhost:5000/classify/batch \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"doc_id": "d1", "text": "Sports news..."}]}'
```

---

### Use Case 2: Background Job Processor

```python
from celery import Celery
from lexiclass.classifier import SVMDocumentClassifier
from lexiclass.plugins import registry
import logging

app = Celery('classifier_tasks', broker='redis://localhost:6379/0')
logger = logging.getLogger(__name__)

# Global classifier (loaded once per worker)
classifier = None

@app.task
def classify_document(doc_id, text):
    """Classify a single document asynchronously."""
    global classifier

    # Lazy load classifier
    if classifier is None:
        logger.info("Loading classifier...")
        classifier = SVMDocumentClassifier.load_model(
            './models/classifier.pkl',
            index_path='./models/index'
        )

    # Classify
    docs = {doc_id: text}
    predictions = classifier.predict(docs)
    label, confidence = predictions[doc_id]

    # Store result in database
    store_prediction(doc_id, label, confidence)

    return {
        'doc_id': doc_id,
        'label': label,
        'confidence': float(confidence)
    }

@app.task
def batch_classify(documents):
    """Classify multiple documents in batch."""
    global classifier

    if classifier is None:
        classifier = SVMDocumentClassifier.load_model(
            './models/classifier.pkl',
            index_path='./models/index'
        )

    # Prepare docs
    docs = {doc['id']: doc['text'] for doc in documents}

    # Classify
    predictions = classifier.predict(docs)

    # Store results
    results = []
    for doc_id, (label, confidence) in predictions.items():
        store_prediction(doc_id, label, confidence)
        results.append({
            'doc_id': doc_id,
            'label': label,
            'confidence': float(confidence)
        })

    return results

def store_prediction(doc_id, label, confidence):
    """Store prediction in database."""
    # Your database logic here
    pass
```

**Usage:**
```python
# Queue a classification task
result = classify_document.delay('doc_123', 'Document text...')

# Get result
prediction = result.get(timeout=10)
```

---

### Use Case 3: Streaming Data Pipeline

```python
from kafka import KafkaConsumer, KafkaProducer
from lexiclass.classifier import SVMDocumentClassifier
import json
import logging

logger = logging.getLogger(__name__)

class ClassificationConsumer:
    """Kafka consumer for real-time classification."""

    def __init__(self, bootstrap_servers, input_topic, output_topic):
        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda m: json.dumps(m).encode('utf-8')
        )

        self.output_topic = output_topic

        # Load classifier
        logger.info("Loading classifier...")
        self.classifier = SVMDocumentClassifier.load_model(
            './models/classifier.pkl',
            index_path='./models/index'
        )
        logger.info("Classifier loaded")

    def process_message(self, message):
        """Process a single message."""
        doc_id = message.get('doc_id')
        text = message.get('text')

        if not doc_id or not text:
            logger.warning(f"Invalid message: {message}")
            return

        try:
            # Classify
            docs = {doc_id: text}
            predictions = self.classifier.predict(docs)
            label, confidence = predictions[doc_id]

            # Send result
            result = {
                'doc_id': doc_id,
                'label': label,
                'confidence': float(confidence),
                'timestamp': message.get('timestamp')
            }

            self.producer.send(self.output_topic, value=result)
            logger.info(f"Classified {doc_id}: {label} ({confidence:.4f})")

        except Exception as e:
            logger.error(f"Error processing {doc_id}: {e}")

    def run(self):
        """Start consuming messages."""
        logger.info("Starting consumer...")
        for message in self.consumer:
            self.process_message(message.value)

if __name__ == '__main__':
    consumer = ClassificationConsumer(
        bootstrap_servers=['localhost:9092'],
        input_topic='documents',
        output_topic='classifications'
    )
    consumer.run()
```

---

### Use Case 4: Microservice with FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from lexiclass.classifier import SVMDocumentClassifier
from lexiclass.plugins import registry
import logging

app = FastAPI(title="LexiClass API", version="0.3.0")
logger = logging.getLogger(__name__)

# Global classifier
classifier = None

class Document(BaseModel):
    doc_id: str
    text: str

class ClassificationRequest(BaseModel):
    documents: List[Document]

class ClassificationResult(BaseModel):
    doc_id: str
    label: str
    confidence: float

class BatchClassificationResponse(BaseModel):
    results: List[ClassificationResult]
    count: int

@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global classifier
    logger.info("Loading LexiClass model...")
    classifier = SVMDocumentClassifier.load_model(
        './models/classifier.pkl',
        index_path='./models/index'
    )
    logger.info("Model loaded successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "version": "0.3.0"
    }

@app.get("/plugins")
async def list_plugins():
    """List available plugins."""
    return {
        "tokenizers": registry.list_plugins(plugin_type='tokenizer'),
        "features": registry.list_plugins(plugin_type='feature_extractor'),
        "classifiers": registry.list_plugins(plugin_type='classifier')
    }

@app.post("/classify", response_model=ClassificationResult)
async def classify_single(doc: Document):
    """Classify a single document."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        docs = {doc.doc_id: doc.text}
        predictions = classifier.predict(docs)
        label, confidence = predictions[doc.doc_id]

        return ClassificationResult(
            doc_id=doc.doc_id,
            label=label,
            confidence=float(confidence)
        )
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_batch(request: ClassificationRequest):
    """Classify multiple documents."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        docs = {doc.doc_id: doc.text for doc in request.documents}
        predictions = classifier.predict(docs)

        results = [
            ClassificationResult(
                doc_id=doc_id,
                label=label,
                confidence=float(confidence)
            )
            for doc_id, (label, confidence) in predictions.items()
        ]

        return BatchClassificationResponse(
            results=results,
            count=len(results)
        )
    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Usage:**
```bash
# Health check
curl http://localhost:8000/health

# List plugins
curl http://localhost:8000/plugins

# Classify
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "doc_123", "text": "Your document text"}'

# Batch classify
curl -X POST http://localhost:8000/classify/batch \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"doc_id": "d1", "text": "Text 1"}]}'
```

---

## Advanced Topics

### Memory-Efficient Processing

```python
from lexiclass.index import DocumentIndex

# Use streaming for large datasets
def document_stream_factory():
    """Generator that yields documents one at a time."""
    import glob
    for filepath in glob.glob('./large_dataset/*.txt'):
        with open(filepath) as f:
            doc_id = filepath.split('/')[-1].replace('.txt', '')
            text = f.read()
            yield (doc_id, text)

# Build index with streaming
index = DocumentIndex()
index.build_index(
    tokenizer=tokenizer,
    feature_extractor=features,
    index_path='./index',
    document_stream_factory=document_stream_factory,
    auto_cache_tokens=True  # Cache tokens to avoid re-tokenization
)
```

---

### GPU Acceleration

```python
from lexiclass.plugins import registry

# Sentence-BERT with GPU
sbert = registry.create(
    'sbert',
    model_name='all-mpnet-base-v2',
    device='cuda',  # or 'mps' for Apple Silicon
    batch_size=128  # Larger batch for GPU
)

# XGBoost with GPU
xgb = registry.create(
    'xgboost',
    use_gpu=True,
    tree_method='gpu_hist'
)

# Transformer with GPU
transformer = registry.create(
    'transformer',
    model_name='roberta-base',
    device='cuda',
    batch_size=32
)
```

---

### Multi-Label Classification

```python
# Multi-label labels format
labels = {
    'doc_1': ['category_a', 'category_b'],
    'doc_2': ['category_c'],
    'doc_3': ['category_a', 'category_c', 'category_d'],
}

# Train (automatically detects multi-label)
classifier.train(labels)

# Predict
predictions = classifier.predict(test_docs)

# Results for multi-label
for doc_id, (labels_list, confidence) in predictions.items():
    print(f"{doc_id}: {labels_list} ({confidence:.4f})")
    # Output: doc_1: ['category_a', 'category_b'] (0.8523)
```

---

### Custom Configuration

```python
import os

# Set environment variables
os.environ['LEXICLASS_LOG_LEVEL'] = 'DEBUG'
os.environ['LEXICLASS_LOG_FORMAT'] = 'json'
os.environ['LEXICLASS_RANDOM_SEED'] = '42'

# Or use config
from lexiclass.config import get_settings

settings = get_settings()
print(settings.random_seed)
print(settings.default_locale)
```

---

## Examples

### Complete Example: AG News Classification

```python
#!/usr/bin/env python3
"""
Complete example: Training and evaluating on AG News dataset
"""
from lexiclass.plugins import registry, PluginType
from lexiclass.index import DocumentIndex
from lexiclass.classifier import SVMDocumentClassifier
from lexiclass.io import DocumentLoader, load_labels
from lexiclass.evaluation import evaluate_predictions
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Setup
    logger.info("Setting up LexiClass pipeline...")

    tokenizer = registry.create('spacy', model_name='en_core_web_sm')
    features = registry.create('tfidf', normalize=True)

    # 2. Build index
    logger.info("Building index...")

    def doc_stream():
        return DocumentLoader.iter_documents_from_directory('./data/agnews/train')

    index = DocumentIndex()
    index.build_index(
        tokenizer=tokenizer,
        feature_extractor=features,
        index_path='./agnews_index',
        document_stream_factory=doc_stream,
        auto_cache_tokens=True
    )

    # 3. Train classifier
    logger.info("Training classifier...")

    labels = load_labels('./data/agnews/train_labels.tsv')
    classifier = SVMDocumentClassifier(
        tokenizer=tokenizer,
        feature_extractor=features,
        document_index=index
    )
    classifier.train(labels)

    # 4. Save model
    classifier.save_model('./agnews_model.pkl', index_path='./agnews_index')
    logger.info("Model saved")

    # 5. Evaluate on test set
    logger.info("Evaluating on test set...")

    test_docs = DocumentLoader.load_documents_from_directory('./data/agnews/test')
    predictions = classifier.predict(test_docs)

    # Load ground truth
    test_labels = load_labels('./data/agnews/test_labels.tsv')

    # Calculate metrics
    from lexiclass.evaluation import load_ground_truth

    pred_dict = {doc_id: label for doc_id, (label, _) in predictions.items()}
    metrics = evaluate_predictions(pred_dict, test_labels)

    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")

    print("\nPer-class metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"{class_name:15} P={class_metrics['precision']:.3f} "
              f"R={class_metrics['recall']:.3f} "
              f"F1={class_metrics['f1']:.3f}")

if __name__ == '__main__':
    main()
```

---

## Troubleshooting

### Common Issues

#### Issue: Plugin not found

**Error:**
```
PluginNotFoundError: Plugin 'sbert' not found
```

**Solution:**
```bash
# Check available plugins
lexiclass plugins list

# Install missing dependencies
pip install sentence-transformers
```

---

#### Issue: Out of memory

**Error:**
```
MemoryError: Cannot allocate memory
```

**Solution:**
```python
# Use streaming instead of loading all at once
def doc_stream():
    # Yield documents one at a time
    for doc in documents:
        yield doc

index.build_index(
    document_stream_factory=doc_stream,
    auto_cache_tokens=True  # Cache to disk
)
```

---

#### Issue: Slow transformer inference

**Solution:**
```python
# Use GPU
sbert = registry.create('sbert', device='cuda', batch_size=128)

# Or use smaller model
sbert = registry.create('sbert', model_name='all-MiniLM-L6-v2')  # Faster

# Or use FastText instead
fasttext = registry.create('fasttext', vector_size=100)
```

---

## Additional Resources

- **Plugin Development:** See `docs/PLUGIN_DEVELOPMENT_GUIDE.md`
- **Benchmarks:** See `docs/BENCHMARKS.md`
- **Migration Guide:** See `docs/MIGRATION_GUIDE.md`
- **Phase Documentation:** See `PHASE1-4_COMPLETE.md` files

---

**Last Updated:** 2025-11-01
**Version:** 0.3.0

For questions or contributions, please see the GitHub repository.
