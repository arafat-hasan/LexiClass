# Migration Guide

**Target Audience:** Users upgrading to LexiClass 0.3.0

This guide helps you migrate from older versions of LexiClass to the new plugin-based architecture.

---

## Table of Contents

1. [Overview of Changes](#overview-of-changes)
2. [Breaking Changes](#breaking-changes)
3. [Migration Paths](#migration-paths)
4. [CLI Changes](#cli-changes)
5. [Library API Changes](#library-api-changes)
6. [New Features](#new-features)
7. [Deprecations](#deprecations)
8. [Troubleshooting](#troubleshooting)

---

## Overview of Changes

### What's New in 0.3.0

**Major Changes:**
- ✅ Complete plugin system (11 plugins)
- ✅ Enhanced CLI with `plugins` commands
- ✅ Modern ML alternatives (XGBoost, Sentence-BERT, Transformers)
- ✅ Better extensibility and flexibility

**Version Timeline:**
- **v0.1.0:** Original hardcoded SVM + BoW
- **v0.2.0:** Plugin system with Phase 1 & 2 (6 plugins)
- **v0.3.0:** Complete ecosystem with Phase 3 (11 plugins)

---

## Breaking Changes

### ⚠️ Important: No Backward Compatibility

Based on project requirements, **v0.3.0 does NOT maintain backward compatibility** with v0.1.0.

**This means:**
- Old code **will** need updates
- Old model files **may** not load
- CLI commands have changed

**Why?** To enable a cleaner, more modern architecture without legacy constraints.

---

## Migration Paths

### From v0.1.0 → v0.3.0

#### 1. Update Imports

**Old (v0.1.0):**
```python
from lexiclass.classifier import SVMDocumentClassifier
from lexiclass.features import FeatureExtractor
from lexiclass.tokenization import ICUTokenizer
```

**New (v0.3.0):**
```python
# Option 1: Use plugin system (recommended)
from lexiclass.plugins import registry, PluginType

tokenizer = registry.create('icu', locale='en')
features = registry.create('bow')  # or 'tfidf'
classifier = registry.create('svm')

# Option 2: Direct imports still work
from lexiclass.plugins.tokenizers.icu import ICUTokenizer
from lexiclass.plugins.features.bow import FeatureExtractor
from lexiclass.plugins.classifiers.svm import SVMClassifier
```

---

#### 2. Update Plugin Access

**Old (v0.1.0):**
```python
from lexiclass.plugins import registry

# Old simple registry
tokenizer = registry.tokenizers['icu'](locale='en')
features = registry.features['bow']()
```

**New (v0.3.0):**
```python
from lexiclass.plugins import registry, PluginType

# New enhanced registry
tokenizer = registry.create('icu', plugin_type=PluginType.TOKENIZER, locale='en')
features = registry.create('bow', plugin_type=PluginType.FEATURE_EXTRACTOR)
```

**Why?** New registry provides:
- Dependency checking
- Metadata access
- Better error messages
- Plugin discovery

---

#### 3. Update CLI Commands

**Old (v0.1.0):**
```bash
# Build index with default plugins
lexiclass build-index ./data ./index
```

**New (v0.3.0):**
```bash
# Same command works, but now you can customize!
lexiclass build-index ./data ./index \
  --tokenizer icu \
  --features bow

# Explore plugins
lexiclass plugins list
lexiclass plugins describe tfidf

# Use better plugins
lexiclass build-index ./data ./index \
  --tokenizer spacy \
  --features tfidf
```

---

### From v0.2.0 → v0.3.0

Good news! **v0.3.0 is mostly backward compatible with v0.2.0**.

**Minor changes:**
- Added 5 new plugins (sentencepiece, huggingface, sbert, transformer)
- Package version updated to 0.3.0
- Documentation enhanced

**Your code should work as-is** if you're on v0.2.0.

---

## CLI Changes

### New Commands

#### `lexiclass plugins` - Plugin Management

```bash
# List all plugins
lexiclass plugins list

# List by type
lexiclass plugins list --type tokenizer
lexiclass plugins list --type feature_extractor
lexiclass plugins list --type classifier

# List only available
lexiclass plugins list --available-only

# Describe plugin
lexiclass plugins describe sbert
lexiclass plugins describe transformer
```

#### Enhanced `build-index`

```bash
# Old: Only default options
lexiclass build-index ./data ./index

# New: Full customization
lexiclass build-index ./data ./index \
  --tokenizer spacy \
  --features tfidf \
  --token-cache-path ./cache.jsonl.gz
```

---

## Library API Changes

### Plugin Creation

**Old (v0.1.0):**
```python
# Manual instantiation
from lexiclass.tokenization import ICUTokenizer
from lexiclass.features import FeatureExtractor

tokenizer = ICUTokenizer(locale='en')
features = FeatureExtractor()
```

**New (v0.3.0):**
```python
# Via registry (recommended)
from lexiclass.plugins import registry

tokenizer = registry.create('icu', locale='en')
features = registry.create('tfidf')  # Now have choices!

# Still can do manual if needed
from lexiclass.plugins.tokenizers.icu import ICUTokenizer
tokenizer = ICUTokenizer(locale='en')
```

---

### Plugin Discovery

**New in v0.3.0:**
```python
from lexiclass.plugins import registry, PluginType

# List available plugins
tokenizers = registry.list_plugins(PluginType.TOKENIZER)
print(tokenizers)  # ['icu', 'spacy', 'sentencepiece', 'huggingface']

# Get metadata
meta = registry.get_metadata('tfidf')
print(meta.quality_tier)  # 'good'
print(meta.dependencies)  # ['gensim>=4.3', 'scipy>=1.8']

# Check availability
registration = registry.get('sbert')
if not registration.is_available():
    missing = registration.get_missing_dependencies()
    print(f"Install: pip install {' '.join(missing)}")
```

---

### Classifier Interface Changes

**v0.1.0 SVMDocumentClassifier:**
```python
from lexiclass.classifier import SVMDocumentClassifier
from lexiclass.index import DocumentIndex

# Build index
index = DocumentIndex()
index.build_index(documents=docs, ...)

# Create classifier with index
classifier = SVMDocumentClassifier(document_index=index)
classifier.train(labels)
predictions = classifier.predict(docs)
```

**v0.3.0 - Now separated:**
```python
from lexiclass.plugins import registry
from lexiclass.index import DocumentIndex

# Build index (same)
index = DocumentIndex()
index.build_index(...)

# Classifiers now work independently
svm = registry.create('svm')

# Get vectors from index
doc_vectors = [index.index.vector_by_id(idx) for idx in doc_ids]

# Train classifier
svm.train(feature_matrix, labels)

# Or use SVMDocumentClassifier wrapper (still exists)
from lexiclass.classifier import SVMDocumentClassifier
classifier = SVMDocumentClassifier(document_index=index)
classifier.train(labels)
```

---

## New Features

### 1. Modern Tokenizers

```python
# spaCy tokenizer
spacy_tok = registry.create('spacy', model_name='en_core_web_sm')

# SentencePiece (trainable!)
sp = registry.create('sentencepiece', vocab_size=8000)
sp.train(texts, output_path='my_model.model')

# HuggingFace (1000+ models)
hf_tok = registry.create('huggingface', model_name='bert-base-uncased')
```

---

### 2. Better Feature Extraction

```python
# TF-IDF (better than BoW)
tfidf = registry.create('tfidf', normalize=True)

# FastText (subword embeddings)
fasttext = registry.create('fasttext', vector_size=200)

# Sentence-BERT (SOTA!)
sbert = registry.create('sbert', model_name='all-mpnet-base-v2')
```

---

### 3. Advanced Classifiers

```python
# XGBoost (better than SVM)
xgb = registry.create('xgboost', n_estimators=200, use_gpu=True)

# Transformer fine-tuning (SOTA!)
transformer = registry.create(
    'transformer',
    model_name='roberta-base',
    num_epochs=5
)
```

---

## Deprecations

### Removed in v0.3.0

**Old `plugins.py` module:**
```python
# ❌ This no longer works
from lexiclass.plugins import Registry
registry = Registry()  # Old simple registry
```

**Why?** Replaced with enhanced plugin system in `plugins/` package.

---

### Still Supported

**These still work:**
- `SVMDocumentClassifier` wrapper class
- `DocumentIndex` class
- `load_labels()` and `DocumentLoader`
- All core functionality

**But recommended:** Use the new plugin system for flexibility!

---

## Troubleshooting

### Issue: ImportError for old plugins

**Error:**
```python
ImportError: cannot import name 'Registry' from 'lexiclass.plugins'
```

**Solution:**
```python
# Old code
from lexiclass.plugins import Registry

# New code
from lexiclass.plugins.registry import PluginRegistry
# Or just use the global registry
from lexiclass.plugins import registry
```

---

### Issue: Plugin not found

**Error:**
```
PluginNotFoundError: Plugin 'my_plugin' not found
```

**Solution:**
```python
# List available plugins
from lexiclass.plugins import registry
print(registry.list_plugins())

# Check if dependencies installed
registration = registry.get('sbert')
if not registration.is_available():
    print(registration.get_missing_dependencies())
```

---

### Issue: Old model files won't load

**Error:**
```
AttributeError: 'SVMDocumentClassifier' object has no attribute 'X'
```

**Solution:**
Unfortunately, old model files may not be compatible. You'll need to:

1. **Re-train your models** with v0.3.0
2. **Export predictions** from old models, if needed
3. Use the new plugin system going forward

**Migration script:**
```python
# Save predictions from old model
old_classifier = load_old_model('old_model.pkl')
predictions = old_classifier.predict(docs)

# Re-train with new system
from lexiclass.plugins import registry
new_classifier = registry.create('svm')
new_classifier.train(feature_matrix, labels)
```

---

### Issue: Different results after upgrade

**Possible causes:**

1. **Different tokenizer behavior:** spaCy vs ICU
2. **Different feature extraction:** TF-IDF vs BoW
3. **Random seed changes:** Set explicitly

**Solution:**
```python
from lexiclass.config import get_settings

# Check current seed
settings = get_settings()
print(settings.random_seed)  # Default: 42

# Set in environment
import os
os.environ['LEXICLASS_RANDOM_SEED'] = '42'

# Or use same plugins as before
tokenizer = registry.create('icu')  # Not spacy
features = registry.create('bow')   # Not tfidf
```

---

## Step-by-Step Migration Example

### Complete v0.1.0 → v0.3.0 Migration

**Old Code (v0.1.0):**
```python
from lexiclass.classifier import SVMDocumentClassifier
from lexiclass.index import DocumentIndex
from lexiclass.io import DocumentLoader, load_labels

# Build index
docs = DocumentLoader.load_documents_from_directory('./data')
index = DocumentIndex()
index.build_index(
    documents=docs,
    index_path='./my_index'
)

# Train classifier
labels = load_labels('./labels.tsv')
classifier = SVMDocumentClassifier(document_index=index)
classifier.train(labels)
classifier.save_model('./model.pkl', index_path='./my_index')

# Predict
test_docs = DocumentLoader.load_documents_from_directory('./test')
predictions = classifier.predict(test_docs)
```

**New Code (v0.3.0) - Improved:**
```python
from lexiclass.plugins import registry, PluginType
from lexiclass.index import DocumentIndex
from lexiclass.io import DocumentLoader, load_labels

# Create plugins (now customizable!)
tokenizer = registry.create('spacy', model_name='en_core_web_sm')
features = registry.create('tfidf', normalize=True)

# Build index with better plugins
docs = DocumentLoader.load_documents_from_directory('./data')
index = DocumentIndex()

def doc_stream():
    return DocumentLoader.iter_documents_from_directory('./data')

index.build_index(
    feature_extractor=features,
    tokenizer=tokenizer,
    index_path='./my_index',
    document_stream_factory=doc_stream
)

# Train with better classifier
labels = load_labels('./labels.tsv')

# Option 1: Use wrapper (compatible with old code)
from lexiclass.classifier import SVMDocumentClassifier
classifier = SVMDocumentClassifier(
    tokenizer=tokenizer,
    feature_extractor=features,
    document_index=index
)
classifier.train(labels)
classifier.save_model('./model.pkl', index_path='./my_index')

# Option 2: Use XGBoost for better accuracy
xgb = registry.create('xgboost', n_estimators=200)
# ... (need to extract features and train)

# Predict
test_docs = DocumentLoader.load_documents_from_directory('./test')
predictions = classifier.predict(test_docs)
```

---

## Recommended Upgrade Path

### Step 1: Test Current Codebase
```bash
# Backup your current environment
pip freeze > requirements_old.txt

# Note your current version
python -c "import lexiclass; print(lexiclass.__version__)"
```

### Step 2: Install v0.3.0
```bash
pip install --upgrade lexiclass==0.3.0

# Install optional dependencies as needed
pip install xgboost spacy sentence-transformers
```

### Step 3: Update Code
1. Replace imports with plugin system
2. Update CLI commands if using scripts
3. Test on small dataset first

### Step 4: Re-train Models
```bash
# Re-build indices
lexiclass build-index ./data ./index \
  --tokenizer spacy \
  --features tfidf

# Re-train models
lexiclass train ./index ./labels.tsv ./model.pkl
```

### Step 5: Validate Results
- Compare accuracy with old models
- Check prediction outputs
- Verify performance is acceptable

---

## Benefits of Upgrading

### Why Upgrade to v0.3.0?

1. **Better Accuracy** - Modern plugins (XGBoost, Sentence-BERT) give 2-5% improvement
2. **More Options** - 11 plugins to choose from
3. **Better Extensibility** - Easy to add custom plugins
4. **Modern ML** - Access to transformers and deep learning
5. **Better Documentation** - Comprehensive guides and benchmarks
6. **Active Development** - Latest features and bug fixes

---

## Getting Help

### Resources

- **Documentation:** `docs/` directory
- **Examples:** See built-in plugins in `src/lexiclass/plugins/`
- **Benchmarks:** `docs/BENCHMARKS.md`
- **Plugin Guide:** `docs/PLUGIN_DEVELOPMENT_GUIDE.md`

### Common Questions

**Q: Do I need to upgrade?**
A: Only if you want new features or better accuracy. v0.1.0 still works.

**Q: Will my old models work?**
A: Probably not. Plan to re-train.

**Q: Can I keep using the old API?**
A: Some parts yes (SVMDocumentClassifier), but recommended to upgrade.

**Q: How long does migration take?**
A: Simple projects: 30-60 minutes. Complex projects: 2-4 hours.

---

## Migration Checklist

- [ ] Backup current code and models
- [ ] Install v0.3.0
- [ ] Update imports to plugin system
- [ ] Test on small dataset
- [ ] Re-build indices
- [ ] Re-train models
- [ ] Validate accuracy
- [ ] Update production systems
- [ ] Update documentation/README
- [ ] Train team on new API

---

**Last Updated:** 2025-11-01
**Version:** 0.3.0

For questions or issues, please see the GitHub repository.
