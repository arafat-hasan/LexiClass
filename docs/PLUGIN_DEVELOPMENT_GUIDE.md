# Plugin Development Guide

**Version:** 0.3.0
**Audience:** Developers who want to create custom plugins for LexiClass

This guide shows you how to create custom tokenizers, feature extractors, and classifiers that integrate seamlessly with LexiClass.

---

## Table of Contents

1. [Overview](#overview)
2. [Plugin Architecture](#plugin-architecture)
3. [Creating a Tokenizer Plugin](#creating-a-tokenizer-plugin)
4. [Creating a Feature Extractor Plugin](#creating-a-feature-extractor-plugin)
5. [Creating a Classifier Plugin](#creating-a-classifier-plugin)
6. [Plugin Metadata](#plugin-metadata)
7. [Testing Your Plugin](#testing-your-plugin)
8. [Best Practices](#best-practices)
9. [Examples](#examples)

---

## Overview

LexiClass uses a protocol-based plugin system that allows you to:

- ✅ Add new algorithms without modifying core code
- ✅ Share plugins as separate packages
- ✅ Override existing plugins
- ✅ Mix and match components freely

### Plugin Types

| Type | Protocol | Purpose |
|------|----------|---------|
| **Tokenizer** | `TokenizerProtocol` | Convert text to tokens |
| **Feature Extractor** | `FeatureExtractorProtocol` | Convert tokens to vectors |
| **Classifier** | `ClassifierProtocol` | Predict labels from vectors |

---

## Plugin Architecture

### Core Concepts

```python
# 1. Protocol defines interface
class TokenizerProtocol(Protocol):
    def tokenize(self, text: str) -> List[str]: ...

# 2. Metadata describes plugin
metadata = PluginMetadata(
    name="my_tokenizer",
    description="My custom tokenizer",
    plugin_type=PluginType.TOKENIZER,
    dependencies=["nltk>=3.8"],
    ...
)

# 3. Factory creates instances
def factory(**kwargs):
    return MyTokenizer(**kwargs)

# 4. Registration makes it available
registry.register(metadata, factory)
```

### File Structure

```
my_plugin_package/
├── __init__.py
├── my_tokenizer.py
├── my_feature_extractor.py
└── my_classifier.py
```

---

## Creating a Tokenizer Plugin

### Step 1: Implement the Protocol

```python
# my_tokenizer.py
from typing import List

class MyTokenizer:
    """My custom tokenizer."""

    def __init__(self, language: str = 'en'):
        self.language = language
        # Initialize your tokenizer here

    def tokenize(self, text: str) -> List[str]:
        """Split text into tokens.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        # Your tokenization logic here
        tokens = text.lower().split()
        return tokens
```

### Step 2: Add Metadata and Register

```python
# my_tokenizer.py (continued)
from lexiclass.plugins.base import PluginMetadata, PluginType
from lexiclass.plugins.registry import registry

metadata = PluginMetadata(
    name="my_tokenizer",
    display_name="My Tokenizer",
    description="A simple whitespace tokenizer with lowercasing",
    plugin_type=PluginType.TOKENIZER,
    dependencies=[],  # List required packages
    optional_dependencies=["nltk>=3.8"],  # Optional enhancements
    supports_streaming=True,
    performance_tier="fast",  # fast, medium, slow
    quality_tier="basic",  # basic, good, excellent
    memory_usage="low",  # low, medium, high
    default_params={"language": "en"},
)

registry.register(
    metadata=metadata,
    factory=lambda **kwargs: MyTokenizer(**kwargs)
)
```

### Step 3: Make it Picklable (Optional but Recommended)

```python
class MyTokenizer:
    # ... __init__ and tokenize methods ...

    def __getstate__(self):
        """Support pickling."""
        return {'language': self.language}

    def __setstate__(self, state):
        """Restore from pickle."""
        self.language = state['language']
        # Re-initialize any non-serializable objects
```

### Step 4: Use Your Plugin

```python
from lexiclass.plugins import registry

# Your plugin is now available!
tokenizer = registry.create('my_tokenizer', language='en')
tokens = tokenizer.tokenize("Hello, world!")
```

---

## Creating a Feature Extractor Plugin

### Step 1: Implement the Protocol

```python
# my_feature_extractor.py
from typing import List, Iterable
from scipy import sparse
import numpy as np

class MyFeatureExtractor:
    """My custom feature extractor."""

    def __init__(self, max_features: int = 10000):
        self.max_features = max_features
        self.fitted = False
        self.vocabulary = {}

    def fit(self, documents: List[List[str]]) -> "MyFeatureExtractor":
        """Build vocabulary from documents.

        Args:
            documents: List of tokenized documents

        Returns:
            Self for chaining
        """
        # Build vocabulary
        all_tokens = set()
        for doc in documents:
            all_tokens.update(doc)

        # Limit to max_features most common
        # (Add your logic here)
        self.vocabulary = {
            token: idx for idx, token in enumerate(sorted(all_tokens)[:self.max_features])
        }

        self.fitted = True
        return self

    def fit_streaming(
        self,
        tokenized_documents_iter: Iterable[List[str]]
    ) -> "MyFeatureExtractor":
        """Fit using streaming (optional but recommended).

        Args:
            tokenized_documents_iter: Iterator of tokenized documents

        Returns:
            Self for chaining
        """
        # Implement streaming fit if possible
        # For now, collect and use regular fit
        documents = list(tokenized_documents_iter)
        return self.fit(documents)

    def transform(self, documents: List[List[str]]) -> sparse.csr_matrix:
        """Transform documents to feature matrix.

        Args:
            documents: List of tokenized documents

        Returns:
            Sparse feature matrix (num_docs x num_features)
        """
        if not self.fitted:
            raise ValueError("Must fit before transform")

        # Convert to sparse matrix
        # (Add your logic here)
        rows, cols, data = [], [], []

        for doc_idx, tokens in enumerate(documents):
            for token in tokens:
                if token in self.vocabulary:
                    rows.append(doc_idx)
                    cols.append(self.vocabulary[token])
                    data.append(1.0)  # Or TF-IDF weight, etc.

        matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(documents), len(self.vocabulary))
        )

        return matrix

    def tokens_to_bow(self, tokens: List[str]) -> List[tuple[int, float]]:
        """Convert tokens to bag-of-words vector.

        Args:
            tokens: List of tokens

        Returns:
            List of (feature_id, value) tuples
        """
        bow = []
        for token in tokens:
            if token in self.vocabulary:
                bow.append((self.vocabulary[token], 1.0))
        return bow

    def num_features(self) -> int:
        """Return number of features."""
        return len(self.vocabulary)
```

### Step 2: Register the Plugin

```python
# my_feature_extractor.py (continued)
from lexiclass.plugins.base import PluginMetadata, PluginType
from lexiclass.plugins.registry import registry

metadata = PluginMetadata(
    name="my_features",
    display_name="My Feature Extractor",
    description="Custom feature extraction algorithm",
    plugin_type=PluginType.FEATURE_EXTRACTOR,
    dependencies=["scipy>=1.8", "numpy>=1.22"],
    supports_streaming=True,
    performance_tier="fast",
    quality_tier="good",
    memory_usage="medium",
    default_params={"max_features": 10000},
)

registry.register(
    metadata=metadata,
    factory=lambda **kwargs: MyFeatureExtractor(**kwargs)
)
```

---

## Creating a Classifier Plugin

### Step 1: Implement the Protocol

```python
# my_classifier.py
from typing import List, Tuple, Union
import numpy as np
from scipy import sparse

class MyClassifier:
    """My custom classifier."""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.is_fitted = False
        self.model = None
        self.label_encoder = None

    def train(
        self,
        feature_matrix: np.ndarray | sparse.spmatrix,
        labels: List[Union[str, List[str]]],
    ) -> "MyClassifier":
        """Train the classifier.

        Args:
            feature_matrix: Document features (num_docs x num_features)
            labels: Document labels (string or list for multi-label)

        Returns:
            Self for chaining
        """
        from sklearn.preprocessing import LabelEncoder

        # Encode labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)

        # Train your model
        # (Add your training logic here)

        self.is_fitted = True
        return self

    def predict(
        self,
        feature_matrix: np.ndarray | sparse.spmatrix,
    ) -> Tuple[List[Union[str, List[str]]], np.ndarray]:
        """Predict labels.

        Args:
            feature_matrix: Document features

        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.is_fitted:
            raise ValueError("Must train before predict")

        # Make predictions
        # (Add your prediction logic here)
        pred_encoded = np.zeros(feature_matrix.shape[0])

        # Decode labels
        predictions = self.label_encoder.inverse_transform(pred_encoded.astype(int))

        # Compute confidence scores
        scores = np.ones(len(predictions))  # Dummy scores

        return list(predictions), scores
```

### Step 2: Register the Plugin

```python
# my_classifier.py (continued)
from lexiclass.plugins.base import PluginMetadata, PluginType
from lexiclass.plugins.registry import registry

metadata = PluginMetadata(
    name="my_classifier",
    display_name="My Classifier",
    description="Custom classification algorithm",
    plugin_type=PluginType.CLASSIFIER,
    dependencies=["scikit-learn>=1.0", "numpy>=1.22"],
    supports_streaming=False,
    supports_multilabel=False,  # Set True if you support multi-label
    performance_tier="fast",
    quality_tier="good",
    memory_usage="medium",
    default_params={"learning_rate": 0.01},
)

registry.register(
    metadata=metadata,
    factory=lambda **kwargs: MyClassifier(**kwargs)
)
```

---

## Plugin Metadata

### Required Fields

```python
PluginMetadata(
    name="my_plugin",              # Unique identifier (lowercase, no spaces)
    display_name="My Plugin",      # Human-readable name
    description="What it does",    # Brief description
    plugin_type=PluginType.TOKENIZER,  # Type of plugin
)
```

### Optional Fields

```python
PluginMetadata(
    # ... required fields ...

    # Dependencies
    dependencies=["package>=1.0"],           # Required packages
    optional_dependencies=["extra>=2.0"],    # Optional enhancements

    # Capabilities
    supports_streaming=True,                 # Can handle streaming data
    supports_multilabel=True,                # Classifier supports multi-label

    # Performance characteristics
    performance_tier="fast",                 # "fast", "medium", "slow"
    quality_tier="excellent",                # "basic", "good", "excellent"
    memory_usage="low",                      # "low", "medium", "high"

    # Pre-trained models
    requires_pretrained=True,                # Needs pre-trained weights
    pretrained_models=["model-v1", ...],     # Available models

    # Default parameters
    default_params={"param": "value"},       # Default initialization params
)
```

---

## Testing Your Plugin

### Unit Tests

```python
# test_my_plugin.py
import pytest
from my_plugin import MyTokenizer

def test_tokenizer_basic():
    tokenizer = MyTokenizer()
    tokens = tokenizer.tokenize("Hello, world!")
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)

def test_tokenizer_empty():
    tokenizer = MyTokenizer()
    tokens = tokenizer.tokenize("")
    assert tokens == []

def test_tokenizer_picklable():
    import pickle
    tokenizer = MyTokenizer(language='en')

    # Pickle and unpickle
    serialized = pickle.dumps(tokenizer)
    restored = pickle.loads(serialized)

    # Should work the same
    assert restored.tokenize("test") == tokenizer.tokenize("test")
```

### Integration Tests

```python
# test_integration.py
from lexiclass.plugins import registry

def test_plugin_registered():
    """Test plugin is registered."""
    assert 'my_tokenizer' in registry.list_plugins()

def test_plugin_creation():
    """Test plugin can be created."""
    tokenizer = registry.create('my_tokenizer', language='en')
    assert tokenizer is not None

def test_plugin_metadata():
    """Test plugin metadata is correct."""
    meta = registry.get_metadata('my_tokenizer')
    assert meta.name == 'my_tokenizer'
    assert meta.plugin_type.value == 'tokenizer'

def test_plugin_in_pipeline():
    """Test plugin works in full pipeline."""
    from lexiclass.plugins import PluginType

    tokenizer = registry.create('my_tokenizer')
    features = registry.create('tfidf')

    # Use in pipeline
    docs = [["hello", "world"], ["test", "document"]]
    features.fit(docs)
    matrix = features.transform(docs)

    assert matrix.shape[0] == len(docs)
```

---

## Best Practices

### 1. Follow the Protocol

Implement **all** required methods of the protocol:

```python
# ✅ Good
class MyTokenizer:
    def tokenize(self, text: str) -> List[str]:
        return text.split()

# ❌ Bad - missing required method
class BadTokenizer:
    def split(self, text: str):  # Wrong method name!
        return text.split()
```

### 2. Handle Edge Cases

```python
def tokenize(self, text: str) -> List[str]:
    # Handle None
    if text is None:
        return []

    # Handle empty string
    if not text:
        return []

    # Handle Unicode properly
    return text.lower().split()
```

### 3. Support Pickling

```python
def __getstate__(self):
    # Return only serializable data
    return {
        'param1': self.param1,
        'param2': self.param2,
    }

def __setstate__(self, state):
    # Restore from state
    self.param1 = state['param1']
    self.param2 = state['param2']
    # Re-initialize non-serializable objects
    self._init_model()
```

### 4. Use Lazy Loading

```python
class MyTokenizer:
    def __init__(self):
        self.model = None  # Don't load yet

    def _load_model(self):
        if self.model is None:
            # Load on first use
            self.model = expensive_loading()

    def tokenize(self, text: str) -> List[str]:
        self._load_model()
        return self.model.tokenize(text)
```

### 5. Provide Good Error Messages

```python
def transform(self, documents):
    if not self.fitted:
        raise ValueError(
            "FeatureExtractor must be fitted before transform. "
            "Call fit() or fit_streaming() first."
        )

    if not documents:
        raise ValueError("Cannot transform empty document list")

    return self._do_transform(documents)
```

### 6. Log Important Events

```python
import logging

logger = logging.getLogger(__name__)

def fit(self, documents):
    logger.info(f"Training on {len(documents)} documents")

    start_time = time.time()
    self._do_fit(documents)

    logger.info(f"Training completed in {time.time() - start_time:.2f}s")
    return self
```

### 7. Document Your Code

```python
class MyTokenizer:
    """Brief description.

    Longer explanation of what the tokenizer does,
    its advantages, and when to use it.

    Examples:
        >>> tokenizer = MyTokenizer()
        >>> tokens = tokenizer.tokenize("Hello, world!")
        >>> print(tokens)
        ['hello', 'world']
    """

    def __init__(self, language: str = 'en'):
        """Initialize tokenizer.

        Args:
            language: Language code (ISO 639-1)
        """
        self.language = language
```

---

## Examples

### Example 1: NLTK Tokenizer

```python
# nltk_tokenizer.py
from typing import List
import logging

logger = logging.getLogger(__name__)

class NLTKTokenizer:
    """NLTK-based tokenizer."""

    def __init__(self, language: str = 'english'):
        self.language = language
        self._tokenizer = None

    def _load_tokenizer(self):
        if self._tokenizer is not None:
            return

        try:
            import nltk
            from nltk.tokenize import word_tokenize

            # Download data if needed
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.info("Downloading NLTK punkt tokenizer")
                nltk.download('punkt', quiet=True)

            self._tokenizer = word_tokenize

        except ImportError:
            raise ImportError(
                "NLTK tokenizer requires nltk. "
                "Install with: pip install nltk"
            )

    def tokenize(self, text: str) -> List[str]:
        self._load_tokenizer()
        tokens = self._tokenizer(text, language=self.language)
        return [t.lower() for t in tokens]

    def __getstate__(self):
        return {'language': self.language}

    def __setstate__(self, state):
        self.language = state['language']
        self._tokenizer = None

# Register
from lexiclass.plugins.base import PluginMetadata, PluginType
from lexiclass.plugins.registry import registry

metadata = PluginMetadata(
    name="nltk",
    display_name="NLTK",
    description="NLTK word_tokenize with Penn Treebank conventions",
    plugin_type=PluginType.TOKENIZER,
    dependencies=["nltk>=3.8"],
    supports_streaming=True,
    performance_tier="fast",
    quality_tier="good",
    memory_usage="low",
    default_params={"language": "english"},
)

registry.register(metadata, lambda **kw: NLTKTokenizer(**kw))
```

### Example 2: Count Vectorizer

```python
# count_vectorizer.py
from typing import List, Iterable
from collections import Counter
from scipy import sparse
import numpy as np

class CountVectorizer:
    """Simple count-based feature extraction."""

    def __init__(self, max_features: int = 10000, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary = {}
        self.fitted = False

    def fit(self, documents: List[List[str]]) -> "CountVectorizer":
        # Count token frequencies across documents
        token_doc_count = Counter()
        for doc in documents:
            unique_tokens = set(doc)
            token_doc_count.update(unique_tokens)

        # Filter by min_df and limit to max_features
        valid_tokens = [
            token for token, count in token_doc_count.most_common()
            if count >= self.min_df
        ][:self.max_features]

        self.vocabulary = {token: idx for idx, token in enumerate(valid_tokens)}
        self.fitted = True
        return self

    def fit_streaming(self, tokenized_documents_iter: Iterable[List[str]]) -> "CountVectorizer":
        documents = list(tokenized_documents_iter)
        return self.fit(documents)

    def transform(self, documents: List[List[str]]) -> sparse.csr_matrix:
        if not self.fitted:
            raise ValueError("Must fit before transform")

        rows, cols, data = [], [], []

        for doc_idx, tokens in enumerate(documents):
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                if token in self.vocabulary:
                    rows.append(doc_idx)
                    cols.append(self.vocabulary[token])
                    data.append(count)

        return sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(documents), len(self.vocabulary))
        )

    def tokens_to_bow(self, tokens: List[str]) -> List[tuple[int, float]]:
        token_counts = Counter(tokens)
        bow = []
        for token, count in token_counts.items():
            if token in self.vocabulary:
                bow.append((self.vocabulary[token], count))
        return bow

    def num_features(self) -> int:
        return len(self.vocabulary)

# Register
from lexiclass.plugins.base import PluginMetadata, PluginType
from lexiclass.plugins.registry import registry

metadata = PluginMetadata(
    name="count",
    display_name="Count Vectorizer",
    description="Simple count-based feature extraction (similar to sklearn CountVectorizer)",
    plugin_type=PluginType.FEATURE_EXTRACTOR,
    dependencies=["scipy>=1.8", "numpy>=1.22"],
    supports_streaming=True,
    performance_tier="fast",
    quality_tier="basic",
    memory_usage="low",
    default_params={"max_features": 10000, "min_df": 2},
)

registry.register(metadata, lambda **kw: CountVectorizer(**kw))
```

---

## Sharing Your Plugin

### As a Separate Package

```python
# my_lexiclass_plugin/
#   setup.py
#   my_lexiclass_plugin/
#     __init__.py
#     tokenizer.py

# my_lexiclass_plugin/__init__.py
from .tokenizer import MyTokenizer, metadata
from lexiclass.plugins.registry import registry

# Auto-register on import
registry.register(metadata, lambda **kw: MyTokenizer(**kw))

__all__ = ['MyTokenizer']
```

Users can install and use:

```bash
pip install my-lexiclass-plugin
```

```python
# Plugin auto-registers on import
import my_lexiclass_plugin

from lexiclass.plugins import registry
tokenizer = registry.create('my_tokenizer')
```

---

## Conclusion

Creating custom plugins for LexiClass is straightforward:

1. **Implement the protocol** methods
2. **Add metadata** with dependencies and characteristics
3. **Register** the plugin
4. **Test** thoroughly
5. **Document** well

For more examples, check the built-in plugins in `src/lexiclass/plugins/`.

Happy plugin development! 🚀
