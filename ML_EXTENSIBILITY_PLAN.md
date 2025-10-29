# LexiClass ML Extensibility Plan

**Date:** October 30, 2025
**Version:** 1.0
**Goal:** Extend ML capabilities with modern alternatives while maintaining backward compatibility

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State](#current-state)
3. [Proposed Architecture](#proposed-architecture)
4. [Feature Extractors](#feature-extractors)
5. [Tokenizers](#tokenizers)
6. [Classifiers](#classifiers)
7. [Implementation Strategy](#implementation-strategy)
8. [API Design](#api-design)
9. [Configuration System](#configuration-system)
10. [Testing Strategy](#testing-strategy)
11. [Performance Considerations](#performance-considerations)
12. [Migration Path](#migration-path)

---

## Executive Summary

### Goals
- Add modern ML alternatives for each component (tokenization, features, classification)
- Maintain backward compatibility with existing implementations
- Provide easy switching via CLI flags and library parameters
- Keep consistent Protocol-based interfaces
- Enable mixing and matching of components

### Current Components
- **Tokenization:** PyICU (locale-aware, Unicode-compliant)
- **Features:** Bag-of-Words with Gensim (sparse, memory-efficient)
- **Classification:** Linear SVM (scikit-learn, supports binary/multi-class/multi-label)

### Proposed Additions

**Tokenizers:**
- spaCy (modern, fast, multilingual)
- SentencePiece (subword, neural-friendly)
- Hugging Face (transformer-compatible)
- NLTK (classic, educational)

**Feature Extractors:**
- TF-IDF (weighted term frequency)
- FastText (subword embeddings)
- Sentence-BERT (transformer-based)
- Doc2Vec (document embeddings)

**Classifiers:**
- XGBoost/LightGBM (gradient boosting)
- Logistic Regression (fast, interpretable)
- Transformer-based (BERT, RoBERTa, DistilBERT)
- SetFit (few-shot learning)

---

## Current State

### Plugin Registry Architecture

Current `plugins.py`:
```python
class PluginRegistry:
    def __init__(self):
        self.tokenizers: Dict[str, Callable] = {}
        self.features: Dict[str, Callable] = {}

registry = PluginRegistry()

# Built-in registrations
registry.tokenizers["icu"] = lambda locale='en': ICUTokenizer(locale)
registry.features["bow"] = lambda: FeatureExtractor()
```

### Strengths
✅ Simple, extensible design
✅ Protocol-based interfaces
✅ Easy registration mechanism

### Limitations
❌ No classifier registry (hardcoded SVM)
❌ No parameter validation for plugins
❌ No automatic discovery
❌ No dependency checking
❌ Limited plugin metadata

---

## Proposed Architecture

### Enhanced Plugin System

#### 1. Plugin Metadata System

Create `src/lexiclass/plugins/base.py`:
```python
"""Base classes and metadata for plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol
from enum import Enum


class PluginType(str, Enum):
    """Types of plugins supported by LexiClass."""
    TOKENIZER = "tokenizer"
    FEATURE_EXTRACTOR = "feature_extractor"
    CLASSIFIER = "classifier"


@dataclass
class PluginMetadata:
    """Metadata about a plugin.

    Attributes:
        name: Unique identifier for the plugin
        display_name: Human-readable name
        description: Brief description of the plugin
        plugin_type: Type of plugin (tokenizer, feature, classifier)
        dependencies: List of required packages
        optional_dependencies: List of optional packages for enhanced functionality
        supports_streaming: Whether the plugin supports streaming data
        supports_multilabel: Whether classifier supports multi-label
        default_params: Default parameters for initialization
        performance_tier: Relative speed (fast/medium/slow)
        quality_tier: Relative quality (basic/good/excellent)
    """
    name: str
    display_name: str
    description: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    optional_dependencies: List[str] = field(default_factory=list)
    supports_streaming: bool = False
    supports_multilabel: bool = False
    default_params: Dict[str, Any] = field(default_factory=dict)
    performance_tier: str = "medium"  # fast, medium, slow
    quality_tier: str = "good"  # basic, good, excellent
    memory_usage: str = "medium"  # low, medium, high
    requires_pretrained: bool = False
    pretrained_models: List[str] = field(default_factory=list)

    def check_dependencies(self) -> tuple[bool, List[str]]:
        """Check if required dependencies are installed.

        Returns:
            Tuple of (all_installed, missing_packages)
        """
        import importlib.util

        missing = []
        for dep in self.dependencies:
            # Parse package name (handle versions like "numpy>=1.22")
            package_name = dep.split(">=")[0].split("==")[0].split("<")[0]
            if importlib.util.find_spec(package_name) is None:
                missing.append(dep)

        return len(missing) == 0, missing


@dataclass
class PluginRegistration:
    """Complete plugin registration with metadata and factory."""
    metadata: PluginMetadata
    factory: Callable[..., Any]

    def create(self, **kwargs) -> Any:
        """Create plugin instance with parameters."""
        # Merge default params with provided kwargs
        params = {**self.metadata.default_params, **kwargs}
        return self.factory(**params)

    def is_available(self) -> bool:
        """Check if plugin can be used (dependencies installed)."""
        available, _ = self.metadata.check_dependencies()
        return available

    def get_missing_dependencies(self) -> List[str]:
        """Get list of missing dependencies."""
        _, missing = self.metadata.check_dependencies()
        return missing
```

#### 2. Enhanced Plugin Registry

Update `src/lexiclass/plugins/registry.py`:
```python
"""Enhanced plugin registry with metadata and validation."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .base import PluginMetadata, PluginRegistration, PluginType
from ..exceptions import PluginNotFoundError, PluginRegistrationError

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for LexiClass plugins with metadata support."""

    def __init__(self):
        self._plugins: Dict[str, PluginRegistration] = {}

    def register(
        self,
        metadata: PluginMetadata,
        factory: Callable[..., Any],
        override: bool = False,
    ) -> None:
        """Register a plugin with metadata.

        Args:
            metadata: Plugin metadata
            factory: Factory function to create plugin instance
            override: Whether to override existing plugin with same name

        Raises:
            PluginRegistrationError: If plugin already exists and override=False
        """
        if metadata.name in self._plugins and not override:
            raise PluginRegistrationError(
                f"Plugin '{metadata.name}' already registered. "
                f"Use override=True to replace."
            )

        registration = PluginRegistration(metadata=metadata, factory=factory)
        self._plugins[metadata.name] = registration
        logger.debug(f"Registered {metadata.plugin_type} plugin: {metadata.name}")

    def get(self, name: str, plugin_type: Optional[PluginType] = None) -> PluginRegistration:
        """Get plugin registration by name.

        Args:
            name: Plugin name
            plugin_type: Optional type filter

        Returns:
            PluginRegistration

        Raises:
            PluginNotFoundError: If plugin not found
        """
        if name not in self._plugins:
            available = self.list_plugins(plugin_type)
            raise PluginNotFoundError(
                f"Plugin '{name}' not found. Available plugins: {available}"
            )

        registration = self._plugins[name]

        if plugin_type and registration.metadata.plugin_type != plugin_type:
            raise PluginNotFoundError(
                f"Plugin '{name}' is a {registration.metadata.plugin_type}, "
                f"not a {plugin_type}"
            )

        return registration

    def create(
        self,
        name: str,
        plugin_type: Optional[PluginType] = None,
        **kwargs
    ) -> Any:
        """Create plugin instance.

        Args:
            name: Plugin name
            plugin_type: Optional type filter
            **kwargs: Parameters for plugin initialization

        Returns:
            Plugin instance

        Raises:
            PluginNotFoundError: If plugin not found
            ImportError: If required dependencies missing
        """
        registration = self.get(name, plugin_type)

        # Check dependencies
        if not registration.is_available():
            missing = registration.get_missing_dependencies()
            raise ImportError(
                f"Plugin '{name}' requires missing dependencies: {missing}\n"
                f"Install with: pip install {' '.join(missing)}"
            )

        return registration.create(**kwargs)

    def list_plugins(
        self,
        plugin_type: Optional[PluginType] = None,
        available_only: bool = False,
    ) -> List[str]:
        """List registered plugin names.

        Args:
            plugin_type: Filter by plugin type
            available_only: Only list plugins with dependencies installed

        Returns:
            List of plugin names
        """
        plugins = []
        for name, registration in self._plugins.items():
            if plugin_type and registration.metadata.plugin_type != plugin_type:
                continue
            if available_only and not registration.is_available():
                continue
            plugins.append(name)
        return sorted(plugins)

    def get_metadata(self, name: str) -> PluginMetadata:
        """Get plugin metadata."""
        return self.get(name).metadata

    def describe(self, name: str) -> str:
        """Get human-readable description of plugin."""
        registration = self.get(name)
        meta = registration.metadata

        status = "✓ Available" if registration.is_available() else "✗ Unavailable"
        missing = registration.get_missing_dependencies()

        lines = [
            f"Plugin: {meta.display_name} ({meta.name})",
            f"Type: {meta.plugin_type.value}",
            f"Status: {status}",
            f"Description: {meta.description}",
            f"Performance: {meta.performance_tier}",
            f"Quality: {meta.quality_tier}",
            f"Memory: {meta.memory_usage}",
            f"Streaming: {'Yes' if meta.supports_streaming else 'No'}",
        ]

        if meta.plugin_type == PluginType.CLASSIFIER:
            lines.append(f"Multi-label: {'Yes' if meta.supports_multilabel else 'No'}")

        if meta.dependencies:
            lines.append(f"Dependencies: {', '.join(meta.dependencies)}")

        if missing:
            lines.append(f"Missing: {', '.join(missing)}")
            lines.append(f"Install: pip install {' '.join(missing)}")

        if meta.default_params:
            lines.append(f"Default params: {meta.default_params}")

        return "\n".join(lines)


# Global registry instance
registry = PluginRegistry()
```

#### 3. Directory Structure

```
src/lexiclass/
├── plugins/
│   ├── __init__.py                 # Re-export registry
│   ├── base.py                     # PluginMetadata, PluginRegistration
│   ├── registry.py                 # PluginRegistry
│   ├── tokenizers/
│   │   ├── __init__.py
│   │   ├── icu.py                  # ICUTokenizer (existing)
│   │   ├── spacy.py                # SpacyTokenizer (new)
│   │   ├── sentencepiece.py        # SentencePieceTokenizer (new)
│   │   ├── huggingface.py          # HFTokenizer (new)
│   │   └── nltk.py                 # NLTKTokenizer (new)
│   ├── features/
│   │   ├── __init__.py
│   │   ├── bow.py                  # BoW (existing, refactored)
│   │   ├── tfidf.py                # TF-IDF (new)
│   │   ├── fasttext.py             # FastText embeddings (new)
│   │   ├── sbert.py                # Sentence-BERT (new)
│   │   └── doc2vec.py              # Doc2Vec (new)
│   └── classifiers/
│       ├── __init__.py
│       ├── svm.py                  # SVM (existing, refactored)
│       ├── xgboost.py              # XGBoost (new)
│       ├── logistic.py             # LogisticRegression (new)
│       └── transformer.py          # Transformer-based (new)
```

---

## Feature Extractors

### 1. TF-IDF (High Priority)

**Why:** Classic upgrade from BoW, better weighting

**Implementation:**
```python
# src/lexiclass/plugins/features/tfidf.py

from __future__ import annotations

import logging
from typing import Iterable, List, Tuple

from gensim import corpora, models
from scipy import sparse

from ...interfaces import FeatureExtractorProtocol

logger = logging.getLogger(__name__)


class TfidfFeatureExtractor:
    """TF-IDF feature extraction using Gensim.

    Converts documents to TF-IDF weighted sparse vectors.
    More informative than raw bag-of-words as it downweights
    common terms and upweights distinctive terms.
    """

    def __init__(
        self,
        normalize: bool = True,
        smartirs: str = 'ntc',  # TF-IDF weighting scheme
        num_workers: int | None = None,
    ):
        """Initialize TF-IDF extractor.

        Args:
            normalize: Whether to L2-normalize vectors
            smartirs: TF-IDF weighting scheme (n=natural, t=idf, c=cosine norm)
            num_workers: Number of workers for parallel processing
        """
        self.dictionary: corpora.Dictionary | None = None
        self.tfidf_model: models.TfidfModel | None = None
        self.normalize = normalize
        self.smartirs = smartirs
        self.num_workers = num_workers
        self.fitted = False

    def fit(self, documents: List[List[str]]) -> "TfidfFeatureExtractor":
        """Fit dictionary and TF-IDF model."""
        logger.info(f"Creating TF-IDF model from {len(documents)} documents")

        # Build dictionary
        self.dictionary = corpora.Dictionary(documents)

        # Filter extremes (same as BoW)
        self.dictionary.filter_extremes(no_below=3, no_above=0.5, keep_n=None)

        # Create corpus for fitting TF-IDF
        corpus = [self.dictionary.doc2bow(doc) for doc in documents]

        # Fit TF-IDF model
        self.tfidf_model = models.TfidfModel(
            corpus,
            normalize=self.normalize,
            smartirs=self.smartirs,
        )

        self.fitted = True
        logger.info(f"TF-IDF model created with {len(self.dictionary)} features")
        return self

    def fit_streaming(
        self,
        tokenized_documents_iter: Iterable[List[str]]
    ) -> "TfidfFeatureExtractor":
        """Fit using streaming approach."""
        logger.info("Creating TF-IDF model from streaming documents")

        # First pass: build dictionary
        self.dictionary = corpora.Dictionary()
        batch = []
        batch_size = 1000

        for tokens in tokenized_documents_iter:
            batch.append(tokens)
            if len(batch) >= batch_size:
                self.dictionary.add_documents(batch, prune_at=2_000_000)
                batch.clear()

        if batch:
            self.dictionary.add_documents(batch)

        self.dictionary.filter_extremes(no_below=3, no_above=0.5)

        # Second pass would require re-streaming, so we use initialize=True
        # This creates a default TF-IDF model without corpus statistics
        # For better results with streaming, use incremental updates
        self.tfidf_model = models.TfidfModel(
            dictionary=self.dictionary,
            normalize=self.normalize,
            smartirs=self.smartirs,
        )

        self.fitted = True
        logger.info(f"TF-IDF model created with {len(self.dictionary)} features (streaming)")
        return self

    def transform(self, documents: List[List[str]]) -> sparse.csr_matrix:
        """Transform documents to TF-IDF sparse matrix."""
        if not self.fitted:
            raise ValueError("TfidfFeatureExtractor must be fitted before transform")

        # Convert to bag-of-words
        corpus = [self.dictionary.doc2bow(doc) for doc in documents]

        # Apply TF-IDF
        tfidf_corpus = [self.tfidf_model[bow] for bow in corpus]

        # Convert to sparse matrix
        num_docs = len(tfidf_corpus)
        num_features = len(self.dictionary)

        rows, cols, data = [], [], []
        for doc_idx, tfidf_vec in enumerate(tfidf_corpus):
            for token_id, weight in tfidf_vec:
                rows.append(doc_idx)
                cols.append(token_id)
                data.append(weight)

        matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(num_docs, num_features)
        )

        return matrix

    def tokens_to_bow(self, tokens: List[str]) -> List[Tuple[int, float]]:
        """Convert tokens to TF-IDF vector."""
        if not self.fitted:
            raise ValueError("TfidfFeatureExtractor must be fitted")

        bow = self.dictionary.doc2bow(tokens)
        return self.tfidf_model[bow]

    def num_features(self) -> int:
        """Return number of features."""
        return len(self.dictionary) if self.dictionary else 0


# Plugin registration
from ..base import PluginMetadata, PluginType
from ..registry import registry

metadata = PluginMetadata(
    name="tfidf",
    display_name="TF-IDF",
    description="Term Frequency-Inverse Document Frequency weighting",
    plugin_type=PluginType.FEATURE_EXTRACTOR,
    dependencies=["gensim>=4.3", "scipy>=1.8"],
    supports_streaming=True,
    performance_tier="fast",
    quality_tier="good",
    memory_usage="medium",
    default_params={"normalize": True, "smartirs": "ntc"},
)

registry.register(
    metadata=metadata,
    factory=lambda **kwargs: TfidfFeatureExtractor(**kwargs)
)
```

**Metadata:**
- Performance: Fast (same as BoW)
- Quality: Good (better than BoW)
- Memory: Medium
- Dependencies: gensim, scipy (already installed)

---

### 2. FastText Embeddings (High Priority)

**Why:** Handles OOV words via subword embeddings, better semantic representation

**Implementation:**
```python
# src/lexiclass/plugins/features/fasttext.py

from __future__ import annotations

import logging
import numpy as np
from typing import Iterable, List, Tuple
from scipy import sparse

logger = logging.getLogger(__name__)


class FastTextFeatureExtractor:
    """FastText subword embeddings for feature extraction.

    Uses pre-trained FastText models or trains custom embeddings.
    Handles out-of-vocabulary words via character n-grams.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 3,
        pretrained_model: str | None = None,
        pretrained_path: str | None = None,
    ):
        """Initialize FastText extractor.

        Args:
            vector_size: Dimensionality of embeddings
            window: Context window size
            min_count: Minimum word frequency
            pretrained_model: Name of pre-trained model (e.g., 'en')
            pretrained_path: Path to pre-trained model file
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.pretrained_model = pretrained_model
        self.pretrained_path = pretrained_path
        self.model = None
        self.fitted = False

    def fit(self, documents: List[List[str]]) -> "FastTextFeatureExtractor":
        """Fit FastText model."""
        try:
            from gensim.models import FastText
        except ImportError:
            raise ImportError(
                "FastText requires gensim with FastText support. "
                "Install with: pip install gensim"
            )

        if self.pretrained_path:
            logger.info(f"Loading pre-trained FastText model from {self.pretrained_path}")
            self.model = FastText.load(self.pretrained_path)
        elif self.pretrained_model:
            logger.info(f"Downloading pre-trained FastText model: {self.pretrained_model}")
            import gensim.downloader as api
            self.model = api.load(f"fasttext-wiki-news-subwords-300")
        else:
            logger.info(f"Training FastText model on {len(documents)} documents")
            self.model = FastText(
                sentences=documents,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=4,
                sg=1,  # Skip-gram
            )

        self.fitted = True
        logger.info(f"FastText model ready with {self.vector_size}-dim vectors")
        return self

    def fit_streaming(
        self,
        tokenized_documents_iter: Iterable[List[str]]
    ) -> "FastTextFeatureExtractor":
        """Fit using streaming approach."""
        from gensim.models import FastText

        if self.pretrained_path or self.pretrained_model:
            # Pre-trained models don't need fitting
            return self.fit([])

        logger.info("Training FastText model from streaming documents")

        # Collect sentences for training
        sentences = list(tokenized_documents_iter)

        self.model = FastText(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            sg=1,
        )

        self.fitted = True
        return self

    def transform(self, documents: List[List[str]]) -> np.ndarray:
        """Transform documents to dense embeddings.

        Returns dense matrix (not sparse) as embeddings are dense.
        """
        if not self.fitted:
            raise ValueError("FastTextFeatureExtractor must be fitted")

        vectors = []
        for doc_tokens in documents:
            if not doc_tokens:
                # Empty document -> zero vector
                vectors.append(np.zeros(self.vector_size))
            else:
                # Average word vectors
                word_vecs = [
                    self.model.wv[token]
                    for token in doc_tokens
                    if token in self.model.wv
                ]
                if word_vecs:
                    doc_vec = np.mean(word_vecs, axis=0)
                else:
                    doc_vec = np.zeros(self.vector_size)
                vectors.append(doc_vec)

        return np.vstack(vectors)

    def tokens_to_bow(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens to embedding vector.

        Note: Returns dense vector, not BoW format.
        """
        if not self.fitted:
            raise ValueError("FastTextFeatureExtractor must be fitted")

        if not tokens:
            return np.zeros(self.vector_size)

        word_vecs = [
            self.model.wv[token]
            for token in tokens
            if token in self.model.wv
        ]

        if word_vecs:
            return np.mean(word_vecs, axis=0)
        else:
            return np.zeros(self.vector_size)

    def num_features(self) -> int:
        """Return dimensionality of embeddings."""
        return self.vector_size


# Plugin registration
from ..base import PluginMetadata, PluginType
from ..registry import registry

metadata = PluginMetadata(
    name="fasttext",
    display_name="FastText",
    description="Subword embeddings with character n-grams (handles OOV)",
    plugin_type=PluginType.FEATURE_EXTRACTOR,
    dependencies=["gensim>=4.3", "numpy>=1.22"],
    supports_streaming=True,
    performance_tier="medium",
    quality_tier="excellent",
    memory_usage="medium",
    requires_pretrained=False,
    pretrained_models=["fasttext-wiki-news-subwords-300"],
    default_params={"vector_size": 100, "window": 5, "min_count": 3},
)

registry.register(metadata=metadata, factory=FastTextFeatureExtractor)
```

**Metadata:**
- Performance: Medium (embedding generation is slower than sparse)
- Quality: Excellent (semantic understanding)
- Memory: Medium (dense vectors)
- Dependencies: gensim

---

### 3. Sentence-BERT (Highest Quality, Modern)

**Why:** State-of-the-art sentence embeddings, best quality

**Implementation:**
```python
# src/lexiclass/plugins/features/sbert.py

from __future__ import annotations

import logging
import numpy as np
from typing import Iterable, List

logger = logging.getLogger(__name__)


class SentenceBERTFeatureExtractor:
    """Sentence-BERT transformer-based embeddings.

    Uses pre-trained Sentence-Transformers for high-quality
    semantic document embeddings.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ):
        """Initialize Sentence-BERT extractor.

        Args:
            model_name: Pre-trained model name
                - "all-MiniLM-L6-v2" (default, fast, 384-dim)
                - "all-mpnet-base-v2" (best quality, 768-dim)
                - "paraphrase-multilingual-MiniLM-L12-v2" (multilingual)
            device: Device to use (None=auto, "cuda", "cpu")
            batch_size: Batch size for encoding
            normalize_embeddings: L2-normalize embeddings
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.model = None
        self.fitted = False
        self._vector_size = None

    def fit(self, documents: List[List[str]]) -> "SentenceBERTFeatureExtractor":
        """Load pre-trained model (no training needed)."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Sentence-BERT requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )

        logger.info(f"Loading Sentence-BERT model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self._vector_size = self.model.get_sentence_embedding_dimension()
        self.fitted = True

        logger.info(
            f"Sentence-BERT model loaded ({self._vector_size}-dim embeddings)"
        )
        return self

    def fit_streaming(
        self,
        tokenized_documents_iter: Iterable[List[str]]
    ) -> "SentenceBERTFeatureExtractor":
        """Load model (streaming not needed for pre-trained)."""
        # Pre-trained models don't need documents for fitting
        # Just consume the iterator
        _ = list(tokenized_documents_iter)
        return self.fit([])

    def transform(self, documents: List[List[str]]) -> np.ndarray:
        """Transform documents to Sentence-BERT embeddings.

        Args:
            documents: List of tokenized documents

        Returns:
            Dense numpy array of embeddings
        """
        if not self.fitted:
            raise ValueError("SentenceBERTFeatureExtractor must be fitted")

        # Reconstruct text from tokens (Sentence-BERT works on text)
        texts = [" ".join(tokens) for tokens in documents]

        # Encode in batches
        logger.info(f"Encoding {len(texts)} documents with Sentence-BERT")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        return embeddings

    def tokens_to_bow(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens to embedding vector."""
        if not self.fitted:
            raise ValueError("SentenceBERTFeatureExtractor must be fitted")

        text = " ".join(tokens)
        embedding = self.model.encode(
            [text],
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embedding[0]

    def num_features(self) -> int:
        """Return embedding dimensionality."""
        return self._vector_size if self._vector_size else 0


# Plugin registration
from ..base import PluginMetadata, PluginType
from ..registry import registry

metadata = PluginMetadata(
    name="sbert",
    display_name="Sentence-BERT",
    description="Transformer-based sentence embeddings (state-of-the-art)",
    plugin_type=PluginType.FEATURE_EXTRACTOR,
    dependencies=["sentence-transformers>=2.0"],
    optional_dependencies=["torch", "transformers"],
    supports_streaming=False,  # Loads pre-trained, doesn't need training
    performance_tier="slow",  # Transformer inference is slower
    quality_tier="excellent",
    memory_usage="high",
    requires_pretrained=True,
    pretrained_models=[
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-multilingual-MiniLM-L12-v2",
    ],
    default_params={
        "model_name": "all-MiniLM-L6-v2",
        "batch_size": 32,
        "normalize_embeddings": True,
    },
)

registry.register(metadata=metadata, factory=SentenceBERTFeatureExtractor)
```

**Metadata:**
- Performance: Slow (transformer inference)
- Quality: Excellent (SOTA)
- Memory: High (large models)
- Dependencies: sentence-transformers, torch

---

### 4. Doc2Vec (Medium Priority)

**Implementation Summary:**
```python
# Uses Gensim's Doc2Vec for document-level embeddings
# Good middle ground between BoW and transformers
```

**Metadata:**
- Performance: Medium
- Quality: Good
- Memory: Medium
- Dependencies: gensim

---

### Feature Extractor Comparison Table

| Feature | BoW | TF-IDF | Doc2Vec | FastText | Sentence-BERT |
|---------|-----|--------|---------|----------|---------------|
| **Quality** | Basic | Good | Good | Excellent | Excellent |
| **Speed** | Fast | Fast | Medium | Medium | Slow |
| **Memory** | Low | Low | Medium | Medium | High |
| **Semantics** | No | No | Yes | Yes | Yes |
| **OOV Handling** | Poor | Poor | Poor | Excellent | Good |
| **Dependencies** | gensim | gensim | gensim | gensim | sentence-transformers |
| **Training Required** | Yes | Yes | Yes | Optional | No (pre-trained) |
| **Dimensionality** | ~10k-50k | ~10k-50k | 50-300 | 50-300 | 384-768 |
| **Best For** | Baselines | Classic ML | Mid-tier | Production | SOTA results |

---

## Tokenizers

### 1. spaCy Tokenizer (High Priority)

**Why:** Modern, fast, multilingual, industry standard

**Implementation:**
```python
# src/lexiclass/plugins/tokenizers/spacy.py

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


class SpacyTokenizer:
    """spaCy-based tokenizer with multilingual support.

    Fast, accurate tokenization using spaCy's linguistic models.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        disable: List[str] | None = None,
        lowercase: bool = True,
        remove_punct: bool = True,
        remove_stop: bool = False,
    ):
        """Initialize spaCy tokenizer.

        Args:
            model_name: spaCy model name
                - "en_core_web_sm" (English, small, fast)
                - "en_core_web_lg" (English, large, more accurate)
                - "xx_ent_wiki_sm" (Multilingual)
                - etc.
            disable: Pipeline components to disable (for speed)
            lowercase: Convert to lowercase
            remove_punct: Remove punctuation tokens
            remove_stop: Remove stop words
        """
        self.model_name = model_name
        self.disable = disable or ["parser", "ner"]  # Disable for speed
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.remove_stop = remove_stop
        self.nlp = None

    def _load_model(self):
        """Lazy load spaCy model."""
        if self.nlp is not None:
            return

        try:
            import spacy
        except ImportError:
            raise ImportError(
                "spaCy tokenizer requires spacy. "
                "Install with: pip install spacy && "
                "python -m spacy download en_core_web_sm"
            )

        try:
            logger.info(f"Loading spaCy model: {self.model_name}")
            self.nlp = spacy.load(self.model_name, disable=self.disable)
        except OSError:
            raise OSError(
                f"spaCy model '{self.model_name}' not found. "
                f"Download with: python -m spacy download {self.model_name}"
            )

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using spaCy."""
        self._load_model()

        doc = self.nlp(text)
        tokens = []

        for token in doc:
            # Skip based on filters
            if self.remove_punct and token.is_punct:
                continue
            if self.remove_stop and token.is_stop:
                continue

            text = token.text
            if self.lowercase:
                text = text.lower()

            tokens.append(text)

        return tokens


# Plugin registration
from ..base import PluginMetadata, PluginType
from ..registry import registry

metadata = PluginMetadata(
    name="spacy",
    display_name="spaCy",
    description="Modern multilingual tokenizer with linguistic features",
    plugin_type=PluginType.TOKENIZER,
    dependencies=["spacy>=3.0"],
    supports_streaming=True,
    performance_tier="fast",
    quality_tier="excellent",
    memory_usage="medium",
    requires_pretrained=True,
    pretrained_models=["en_core_web_sm", "en_core_web_lg", "xx_ent_wiki_sm"],
    default_params={
        "model_name": "en_core_web_sm",
        "lowercase": True,
        "remove_punct": True,
    },
)

registry.register(metadata=metadata, factory=SpacyTokenizer)
```

---

### 2. SentencePiece Tokenizer (High Priority)

**Why:** Subword tokenization, language-agnostic, used by neural models

**Implementation:**
```python
# src/lexiclass/plugins/tokenizers/sentencepiece.py

from __future__ import annotations

import logging
import tempfile
import os
from typing import List, Iterable

logger = logging.getLogger(__name__)


class SentencePieceTokenizer:
    """SentencePiece subword tokenizer.

    Unsupervised tokenizer that learns subword units.
    Language-agnostic and handles any Unicode text.
    """

    def __init__(
        self,
        model_path: str | None = None,
        vocab_size: int = 8000,
        model_type: str = "unigram",  # or "bpe"
        character_coverage: float = 0.9995,
    ):
        """Initialize SentencePiece tokenizer.

        Args:
            model_path: Path to trained model (if None, trains on-demand)
            vocab_size: Vocabulary size for training
            model_type: Model type ("unigram" or "bpe")
            character_coverage: Character coverage for training
        """
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage
        self.sp = None
        self._trained = False

    def train(self, texts: List[str], output_path: str | None = None):
        """Train SentencePiece model on texts.

        Args:
            texts: Training texts
            output_path: Where to save model (if None, uses temp)
        """
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError(
                "SentencePiece requires sentencepiece. "
                "Install with: pip install sentencepiece"
            )

        # Write texts to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            delete=False,
            suffix='.txt'
        ) as f:
            temp_file = f.name
            for text in texts:
                f.write(text + '\n')

        try:
            # Train model
            if output_path is None:
                output_path = tempfile.mktemp(suffix='.model')

            logger.info(f"Training SentencePiece model (vocab_size={self.vocab_size})")
            spm.SentencePieceTrainer.train(
                input=temp_file,
                model_prefix=output_path.replace('.model', ''),
                vocab_size=self.vocab_size,
                model_type=self.model_type,
                character_coverage=self.character_coverage,
            )

            self.model_path = output_path
            self._load_model()
            self._trained = True

        finally:
            os.unlink(temp_file)

    def _load_model(self):
        """Load SentencePiece model."""
        if self.sp is not None:
            return

        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("pip install sentencepiece")

        if self.model_path is None:
            raise ValueError(
                "No model_path specified. Either provide model_path or call train() first."
            )

        logger.info(f"Loading SentencePiece model from {self.model_path}")
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using SentencePiece."""
        self._load_model()
        return self.sp.encode(text, out_type=str)


# Plugin registration
from ..base import PluginMetadata, PluginType
from ..registry import registry

metadata = PluginMetadata(
    name="sentencepiece",
    display_name="SentencePiece",
    description="Subword tokenizer for neural models (language-agnostic)",
    plugin_type=PluginType.TOKENIZER,
    dependencies=["sentencepiece>=0.1.99"],
    supports_streaming=True,
    performance_tier="fast",
    quality_tier="excellent",
    memory_usage="low",
    requires_pretrained=False,
    default_params={
        "vocab_size": 8000,
        "model_type": "unigram",
        "character_coverage": 0.9995,
    },
)

registry.register(metadata=metadata, factory=SentencePieceTokenizer)
```

---

### 3. Hugging Face Tokenizer (Medium Priority)

**Why:** Compatible with transformer models, fast Rust implementation

```python
# Uses tokenizers library (Rust-based, very fast)
# Compatible with all Hugging Face models
```

---

### Tokenizer Comparison Table

| Tokenizer | PyICU | spaCy | SentencePiece | HF Tokenizers |
|-----------|-------|-------|---------------|---------------|
| **Speed** | Fast | Fast | Very Fast | Very Fast |
| **Quality** | Good | Excellent | Excellent | Excellent |
| **Multilingual** | Yes | Yes | Yes | Yes |
| **Subword** | No | No | Yes | Yes |
| **Dependencies** | PyICU | spacy | sentencepiece | tokenizers |
| **Best For** | Classic | General | Neural ML | Transformers |

---

## Classifiers

### 1. XGBoost/LightGBM (Highest Priority)

**Why:** Often best performance for traditional ML, fast, production-ready

**Implementation:**
```python
# src/lexiclass/plugins/classifiers/xgboost.py

from __future__ import annotations

import logging
from typing import Dict, List, Union
import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


class XGBoostClassifier:
    """XGBoost gradient boosting classifier.

    High-performance gradient boosting that often outperforms
    linear models on document classification tasks.
    """

    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        objective: str = "multi:softmax",
        use_gpu: bool = False,
        **kwargs,
    ):
        """Initialize XGBoost classifier.

        Args:
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            n_estimators: Number of boosting rounds
            objective: Loss function (auto-detected from labels)
            use_gpu: Use GPU acceleration
            **kwargs: Additional XGBoost parameters
        """
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.objective = objective
        self.use_gpu = use_gpu
        self.kwargs = kwargs
        self.model = None
        self.label_encoder = None
        self.is_multilabel = False

    def train(
        self,
        feature_matrix: np.ndarray | sparse.spmatrix,
        labels: List[Union[str, List[str]]],
    ) -> "XGBoostClassifier":
        """Train XGBoost classifier.

        Args:
            feature_matrix: Document feature matrix
            labels: Document labels (string or list of strings)
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost requires xgboost. Install with: pip install xgboost"
            )

        from sklearn import preprocessing

        # Detect multi-label
        if isinstance(labels[0], list):
            self.is_multilabel = True
            logger.info("Training XGBoost for multi-label classification")

            # Multi-label: train one classifier per label
            mlb = preprocessing.MultiLabelBinarizer()
            y_encoded = mlb.fit_transform(labels)
            self.label_encoder = mlb

            # Train separate model for each label
            self.model = []
            for i, label in enumerate(mlb.classes_):
                logger.info(f"Training classifier for label: {label}")
                clf = xgb.XGBClassifier(
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    n_estimators=self.n_estimators,
                    objective="binary:logistic",
                    tree_method="gpu_hist" if self.use_gpu else "auto",
                    **self.kwargs,
                )
                clf.fit(feature_matrix, y_encoded[:, i])
                self.model.append(clf)

        else:
            # Single-label classification
            unique_labels = sorted(set(labels))

            if len(unique_labels) == 2:
                logger.info("Training XGBoost for binary classification")
                self.label_encoder = preprocessing.LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(labels)
                objective = "binary:logistic"
            else:
                logger.info(f"Training XGBoost for {len(unique_labels)}-class classification")
                self.label_encoder = preprocessing.LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(labels)
                objective = "multi:softmax"

            self.model = xgb.XGBClassifier(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                objective=objective,
                tree_method="gpu_hist" if self.use_gpu else "auto",
                **self.kwargs,
            )
            self.model.fit(feature_matrix, y_encoded)

        logger.info("XGBoost training completed")
        return self

    def predict(
        self,
        feature_matrix: np.ndarray | sparse.spmatrix,
    ) -> tuple[List[Union[str, List[str]]], np.ndarray]:
        """Predict labels for features.

        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        if self.is_multilabel:
            # Multi-label prediction
            predictions = []
            scores_list = []

            for clf in self.model:
                probs = clf.predict_proba(feature_matrix)[:, 1]
                scores_list.append(probs)

            # Stack scores
            all_scores = np.column_stack(scores_list)

            # Convert to binary predictions (threshold=0.5)
            binary_preds = (all_scores > 0.5).astype(int)

            # Decode to labels
            predictions = self.label_encoder.inverse_transform(binary_preds)

            # Max score per sample
            max_scores = np.max(all_scores, axis=1)

            return list(predictions), max_scores
        else:
            # Single-label prediction
            y_pred = self.model.predict(feature_matrix)

            # Get probabilities for confidence
            probs = self.model.predict_proba(feature_matrix)
            max_probs = np.max(probs, axis=1)

            # Decode labels
            predictions = self.label_encoder.inverse_transform(y_pred)

            return list(predictions), max_probs


# Plugin registration
from ..base import PluginMetadata, PluginType
from ..registry import registry

metadata = PluginMetadata(
    name="xgboost",
    display_name="XGBoost",
    description="Gradient boosting classifier (high performance)",
    plugin_type=PluginType.CLASSIFIER,
    dependencies=["xgboost>=1.7"],
    optional_dependencies=["cupy"],  # For GPU
    supports_multilabel=True,
    performance_tier="fast",
    quality_tier="excellent",
    memory_usage="medium",
    default_params={
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
    },
)

registry.register(metadata=metadata, factory=XGBoostClassifier)
```

---

### 2. Transformer-Based Classifier (Highest Quality)

**Why:** SOTA results, fine-tunable, production-ready

**Implementation:**
```python
# src/lexiclass/plugins/classifiers/transformer.py

from __future__ import annotations

import logging
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)


class TransformerClassifier:
    """Transformer-based text classifier using Hugging Face.

    Fine-tunes pre-trained transformers (BERT, RoBERTa, DistilBERT, etc.)
    for document classification.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_length: int = 512,
        device: str | None = None,
    ):
        """Initialize transformer classifier.

        Args:
            model_name: Pre-trained model name
                - "distilbert-base-uncased" (fast, lightweight)
                - "bert-base-uncased" (balanced)
                - "roberta-base" (high quality)
            num_epochs: Training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            max_length: Maximum sequence length
            device: Device (None=auto, "cuda", "cpu")
        """
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.device = device
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.is_multilabel = False

    def train(
        self,
        texts: List[str],
        labels: List[Union[str, List[str]]],
    ) -> "TransformerClassifier":
        """Train transformer classifier.

        Args:
            texts: Document texts
            labels: Document labels
        """
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                TrainingArguments,
                Trainer,
            )
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "Transformer classifier requires transformers and datasets. "
                "Install with: pip install transformers datasets torch"
            )

        from sklearn import preprocessing

        # Detect multi-label
        self.is_multilabel = isinstance(labels[0], list)

        if self.is_multilabel:
            mlb = preprocessing.MultiLabelBinarizer()
            encoded_labels = mlb.fit_transform(labels)
            self.label_encoder = mlb
            num_labels = len(mlb.classes_)
        else:
            le = preprocessing.LabelEncoder()
            encoded_labels = le.fit_transform(labels)
            self.label_encoder = le
            num_labels = len(le.classes_)

        # Load tokenizer and model
        logger.info(f"Loading transformer model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification" if self.is_multilabel else "single_label_classification",
        )

        # Prepare dataset
        dataset = Dataset.from_dict({
            "text": texts,
            "labels": encoded_labels.tolist() if self.is_multilabel else encoded_labels,
        })

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            logging_steps=100,
            save_strategy="epoch",
        )

        # Train
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        logger.info("Starting transformer training...")
        trainer.train()
        logger.info("Training completed")

        return self

    def predict(
        self,
        texts: List[str],
    ) -> tuple[List[Union[str, List[str]]], np.ndarray]:
        """Predict labels for texts."""
        if self.model is None:
            raise ValueError("Model must be trained first")

        import torch

        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits

        if self.is_multilabel:
            # Multi-label: sigmoid + threshold
            probs = torch.sigmoid(logits).cpu().numpy()
            binary_preds = (probs > 0.5).astype(int)
            predictions = self.label_encoder.inverse_transform(binary_preds)
            scores = np.max(probs, axis=1)
        else:
            # Single-label: softmax
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_indices = np.argmax(probs, axis=1)
            predictions = self.label_encoder.inverse_transform(pred_indices)
            scores = np.max(probs, axis=1)

        return list(predictions), scores


# Plugin registration
from ..base import PluginMetadata, PluginType
from ..registry import registry

metadata = PluginMetadata(
    name="transformer",
    display_name="Transformer (BERT/RoBERTa)",
    description="Fine-tuned transformer models (state-of-the-art)",
    plugin_type=PluginType.CLASSIFIER,
    dependencies=["transformers>=4.30", "torch>=2.0", "datasets>=2.12"],
    supports_multilabel=True,
    performance_tier="slow",
    quality_tier="excellent",
    memory_usage="high",
    requires_pretrained=True,
    pretrained_models=[
        "distilbert-base-uncased",
        "bert-base-uncased",
        "roberta-base",
    ],
    default_params={
        "model_name": "distilbert-base-uncased",
        "num_epochs": 3,
        "batch_size": 16,
    },
)

registry.register(metadata=metadata, factory=TransformerClassifier)
```

---

### Classifier Comparison Table

| Classifier | SVM | Logistic | XGBoost | Transformer |
|------------|-----|----------|---------|-------------|
| **Speed (Train)** | Fast | Very Fast | Fast | Slow |
| **Speed (Predict)** | Fast | Very Fast | Fast | Medium |
| **Quality** | Good | Good | Excellent | Excellent |
| **Interpretability** | Medium | High | Low | Low |
| **Memory** | Medium | Low | Medium | High |
| **Multi-label** | Yes | Yes | Yes | Yes |
| **Dependencies** | sklearn | sklearn | xgboost | transformers, torch |
| **Best For** | Baseline | Fast | Production | SOTA |

---

## Implementation Strategy

### Phase 1: Foundation (Week 1-2)

**Goal:** Establish enhanced plugin system

1. **Refactor Plugin System**
   - Create `plugins/base.py` with metadata classes
   - Update `plugins/registry.py` with enhanced registry
   - Update existing plugins to use new system

2. **Move Existing Implementations**
   - `tokenization.py` → `plugins/tokenizers/icu.py`
   - `features.py` → `plugins/features/bow.py`
   - `classifier.py` → Extract SVM to `plugins/classifiers/svm.py`

3. **Update Interfaces**
   - Add `ClassifierProtocol` to `interfaces.py`
   - Ensure all protocols support new features

**Deliverables:**
- Enhanced plugin system with metadata
- Existing components as plugins
- Backward compatible

---

### Phase 2: High-Priority Additions (Week 3-4)

**Goal:** Add most valuable alternatives

1. **Feature Extractors**
   - Implement TF-IDF
   - Implement FastText (optional deps)

2. **Tokenizers**
   - Implement spaCy tokenizer

3. **Classifiers**
   - Implement XGBoost classifier
   - Extract SVM to plugin

**Deliverables:**
- TF-IDF, FastText features
- spaCy tokenizer
- XGBoost classifier
- CLI support for all

---

### Phase 3: Advanced Features (Week 5-6)

**Goal:** Add SOTA capabilities

1. **Sentence-BERT**
   - Implement SBERT feature extractor
   - Add model download utilities

2. **Transformer Classifier**
   - Implement transformer classifier
   - Add training utilities

3. **Additional Tokenizers**
   - SentencePiece
   - Hugging Face tokenizers

**Deliverables:**
- Sentence-BERT integration
- Transformer fine-tuning
- Full plugin ecosystem

---

### Phase 4: Polish & Documentation (Week 7-8)

**Goal:** Production-ready

1. **CLI Enhancements**
   - `lexiclass plugins list` command
   - `lexiclass plugins describe <name>` command
   - Better help messages

2. **Documentation**
   - Plugin development guide
   - Comparison benchmarks
   - Migration guide

3. **Testing**
   - Test all combinations
   - Performance benchmarks
   - Integration tests

**Deliverables:**
- Complete documentation
- Benchmark results
- Stable release

---

## API Design

### CLI Interface

#### List Available Plugins
```bash
# List all plugins
lexiclass plugins list

# List by type
lexiclass plugins list --type tokenizer
lexiclass plugins list --type feature
lexiclass plugins list --type classifier

# Show only available (dependencies installed)
lexiclass plugins list --available-only
```

Output:
```
Tokenizers:
  icu              PyICU (locale-aware)                     ✓ Available
  spacy            spaCy (modern multilingual)              ✗ Missing: spacy
  sentencepiece    SentencePiece (subword)                  ✓ Available

Feature Extractors:
  bow              Bag-of-Words                             ✓ Available
  tfidf            TF-IDF                                   ✓ Available
  fasttext         FastText embeddings                      ✓ Available
  sbert            Sentence-BERT                            ✗ Missing: sentence-transformers

Classifiers:
  svm              Linear SVM                               ✓ Available
  xgboost          XGBoost                                  ✓ Available
  transformer      Transformer (BERT/RoBERTa)               ✗ Missing: transformers, torch
```

#### Describe Plugin
```bash
lexiclass plugins describe sbert
```

Output:
```
Plugin: Sentence-BERT (sbert)
Type: feature_extractor
Status: ✗ Unavailable
Description: Transformer-based sentence embeddings (state-of-the-art)
Performance: slow
Quality: excellent
Memory: high
Streaming: No
Dependencies: sentence-transformers>=2.0
Optional: torch, transformers
Missing: sentence-transformers
Install: pip install sentence-transformers

Default parameters:
  model_name: all-MiniLM-L6-v2
  batch_size: 32
  normalize_embeddings: True

Pre-trained models available:
  - all-MiniLM-L6-v2 (384-dim, fast)
  - all-mpnet-base-v2 (768-dim, best quality)
  - paraphrase-multilingual-MiniLM-L12-v2 (multilingual)
```

#### Build Index with Different Plugins
```bash
# Default (BoW + ICU)
lexiclass build-index ./texts ./index

# TF-IDF + spaCy
lexiclass build-index ./texts ./index \
  --tokenizer spacy \
  --features tfidf

# Sentence-BERT + spaCy
lexiclass build-index ./texts ./index \
  --tokenizer spacy \
  --features sbert \
  --features-params model_name=all-mpnet-base-v2

# FastText with custom params
lexiclass build-index ./texts ./index \
  --features fasttext \
  --features-params vector_size=200,window=7
```

#### Train with Different Classifiers
```bash
# Default (SVM)
lexiclass train ./index ./labels.tsv ./model.pkl

# XGBoost
lexiclass train ./index ./labels.tsv ./model.pkl \
  --classifier xgboost

# XGBoost with GPU
lexiclass train ./index ./labels.tsv ./model.pkl \
  --classifier xgboost \
  --classifier-params use_gpu=true,n_estimators=200

# Transformer (requires text, not pre-built index)
lexiclass train-transformer ./texts ./labels.tsv ./model \
  --model-name distilbert-base-uncased \
  --num-epochs 5
```

---

### Library Interface

#### Using Different Plugins Programmatically

```python
from lexiclass.classifier import SVMDocumentClassifier
from lexiclass.plugins import registry
from lexiclass.io import DocumentLoader, load_labels

# Option 1: Create plugins manually
tokenizer = registry.create("spacy", model_name="en_core_web_sm")
features = registry.create("tfidf", normalize=True)

clf = SVMDocumentClassifier(
    tokenizer=tokenizer,
    feature_extractor=features,
)

# Build index
def stream_factory():
    return DocumentLoader.iter_documents_from_directory("./texts")

clf.build_index(
    index_path="my_index",
    document_stream_factory=stream_factory,
)

# Train with XGBoost
from lexiclass.plugins.classifiers import XGBoostClassifier

xgb_clf = XGBoostClassifier(n_estimators=200, use_gpu=True)
labels = load_labels("labels.tsv")

# Extract features
docs = DocumentLoader.load_documents_from_directory("./texts")
feature_matrix = features.transform([
    tokenizer.tokenize(text) for text in docs.values()
])

# Train
xgb_clf.train(feature_matrix, list(labels.values()))

# Predict
preds, scores = xgb_clf.predict(feature_matrix)
```

#### Configuration-Based Approach

```python
# config.yaml
tokenizer:
  name: spacy
  params:
    model_name: en_core_web_sm
    lowercase: true

features:
  name: tfidf
  params:
    normalize: true
    smartirs: ntc

classifier:
  name: xgboost
  params:
    n_estimators: 200
    max_depth: 8
    use_gpu: false
```

```python
import yaml
from lexiclass.plugins import registry

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Create from config
tokenizer = registry.create(
    config["tokenizer"]["name"],
    **config["tokenizer"]["params"]
)

features = registry.create(
    config["features"]["name"],
    **config["features"]["params"]
)

classifier = registry.create(
    config["classifier"]["name"],
    **config["classifier"]["params"]
)
```

---

## Configuration System

### Plugin Configuration File

Create `plugin_configs.yaml` for common presets:

```yaml
# plugin_configs.yaml

presets:
  # Fast baseline
  fast:
    tokenizer:
      name: icu
      params:
        locale: en
    features:
      name: bow
    classifier:
      name: svm

  # Balanced quality/speed
  balanced:
    tokenizer:
      name: spacy
      params:
        model_name: en_core_web_sm
    features:
      name: tfidf
      params:
        normalize: true
    classifier:
      name: xgboost
      params:
        n_estimators: 100

  # Best quality
  best:
    tokenizer:
      name: spacy
      params:
        model_name: en_core_web_lg
    features:
      name: sbert
      params:
        model_name: all-mpnet-base-v2
    classifier:
      name: transformer
      params:
        model_name: roberta-base
        num_epochs: 5

  # Multilingual
  multilingual:
    tokenizer:
      name: sentencepiece
      params:
        vocab_size: 16000
    features:
      name: sbert
      params:
        model_name: paraphrase-multilingual-MiniLM-L12-v2
    classifier:
      name: xgboost
```

#### CLI Usage with Presets

```bash
# Use preset
lexiclass build-index ./texts ./index --preset balanced

# Override specific params
lexiclass build-index ./texts ./index \
  --preset balanced \
  --features-params smartirs=nfc
```

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/plugins/test_plugin_registry.py

def test_register_plugin():
    """Test plugin registration."""
    registry = PluginRegistry()

    metadata = PluginMetadata(
        name="test",
        display_name="Test",
        description="Test plugin",
        plugin_type=PluginType.TOKENIZER,
    )

    registry.register(metadata, lambda: "test")
    assert "test" in registry.list_plugins()

def test_create_plugin_with_missing_deps():
    """Test error when dependencies missing."""
    # Mock plugin with missing dependency
    with pytest.raises(ImportError, match="missing dependencies"):
        registry.create("plugin_with_missing_deps")
```

### Integration Tests

```python
# tests/integration/test_plugin_combinations.py

@pytest.mark.parametrize("tokenizer,features,classifier", [
    ("icu", "bow", "svm"),
    ("spacy", "tfidf", "svm"),
    ("spacy", "tfidf", "xgboost"),
])
def test_plugin_combination(tokenizer, features, classifier, sample_data):
    """Test different plugin combinations."""
    tok = registry.create(tokenizer)
    feat = registry.create(features)
    clf = registry.create(classifier)

    # Test end-to-end pipeline
    # ... implementation
```

### Performance Benchmarks

```python
# benchmarks/compare_features.py

def benchmark_all_features(corpus_size=10000):
    """Benchmark all feature extractors."""
    results = {}

    for feature_name in ["bow", "tfidf", "fasttext", "sbert"]:
        if not is_available(feature_name):
            continue

        extractor = registry.create(feature_name)

        start = time.time()
        # ... benchmark code
        elapsed = time.time() - start

        results[feature_name] = {
            "time": elapsed,
            "throughput": corpus_size / elapsed,
        }

    return results
```

---

## Performance Considerations

### Memory Management

**Problem:** Different features have vastly different memory requirements

**Solution:** Add memory budgeting

```python
# In DocumentIndex.build_index()

def build_index(
    self,
    *,
    feature_extractor: FeatureExtractorProtocol,
    max_memory_gb: float = 4.0,
    **kwargs
):
    """Build index with memory budget."""

    # Estimate memory requirements
    feature_meta = registry.get_metadata(feature_extractor.__class__.__name__)

    if feature_meta.memory_usage == "high" and max_memory_gb < 8.0:
        logger.warning(
            f"{feature_meta.display_name} requires ~8GB memory, "
            f"but budget is {max_memory_gb}GB. Consider using batch processing."
        )
        # Enable batch processing automatically
        kwargs["use_batching"] = True
```

### Caching Strategy

**Different extractors need different caching:**

```python
# Sparse features (BoW, TF-IDF): Cache tokens
# Dense features (FastText, SBERT): Cache embeddings

class CacheManager:
    """Smart caching based on feature type."""

    def should_cache_tokens(self, feature_extractor):
        """Sparse features cache tokens."""
        return feature_extractor.num_features() > 10000

    def should_cache_embeddings(self, feature_extractor):
        """Dense features cache embeddings."""
        return hasattr(feature_extractor, "vector_size")
```

---

## Migration Path

### For Existing Users

**Backward Compatibility:**
- Old code continues to work unchanged
- Existing indexes remain compatible
- Default behavior unchanged

**Migration Guide:**

```python
# Old way (still works)
from lexiclass.classifier import SVMDocumentClassifier
from lexiclass.tokenization import ICUTokenizer
from lexiclass.features import FeatureExtractor

clf = SVMDocumentClassifier(
    tokenizer=ICUTokenizer(),
    feature_extractor=FeatureExtractor(),
)

# New way (recommended)
from lexiclass.plugins import registry

tokenizer = registry.create("icu", locale="en")
features = registry.create("tfidf")  # Upgrade to TF-IDF

clf = SVMDocumentClassifier(
    tokenizer=tokenizer,
    feature_extractor=features,
)
```

### Deprecation Timeline

**v0.2.0:** Add new plugin system, keep old imports
**v0.3.0:** Deprecation warnings for direct imports
**v1.0.0:** Remove direct imports, plugin-only

---

## Documentation Updates

### New Documentation Sections

1. **Plugin Development Guide**
   - How to create custom plugins
   - Plugin interface requirements
   - Registration process

2. **Plugin Comparison Guide**
   - Performance benchmarks
   - Quality comparisons
   - Use case recommendations

3. **Migration Guide**
   - From BoW to TF-IDF
   - From SVM to XGBoost
   - From traditional to transformers

4. **Configuration Guide**
   - Using presets
   - Custom configurations
   - Environment-based selection

---

## Success Criteria

### Technical
- ✅ All existing functionality preserved
- ✅ At least 2 alternatives per component type
- ✅ Plugin system with metadata
- ✅ CLI support for all plugins
- ✅ Comprehensive tests (80%+ coverage)

### Documentation
- ✅ Plugin development guide
- ✅ Comparison benchmarks published
- ✅ Migration guide for users
- ✅ API documentation updated

### Performance
- ✅ No regression in default configuration
- ✅ Faster alternatives available (XGBoost)
- ✅ Higher quality alternatives available (Transformers)

---

## Risks & Mitigation

### Risk 1: Dependency Bloat
**Mitigation:** All alternatives are optional dependencies

```toml
[project.optional-dependencies]
features-all = ["sentence-transformers>=2.0"]
classifiers-all = ["xgboost>=1.7", "transformers>=4.30"]
tokenizers-all = ["spacy>=3.0", "sentencepiece>=0.1.99"]
```

### Risk 2: Breaking Changes
**Mitigation:** Careful versioning, deprecation warnings, migration guide

### Risk 3: Maintenance Burden
**Mitigation:** Start with high-value plugins, add more based on demand

---

## Next Steps

### Immediate (This Week)
1. Review and approve plan
2. Set up development branch
3. Create issue tracker for plugin implementations

### Short-term (Next 2 Weeks)
1. Implement enhanced plugin system
2. Refactor existing components as plugins
3. Add TF-IDF and XGBoost

### Medium-term (Next 1-2 Months)
1. Add remaining high-priority plugins
2. Comprehensive testing
3. Documentation and examples

### Long-term (Next 3-6 Months)
1. Community plugin contributions
2. Performance optimizations
3. Advanced features (AutoML, hyperparameter tuning)

---

## Conclusion

This plan provides a path to extend LexiClass with modern ML alternatives while maintaining its clean architecture and ease of use. The plugin system allows users to choose the right tool for their use case, from fast baselines to SOTA transformers.

**Key Benefits:**
- 🚀 Modern ML techniques available
- 🔌 Plugin architecture for easy extension
- 📊 Clear performance/quality tradeoffs
- 🔄 Backward compatible
- 📚 Well-documented

**Recommended Starting Order:**
1. Enhanced plugin system (foundation)
2. TF-IDF + XGBoost (high ROI)
3. Sentence-BERT (SOTA capability)
4. Additional tokenizers (flexibility)
5. Transformers (cutting-edge)
