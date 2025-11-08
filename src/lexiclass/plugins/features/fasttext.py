"""FastText feature extractor plugin."""

from __future__ import annotations

import logging
from typing import Iterable, List, Tuple

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


class FastTextFeatureExtractor:
    """FastText subword embeddings for feature extraction.

    Uses Gensim's FastText implementation to create document embeddings.
    Handles out-of-vocabulary words via character n-grams.
    Returns dense vectors instead of sparse.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 3,
        workers: int = 4,
        sg: int = 1,  # Skip-gram (1) or CBOW (0)
        min_n: int = 3,  # Min character n-gram length
        max_n: int = 6,  # Max character n-gram length
    ) -> None:
        """Initialize FastText extractor.

        Args:
            vector_size: Dimensionality of embeddings
            window: Context window size
            min_count: Minimum word frequency
            workers: Number of worker threads for training
            sg: Training algorithm (1=skip-gram, 0=CBOW)
            min_n: Minimum character n-gram length
            max_n: Maximum character n-gram length
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.min_n = min_n
        self.max_n = max_n
        self.model = None
        self.fitted = False

    def fit(self, documents: List[List[str]]) -> "FastTextFeatureExtractor":
        """Fit FastText model.

        Args:
            documents: List of tokenized documents

        Returns:
            Self for chaining
        """
        try:
            from gensim.models import FastText
        except ImportError:
            raise ImportError(
                "FastText requires gensim with FastText support. "
                "Install with: pip install gensim"
            )

        logger.info(f"Training FastText model on {len(documents)} documents")
        logger.info(f"Parameters: vector_size={self.vector_size}, window={self.window}, sg={self.sg}")

        self.model = FastText(
            sentences=documents,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            min_n=self.min_n,
            max_n=self.max_n,
        )

        self.fitted = True
        vocab_size = len(self.model.wv)
        logger.info(f"FastText model ready with {self.vector_size}-dim vectors, vocabulary size: {vocab_size}")
        return self

    def fit_streaming(
        self,
        tokenized_documents_iter: Iterable[List[str]]
    ) -> "FastTextFeatureExtractor":
        """Fit using streaming approach.

        Note: FastText requires all documents for training, so we collect them.

        Args:
            tokenized_documents_iter: Iterator of tokenized documents

        Returns:
            Self for chaining
        """
        from gensim.models import FastText

        logger.info("Training FastText model from streaming documents")

        # Collect sentences for training
        # FastText training requires the full corpus
        sentences = list(tokenized_documents_iter)
        logger.info(f"Collected {len(sentences)} documents for FastText training")

        self.model = FastText(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            min_n=self.min_n,
            max_n=self.max_n,
        )

        self.fitted = True
        vocab_size = len(self.model.wv)
        logger.info(f"FastText model ready with {self.vector_size}-dim vectors, vocabulary size: {vocab_size}")
        return self

    def transform(self, documents: List[List[str]]) -> np.ndarray:
        """Transform documents to dense embeddings.

        Args:
            documents: List of tokenized documents

        Returns:
            Dense numpy array (not sparse) of document embeddings
        """
        if not self.fitted:
            raise ValueError("FastTextFeatureExtractor must be fitted before transform")

        logger.info(f"Transforming {len(documents)} documents to FastText embeddings")

        vectors = []
        for doc_tokens in documents:
            if not doc_tokens:
                # Empty document -> zero vector
                vectors.append(np.zeros(self.vector_size))
            else:
                # Average word vectors (handles OOV via character n-grams)
                word_vecs = []
                for token in doc_tokens:
                    # FastText handles OOV words via character n-grams
                    try:
                        word_vecs.append(self.model.wv[token])
                    except KeyError:
                        # Should not happen with FastText, but handle gracefully
                        pass

                if word_vecs:
                    doc_vec = np.mean(word_vecs, axis=0)
                else:
                    doc_vec = np.zeros(self.vector_size)

                vectors.append(doc_vec)

        result = np.vstack(vectors)
        logger.info(f"Transformation complete: shape {result.shape}")
        return result

    def fit_transform(self, documents: List[List[str]]) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            documents: List of tokenized documents

        Returns:
            Dense embedding matrix
        """
        return self.fit(documents).transform(documents)

    def tokens_to_bow(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens to embedding vector.

        Note: Returns dense vector, not sparse BoW format.
        The name is kept for interface compatibility.

        Args:
            tokens: List of tokens

        Returns:
            Dense embedding vector
        """
        if not self.fitted:
            raise ValueError("FastTextFeatureExtractor must be fitted")

        if not tokens:
            return np.zeros(self.vector_size)

        word_vecs = []
        for token in tokens:
            try:
                word_vecs.append(self.model.wv[token])
            except KeyError:
                pass

        if word_vecs:
            return np.mean(word_vecs, axis=0)
        else:
            return np.zeros(self.vector_size)

    def num_features(self) -> int:
        """Return dimensionality of embeddings.

        Returns:
            Embedding dimension size
        """
        return self.vector_size

    def save(self, path: str) -> None:
        """Save FastText feature extractor to disk.

        Args:
            path: Path to save the extractor
        """
        import pickle

        if not self.fitted:
            logger.warning("Saving unfitted FastTextFeatureExtractor")

        # Save using Gensim's native save method
        model_path = f"{path}.model"
        if self.model:
            self.model.save(model_path)

        # Save metadata
        metadata_path = f"{path}.metadata.pkl"
        metadata = {
            'vector_size': self.vector_size,
            'window': self.window,
            'min_count': self.min_count,
            'workers': self.workers,
            'sg': self.sg,
            'min_n': self.min_n,
            'max_n': self.max_n,
            'fitted': self.fitted,
        }

        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"FastTextFeatureExtractor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FastTextFeatureExtractor":
        """Load FastText feature extractor from disk.

        Args:
            path: Path to the saved extractor

        Returns:
            Loaded FastTextFeatureExtractor instance
        """
        import pickle

        try:
            from gensim.models import FastText
        except ImportError:
            raise ImportError(
                "Loading FastText requires gensim. Install with: pip install gensim"
            )

        # Load metadata
        metadata_path = f"{path}.metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        # Create instance with saved parameters
        instance = cls(
            vector_size=metadata['vector_size'],
            window=metadata['window'],
            min_count=metadata['min_count'],
            workers=metadata['workers'],
            sg=metadata['sg'],
            min_n=metadata['min_n'],
            max_n=metadata['max_n'],
        )

        # Load the model using Gensim's load method
        model_path = f"{path}.model"
        instance.model = FastText.load(model_path)

        # Restore state
        instance.fitted = metadata['fitted']

        logger.info(f"FastTextFeatureExtractor loaded from {path}")
        return instance


# Plugin registration
from ..base import PluginMetadata, PluginType
from ..registry import registry

metadata = PluginMetadata(
    name="fasttext",
    display_name="FastText",
    description="Subword embeddings with character n-grams (handles OOV, dense vectors)",
    plugin_type=PluginType.FEATURE_EXTRACTOR,
    dependencies=["gensim>=4.3", "numpy>=1.22"],
    supports_streaming=True,
    performance_tier="medium",
    quality_tier="excellent",
    memory_usage="medium",
    requires_pretrained=False,
    default_params={
        "vector_size": 100,
        "window": 5,
        "min_count": 3,
        "sg": 1,
    },
)

registry.register(
    metadata=metadata,
    factory=lambda **kwargs: FastTextFeatureExtractor(**kwargs)
)
