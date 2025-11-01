"""Sentence-BERT feature extractor plugin."""

from __future__ import annotations

import logging
from typing import Iterable, List

import numpy as np

logger = logging.getLogger(__name__)


class SentenceBERTFeatureExtractor:
    """Sentence-BERT transformer-based embeddings.

    Uses pre-trained Sentence-Transformers for high-quality
    semantic document embeddings. State-of-the-art quality
    but slower than traditional methods.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress: bool = False,
    ) -> None:
        """Initialize Sentence-BERT extractor.

        Args:
            model_name: Pre-trained model name
                - "all-MiniLM-L6-v2" (default, fast, 384-dim, good quality)
                - "all-mpnet-base-v2" (best quality, 768-dim, slower)
                - "all-MiniLM-L12-v2" (balanced, 384-dim)
                - "paraphrase-multilingual-MiniLM-L12-v2" (multilingual)
                - "paraphrase-multilingual-mpnet-base-v2" (multilingual, best)
            device: Device to use (None=auto, "cuda", "cpu", "mps")
            batch_size: Batch size for encoding
            normalize_embeddings: L2-normalize embeddings
            show_progress: Show progress bar during encoding
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress = show_progress
        self.model = None
        self.fitted = False
        self._vector_size = None

    def fit(self, documents: List[List[str]]) -> "SentenceBERTFeatureExtractor":
        """Load pre-trained model (no training needed).

        Args:
            documents: List of tokenized documents (not used, for API compatibility)

        Returns:
            Self for chaining
        """
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
        """Load model (streaming not needed for pre-trained).

        Args:
            tokenized_documents_iter: Iterator of tokenized documents (consumed but not used)

        Returns:
            Self for chaining
        """
        # Pre-trained models don't need documents for fitting
        # Just consume the iterator to maintain API compatibility
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
            raise ValueError("SentenceBERTFeatureExtractor must be fitted before transform")

        # Reconstruct text from tokens (Sentence-BERT works on text)
        texts = [" ".join(tokens) for tokens in documents]

        # Encode in batches
        logger.info(f"Encoding {len(texts)} documents with Sentence-BERT")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress and len(texts) > 100,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        logger.info(f"Encoding complete: shape {embeddings.shape}")
        return embeddings

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

        Note: Returns dense vector, not BoW format.
        Name kept for interface compatibility.

        Args:
            tokens: List of tokens

        Returns:
            Dense embedding vector
        """
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
        """Return embedding dimensionality.

        Returns:
            Embedding dimension size
        """
        return self._vector_size if self._vector_size else 0


# Plugin registration
from ..base import PluginMetadata, PluginType
from ..registry import registry

metadata = PluginMetadata(
    name="sbert",
    display_name="Sentence-BERT",
    description="Transformer-based sentence embeddings (state-of-the-art quality)",
    plugin_type=PluginType.FEATURE_EXTRACTOR,
    dependencies=["sentence-transformers>=2.0"],
    optional_dependencies=["torch>=2.0", "transformers>=4.30"],
    supports_streaming=False,  # Loads pre-trained, doesn't need training
    performance_tier="slow",  # Transformer inference is slower
    quality_tier="excellent",
    memory_usage="high",
    requires_pretrained=True,
    pretrained_models=[
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "all-MiniLM-L12-v2",
        "paraphrase-multilingual-MiniLM-L12-v2",
        "paraphrase-multilingual-mpnet-base-v2",
    ],
    default_params={
        "model_name": "all-MiniLM-L6-v2",
        "batch_size": 32,
        "normalize_embeddings": True,
        "show_progress": False,
    },
)

registry.register(
    metadata=metadata,
    factory=lambda **kwargs: SentenceBERTFeatureExtractor(**kwargs)
)
