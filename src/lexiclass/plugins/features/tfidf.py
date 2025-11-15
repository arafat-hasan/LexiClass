"""TF-IDF feature extractor plugin."""

from __future__ import annotations

import logging
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Optional, Tuple

from gensim import corpora, models
from scipy import sparse

from ...memory_utils import calculate_batch_size, monitor_memory_usage

logger = logging.getLogger(__name__)


# Token filtering constants (same as BoW)
MAX_CHARS_PER_TOKEN = 40
MIN_ALPHANUMERIC_PCT_PER_TOKEN = 0.5
MAX_NUMERIC_PCT_PER_TOKEN = 0.5
MIN_DOCS_PER_TOKEN = 3
MAX_DOCS_PCT_PER_TOKEN = 0.5


class TfidfFeatureExtractor:
    """TF-IDF feature extraction using Gensim.

    Converts documents to TF-IDF weighted sparse vectors.
    More informative than raw bag-of-words as it downweights
    common terms and upweights distinctive terms.
    """

    def __init__(
        self,
        normalize: bool = True,
        smartirs: str = 'ntc',
        num_workers: Optional[int] = None,
    ) -> None:
        """Initialize TF-IDF extractor.

        Args:
            normalize: Whether to L2-normalize vectors
            smartirs: TF-IDF weighting scheme
                     (n=natural TF, t=IDF, c=cosine normalization)
            num_workers: Number of workers for parallel processing
        """
        self.dictionary: corpora.Dictionary | None = None
        self.tfidf_model: models.TfidfModel | None = None
        self.normalize = normalize
        self.smartirs = smartirs
        self.num_workers = num_workers if num_workers is not None else max(1, mp.cpu_count() - 1)
        self.fitted: bool = False

    def fit(self, documents: List[List[str]]) -> "TfidfFeatureExtractor":
        """Fit dictionary and TF-IDF model.

        Args:
            documents: List of tokenized documents

        Returns:
            Self for chaining
        """
        start_time = time.time()
        logger.info("Creating TF-IDF model from %d documents", len(documents))

        # Build dictionary
        dict_start = time.time()
        self.dictionary = corpora.Dictionary(documents)
        logger.info("Gensim dictionary created in %.2f seconds", time.time() - dict_start)

        # Filter dictionary
        filter_start = time.time()
        self._filter_dictionary()
        logger.info("Dictionary filtering completed in %.2f seconds", time.time() - filter_start)

        # Create corpus for fitting TF-IDF
        logger.info("Creating bag-of-words corpus for TF-IDF training...")
        corpus = [self.dictionary.doc2bow(doc) for doc in documents]

        # Fit TF-IDF model
        tfidf_start = time.time()
        self.tfidf_model = models.TfidfModel(
            corpus,
            normalize=self.normalize,
            smartirs=self.smartirs,
        )
        logger.info("TF-IDF model fitted in %.2f seconds", time.time() - tfidf_start)

        self.fitted = True
        logger.info(
            "TF-IDF feature extractor created with %d features in %.2f seconds total",
            len(self.dictionary), time.time() - start_time
        )
        return self

    def fit_streaming(
        self,
        tokenized_documents_iter: Iterable[List[str]]
    ) -> "TfidfFeatureExtractor":
        """Fit using streaming approach.

        Note: For best TF-IDF results, we need corpus statistics.
        This implementation builds the dictionary in streaming mode,
        then makes a second pass to compute TF-IDF statistics.

        Args:
            tokenized_documents_iter: Iterator of tokenized documents

        Returns:
            Self for chaining
        """
        start_time = time.time()
        logger.info("Creating TF-IDF model from streaming documents")

        # First pass: build dictionary
        BATCH_SIZE = 1000
        PRUNE_AT = 2_000_000

        self.dictionary = corpora.Dictionary()
        batch: list[list[str]] = []
        num_docs_seen = 0

        # We need to cache documents for the second pass
        cached_docs: list[list[str]] = []

        for tokens in tokenized_documents_iter:
            batch.append(tokens)
            cached_docs.append(tokens)
            num_docs_seen += 1

            if len(batch) >= BATCH_SIZE:
                self.dictionary.add_documents(batch, prune_at=PRUNE_AT)
                batch.clear()

            if num_docs_seen % 10000 == 0:
                logger.info("Added %d documents to dictionary so far", num_docs_seen)

        if batch:
            self.dictionary.add_documents(batch)

        logger.info(
            "Gensim dictionary (streaming) created in %.2f seconds from %d documents",
            time.time() - start_time,
            num_docs_seen,
        )

        # Filter dictionary
        filter_start = time.time()
        self._filter_dictionary()
        logger.info("Dictionary filtering completed in %.2f seconds", time.time() - filter_start)

        # Second pass: compute TF-IDF statistics
        logger.info("Computing TF-IDF statistics from %d documents...", len(cached_docs))
        tfidf_start = time.time()

        # Create corpus from cached documents
        corpus = [self.dictionary.doc2bow(doc) for doc in cached_docs]

        self.tfidf_model = models.TfidfModel(
            corpus,
            normalize=self.normalize,
            smartirs=self.smartirs,
        )

        logger.info("TF-IDF model fitted in %.2f seconds", time.time() - tfidf_start)

        self.fitted = True
        logger.info(
            "TF-IDF feature extractor created with %d features in %.2f seconds total (streaming)",
            len(self.dictionary),
            time.time() - start_time,
        )
        return self

    def transform(
        self,
        documents: List[List[str]],
        batch_size: Optional[int] = None,
        target_memory_usage: float = 0.25,
    ) -> sparse.csr_matrix:
        """Transform documents to TF-IDF sparse matrix.

        Args:
            documents: List of tokenized documents
            batch_size: Number of documents to process in each batch
            target_memory_usage: Target memory usage fraction

        Returns:
            Sparse matrix of TF-IDF weighted document vectors
        """
        if not self.fitted:
            raise ValueError("TfidfFeatureExtractor must be fitted before transform")

        start_time = time.time()
        logger.info(
            "Transforming %d documents to TF-IDF vectors using %d workers",
            len(documents), self.num_workers
        )

        # Process documents in parallel batches
        bow_start = time.time()

        def process_batch(batch: List[List[str]]) -> List[List[Tuple[int, float]]]:
            # Convert to bag-of-words then apply TF-IDF
            bow_corpus = [self.dictionary.doc2bow(doc) for doc in batch]  # type: ignore[arg-type]
            return [self.tfidf_model[bow] for bow in bow_corpus]  # type: ignore[index]

        # Calculate optimal batch size if not provided
        if batch_size is None:
            sample_size = min(1000, len(documents))
            avg_doc_size = sum(
                sum(len(token.encode('utf-8')) for token in doc)
                for doc in documents[:sample_size]
            ) / sample_size
            batch_size = calculate_batch_size(
                num_docs=len(documents),
                avg_doc_size=avg_doc_size,
                target_memory_usage=target_memory_usage,
            )
            logger.info("Using adaptive batch size: %d", batch_size)

        # Split documents into batches
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

        # Process batches in parallel
        tfidf_corpus: List[List[Tuple[int, float]]] = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for i, batch_tfidf in enumerate(executor.map(process_batch, batches)):
                tfidf_corpus.extend(batch_tfidf)

                # Monitor memory usage every 10 batches
                if i > 0 and i % 10 == 0:
                    monitor_memory_usage()
                    logger.debug(
                        "Processed %d/%d batches (%.1f%%)",
                        i, len(batches), i * 100 / len(batches)
                    )

        logger.info(
            "Parallel TF-IDF transformation completed in %.2f seconds",
            time.time() - bow_start
        )

        # Create sparse matrix
        sparse_start = time.time()
        num_docs = len(tfidf_corpus)
        num_features = len(self.dictionary)  # type: ignore[arg-type]

        # Pre-allocate lists with estimated size
        estimated_nnz = sum(len(tfidf_vec) for tfidf_vec in tfidf_corpus)
        rows = [0] * estimated_nnz
        cols = [0] * estimated_nnz
        data = [0.0] * estimated_nnz
        pos = 0

        for doc_idx, tfidf_vec in enumerate(tfidf_corpus):
            for token_id, weight in tfidf_vec:
                rows[pos] = doc_idx
                cols[pos] = token_id
                data[pos] = weight
                pos += 1

        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(num_docs, num_features))
        logger.info("Sparse matrix creation completed in %.2f seconds", time.time() - sparse_start)

        logger.info("Document transformation completed in %.2f seconds total", time.time() - start_time)
        return matrix

    def fit_transform(self, documents: List[List[str]]) -> sparse.csr_matrix:
        """Fit and transform in one step.

        Args:
            documents: List of tokenized documents

        Returns:
            Sparse TF-IDF matrix
        """
        return self.fit(documents).transform(documents)

    def tokens_to_bow(self, tokens: List[str]) -> List[Tuple[int, float]]:
        """Convert tokens to TF-IDF vector.

        Args:
            tokens: List of tokens

        Returns:
            List of (token_id, tfidf_weight) tuples
        """
        if not self.fitted:
            raise ValueError("TfidfFeatureExtractor must be fitted before tokens_to_bow")

        bow = self.dictionary.doc2bow(tokens)  # type: ignore[union-attr]
        return self.tfidf_model[bow]  # type: ignore[index]

    def num_features(self) -> int:
        """Return number of features.

        Returns:
            Number of features in the dictionary
        """
        return len(self.dictionary) if self.dictionary else 0

    def _filter_dictionary(self) -> None:
        """Filter dictionary (same logic as BoW)."""
        assert self.dictionary is not None

        # Determine adaptive filtering parameters based on corpus size
        num_docs = self.dictionary.num_docs

        # For small corpora, use more lenient filtering
        if num_docs < 10:
            min_docs = 2
            max_docs_pct = 1.0
        elif num_docs < 50:
            min_docs = 2
            max_docs_pct = 0.8
        else:
            min_docs = MIN_DOCS_PER_TOKEN
            max_docs_pct = MAX_DOCS_PCT_PER_TOKEN

        logger.info(
            "Applying dictionary filtering: min_docs=%d, max_docs_pct=%.2f (corpus size: %d documents)",
            min_docs, max_docs_pct, num_docs
        )

        # Apply frequency-based filtering
        self.dictionary.filter_extremes(
            no_below=min_docs,
            no_above=max_docs_pct,
            keep_n=None,
        )

        # Filter garbage tokens
        garbage_ids = []
        for token_id, token in self.dictionary.id2token.items():
            if self._is_garbage_token(token):
                garbage_ids.append(token_id)
                if len(garbage_ids) >= 10000:
                    self.dictionary.filter_tokens(garbage_ids)
                    garbage_ids = []

        if garbage_ids:
            self.dictionary.filter_tokens(garbage_ids)

        # Log warning if dictionary is empty
        if len(self.dictionary) == 0:
            logger.warning(
                "Dictionary is empty after filtering. Consider using longer documents with more varied content."
            )

    def _is_garbage_token(self, token: str) -> bool:
        """Check if token should be filtered out."""
        token_len = len(token)
        alpha_count = sum(c.isalpha() for c in token)
        num_count = sum(c.isnumeric() for c in token)

        if token_len > MAX_CHARS_PER_TOKEN:
            return True
        if (alpha_count + num_count) < MIN_ALPHANUMERIC_PCT_PER_TOKEN * token_len:
            return True
        if num_count > MAX_NUMERIC_PCT_PER_TOKEN * token_len:
            return True
        return False

    def save(self, path: str) -> None:
        """Save TF-IDF feature extractor to disk.

        Args:
            path: Path to save the feature extractor
        """
        import pickle

        if not self.fitted:
            logger.warning("Saving unfitted TfidfFeatureExtractor")

        model_data = {
            'dictionary': self.dictionary,
            'tfidf_model': self.tfidf_model,
            'normalize': self.normalize,
            'smartirs': self.smartirs,
            'num_workers': self.num_workers,
            'fitted': self.fitted,
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"TfidfFeatureExtractor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TfidfFeatureExtractor":
        """Load TF-IDF feature extractor from disk.

        Args:
            path: Path to the saved feature extractor

        Returns:
            Loaded TfidfFeatureExtractor instance
        """
        import pickle

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        # Create instance with saved parameters
        instance = cls(
            normalize=model_data['normalize'],
            smartirs=model_data['smartirs'],
            num_workers=model_data.get('num_workers'),
        )

        # Restore state
        instance.dictionary = model_data['dictionary']
        instance.tfidf_model = model_data['tfidf_model']
        instance.fitted = model_data['fitted']

        logger.info(f"TfidfFeatureExtractor loaded from {path}")
        return instance


# Plugin registration
from ..base import PluginMetadata, PluginType
from ..registry import registry

metadata = PluginMetadata(
    name="tfidf",
    display_name="TF-IDF",
    description="Term Frequency-Inverse Document Frequency weighting (better than BoW)",
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
