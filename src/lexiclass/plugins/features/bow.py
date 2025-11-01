"""Bag-of-Words feature extractor plugin."""

from __future__ import annotations

import logging
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Optional, Tuple

from gensim import corpora
from scipy import sparse

from ...memory_utils import calculate_batch_size, monitor_memory_usage

logger = logging.getLogger(__name__)


# Token filtering constants
MAX_CHARS_PER_TOKEN = 40
MIN_ALPHANUMERIC_PCT_PER_TOKEN = 0.5
MAX_NUMERIC_PCT_PER_TOKEN = 0.5
MIN_DOCS_PER_TOKEN = 3
MAX_DOCS_PCT_PER_TOKEN = 0.5


class FeatureExtractor:
    """Bag-of-words feature extraction using a Gensim Dictionary.

    Supports both in-memory and streaming dictionary construction.
    """

    def __init__(self, num_workers: Optional[int] = None) -> None:
        self.dictionary: corpora.Dictionary | None = None
        self.fitted: bool = False
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)

    def fit(self, documents: List[List[str]]) -> "FeatureExtractor":
        start_time = time.time()
        logger.info("Creating dictionary from %d documents", len(documents))

        dict_start = time.time()
        self.dictionary = corpora.Dictionary(documents)
        logger.info("Gensim dictionary created in %.2f seconds", time.time() - dict_start)

        filter_start = time.time()
        self._filter_dictionary()
        logger.info("Dictionary filtering completed in %.2f seconds", time.time() - filter_start)

        self.fitted = True
        logger.info("Dictionary created with %d features in %.2f seconds total", len(self.dictionary), time.time() - start_time)
        return self

    def fit_streaming(self, tokenized_documents_iter: Iterable[List[str]]) -> "FeatureExtractor":
        start_time = time.time()
        logger.info("Creating dictionary from streaming documents")

        # Internal batching to reduce Python overhead of add_documents per doc
        # and to keep memory usage in check for very large corpora.
        BATCH_SIZE = 1000
        PRUNE_AT = 2_000_000

        self.dictionary = corpora.Dictionary()
        batch: list[list[str]] = []
        num_docs_seen = 0

        for tokens in tokenized_documents_iter:
            batch.append(tokens)
            num_docs_seen += 1
            if len(batch) >= BATCH_SIZE:
                self.dictionary.add_documents(batch, prune_at=PRUNE_AT)
                batch.clear()
            if num_docs_seen % 10000 == 0:
                logger.info("Added %d documents to dictionary so far", num_docs_seen)

        if batch:
            self.dictionary.add_documents(batch, prune_at=PRUNE_AT)

        logger.info(
            "Gensim dictionary (streaming) created in %.2f seconds from %d documents",
            time.time() - start_time,
            num_docs_seen,
        )

        filter_start = time.time()
        self._filter_dictionary()
        logger.info("Dictionary filtering completed in %.2f seconds", time.time() - filter_start)

        self.fitted = True
        logger.info(
            "Dictionary created with %d features in %.2f seconds total (streaming)",
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
        """Transform documents to sparse matrix using parallel processing.

        Args:
            documents: List of tokenized documents
            batch_size: Number of documents to process in each batch

        Returns:
            Sparse matrix of document vectors
        """
        if not self.fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")

        start_time = time.time()
        logger.info("Transforming %d documents to feature vectors using %d workers", len(documents), self.num_workers)

        # Process documents in parallel batches
        bow_start = time.time()

        def process_batch(batch: List[List[str]]) -> List[List[Tuple[int, float]]]:
            return [self.dictionary.doc2bow(doc) for doc in batch]  # type: ignore[arg-type]

        # Calculate optimal batch size if not provided
        if batch_size is None:
            # Estimate average document size from first 1000 docs
            sample_size = min(1000, len(documents))
            avg_doc_size = sum(sum(len(token.encode('utf-8')) for token in doc) for doc in documents[:sample_size]) / sample_size
            batch_size = calculate_batch_size(
                num_docs=len(documents),
                avg_doc_size=avg_doc_size,
                target_memory_usage=target_memory_usage,
            )
            logger.info("Using adaptive batch size: %d", batch_size)

        # Split documents into batches
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

        # Process batches in parallel with memory monitoring
        bow_corpus: List[List[Tuple[int, float]]] = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for i, batch_bows in enumerate(executor.map(process_batch, batches)):
                bow_corpus.extend(batch_bows)

                # Monitor memory usage every 10 batches
                if i > 0 and i % 10 == 0:
                    monitor_memory_usage()
                    logger.debug(
                        "Processed %d/%d batches (%.1f%%)",
                        i, len(batches), i * 100 / len(batches)
                    )

        logger.info("Parallel bag-of-words conversion completed in %.2f seconds", time.time() - bow_start)

        # Create sparse matrix
        sparse_start = time.time()
        num_docs = len(bow_corpus)
        num_features = len(self.dictionary)  # type: ignore[arg-type]

        # Pre-allocate lists with estimated size
        estimated_nnz = sum(len(bow) for bow in bow_corpus)
        rows = [0] * estimated_nnz
        cols = [0] * estimated_nnz
        data = [0.0] * estimated_nnz
        pos = 0

        for doc_idx, bow in enumerate(bow_corpus):
            for token_id, count in bow:
                rows[pos] = doc_idx
                cols[pos] = token_id
                data[pos] = count
                pos += 1

        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(num_docs, num_features))
        logger.info("Sparse matrix creation completed in %.2f seconds", time.time() - sparse_start)

        logger.info("Document transformation completed in %.2f seconds total", time.time() - start_time)
        return matrix

    def fit_transform(self, documents: List[List[str]]) -> sparse.csr_matrix:
        return self.fit(documents).transform(documents)

    def tokens_to_bow(self, tokens: List[str]) -> List[tuple[int, float]]:
        if self.dictionary is None:
            raise ValueError("FeatureExtractor must be fitted before tokens_to_bow")
        return self.dictionary.doc2bow(tokens)

    def _filter_dictionary(self) -> None:
        """Filter dictionary using streaming approach to minimize memory usage."""
        assert self.dictionary is not None

        # Determine adaptive filtering parameters based on corpus size
        num_docs = self.dictionary.num_docs

        # For small corpora, use more lenient filtering
        if num_docs < 10:
            # Very small corpus: only filter tokens that appear in exactly 1 document
            min_docs = 2
            max_docs_pct = 1.0  # Allow tokens in all documents
        elif num_docs < 50:
            # Small corpus: require at least 2 docs, allow up to 80%
            min_docs = 2
            max_docs_pct = 0.8
        else:
            # Normal corpus: use standard filtering
            min_docs = MIN_DOCS_PER_TOKEN
            max_docs_pct = MAX_DOCS_PCT_PER_TOKEN

        logger.info(
            "Applying dictionary filtering: min_docs=%d, max_docs_pct=%.2f (corpus size: %d documents)",
            min_docs, max_docs_pct, num_docs
        )

        # First apply frequency-based filtering
        self.dictionary.filter_extremes(
            no_below=min_docs,
            no_above=max_docs_pct,
            keep_n=None,
        )

        # Stream through tokens and collect garbage IDs directly
        # This avoids materializing two separate lists (tokens and their IDs)
        garbage_ids = []
        for token_id, token in self.dictionary.id2token.items():
            if self._is_garbage_token(token):
                garbage_ids.append(token_id)
                # Free memory as we go by removing from both mappings
                # This helps when processing very large dictionaries
                if len(garbage_ids) >= 10000:
                    self.dictionary.filter_tokens(garbage_ids)
                    garbage_ids = []

        # Filter any remaining garbage tokens
        if garbage_ids:
            self.dictionary.filter_tokens(garbage_ids)

        # Log warning if dictionary is empty after filtering
        if len(self.dictionary) == 0:
            logger.warning(
                "Dictionary is empty after filtering. This means no valid features could be extracted. "
                "Possible causes: (1) Documents are too short or contain only stop words, "
                "(2) All tokens appear in too many or too few documents, "
                "(3) Documents have insufficient vocabulary diversity. "
                f"Original corpus size: {num_docs} documents. "
                "Consider using longer documents with more varied content."
            )

    def _is_garbage_token(self, token: str) -> bool:
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

    def num_features(self) -> int:
        return len(self.dictionary) if self.dictionary is not None else 0


# Plugin registration
from ..base import PluginMetadata, PluginType
from ..registry import registry

metadata = PluginMetadata(
    name="bow",
    display_name="Bag-of-Words",
    description="Basic bag-of-words feature extraction using Gensim dictionary",
    plugin_type=PluginType.FEATURE_EXTRACTOR,
    dependencies=["gensim>=4.3", "scipy>=1.8"],
    supports_streaming=True,
    performance_tier="fast",
    quality_tier="basic",
    memory_usage="low",
    default_params={},
)

registry.register(
    metadata=metadata,
    factory=lambda **kwargs: FeatureExtractor(**kwargs)
)
