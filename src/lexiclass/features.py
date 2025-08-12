from __future__ import annotations

import logging
import time
from typing import Iterable, List

from gensim import corpora
from scipy import sparse

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

    def __init__(self) -> None:
        self.dictionary: corpora.Dictionary | None = None
        self.fitted: bool = False

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

    def transform(self, documents: List[List[str]]) -> sparse.csr_matrix:
        if not self.fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")

        start_time = time.time()
        logger.info("Transforming %d documents to feature vectors", len(documents))

        bow_start = time.time()
        bow_corpus = [self.dictionary.doc2bow(doc) for doc in documents]  # type: ignore[arg-type]
        logger.info("Bag-of-words conversion completed in %.2f seconds", time.time() - bow_start)

        sparse_start = time.time()
        num_docs = len(bow_corpus)
        num_features = len(self.dictionary)  # type: ignore[arg-type]

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for doc_idx, bow in enumerate(bow_corpus):
            for token_id, count in bow:
                rows.append(doc_idx)
                cols.append(token_id)
                data.append(count)

        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(num_docs, num_features))
        logger.info("Sparse matrix creation completed in %.2f seconds", time.time() - sparse_start)

        logger.info("Document transformation completed in %.2f seconds total", time.time() - start_time)
        return matrix

    def fit_transform(self, documents: List[List[str]]) -> sparse.csr_matrix:
        return self.fit(documents).transform(documents)

    def tokens_to_bow(self, tokens: List[str]) -> List[tuple[int, float]]:
        if not self.dictionary:
            raise ValueError("FeatureExtractor must be fitted before tokens_to_bow")
        return self.dictionary.doc2bow(tokens)

    def _filter_dictionary(self) -> None:
        assert self.dictionary is not None
        self.dictionary.filter_extremes(
            no_below=MIN_DOCS_PER_TOKEN,
            no_above=MAX_DOCS_PCT_PER_TOKEN,
            keep_n=None,
        )

        garbage_tokens: list[str] = []
        for token in self.dictionary.token2id:
            if self._is_garbage_token(token):
                garbage_tokens.append(token)
        if garbage_tokens:
            garbage_ids = [self.dictionary.token2id[token] for token in garbage_tokens]
            self.dictionary.filter_tokens(garbage_ids)

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
        return len(self.dictionary) if self.dictionary else 0


