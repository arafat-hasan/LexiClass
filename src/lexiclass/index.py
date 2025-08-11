from __future__ import annotations

import logging
import pickle
import time
import os
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from gensim import similarities

from .features import FeatureExtractor
from .interfaces import FeatureExtractorProtocol, TokenizerProtocol
from .logging_utils import configure_logging

logger = logging.getLogger(__name__)


class DocumentIndex:
    """Similarity index backed by gensim.similarities.Similarity.

    Two-pass, streaming-friendly build that avoids materializing the whole corpus.
    """

    def __init__(self, doc2idx: Dict[str, int] | None = None) -> None:
        self.doc2idx: Dict[str, int] = doc2idx or {}
        self.idx2doc: Dict[int, str] = {v: k for k, v in self.doc2idx.items()} if doc2idx else {}
        self.index = None
        self.index_length = 0

    def build_index(
        self,
        *,
        documents: Dict[str, str] | None = None,
        feature_extractor: FeatureExtractorProtocol,
        tokenizer: TokenizerProtocol,
        index_path: str | None = None,
        document_stream_factory: Optional[Callable[[], Iterator[Tuple[str, str]]]] = None,
    ) -> "DocumentIndex":
        # Ensure logging is configured if the library is used programmatically
        configure_logging()
        total_start_time = time.time()
        if documents is not None:
            logger.info("Building document index (in-memory documents: %d)", len(documents))

            def _make_stream() -> Iterator[Tuple[str, str]]:
                for _doc_id, _text in documents.items():
                    yield _doc_id, _text

            make_stream = _make_stream
        elif document_stream_factory is not None:
            logger.info("Building document index from streaming source")
            make_stream = document_stream_factory
        else:
            raise ValueError("Either 'documents' or 'document_stream_factory' must be provided")

        tokenize_start = time.time()
        logger.info("Tokenizing documents (pass 1) to build dictionary...")
        feature_extractor.fit_streaming((tokenizer.tokenize(text) for _, text in make_stream()))
        logger.info("Dictionary built from streaming tokens in %.2f seconds", time.time() - tokenize_start)

        def _bow_stream() -> Iterator[List[Tuple[int, float]]]:
            self.doc2idx = {}
            self.idx2doc = {}
            idx_local = 0
            for doc_id, text in make_stream():
                tokens = tokenizer.tokenize(text)
                bow = feature_extractor.tokens_to_bow(tokens)
                self.doc2idx[doc_id] = idx_local
                self.idx2doc[idx_local] = doc_id
                idx_local += 1
                yield bow

        index_start = time.time()
        logger.info("Building similarity index from streaming corpus...")
        num_features = feature_extractor.num_features()
        self.index = similarities.Similarity(
            output_prefix=index_path or 'temp_index',
            corpus=_bow_stream(),
            num_features=num_features,
        )
        logger.info("Similarity index building completed in %.2f seconds", time.time() - index_start)

        self.index_length = len(self.doc2idx)

        if index_path:
            self.save_index(index_path)

        logger.info(
            "Document index built successfully with %d documents in %.2f seconds total",
            self.index_length,
            time.time() - total_start_time,
        )
        return self

    def save_index(self, index_path: str) -> None:
        if self.index is None:
            raise ValueError("No index to save")
        # Ensure the parent directory exists when using a nested prefix
        parent_dir = os.path.dirname(index_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self.index.save(index_path)
        doc2idx_path = index_path + '.doc2idx'
        with open(doc2idx_path, 'wb') as f:
            pickle.dump(self.doc2idx, f, protocol=2)
        logger.info("Index saved to %s", index_path)

    @classmethod
    def load_index(cls, index_path: str) -> "DocumentIndex":
        doc_index = cls()
        doc_index.index = similarities.Similarity.load(index_path)
        doc2idx_path = index_path + '.doc2idx'
        with open(doc2idx_path, 'rb') as f:
            doc_index.doc2idx = pickle.load(f)
        doc_index.idx2doc = {v: k for k, v in doc_index.doc2idx.items()}
        doc_index.index_length = len(doc_index.doc2idx)
        logger.info("Index loaded from %s with %d documents", index_path, doc_index.index_length)
        return doc_index

    def query_by_id(self, doc_id: str, threshold: float | None = None) -> List[Tuple[str, float]]:
        if doc_id not in self.doc2idx:
            raise ValueError(f"Document {doc_id} not in index")
        doc_idx = self.doc2idx[doc_id]
        similarities_scores = self.index.similarity_by_id(doc_idx)
        if threshold is not None:
            similarities_scores[similarities_scores < threshold] = 0
        results: list[tuple[str, float]] = []
        for idx, score in enumerate(similarities_scores):
            if idx in self.idx2doc:
                results.append((self.idx2doc[idx], float(score)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def query_by_vector(self, query_vector: List[Tuple[int, float]], threshold: float | None = None) -> List[Tuple[str, float]]:
        similarities_scores = self.index[query_vector]
        if threshold is not None:
            similarities_scores[similarities_scores < threshold] = 0
        results: list[tuple[str, float]] = []
        for idx, score in enumerate(similarities_scores):
            if idx in self.idx2doc:
                results.append((self.idx2doc[idx], float(score)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_docids(self) -> List[str]:
        return list(self.doc2idx.keys())

    def num_docs(self) -> int:
        return len(self.doc2idx)


