from __future__ import annotations

import logging
import pickle
import time
import os
import json
import gzip
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from gensim import similarities

from .interfaces import FeatureExtractorProtocol, TokenizerProtocol
from .logging_utils import configure_logging

logger = logging.getLogger(__name__)


class DocumentIndex:
    """Similarity index backed by gensim.similarities.Similarity.

    Two-pass, streaming-friendly build that avoids materializing the whole corpus.
    """

    def __init__(self, doc2idx: Dict[str, int] | None = None) -> None:
        self.doc2idx: Dict[str, int] = doc2idx or {}
        # Use list for contiguous indices for better performance and lower memory
        self.idx2doc: List[str] = []
        if doc2idx:
            max_idx = max(doc2idx.values()) if doc2idx else -1
            self.idx2doc = [""] * (max_idx + 1)
            for _doc_id, _idx in doc2idx.items():
                self.idx2doc[_idx] = _doc_id
        self.index = None
        self.index_length = 0

    def _create_document_stream(
        self,
        documents: Dict[str, str] | None = None,
        document_stream_factory: Optional[Callable[[], Iterator[Tuple[str, str]]]] = None,
    ) -> Callable[[], Iterator[Tuple[str, str]]]:
        """Create a document stream factory from either documents dict or stream factory."""
        if documents is not None:
            logger.info("Building document index (in-memory documents: %d)", len(documents))
            def _make_stream() -> Iterator[Tuple[str, str]]:
                for _doc_id, _text in documents.items():
                    yield _doc_id, _text
            return _make_stream
        elif document_stream_factory is not None:
            logger.info("Building document index from streaming source")
            return document_stream_factory
        else:
            raise ValueError("Either 'documents' or 'document_stream_factory' must be provided")

    def _process_tokens_and_extract_features(
        self,
        make_stream: Callable[[], Iterator[Tuple[str, str]]],
        tokenizer: TokenizerProtocol,
        feature_extractor: FeatureExtractorProtocol,
        token_cache_path: Optional[str] = None,
    ) -> None:
        """Process tokens and extract features, optionally using token cache."""
        tokenize_start = time.time()
        logger.info("Tokenizing documents (pass 1) to build dictionary%s...",
                    " with token cache" if token_cache_path else "")

        if token_cache_path:
            parent_dir = os.path.dirname(token_cache_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            def _token_stream_and_cache() -> Iterator[List[str]]:
                """Stream and cache tokens with efficient compression.
                
                Uses a higher compression level for gzip and batched writes to improve
                compression ratio and reduce I/O overhead.
                """
                # Use higher compression level (9) for better compression
                def make_opener(path: str):
                    if path.endswith('.gz'):
                        return gzip.open(path, 'wt', encoding='utf-8', compresslevel=9)
                    return open(path, 'wt', encoding='utf-8')
                
                # Buffer for batched writes
                buffer = []
                buffer_size = 0
                max_buffer_size = 10 * 1024 * 1024  # 10MB buffer
                
                def flush_buffer(f):
                    if buffer:
                        f.write(''.join(buffer))
                        buffer.clear()
                        nonlocal buffer_size
                        buffer_size = 0
                
                with make_opener(token_cache_path) as f:
                    for doc_id, text in make_stream:  # make_stream is already a generator
                        tokens = tokenizer.tokenize(text)
                        # Use more compact JSON format
                        line = json.dumps([doc_id, tokens], separators=(',', ':')) + "\n"
                        buffer.append(line)
                        buffer_size += len(line.encode('utf-8'))
                        
                        # Flush buffer if it gets too large
                        if buffer_size >= max_buffer_size:
                            flush_buffer(f)
                            
                        yield tokens
                    
                    # Flush any remaining entries
                    flush_buffer(f)

            feature_extractor.fit_streaming(_token_stream_and_cache())
        else:
            feature_extractor.fit_streaming((tokenizer.tokenize(text) for _, text in make_stream))

        logger.info("Dictionary built from streaming tokens in %.2f seconds", time.time() - tokenize_start)

    def _create_bow_stream(
        self,
        make_stream: Callable[[], Iterator[Tuple[str, str]]],
        tokenizer: TokenizerProtocol,
        feature_extractor: FeatureExtractorProtocol,
        token_cache_path: Optional[str] = None,
    ) -> Iterator[List[Tuple[int, float]]]:
        """Create BOW stream from documents, handling token cache if provided."""
        self.doc2idx = {}
        self.idx2doc = []
        idx_local = 0

        if token_cache_path:
            # Use buffered reading for better performance
            def make_opener(path: str):
                if path.endswith('.gz'):
                    return gzip.open(path, 'rt', encoding='utf-8')
                return open(path, 'rt', encoding='utf-8')
            
            with make_opener(token_cache_path) as f:
                # Process tokens in batches to reduce Python overhead
                batch = []
                batch_size = 1000  # Process 1000 documents at a time
                
                for line in f:
                    doc_id, tokens = json.loads(line)
                    batch.append((doc_id, tokens))
                    
                    if len(batch) >= batch_size:
                        # Process batch
                        for b_doc_id, b_tokens in batch:
                            bow = feature_extractor.tokens_to_bow(b_tokens)
                            self.doc2idx[b_doc_id] = idx_local
                            self.idx2doc.append(b_doc_id)
                            idx_local += 1
                            yield bow
                        batch = []
                
                # Process remaining documents
                for b_doc_id, b_tokens in batch:
                    bow = feature_extractor.tokens_to_bow(b_tokens)
                    self.doc2idx[b_doc_id] = idx_local
                    self.idx2doc.append(b_doc_id)
                    idx_local += 1
                    yield bow
        else:
            for doc_id, text in make_stream:  # make_stream is already a generator
                tokens = tokenizer.tokenize(text)
                bow = feature_extractor.tokens_to_bow(tokens)
                self.doc2idx[doc_id] = idx_local
                self.idx2doc.append(doc_id)
                idx_local += 1
                yield bow

    def _build_similarity_index(
        self,
        bow_stream: Iterator[List[Tuple[int, float]]],
        feature_extractor: FeatureExtractorProtocol,
        index_path: str | None = None,
        similarity_chunksize: int = 1024,
    ) -> None:
        """Build the similarity index from BOW stream."""
        index_start = time.time()
        logger.info("Building similarity index from streaming corpus...")
        num_features = feature_extractor.num_features()
        # Convert index_path to string to handle pathlib.Path objects
        prefix = str(index_path) if index_path is not None else 'temp_index'
        self.index = similarities.Similarity(
            output_prefix=prefix,
            corpus=bow_stream,
            num_features=num_features,
            chunksize=similarity_chunksize,
        )
        logger.info("Similarity index building completed in %.2f seconds", time.time() - index_start)

    def build_index(
        self,
        *,
        documents: Dict[str, str] | None = None,
        feature_extractor: FeatureExtractorProtocol,
        tokenizer: TokenizerProtocol,
        index_path: str | None = None,
        document_stream_factory: Optional[Callable[[], Iterator[Tuple[str, str]]]] = None,
        token_cache_path: Optional[str] = None,
        similarity_chunksize: int = 1024,
    ) -> "DocumentIndex":
        """Build document index from documents or stream factory."""
        # Ensure logging is configured if the library is used programmatically
        configure_logging()
        # Convert to string to handle pathlib.Path objects
        if index_path is not None:
            index_path = str(index_path)
        if token_cache_path is not None:
            token_cache_path = str(token_cache_path)
        total_start_time = time.time()

        # Create document stream
        make_stream = self._create_document_stream(documents, document_stream_factory)

        # Process tokens and extract features
        self._process_tokens_and_extract_features(
            make_stream, tokenizer, feature_extractor, token_cache_path
        )

        # Create BOW stream and build similarity index
        bow_stream = self._create_bow_stream(
            make_stream, tokenizer, feature_extractor, token_cache_path
        )
        self._build_similarity_index(
            bow_stream, feature_extractor, index_path, similarity_chunksize
        )

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
        # Convert to string to handle pathlib.Path objects
        index_path = str(index_path)
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
        # Convert to string to handle pathlib.Path objects
        index_path = str(index_path)
        doc_index = cls()
        doc_index.index = similarities.Similarity.load(index_path)
        doc2idx_path = index_path + '.doc2idx'
        with open(doc2idx_path, 'rb') as f:
            doc_index.doc2idx = pickle.load(f)
        # Rebuild idx2doc list from mapping
        max_idx = max(doc_index.doc2idx.values()) if doc_index.doc2idx else -1
        doc_index.idx2doc = [""] * (max_idx + 1)
        for k, v in doc_index.doc2idx.items():
            doc_index.idx2doc[v] = k
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
            if 0 <= idx < len(self.idx2doc):
                results.append((self.idx2doc[idx], float(score)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def query_by_vector(self, query_vector: List[Tuple[int, float]], threshold: float | None = None) -> List[Tuple[str, float]]:
        similarities_scores = self.index[query_vector]
        if threshold is not None:
            similarities_scores[similarities_scores < threshold] = 0
        results: list[tuple[str, float]] = []
        for idx, score in enumerate(similarities_scores):
            if 0 <= idx < len(self.idx2doc):
                results.append((self.idx2doc[idx], float(score)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_docids(self) -> List[str]:
        return list(self.doc2idx.keys())

    def num_docs(self) -> int:
        return len(self.doc2idx)


