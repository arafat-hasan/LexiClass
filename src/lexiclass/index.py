from __future__ import annotations

import logging
import pickle
import time
import os
import json
import gzip
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Any

from gensim import similarities

from .interfaces import FeatureExtractorProtocol, TokenizerProtocol
from .logging_utils import configure_logging

logger = logging.getLogger(__name__)


@dataclass
class IndexMetadata:
    """Metadata about how an index was built.

    Captures plugin names and parameters used to build the index,
    enabling reproducibility and validation.
    """

    # Plugin identifiers
    tokenizer_name: str
    feature_extractor_name: str

    # Plugin parameters used
    tokenizer_params: Dict[str, Any] = field(default_factory=dict)
    feature_extractor_params: Dict[str, Any] = field(default_factory=dict)

    # Build information
    index_version: str = "0.3.0"
    build_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    num_documents: int = 0
    num_features: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexMetadata":
        """Create IndexMetadata from dictionary."""
        return cls(**data)


class DocumentIndex:
    """Similarity index backed by gensim.similarities.Similarity.

    Two-pass, streaming-friendly build that avoids materializing the whole corpus.
    Stores tokenizer and feature extractor for reproducibility.
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

        # Store components and metadata
        self.tokenizer: Optional[TokenizerProtocol] = None
        self.feature_extractor: Optional[FeatureExtractorProtocol] = None
        self.metadata: Optional[IndexMetadata] = None

    def _create_document_stream(
        self,
        documents: Dict[str, str] | None = None,
        document_stream_factory: Optional[Callable[[], Iterator[Tuple[str, str]]]] = None,
    ) -> Callable[[], Iterator[Tuple[str, str]]]:
        """Create a document stream factory from either documents dict or stream factory.

        Ensures that the result is always a callable factory, not a generator instance.
        """
        if documents is not None:
            logger.info("Building document index (in-memory documents: %d)", len(documents))
            def _make_stream() -> Iterator[Tuple[str, str]]:
                for _doc_id, _text in documents.items():
                    yield _doc_id, _text
            return _make_stream
        elif document_stream_factory is not None:
            logger.info("Building document index from streaming source")
            # Ensure document_stream_factory is callable, not already a generator
            if not callable(document_stream_factory):
                raise TypeError("document_stream_factory must be a callable that returns an iterator, not an iterator itself")
            return document_stream_factory

        raise ValueError("Either 'documents' or 'document_stream_factory' must be provided")

    def _get_token_cache_path(self, index_path: Optional[str], auto_cache: bool) -> Optional[str]:
        """Generate token cache path if auto_cache is enabled and index_path is provided."""
        if auto_cache and index_path:
            return f"{index_path}.tokens.jsonl.gz"
        return None

    def _save_tokens_and_build_dict(
        self,
        make_stream: Callable[[], Iterator[Tuple[str, str]]],
        tokenizer: TokenizerProtocol,
        feature_extractor: FeatureExtractorProtocol,
        token_cache_path: str,
    ) -> None:
        """Tokenize documents, save to cache, and build dictionary."""
        tokenize_start = time.time()
        logger.info("Tokenizing documents and caching to %s...", token_cache_path)

        parent_dir = os.path.dirname(token_cache_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        def _token_stream_and_cache() -> Iterator[List[str]]:
            """Stream and cache tokens with efficient compression."""
            def make_opener(path: str):
                if path.endswith('.gz'):
                    return gzip.open(path, 'wt', encoding='utf-8', compresslevel=9)
                return open(path, 'wt', encoding='utf-8')

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
                for doc_id, text in make_stream():
                    tokens = tokenizer.tokenize(text)
                    line = json.dumps([doc_id, tokens], separators=(',', ':')) + "\n"
                    buffer.append(line)
                    buffer_size += len(line.encode('utf-8'))

                    if buffer_size >= max_buffer_size:
                        flush_buffer(f)

                    yield tokens

                flush_buffer(f)

        feature_extractor.fit_streaming(_token_stream_and_cache())
        logger.info("Tokens cached and dictionary built in %.2f seconds", time.time() - tokenize_start)

    def _build_dict_from_stream(
        self,
        make_stream: Callable[[], Iterator[Tuple[str, str]]],
        tokenizer: TokenizerProtocol,
        feature_extractor: FeatureExtractorProtocol,
    ) -> None:
        """Build dictionary directly from document stream without caching."""
        tokenize_start = time.time()
        logger.info("Tokenizing documents to build dictionary...")
        feature_extractor.fit_streaming((tokenizer.tokenize(text) for _, text in make_stream()))
        logger.info("Dictionary built in %.2f seconds", time.time() - tokenize_start)

    def _process_tokens_and_extract_features(
        self,
        make_stream: Callable[[], Iterator[Tuple[str, str]]],
        tokenizer: TokenizerProtocol,
        feature_extractor: FeatureExtractorProtocol,
        token_cache_path: Optional[str],
    ) -> None:
        """Process tokens and extract features, with optional token caching."""
        if token_cache_path:
            self._save_tokens_and_build_dict(make_stream, tokenizer, feature_extractor, token_cache_path)
        else:
            self._build_dict_from_stream(make_stream, tokenizer, feature_extractor)

    def _create_bow_stream_from_cache(
        self,
        token_cache_path: str,
        feature_extractor: FeatureExtractorProtocol,
    ) -> Iterator[List[Tuple[int, float]]]:
        """Create BOW stream from cached tokens."""
        logger.info("Creating BOW stream from token cache...")
        self.doc2idx = {}
        self.idx2doc = []
        idx_local = 0

        def make_opener(path: str):
            if path.endswith('.gz'):
                return gzip.open(path, 'rt', encoding='utf-8')
            return open(path, 'rt', encoding='utf-8')

        with make_opener(token_cache_path) as f:
            batch = []
            batch_size = 1000

            for line in f:
                doc_id, tokens = json.loads(line)
                batch.append((doc_id, tokens))

                if len(batch) >= batch_size:
                    for b_doc_id, b_tokens in batch:
                        bow = feature_extractor.tokens_to_bow(b_tokens)
                        self.doc2idx[b_doc_id] = idx_local
                        self.idx2doc.append(b_doc_id)
                        idx_local += 1
                        yield bow
                    batch = []

            for b_doc_id, b_tokens in batch:
                bow = feature_extractor.tokens_to_bow(b_tokens)
                self.doc2idx[b_doc_id] = idx_local
                self.idx2doc.append(b_doc_id)
                idx_local += 1
                yield bow

    def _create_bow_stream_from_documents(
        self,
        make_stream: Callable[[], Iterator[Tuple[str, str]]],
        tokenizer: TokenizerProtocol,
        feature_extractor: FeatureExtractorProtocol,
    ) -> Iterator[List[Tuple[int, float]]]:
        """Create BOW stream by tokenizing documents on the fly."""
        logger.info("Creating BOW stream from documents (re-tokenizing)...")
        self.doc2idx = {}
        self.idx2doc = []
        idx_local = 0

        for doc_id, text in make_stream():
            tokens = tokenizer.tokenize(text)
            bow = feature_extractor.tokens_to_bow(tokens)
            self.doc2idx[doc_id] = idx_local
            self.idx2doc.append(doc_id)
            idx_local += 1
            yield bow

    def _create_bow_stream(
        self,
        make_stream: Callable[[], Iterator[Tuple[str, str]]],
        tokenizer: TokenizerProtocol,
        feature_extractor: FeatureExtractorProtocol,
        token_cache_path: Optional[str],
    ) -> Iterator[List[Tuple[int, float]]]:
        """Create BOW stream from documents, using token cache if available."""
        if token_cache_path and os.path.exists(token_cache_path):
            return self._create_bow_stream_from_cache(token_cache_path, feature_extractor)
        else:
            return self._create_bow_stream_from_documents(make_stream, tokenizer, feature_extractor)

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
        auto_cache_tokens: bool = False,
        similarity_chunksize: int = 1024,
        tokenizer_name: str = "unknown",
        feature_extractor_name: str = "unknown",
        tokenizer_params: Optional[Dict[str, Any]] = None,
        feature_extractor_params: Optional[Dict[str, Any]] = None,
    ) -> "DocumentIndex":
        """Build document index from documents or stream factory.

        Args:
            documents: Dictionary of doc_id -> text (for in-memory corpora)
            feature_extractor: Feature extractor to build dictionary and transform documents
            tokenizer: Tokenizer to convert text to tokens
            index_path: Path prefix for saving index artifacts
            document_stream_factory: Factory function that returns an iterator of (doc_id, text) tuples
            token_cache_path: Explicit path to save/load tokenized documents
            auto_cache_tokens: If True and index_path is provided, automatically cache tokens
                              to {index_path}.tokens.jsonl.gz to avoid re-tokenization
            similarity_chunksize: Chunk size for Gensim Similarity index
            tokenizer_name: Name of tokenizer plugin used (for metadata)
            feature_extractor_name: Name of feature extractor plugin used (for metadata)
            tokenizer_params: Parameters used to create tokenizer (for metadata)
            feature_extractor_params: Parameters used to create feature extractor (for metadata)

        Returns:
            Self for method chaining

        Note:
            - If token_cache_path is provided, it takes precedence over auto_cache_tokens
            - Token caching avoids tokenizing documents twice (once for dict building, once for indexing)
            - For library use, set auto_cache_tokens=True for better performance on large corpora
        """
        # Ensure logging is configured if the library is used programmatically
        configure_logging()

        # Convert to string to handle pathlib.Path objects
        if index_path is not None:
            index_path = str(index_path)
        if token_cache_path is not None:
            token_cache_path = str(token_cache_path)

        # Determine effective token cache path
        effective_cache_path = token_cache_path or self._get_token_cache_path(index_path, auto_cache_tokens)

        total_start_time = time.time()

        # Create document stream factory
        make_stream = self._create_document_stream(documents, document_stream_factory)

        # Phase 1: Tokenize and build dictionary (with optional caching)
        self._process_tokens_and_extract_features(
            make_stream, tokenizer, feature_extractor, effective_cache_path
        )

        # Phase 2: Create BOW stream and build similarity index
        bow_stream = self._create_bow_stream(
            make_stream, tokenizer, feature_extractor, effective_cache_path
        )
        self._build_similarity_index(
            bow_stream, feature_extractor, index_path, similarity_chunksize
        )

        self.index_length = len(self.doc2idx)

        # Store components in the index instance
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

        # Create metadata
        from . import __version__
        self.metadata = IndexMetadata(
            tokenizer_name=tokenizer_name,
            feature_extractor_name=feature_extractor_name,
            tokenizer_params=tokenizer_params or {},
            feature_extractor_params=feature_extractor_params or {},
            index_version=__version__,
            build_timestamp=datetime.now().isoformat(),
            num_documents=self.index_length,
            num_features=feature_extractor.num_features(),
        )

        if index_path:
            self.save_index(index_path)

        logger.info(
            "Document index built successfully with %d documents in %.2f seconds total",
            self.index_length,
            time.time() - total_start_time,
        )
        return self

    def save_index(self, index_path: str) -> None:
        """Save index with all components (similarity index, doc2idx, tokenizer, feature_extractor, metadata).

        Args:
            index_path: Path prefix for saving index artifacts

        Saves the following files:
            - {index_path}: Gensim Similarity index (Similarity.* shards)
            - {index_path}.doc2idx: Pickled doc_id → index mapping
            - {index_path}.tokenizer: Pickled tokenizer
            - {index_path}.extractor: Pickled feature extractor
            - {index_path}.metadata.json: JSON metadata about the index
        """
        if self.index is None:
            raise ValueError("No index to save")

        # Convert to string to handle pathlib.Path objects
        index_path = str(index_path)

        # Ensure the parent directory exists when using a nested prefix
        parent_dir = os.path.dirname(index_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Save similarity index
        self.index.save(index_path)

        # Save doc2idx mapping
        doc2idx_path = index_path + '.doc2idx'
        with open(doc2idx_path, 'wb') as f:
            pickle.dump(self.doc2idx, f, protocol=2)

        # Save tokenizer
        if self.tokenizer is not None:
            tokenizer_path = index_path + '.tokenizer'
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(self.tokenizer, f, protocol=2)
            logger.debug("Tokenizer saved to %s", tokenizer_path)

        # Save feature extractor
        if self.feature_extractor is not None:
            extractor_path = index_path + '.extractor'
            with open(extractor_path, 'wb') as f:
                pickle.dump(self.feature_extractor, f, protocol=2)
            logger.debug("Feature extractor saved to %s", extractor_path)

        # Save metadata
        if self.metadata is not None:
            metadata_path = index_path + '.metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata.to_dict(), f, indent=2)
            logger.debug("Metadata saved to %s", metadata_path)

        logger.info("Index and all components saved to %s", index_path)

    @classmethod
    def load_index(cls, index_path: str, load_tokenizer: bool = True, load_feature_extractor: bool = True) -> "DocumentIndex":
        """Load index with all components.

        Args:
            index_path: Path prefix for loading index artifacts
            load_tokenizer: Whether to load the tokenizer (default: True)
            load_feature_extractor: Whether to load the feature extractor (default: True)

        Returns:
            DocumentIndex with loaded components

        Note:
            If tokenizer or feature extractor files don't exist, they will be set to None
            with a warning logged. The metadata will still be loaded if available.
        """
        # Convert to string to handle pathlib.Path objects
        index_path = str(index_path)
        doc_index = cls()

        # Load similarity index
        doc_index.index = similarities.Similarity.load(index_path)

        # Load doc2idx mapping
        doc2idx_path = index_path + '.doc2idx'
        with open(doc2idx_path, 'rb') as f:
            doc_index.doc2idx = pickle.load(f)

        # Rebuild idx2doc list from mapping
        max_idx = max(doc_index.doc2idx.values()) if doc_index.doc2idx else -1
        doc_index.idx2doc = [""] * (max_idx + 1)
        for k, v in doc_index.doc2idx.items():
            doc_index.idx2doc[v] = k
        doc_index.index_length = len(doc_index.doc2idx)

        # Load tokenizer
        tokenizer_path = index_path + '.tokenizer'
        if load_tokenizer and os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as f:
                doc_index.tokenizer = pickle.load(f)
            logger.debug("Tokenizer loaded from %s", tokenizer_path)
        elif load_tokenizer:
            logger.warning("Tokenizer file not found at %s (index may be from older version)", tokenizer_path)

        # Load feature extractor
        extractor_path = index_path + '.extractor'
        if load_feature_extractor and os.path.exists(extractor_path):
            with open(extractor_path, 'rb') as f:
                doc_index.feature_extractor = pickle.load(f)
            logger.debug("Feature extractor loaded from %s", extractor_path)
        elif load_feature_extractor:
            logger.warning("Feature extractor file not found at %s", extractor_path)

        # Load metadata
        metadata_path = index_path + '.metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
                doc_index.metadata = IndexMetadata.from_dict(metadata_dict)
            logger.debug("Metadata loaded from %s", metadata_path)
            logger.info(
                "Index loaded from %s with %d documents (tokenizer: %s, features: %s)",
                index_path,
                doc_index.index_length,
                doc_index.metadata.tokenizer_name,
                doc_index.metadata.feature_extractor_name,
            )
        else:
            logger.warning("Metadata file not found at %s (index may be from older version)", metadata_path)
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


