from __future__ import annotations

from typing import Callable, Iterable, Iterator, List, Optional, Protocol, Tuple, Dict, Union

from scipy import sparse  # type: ignore


class TokenizerProtocol(Protocol):
    def tokenize(self, text: str) -> List[str]:
        ...

    def save(self, path: str) -> None:
        """Save tokenizer to disk."""
        ...

    @classmethod
    def load(cls, path: str) -> "TokenizerProtocol":
        """Load tokenizer from disk."""
        ...


class FeatureExtractorProtocol(Protocol):
    def fit(self, documents: List[List[str]]) -> "FeatureExtractorProtocol":
        ...

    def fit_streaming(self, tokenized_documents_iter: Iterable[List[str]]) -> "FeatureExtractorProtocol":
        ...

    def transform(self, documents: List[List[str]]) -> sparse.csr_matrix:
        ...

    def num_features(self) -> int:
        ...

    def tokens_to_bow(self, tokens: List[str]) -> List[Tuple[int, float]]:
        ...

    def save(self, path: str) -> None:
        """Save feature extractor to disk."""
        ...

    @classmethod
    def load(cls, path: str) -> "FeatureExtractorProtocol":
        """Load feature extractor from disk."""
        ...


class DocumentIndexProtocol(Protocol):
    def build_index(
        self,
        *,
        documents: Dict[str, str] | None,
        feature_extractor: FeatureExtractorProtocol,
        tokenizer: TokenizerProtocol,
        index_path: Optional[str],
        document_stream_factory: Optional[Callable[[], Iterator[Tuple[str, str]]]],
    ) -> "DocumentIndexProtocol":
        ...

    def query_by_id(self, doc_id: str, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        ...

    def query_by_vector(self, query_vector: List[Tuple[int, float]], threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        ...


class ClassifierProtocol(Protocol):
    def train(self, labels: Dict[str, Union[str, List[str]]]) -> "ClassifierProtocol":
        ...

    def predict(self, documents: Dict[str, str]):
        ...

    def save(self, path: str) -> None:
        """Save classifier to disk."""
        ...

    @classmethod
    def load(cls, path: str) -> "ClassifierProtocol":
        """Load classifier from disk."""
        ...


