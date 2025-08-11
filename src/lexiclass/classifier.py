from __future__ import annotations

import logging
import os
import pickle
import time
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import preprocessing

from .encoding import BinaryClassEncoder, MultiClassEncoder
from .features import FeatureExtractor
from .index import DocumentIndex
from .tokenization import ICUTokenizer
from .interfaces import FeatureExtractorProtocol
from .config import get_settings

logger = logging.getLogger(__name__)


class SVMDocumentClassifier:
    """End-to-end SVM-based document classifier with retrieval index."""

    def __init__(self, tokenizer=None, feature_extractor: FeatureExtractorProtocol | None = None, document_index: DocumentIndex | None = None, is_multilabel: bool = False) -> None:
        settings = get_settings()
        self.tokenizer = tokenizer or ICUTokenizer(settings.default_locale)
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.document_index = document_index
        self.is_multilabel = is_multilabel
        self.classifier = None
        self.encoder = None
        self.is_fitted = False
        self.index_built = False

    def build_index(
        self,
        *,
        documents: Dict[str, str] | None = None,
        index_path: str | None = None,
        document_stream_factory: Optional[Callable[[], Iterator[Tuple[str, str]]]] = None,  # type: ignore[name-defined]
    ) -> "SVMDocumentClassifier":
        start_time = time.time()
        if documents is not None:
            logger.info("Building document index for %d documents", len(documents))
        else:
            logger.info("Building document index from streaming source")
        if self.document_index is None:
            self.document_index = DocumentIndex()
        self.document_index.build_index(
            documents=documents,
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            index_path=index_path,
            document_stream_factory=document_stream_factory,
        )
        self.index_built = True
        logger.info("Document index built successfully in %.2f seconds", time.time() - start_time)
        return self

    def load_index(self, index_path: str) -> "SVMDocumentClassifier":
        self.document_index = DocumentIndex.load_index(index_path)
        extractor_path = index_path + '.extractor'
        if os.path.exists(extractor_path):
            with open(extractor_path, 'rb') as f:
                self.feature_extractor = pickle.load(f)
        self.index_built = True
        logger.info("Document index loaded successfully")
        return self

    def train(self, labels: Dict[str, Union[str, List[str]]]) -> "SVMDocumentClassifier":
        if not self.index_built or self.document_index is None:
            raise ValueError("Document index must be built before training")
        start_time = time.time()
        logger.info("Training classifier on %d labeled documents", len(labels))
        doc_vectors: list[np.ndarray | sparse.csr_matrix] = []
        doc_labels: list[Union[str, List[str]]] = []
        for doc_id, label in labels.items():
            if doc_id in self.document_index.doc2idx:
                doc_idx = self.document_index.doc2idx[doc_id]
                vector = self.document_index.index.vector_by_id(doc_idx)
                if vector is not None:
                    doc_vectors.append(vector)
                    doc_labels.append(label)
        if not doc_vectors:
            raise ValueError("No labeled documents found in index")
        if hasattr(doc_vectors[0], 'toarray'):
            feature_matrix = sparse.vstack(doc_vectors, format='csr')
        else:
            feature_matrix = np.vstack(doc_vectors)

        self._setup_encoder_and_classifier(doc_labels)
        encoded_labels = self.encoder.encode(doc_labels)
        logger.info("Training SVM classifier...")
        self.classifier.fit(feature_matrix, encoded_labels)
        self.is_fitted = True
        logger.info("Training completed successfully on %d documents in %.2f seconds", len(doc_vectors), time.time() - start_time)
        return self

    def predict(self, documents: Dict[str, str]) -> Dict[str, Tuple[str, float]]:
        if not self.is_fitted:
            raise ValueError("Classifier must be trained before prediction")
        logger.info("Predicting labels for %d documents", len(documents))
        doc_ids = list(documents.keys())
        doc_texts = [documents[doc_id] for doc_id in doc_ids]
        tokenized_docs = [self.tokenizer.tokenize(text) for text in doc_texts]
        feature_matrix = self.feature_extractor.transform(tokenized_docs)
        scores = self.classifier.decision_function(feature_matrix)
        if hasattr(self.encoder, 'predict'):
            predictions = self.encoder.predict(scores)
            if scores.ndim == 1:
                result = {doc_ids[i]: (predictions[i], float(scores[i])) for i in range(len(doc_ids))}
            else:
                max_scores = np.max(scores, axis=1)
                result = {doc_ids[i]: (predictions[i], float(max_scores[i])) for i in range(len(doc_ids))}
        else:
            binary_predictions = (scores > 0).astype(int)
            predictions = self.encoder.decode(binary_predictions)
            max_scores = np.max(scores, axis=1)
            result = {doc_ids[i]: (predictions[i], float(max_scores[i])) for i in range(len(doc_ids))}
        return result

    def find_similar_documents(self, doc_id: str, threshold: float = 0.1, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.index_built or self.document_index is None:
            raise ValueError("Document index must be built before similarity search")
        results = self.document_index.query_by_id(doc_id, threshold=threshold)
        return results[:top_k]

    def find_similar_to_text(self, text: str, threshold: float = 0.1, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.index_built or self.document_index is None:
            raise ValueError("Document index must be built before similarity search")
        tokens = self.tokenizer.tokenize(text)
        bow = self.feature_extractor.tokens_to_bow(tokens)
        results = self.document_index.query_by_vector(bow, threshold=threshold)
        return results[:top_k]

    def _setup_encoder_and_classifier(self, labels: List[Union[str, List[str]]]) -> None:
        sample_label = labels[0]
        if isinstance(sample_label, list):
            self.is_multilabel = True
            all_classes: set[str] = set()
            for label_list in labels:
                all_classes.update(label_list)
            self.encoder = preprocessing.MultiLabelBinarizer()
            self.encoder.fit([list(all_classes)])
            seed = get_settings().random_seed
            self.classifier = OneVsRestClassifier(
                LinearSVC(loss='squared_hinge', penalty='l2', tol=1e-5, dual=False, random_state=seed)
            )
        else:
            unique_labels = list(set(labels))
            if len(unique_labels) == 2:
                pos_label, neg_label = unique_labels[0], unique_labels[1]
                self.encoder = BinaryClassEncoder(pos_label, neg_label)
                self.classifier = LinearSVC(loss='squared_hinge', penalty='l2', tol=1e-5, dual=False, random_state=get_settings().random_seed)
            else:
                self.encoder = MultiClassEncoder(unique_labels)
                self.classifier = LinearSVC(loss='squared_hinge', penalty='l2', tol=1e-5, dual=False, random_state=get_settings().random_seed)

    def save_model(self, filepath: str, index_path: str | None = None) -> None:
        if not self.is_fitted:
            raise ValueError("Cannot save model that hasn't been trained")
        tokenizer_type = 'icu'
        tokenizer_locale = getattr(self.tokenizer, 'locale', 'en')
        model_data = {
            'tokenizer_type': tokenizer_type,
            'tokenizer_locale': tokenizer_locale,
            'feature_extractor': self.feature_extractor,
            'classifier': self.classifier,
            'encoder': self.encoder,
            'is_multilabel': self.is_multilabel,
            'index_built': self.index_built,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=2)
        if self.index_built and self.document_index and index_path:
            self.document_index.save_index(index_path)
            extractor_path = index_path + '.extractor'
            with open(extractor_path, 'wb') as f:
                pickle.dump(self.feature_extractor, f, protocol=2)
        logger.info("Model saved to %s", filepath)

    @classmethod
    def load_model(cls, filepath: str, index_path: str | None = None) -> "SVMDocumentClassifier":
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        tok = None
        if model_data.get('tokenizer_type') == 'icu':
            try:
                tok = ICUTokenizer(model_data.get('tokenizer_locale', 'en'))
            except Exception:  # noqa: BLE001
                tok = None
        classifier = cls(tokenizer=tok, feature_extractor=model_data['feature_extractor'], is_multilabel=model_data['is_multilabel'])
        classifier.classifier = model_data['classifier']
        classifier.encoder = model_data['encoder']
        classifier.is_fitted = True
        classifier.index_built = model_data.get('index_built', False)
        if index_path and os.path.exists(index_path):
            classifier.load_index(index_path)
        logger.info("Model loaded from %s", filepath)
        return classifier


