"""High-level document classifier API (Facade pattern).

This module provides a simple, user-friendly interface for document classification
that hides the complexity of orchestrating tokenizers, feature extractors, indexes,
and classifier plugins.
"""

from __future__ import annotations

import logging
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from scipy import sparse

from .index import DocumentIndex
from .interfaces import TokenizerProtocol, FeatureExtractorProtocol
from .plugins import registry, PluginType
from .io import DocumentLoader, load_labels
from .evaluation import evaluate_predictions

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """High-level API for document classification workflow.

    This class provides a Facade over the complex orchestration of:
    - Document indexing
    - Tokenization
    - Feature extraction
    - Classification

    Example (Library Usage):
        >>> # Create from plugins
        >>> classifier = DocumentClassifier.from_plugins('icu', 'tfidf', 'svm', locale='en')
        >>>
        >>> # Build index and train
        >>> classifier.build_index(documents, './my_index')
        >>> classifier.train(labels)
        >>>
        >>> # Save and load
        >>> classifier.save('./model.pkl')
        >>> loaded = DocumentClassifier.load('./model.pkl')
        >>>
        >>> # Predict
        >>> predictions = loaded.predict(test_documents)

    Example (With Existing Index):
        >>> # Load existing index
        >>> classifier = DocumentClassifier.load_index('./my_index')
        >>> classifier.set_classifier('svm')
        >>> classifier.train(labels)
        >>> predictions = classifier.predict(test_documents)
    """

    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        feature_extractor: FeatureExtractorProtocol,
        classifier_plugin: Any = None,
        index: Optional[DocumentIndex] = None,
        tokenizer_name: str = "unknown",
        feature_extractor_name: str = "unknown",
        classifier_name: str = "unknown",
        tokenizer_params: Optional[Dict[str, Any]] = None,
        feature_extractor_params: Optional[Dict[str, Any]] = None,
        classifier_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize DocumentClassifier.

        Args:
            tokenizer: Tokenizer instance
            feature_extractor: Feature extractor instance
            classifier_plugin: Classifier plugin instance (optional, can be set later)
            index: DocumentIndex instance (optional, can be built later)
            tokenizer_name: Name of tokenizer plugin (for metadata)
            feature_extractor_name: Name of feature extractor plugin
            classifier_name: Name of classifier plugin
            tokenizer_params: Parameters used to create tokenizer
            feature_extractor_params: Parameters used to create feature extractor
            classifier_params: Parameters used to create classifier
        """
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.classifier = classifier_plugin
        self.index = index

        # Store metadata for reproducibility
        self.tokenizer_name = tokenizer_name
        self.feature_extractor_name = feature_extractor_name
        self.classifier_name = classifier_name
        self.tokenizer_params = tokenizer_params or {}
        self.feature_extractor_params = feature_extractor_params or {}
        self.classifier_params = classifier_params or {}

        self.is_trained = False

    @classmethod
    def from_plugins(
        cls,
        tokenizer_name: str,
        feature_name: str,
        classifier_name: str,
        **kwargs
    ) -> "DocumentClassifier":
        """Create classifier from plugin names.

        Args:
            tokenizer_name: Name of tokenizer plugin (e.g., 'icu', 'spacy')
            feature_name: Name of feature extractor plugin (e.g., 'bow', 'tfidf')
            classifier_name: Name of classifier plugin (e.g., 'svm', 'xgboost')
            **kwargs: Parameters for plugins (use prefix: tokenizer_*, feature_*, classifier_*)

        Returns:
            DocumentClassifier instance

        Example:
            >>> clf = DocumentClassifier.from_plugins(
            ...     'icu', 'tfidf', 'svm',
            ...     tokenizer_locale='en',
            ...     feature_normalize=True,
            ...     classifier_C=1.0
            ... )
        """
        # Separate kwargs by prefix
        tokenizer_kwargs = {k.replace('tokenizer_', ''): v for k, v in kwargs.items() if k.startswith('tokenizer_')}
        feature_kwargs = {k.replace('feature_', ''): v for k, v in kwargs.items() if k.startswith('feature_')}
        classifier_kwargs = {k.replace('classifier_', ''): v for k, v in kwargs.items() if k.startswith('classifier_')}

        # Create plugin instances
        tokenizer = registry.create(tokenizer_name, plugin_type=PluginType.TOKENIZER, **tokenizer_kwargs)
        feature_extractor = registry.create(feature_name, plugin_type=PluginType.FEATURE_EXTRACTOR, **feature_kwargs)
        classifier_plugin = registry.create(classifier_name, plugin_type=PluginType.CLASSIFIER, **classifier_kwargs)

        return cls(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            classifier_plugin=classifier_plugin,
            tokenizer_name=tokenizer_name,
            feature_extractor_name=feature_name,
            classifier_name=classifier_name,
            tokenizer_params=tokenizer_kwargs,
            feature_extractor_params=feature_kwargs,
            classifier_params=classifier_kwargs,
        )

    @classmethod
    def load_index(cls, index_path: str, classifier_name: Optional[str] = None, **classifier_kwargs) -> "DocumentClassifier":
        """Create classifier from existing index.

        Args:
            index_path: Path to saved index
            classifier_name: Optional classifier plugin name
            **classifier_kwargs: Optional classifier parameters

        Returns:
            DocumentClassifier with loaded index and components

        Example:
            >>> clf = DocumentClassifier.load_index('./my_index', classifier_name='svm', C=1.0)
            >>> clf.train(labels)
        """
        # Load index with all components
        index = DocumentIndex.load_index(index_path)

        if index.tokenizer is None or index.feature_extractor is None:
            raise ValueError(
                f"Index at {index_path} does not contain tokenizer/feature extractor. "
                "This may be an old index format. Please rebuild the index."
            )

        # Get metadata
        tokenizer_name = index.metadata.tokenizer_name if index.metadata else "unknown"
        feature_name = index.metadata.feature_extractor_name if index.metadata else "unknown"

        # Create classifier if specified
        classifier_plugin = None
        if classifier_name:
            classifier_plugin = registry.create(classifier_name, plugin_type=PluginType.CLASSIFIER, **classifier_kwargs)

        return cls(
            tokenizer=index.tokenizer,
            feature_extractor=index.feature_extractor,
            classifier_plugin=classifier_plugin,
            index=index,
            tokenizer_name=tokenizer_name,
            feature_extractor_name=feature_name,
            classifier_name=classifier_name or "unknown",
            classifier_params=classifier_kwargs,
        )

    def set_classifier(self, classifier_name: str, **kwargs) -> "DocumentClassifier":
        """Set or change the classifier plugin.

        Args:
            classifier_name: Name of classifier plugin
            **kwargs: Classifier parameters

        Returns:
            Self for chaining
        """
        self.classifier = registry.create(classifier_name, plugin_type=PluginType.CLASSIFIER, **kwargs)
        self.classifier_name = classifier_name
        self.classifier_params = kwargs
        self.is_trained = False
        return self

    def build_index(
        self,
        documents: Union[Dict[str, str], str, Path],
        index_path: Optional[str] = None,
        auto_cache_tokens: bool = True,
    ) -> "DocumentClassifier":
        """Build document index.

        Args:
            documents: Either dict of {doc_id: text} or path to directory of .txt files
            index_path: Optional path to save index (if None, index kept in memory)
            auto_cache_tokens: Whether to cache tokens to avoid re-tokenization

        Returns:
            Self for chaining
        """
        # Handle directory path
        if isinstance(documents, (str, Path)):
            documents_dict = DocumentLoader.load_documents_from_directory(str(documents))

            def doc_stream_factory():
                return DocumentLoader.iter_documents_from_directory(str(documents))

            # Use streaming for directory
            self.index = DocumentIndex()
            self.index.build_index(
                feature_extractor=self.feature_extractor,
                tokenizer=self.tokenizer,
                index_path=index_path,
                document_stream_factory=doc_stream_factory,
                auto_cache_tokens=auto_cache_tokens,
                tokenizer_name=self.tokenizer_name,
                feature_extractor_name=self.feature_extractor_name,
                tokenizer_params=self.tokenizer_params,
                feature_extractor_params=self.feature_extractor_params,
            )
        else:
            # Use in-memory for dict
            self.index = DocumentIndex()
            self.index.build_index(
                documents=documents,
                feature_extractor=self.feature_extractor,
                tokenizer=self.tokenizer,
                index_path=index_path,
                auto_cache_tokens=auto_cache_tokens,
                tokenizer_name=self.tokenizer_name,
                feature_extractor_name=self.feature_extractor_name,
                tokenizer_params=self.tokenizer_params,
                feature_extractor_params=self.feature_extractor_params,
            )

        return self

    def train(self, labels: Union[Dict[str, Union[str, List[str]]], str, Path]) -> "DocumentClassifier":
        """Train the classifier.

        Args:
            labels: Either dict of {doc_id: label(s)} or path to labels TSV file

        Returns:
            Self for chaining

        Raises:
            ValueError: If index not built or classifier not set
        """
        if self.index is None:
            raise ValueError("Index must be built before training. Call build_index() first.")

        if self.classifier is None:
            raise ValueError("Classifier not set. Call set_classifier() or use from_plugins().")

        # Load labels if path provided
        if isinstance(labels, (str, Path)):
            labels = load_labels(str(labels))

        logger.info("Training classifier on %d labeled documents", len(labels))

        # Extract feature vectors for labeled documents
        feature_matrix, valid_labels = self._extract_features_for_labels(labels)

        if feature_matrix.shape[0] == 0:
            raise ValueError("No documents found in index matching the labels")

        # Train classifier
        self.classifier.train(feature_matrix, valid_labels)
        self.is_trained = True

        logger.info("Classifier training completed successfully")
        return self

    def predict(
        self,
        documents: Union[Dict[str, str], str, Path, List[str]]
    ) -> Dict[str, Tuple[Union[str, List[str]], float]]:
        """Predict labels for documents.

        Args:
            documents: Documents to predict on. Can be:
                - Dict of {doc_id: text}
                - Path to directory of .txt files
                - List of doc_ids (must be in index)

        Returns:
            Dict of {doc_id: (predicted_label, confidence_score)}

        Raises:
            ValueError: If classifier not trained
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction. Call train() first.")

        # Handle different input types
        if isinstance(documents, list):
            # List of doc_ids - extract from index
            doc_ids = documents
            feature_matrix = self._extract_features_for_doc_ids(doc_ids)
        elif isinstance(documents, (str, Path)):
            # Directory path
            documents = DocumentLoader.load_documents_from_directory(str(documents))
            doc_ids = list(documents.keys())
            feature_matrix = self._extract_features_for_documents(documents)
        else:
            # Dict of documents
            doc_ids = list(documents.keys())
            feature_matrix = self._extract_features_for_documents(documents)

        if feature_matrix.shape[0] == 0:
            logger.warning("No valid documents found for prediction")
            return {}

        # Predict
        predictions, scores = self.classifier.predict(feature_matrix)

        # Create results dict
        results = {}
        for i, doc_id in enumerate(doc_ids):
            if i < len(predictions):
                results[doc_id] = (predictions[i], float(scores[i]))

        logger.info("Predicted labels for %d documents", len(results))
        return results

    def evaluate(
        self,
        documents: Union[Dict[str, str], str, Path],
        ground_truth: Union[Dict[str, str], str, Path]
    ) -> Dict[str, Any]:
        """Predict and evaluate against ground truth.

        Args:
            documents: Documents to predict on
            ground_truth: Ground truth labels (dict or path to TSV)

        Returns:
            Dict of evaluation metrics (accuracy, precision, recall, f1, etc.)
        """
        # Get predictions
        predictions = self.predict(documents)

        # Load ground truth if path
        if isinstance(ground_truth, (str, Path)):
            ground_truth = load_labels(str(ground_truth))

        # Convert predictions to format expected by evaluate
        pred_dict = {doc_id: label for doc_id, (label, score) in predictions.items()}

        # Evaluate
        metrics = evaluate_predictions(pred_dict, ground_truth)

        logger.info("Evaluation completed - Accuracy: %.4f", metrics['accuracy'])
        return metrics

    def save(self, model_path: str, index_path: Optional[str] = None) -> None:
        """Save trained model.

        Args:
            model_path: Path to save model pickle
            index_path: Optional path to save index (if not already saved)
        """
        if not self.is_trained:
            logger.warning("Saving untrained classifier")

        # Save index if path provided and index exists
        if index_path and self.index:
            self.index.save_index(index_path)

        # Create model bundle
        model_data = {
            'classifier': self.classifier,
            'classifier_name': self.classifier_name,
            'classifier_params': self.classifier_params,
            'tokenizer_name': self.tokenizer_name,
            'feature_extractor_name': self.feature_extractor_name,
            'tokenizer_params': self.tokenizer_params,
            'feature_extractor_params': self.feature_extractor_params,
            'is_trained': self.is_trained,
            'index_path': index_path or getattr(self.index, 'index_path', None) if self.index else None,
        }

        # Save model
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f, protocol=2)

        logger.info("Model saved to %s", model_path)

    @classmethod
    def load(cls, model_path: str, index_path: Optional[str] = None) -> "DocumentClassifier":
        """Load saved model.

        Args:
            model_path: Path to model pickle
            index_path: Optional override for index path

        Returns:
            DocumentClassifier instance
        """
        # Load model data
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Determine index path
        index_path = index_path or model_data.get('index_path')

        if not index_path:
            raise ValueError("Index path not found in model. Specify index_path parameter.")

        # Load index
        index = DocumentIndex.load_index(index_path)

        # Create classifier instance
        classifier_instance = cls(
            tokenizer=index.tokenizer,
            feature_extractor=index.feature_extractor,
            classifier_plugin=model_data['classifier'],
            index=index,
            tokenizer_name=model_data['tokenizer_name'],
            feature_extractor_name=model_data['feature_extractor_name'],
            classifier_name=model_data['classifier_name'],
            tokenizer_params=model_data.get('tokenizer_params', {}),
            feature_extractor_params=model_data.get('feature_extractor_params', {}),
            classifier_params=model_data.get('classifier_params', {}),
        )

        classifier_instance.is_trained = model_data.get('is_trained', False)

        logger.info("Model loaded from %s", model_path)
        return classifier_instance

    # Private helper methods

    def _extract_features_for_labels(self, labels: Dict[str, Union[str, List[str]]]) -> Tuple[Any, List]:
        """Extract feature matrix for documents that have labels."""
        doc_ids = list(labels.keys())
        feature_vectors = []
        valid_labels = []

        for doc_id in doc_ids:
            if doc_id in self.index.doc2idx:
                idx = self.index.doc2idx[doc_id]
                vector = self.index.index.vector_by_id(idx)
                feature_vectors.append(vector)
                valid_labels.append(labels[doc_id])
            else:
                logger.warning("Document %s not found in index", doc_id)

        # Stack into matrix
        if sparse.issparse(feature_vectors[0]) if feature_vectors else False:
            feature_matrix = sparse.vstack(feature_vectors)
        else:
            feature_matrix = np.vstack(feature_vectors) if feature_vectors else np.array([])

        return feature_matrix, valid_labels

    def _extract_features_for_doc_ids(self, doc_ids: List[str]) -> Any:
        """Extract feature matrix for given doc IDs from index."""
        feature_vectors = []

        for doc_id in doc_ids:
            if doc_id in self.index.doc2idx:
                idx = self.index.doc2idx[doc_id]
                vector = self.index.index.vector_by_id(idx)
                feature_vectors.append(vector)
            else:
                logger.warning("Document %s not found in index", doc_id)

        if not feature_vectors:
            return sparse.csr_matrix((0, 0))

        # Stack into matrix
        if sparse.issparse(feature_vectors[0]):
            return sparse.vstack(feature_vectors)
        else:
            return np.vstack(feature_vectors)

    def _extract_features_for_documents(self, documents: Dict[str, str]) -> Any:
        """Extract features for new documents (tokenize + transform)."""
        # Tokenize documents
        tokenized = [self.tokenizer.tokenize(text) for text in documents.values()]

        # Transform to feature vectors
        if hasattr(self.feature_extractor, 'transform'):
            # Use transform if available (for fitted extractors)
            feature_matrix = self.feature_extractor.transform(tokenized)
        else:
            # Use tokens_to_bow for each document
            feature_vectors = [self.feature_extractor.tokens_to_bow(tokens) for tokens in tokenized]

            # Stack into sparse matrix
            if feature_vectors:
                num_features = self.feature_extractor.num_features()
                rows, cols, data = [], [], []
                for doc_idx, bow in enumerate(feature_vectors):
                    for token_id, weight in bow:
                        rows.append(doc_idx)
                        cols.append(token_id)
                        data.append(weight)

                feature_matrix = sparse.csr_matrix(
                    (data, (rows, cols)),
                    shape=(len(documents), num_features)
                )
            else:
                feature_matrix = sparse.csr_matrix((0, 0))

        return feature_matrix
