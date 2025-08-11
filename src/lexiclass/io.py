from __future__ import annotations

import logging
import os
from typing import Dict, Iterator, Optional, Tuple

from bs4 import UnicodeDammit

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load raw text documents from disk (UTF-8 with fallback)."""

    @staticmethod
    def load_text_file(filepath: str) -> Tuple[str, Optional[str]]:
        try:
            with open(filepath, 'rb') as f:
                raw_data = f.read()
            try:
                text = raw_data.decode('utf-8', 'strict')
            except UnicodeError:
                text = UnicodeDammit(raw_data).unicode_markup  # type: ignore[assignment]
            return text, None
        except Exception as e:  # noqa: BLE001
            error_msg = f"Error reading file {filepath}: {str(e)}"
            logger.error(error_msg)
            return "", error_msg

    @staticmethod
    def load_documents_from_directory(directory: str) -> Dict[str, str]:
        documents: Dict[str, str] = {}
        if not os.path.exists(directory):
            logger.error("Directory does not exist: %s", directory)
            return documents
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory, filename)
                doc_id = os.path.splitext(filename)[0]
                text, error = DocumentLoader.load_text_file(filepath)
                if error is None:
                    documents[doc_id] = text
                else:
                    logger.warning("Failed to load document: %s", doc_id)
        logger.info("Loaded %d documents from %s", len(documents), directory)
        return documents

    @staticmethod
    def iter_documents_from_directory(directory: str) -> Iterator[Tuple[str, str]]:
        if not os.path.exists(directory):
            logger.error("Directory does not exist: %s", directory)
            return
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory, filename)
                doc_id = os.path.splitext(filename)[0]
                text, error = DocumentLoader.load_text_file(filepath)
                if error is None:
                    yield doc_id, text
                else:
                    logger.warning("Failed to load document: %s", doc_id)


def load_labels(filepath: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    doc_id, label = parts
                    labels[doc_id] = label
    return labels


