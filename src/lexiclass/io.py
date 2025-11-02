from __future__ import annotations

import logging
import os
from typing import Dict, Iterator, Optional, Tuple

from bs4 import UnicodeDammit
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from collections import deque

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
        except OSError as e:  # file I/O errors
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
        # Deterministic order
        try:
            entries = sorted([e for e in os.scandir(directory) if e.is_file() and e.name.endswith('.txt')], key=lambda e: e.name)
        except OSError:
            # Fallback to os.listdir if scandir fails for any reason
            entries = []
            for filename in sorted(os.listdir(directory)):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath) and filename.endswith('.txt'):
                    entries.append(type('E', (), {'path': filepath, 'name': filename}))  # lightweight shim

        for entry in entries:
            filepath = getattr(entry, 'path', os.path.join(directory, entry.name))
            filename = entry.name
            doc_id = os.path.splitext(filename)[0]
            text, error = DocumentLoader.load_text_file(filepath)
            if error is None:
                yield doc_id, text
            else:
                logger.warning("Failed to load document: %s", doc_id)

    @staticmethod
    def iter_documents_from_paths(
        document_paths: list[tuple[str, str]]
    ) -> Iterator[Tuple[str, str]]:
        """Iterate documents from a list of (doc_id, filepath) tuples.

        Memory-efficient: Streams documents one at a time, only loading one
        document into memory at a time. Ideal for database-driven workflows
        where document paths are stored in a database.

        Args:
            document_paths: List of (doc_id, filepath) tuples where:
                - doc_id: Document identifier (string)
                - filepath: Absolute or relative path to the .txt file

        Yields:
            Tuple of (doc_id, text) for each successfully loaded document

        Example:
            >>> paths = [
            ...     ("doc_1", "/data/docs/doc_1.txt"),
            ...     ("doc_2", "/data/docs/doc_2.txt"),
            ... ]
            >>> for doc_id, text in DocumentLoader.iter_documents_from_paths(paths):
            ...     print(f"Loaded {doc_id}: {len(text)} chars")
        """
        for doc_id, filepath in document_paths:
            text, error = DocumentLoader.load_text_file(filepath)
            if error is None:
                yield doc_id, text
            else:
                logger.warning("Failed to load document %s from %s: %s", doc_id, filepath, error)

    @staticmethod
    def iter_documents_from_directory_parallel(
        directory: str,
        *,
        num_workers: int | None = None,
        prefetch: int = 64,
        ordered: bool = True,
    ) -> Iterator[Tuple[str, str]]:
        """Iterate documents from a directory using a thread pool for I/O.

        - Uses threads to overlap disk reads and decoding.
        - Bounded "prefetch" controls max in-flight documents to cap RAM.
        - If ordered=True, yields documents in deterministic, name-sorted order.
        """
        if not os.path.exists(directory):
            logger.error("Directory does not exist: %s", directory)
            return

        try:
            entries = sorted([e for e in os.scandir(directory) if e.is_file() and e.name.endswith('.txt')], key=lambda e: e.name)
        except OSError:
            # Fallback path
            entries = []
            for filename in sorted(os.listdir(directory)):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath) and filename.endswith('.txt'):
                    entries.append(type('E', (), {'path': filepath, 'name': filename}))

        if not entries:
            return

        max_workers = num_workers or (os.cpu_count() or 4)
        if max_workers <= 1 and prefetch <= 1:
            # Degenerate to sequential iterator
            yield from DocumentLoader.iter_documents_from_directory(directory)
            return

        # Submission window bounded by prefetch
        def read_task(filepath: str) -> Tuple[str, Optional[str], str]:
            text, err = DocumentLoader.load_text_file(filepath)
            filename = os.path.basename(filepath)
            doc_id = os.path.splitext(filename)[0]
            return (doc_id, err, text)

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="lexi-io") as executor:
            if ordered:
                # Submit in deterministic order and yield by original index
                total = len(entries)
                next_yield_idx = 0
                submitted = 0
                # Map index->future and future->index for quick lookup
                inflight_by_idx: dict[int, Future] = {}
                future_to_idx: dict[Future, int] = {}
                results: dict[int, Optional[Tuple[str, str]]] = {}

                def submit_one(i: int) -> None:
                    fp = getattr(entries[i], 'path', os.path.join(directory, entries[i].name))
                    fut = executor.submit(lambda idx=i, path=fp: (idx,) + read_task(path))
                    inflight_by_idx[i] = fut
                    future_to_idx[fut] = i

                # Prime the prefetch window
                while submitted < total and len(inflight_by_idx) < prefetch:
                    submit_one(submitted)
                    submitted += 1

                while next_yield_idx < total:
                    if not inflight_by_idx and submitted >= total and next_yield_idx not in results:
                        break
                    # Wait for any future to complete
                    for fut in as_completed(list(future_to_idx.keys()), timeout=None):
                        idx = future_to_idx.pop(fut)
                        inflight_by_idx.pop(idx, None)
                        _idx_ret, doc_id, err, text = fut.result()
                        if err is None:
                            results[idx] = (doc_id, text)
                        else:
                            logger.warning("Failed to load document: %s", doc_id)
                            results[idx] = None
                        # Top up submissions
                        if submitted < total and len(inflight_by_idx) < prefetch:
                            submit_one(submitted)
                            submitted += 1
                        # After each completion, try to flush in-order results
                        while next_yield_idx in results:
                            item = results.pop(next_yield_idx)
                            if item is not None:
                                yield item
                            next_yield_idx += 1
                        # Break to re-evaluate loop conditions
                        break

            # Unordered: keep bounded set and yield as they finish
            inflight: set[Future] = set()

            def try_submit(i: int) -> None:
                fp = getattr(entries[i], 'path', os.path.join(directory, entries[i].name))
                fut = executor.submit(read_task, fp)
                inflight.add(fut)

            total = len(entries)
            submitted = 0
            yielded = 0

            # Prime submissions
            while submitted < total and len(inflight) < prefetch:
                try_submit(submitted)
                submitted += 1

            while yielded < total:
                if not inflight:
                    break
                for fut in as_completed(list(inflight), timeout=None):
                    inflight.discard(fut)
                    doc_id, err, text = fut.result()
                    if err is None:
                        yield doc_id, text
                    else:
                        logger.warning("Failed to load document: %s", doc_id)
                    yielded += 1
                    # Top up submissions to maintain window
                    if submitted < total:
                        try_submit(submitted)
                        submitted += 1
                    # Break to refresh the as_completed iterator with updated inflight
                    break


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


