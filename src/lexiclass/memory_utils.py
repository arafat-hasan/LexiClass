"""Memory monitoring utilities."""
from __future__ import annotations

import logging
import os
import psutil
from typing import Optional

logger = logging.getLogger(__name__)

def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def get_available_memory() -> float:
    """Get available system memory in GB."""
    return psutil.virtual_memory().available / 1024 / 1024 / 1024

def get_total_memory() -> float:
    """Get total system memory in GB."""
    return psutil.virtual_memory().total / 1024 / 1024 / 1024

def calculate_batch_size(
    num_docs: int,
    avg_doc_size: Optional[float] = None,
    target_memory_usage: float = 0.25,  # Target 25% of available memory
    min_batch_size: int = 100,
    max_batch_size: int = 10000,
) -> int:
    """Calculate optimal batch size based on available memory and document size.
    
    Args:
        num_docs: Total number of documents
        avg_doc_size: Average document size in bytes (if known)
        target_memory_usage: Target memory usage as fraction of available memory
        min_batch_size: Minimum batch size
        max_batch_size: Maximum batch size
        
    Returns:
        Optimal batch size
    """
    available_mem = get_available_memory()
    target_mem = available_mem * target_memory_usage
    
    if avg_doc_size is None:
        # Conservative estimate if we don't know doc size
        batch_size = min(max(min_batch_size, int(target_mem * 1024)), max_batch_size)
    else:
        # Calculate based on actual doc size
        docs_per_gb = 1024 * 1024 * 1024 / avg_doc_size
        batch_size = min(max(min_batch_size, int(target_mem * docs_per_gb)), max_batch_size)
    
    logger.debug(
        "Calculated batch size %d (available memory: %.1f GB, target usage: %.1f GB)",
        batch_size, available_mem, target_mem
    )
    return batch_size

def monitor_memory_usage(threshold: float = 0.8) -> None:
    """Log warning if memory usage exceeds threshold of total memory."""
    used = get_memory_usage()
    total = get_total_memory()
    usage_pct = used / total
    
    if usage_pct > threshold:
        logger.warning(
            "High memory usage: %.1f GB (%.1f%% of %.1f GB total)",
            used, usage_pct * 100, total
        )
