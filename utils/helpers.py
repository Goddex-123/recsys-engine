# Helper Utilities
import time
import functools
from typing import List, Dict, Any, Callable
import numpy as np


def format_number(n: float, precision: int = 2) -> str:
    """Format large numbers with K/M/B suffixes."""
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.{precision}f}B"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.{precision}f}M"
    elif n >= 1_000:
        return f"{n/1_000:.{precision}f}K"
    else:
        return f"{n:.{precision}f}"


def calculate_sparsity(n_interactions: int, n_users: int, n_items: int) -> float:
    """Calculate matrix sparsity."""
    if n_users == 0 or n_items == 0:
        return 1.0
    density = n_interactions / (n_users * n_items)
    return 1 - density


def timer_decorator(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[TIME] {func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide two numbers."""
    if b == 0:
        return default
    return a / b


def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to 0-1 range."""
    if not scores:
        return []
    
    min_s = min(scores)
    max_s = max(scores)
    
    if max_s == min_s:
        return [0.5] * len(scores)
    
    return [(s - min_s) / (max_s - min_s) for s in scores]


def batch_generator(items: List, batch_size: int):
    """Generate batches from a list."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def cosine_similarity_numpy(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))
