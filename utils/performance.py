"""
Performance monitoring and profiling utilities.
This module provides tools for measuring and analyzing performance.
"""

import time
import functools
import logging
import numpy as np
from contextlib import contextmanager

logger = logging.getLogger("taxi-rl.performance")

# Dictionary to store performance metrics
_performance_metrics = {}

@contextmanager
def timer(name, verbose=True):
    """
    Context manager for timing code blocks.
    
    Args:
        name (str): Name of the timed section
        verbose (bool): Whether to print timing information
        
    Yields:
        None
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        if name not in _performance_metrics:
            _performance_metrics[name] = []
        _performance_metrics[name].append(elapsed)
        
        if verbose:
            logger.info(f"Timing '{name}': {elapsed:.4f} seconds")

def time_function(func=None, name=None, verbose=True):
    """
    Decorator for timing function execution.
    
    Args:
        func: Function to time
        name (str, optional): Name to use for the timer
        verbose (bool): Whether to print timing information
        
    Returns:
        Function wrapper
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or func.__name__
            with timer(timer_name, verbose=verbose):
                return func(*args, **kwargs)
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)

def get_metrics(name=None):
    """
    Get collected performance metrics.
    
    Args:
        name (str, optional): Name of the metric to retrieve
        
    Returns:
        dict or list: Performance metrics
    """
    if name is not None:
        return _performance_metrics.get(name, [])
    return _performance_metrics

def clear_metrics():
    """Clear all performance metrics."""
    global _performance_metrics
    _performance_metrics = {}

def summarize_metrics():
    """
    Generate a summary of all performance metrics.
    
    Returns:
        dict: Summary statistics for each metric
    """
    summary = {}
    for name, times in _performance_metrics.items():
        if not times:
            continue
            
        summary[name] = {
            "count": len(times),
            "total": sum(times),
            "mean": np.mean(times),
            "median": np.median(times),
            "min": min(times),
            "max": max(times),
            "std": np.std(times)
        }
    
    return summary

def print_summary():
    """Print a formatted summary of all performance metrics."""
    summary = summarize_metrics()
    
    if not summary:
        logger.info("No performance metrics collected")
        return
    
    logger.info("Performance Metrics Summary:")
    logger.info("---------------------------")
    
    for name, stats in summary.items():
        logger.info(f"{name}:")
        logger.info(f"  Count: {stats['count']}")
        logger.info(f"  Total: {stats['total']:.4f}s")
        logger.info(f"  Mean: {stats['mean']:.4f}s")
        logger.info(f"  Median: {stats['median']:.4f}s")
        logger.info(f"  Min: {stats['min']:.4f}s")
        logger.info(f"  Max: {stats['max']:.4f}s")
        logger.info(f"  Std Dev: {stats['std']:.4f}s")
        logger.info("")
