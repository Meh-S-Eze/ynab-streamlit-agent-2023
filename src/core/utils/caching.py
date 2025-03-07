"""
Caching utilities for API responses and expensive operations.

This module provides caching mechanisms to improve performance and reduce API calls.
"""

import functools
import time
from typing import Any, Callable, Dict, Optional, TypeVar, cast
import logging
from datetime import datetime, timedelta

# Type variable for generic function
T = TypeVar('T')

# Global cache storage
_cache_storage: Dict[str, Dict[str, Any]] = {}
_cache_expiry: Dict[str, Dict[str, float]] = {}
_logger = logging.getLogger(__name__)


def cached_method(maxsize: int = 128, ttl: int = 300) -> Callable:
    """
    Decorator for caching method results with TTL support
    
    Args:
        maxsize (int): Maximum cache size
        ttl (int): Time to live in seconds
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable) -> Callable:
        cache_key = f"{func.__module__}.{func.__qualname__}"
        
        if cache_key not in _cache_storage:
            _cache_storage[cache_key] = {}
            _cache_expiry[cache_key] = {}
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Skip caching for self argument
            cache_args = args[1:] if args and hasattr(args[0], '__dict__') else args
            
            # Create a hashable key from arguments
            key_parts = [str(arg) for arg in cache_args]
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            key = "|".join(key_parts)
            
            # Check if cached result exists and is not expired
            now = time.time()
            if key in _cache_storage[cache_key]:
                if now < _cache_expiry[cache_key].get(key, 0):
                    _logger.debug(f"Cache hit for {cache_key}:{key}")
                    return _cache_storage[cache_key][key]
                else:
                    _logger.debug(f"Cache expired for {cache_key}:{key}")
                    
            # Call the function and cache the result
            result = func(*args, **kwargs)
            
            # Manage cache size
            if len(_cache_storage[cache_key]) >= maxsize:
                # Remove oldest entry
                oldest_key = min(_cache_expiry[cache_key].items(), key=lambda x: x[1])[0]
                _cache_storage[cache_key].pop(oldest_key, None)
                _cache_expiry[cache_key].pop(oldest_key, None)
                
            # Store result and expiry
            _cache_storage[cache_key][key] = result
            _cache_expiry[cache_key][key] = now + ttl
            
            _logger.debug(f"Cached result for {cache_key}:{key}, expires in {ttl}s")
            return result
            
        return wrapper
    
    return decorator


def clear_cache(cache_name: Optional[str] = None) -> None:
    """
    Clear cache entries
    
    Args:
        cache_name (Optional[str]): Specific cache to clear, or all if None
    """
    global _cache_storage, _cache_expiry
    
    if cache_name:
        if cache_name in _cache_storage:
            _cache_storage[cache_name] = {}
            _cache_expiry[cache_name] = {}
            _logger.info(f"Cleared cache: {cache_name}")
    else:
        _cache_storage = {}
        _cache_expiry = {}
        _logger.info("Cleared all caches")


def get_cached_value(cache_name: str, key: str) -> Optional[Any]:
    """
    Get a value from cache if it exists and is not expired
    
    Args:
        cache_name (str): Cache name
        key (str): Cache key
        
    Returns:
        Optional[Any]: Cached value or None
    """
    if cache_name not in _cache_storage or key not in _cache_storage[cache_name]:
        return None
        
    now = time.time()
    if now < _cache_expiry[cache_name].get(key, 0):
        return _cache_storage[cache_name][key]
        
    # Expired
    return None


def set_cached_value(cache_name: str, key: str, value: Any, ttl: int = 300) -> None:
    """
    Set a value in cache with TTL
    
    Args:
        cache_name (str): Cache name
        key (str): Cache key
        value (Any): Value to cache
        ttl (int): Time to live in seconds
    """
    if cache_name not in _cache_storage:
        _cache_storage[cache_name] = {}
        _cache_expiry[cache_name] = {}
        
    _cache_storage[cache_name][key] = value
    _cache_expiry[cache_name][key] = time.time() + ttl
    _logger.debug(f"Set cache value for {cache_name}:{key}, expires in {ttl}s")
