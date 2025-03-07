"""
Base API client for YNAB API interactions.

This module provides the foundation for all API interactions with YNAB,
handling authentication, request formatting, error handling, and response processing.
"""

import logging
import requests
import os
import json
from typing import Any, Dict, Optional

from ..utils.circuit_breaker import CircuitBreaker
from ..utils.caching import clear_cache

from .errors import APIError, AuthenticationError, ResourceNotFoundError, ValidationError, ServerError
from .request_handler import RequestHandler

# Setup logger
logger = logging.getLogger(__name__)

# Load configuration from environment variables with defaults
DEFAULT_API_URL = os.environ.get('YNAB_API_URL', 'https://api.youneedabudget.com/v1')
DEFAULT_TIMEOUT = int(os.environ.get('YNAB_API_TIMEOUT', '30'))
DEFAULT_MAX_RETRIES = int(os.environ.get('YNAB_API_MAX_RETRIES', '3'))
DEFAULT_RETRY_DELAY = int(os.environ.get('YNAB_API_RETRY_DELAY', '1'))
DEFAULT_CIRCUIT_THRESHOLD = int(os.environ.get('YNAB_CIRCUIT_THRESHOLD', '5'))
DEFAULT_CIRCUIT_TIMEOUT = int(os.environ.get('YNAB_CIRCUIT_TIMEOUT', '60'))


class BaseAPIClient:
    """
    Base client for interacting with the YNAB API.
    
    This class provides the foundation for all API interactions, handling:
    - Authentication
    - Request formatting and execution
    - Error handling and retries
    - Response parsing and validation
    - Rate limiting
    - Circuit breaking for fault tolerance
    """
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = DEFAULT_API_URL,
                 timeout: int = DEFAULT_TIMEOUT,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 retry_delay: int = DEFAULT_RETRY_DELAY,
                 circuit_breaker_threshold: int = DEFAULT_CIRCUIT_THRESHOLD,
                 circuit_breaker_timeout: int = DEFAULT_CIRCUIT_TIMEOUT):
        """
        Initialize the API client.
        
        Args:
            api_key: YNAB API key
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Initial delay between retries (seconds)
            circuit_breaker_threshold: Number of failures before circuit opens
            circuit_breaker_timeout: Time circuit stays open (seconds)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # Setup circuit breaker for fault tolerance
        self.circuit_breaker = CircuitBreaker(
            max_failures=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_timeout,
            max_retries=max_retries,
            initial_backoff=retry_delay
        )
        
        # Setup session with default headers
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Initialize request handler
        self.request_handler = RequestHandler(
            session=self.session,
            base_url=self.base_url,
            timeout=self.timeout,
            circuit_breaker=self.circuit_breaker
        )
        
        logger.debug(f"Initialized API client with base URL: {self.base_url}")
    
    def get(self, 
           endpoint: str, 
           params: Optional[Dict[str, Any]] = None,
           headers: Optional[Dict[str, Any]] = None,
           timeout: Optional[int] = None,
           cache: bool = True) -> Dict[str, Any]:
        """
        Make a GET request to the API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout
            cache: Whether to cache the response
            
        Returns:
            Dict: Parsed response data
        """
        if cache:
            # Generate cache key based on endpoint and params
            cache_key = f"get:{endpoint}:{json.dumps(params or {})}"
            return self.request_handler.make_request('GET', endpoint, params, None, headers, timeout, cache_key)
        else:
            # Clear cache for this method to ensure fresh data
            clear_cache(self.request_handler.make_request, f"get:{endpoint}:")
            return self.request_handler.make_request('GET', endpoint, params, None, headers, timeout)
    
    def post(self, 
            endpoint: str, 
            data: Dict[str, Any],
            params: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, Any]] = None,
            timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Make a POST request to the API.
        
        Args:
            endpoint: API endpoint
            data: Request body data
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout
            
        Returns:
            Dict: Parsed response data
        """
        return self.request_handler.make_request('POST', endpoint, params, data, headers, timeout)
    
    def put(self, 
           endpoint: str, 
           data: Dict[str, Any],
           params: Optional[Dict[str, Any]] = None,
           headers: Optional[Dict[str, Any]] = None,
           timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Make a PUT request to the API.
        
        Args:
            endpoint: API endpoint
            data: Request body data
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout
            
        Returns:
            Dict: Parsed response data
        """
        return self.request_handler.make_request('PUT', endpoint, params, data, headers, timeout)
    
    def patch(self, 
             endpoint: str, 
             data: Dict[str, Any],
             params: Optional[Dict[str, Any]] = None,
             headers: Optional[Dict[str, Any]] = None,
             timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Make a PATCH request to the API.
        
        Args:
            endpoint: API endpoint
            data: Request body data
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout
            
        Returns:
            Dict: Parsed response data
        """
        return self.request_handler.make_request('PATCH', endpoint, params, data, headers, timeout)
    
    def delete(self, 
              endpoint: str,
              params: Optional[Dict[str, Any]] = None,
              headers: Optional[Dict[str, Any]] = None,
              timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Make a DELETE request to the API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout
            
        Returns:
            Dict: Parsed response data
        """
        return self.request_handler.make_request('DELETE', endpoint, params, None, headers, timeout)
    
    def clear_cache(self) -> None:
        """Clear the request cache"""
        self.request_handler.clear_cache()
        logger.debug("API client cache cleared")
    
    def health_check(self) -> bool:
        """
        Check if the API is accessible.
        
        Returns:
            bool: True if API is accessible, False otherwise
        """
        try:
            # Make a simple request to check API health
            self.get('user', cache=False)
            return True
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return False
