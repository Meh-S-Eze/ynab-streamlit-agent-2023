"""
API request handler for making HTTP requests.

This module provides a class for handling HTTP requests with circuit breaker protection,
caching, and error handling.
"""

import logging
import time
import requests
import json
import uuid
import os
from typing import Any, Dict, Optional, Callable

from ..utils.circuit_breaker import CircuitBreaker, CircuitOpenError, RateLimitError
from ..utils.caching import cached_method, clear_cache
from ..utils.api_utils import (
    parse_response,
    extract_error_details,
    handle_rate_limits,
    build_query_params,
    validate_response
)
from .errors import APIError, AuthenticationError, ResourceNotFoundError, ValidationError, ServerError

# Setup logger
logger = logging.getLogger(__name__)

# Get the development budget ID from environment variables
DEV_BUDGET_ID = os.environ.get('YNAB_BUDGET_DEV', '7c8d67c8-ed70-4ba8-a25e-931a2f294167')

class BudgetAccessError(ValidationError):
    """Exception raised when attempting to access an unauthorized budget"""
    pass

class RequestHandler:
    """
    Handler for making HTTP requests with circuit breaker protection,
    caching, and error handling.
    """
    
    def __init__(self, 
                 session: requests.Session,
                 base_url: str,
                 timeout: int,
                 circuit_breaker: CircuitBreaker):
        """
        Initialize the request handler.
        
        Args:
            session: Requests session with authentication headers
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            circuit_breaker: Circuit breaker for fault tolerance
        """
        self.session = session
        self.base_url = base_url
        self.timeout = timeout
        self.circuit_breaker = circuit_breaker
        
    def _get_correlation_id(self) -> str:
        """Generate a unique correlation ID for request tracing"""
        return str(uuid.uuid4())
    
    def _handle_response_error(self, response: requests.Response) -> None:
        """
        Handle error responses from the API.
        
        Args:
            response: Response object
            
        Raises:
            AuthenticationError: For 401 errors
            ResourceNotFoundError: For 404 errors
            ValidationError: For 400 and 422 errors
            ServerError: For 5xx errors
            APIError: For other errors
        """
        status_code = response.status_code
        error_details = extract_error_details(response)
        
        error_message = error_details.get('message', 'Unknown error')
        
        if status_code == 401:
            raise AuthenticationError(f"Authentication failed: {error_message}", 
                                     status_code, error_details)
        elif status_code == 404:
            raise ResourceNotFoundError(f"Resource not found: {error_message}", 
                                       status_code, error_details)
        elif status_code in (400, 422):
            raise ValidationError(f"Validation error: {error_message}", 
                                 status_code, error_details)
        elif 500 <= status_code < 600:
            raise ServerError(f"Server error: {error_message}", 
                             status_code, error_details)
        else:
            raise APIError(f"API error ({status_code}): {error_message}", 
                          status_code, error_details)
    
    @cached_method(maxsize=100, ttl=300)  # Cache for 5 minutes
    def make_request(self, 
                    method: str, 
                    endpoint: str, 
                    params: Optional[Dict[str, Any]] = None,
                    data: Optional[Dict[str, Any]] = None,
                    headers: Optional[Dict[str, Any]] = None,
                    timeout: Optional[int] = None,
                    cache_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Make an HTTP request to the API with circuit breaker protection.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            params: Query parameters
            data: Request body data
            headers: Additional headers
            timeout: Request timeout (overrides default)
            cache_key: Optional cache key for the request
            
        Returns:
            Dict: Parsed response data
            
        Raises:
            APIError: For API errors
            CircuitOpenError: When circuit breaker is open
        """
        # Generate correlation ID for request tracing
        correlation_id = self._get_correlation_id()
        
        # Prepare request
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_timeout = timeout or self.timeout
        request_headers = {'X-Correlation-ID': correlation_id}
        if headers:
            request_headers.update(headers)
            
        # Format query parameters
        if params:
            formatted_params = build_query_params(params)
        else:
            formatted_params = None
            
        # Format request body
        request_data = None
        if data:
            request_data = json.dumps(data)
            
        # Log request details
        logger.debug(f"API Request: {method} {url} | Correlation ID: {correlation_id}")
        if params:
            logger.debug(f"Query params: {formatted_params}")
        
        # Use circuit breaker to make the request
        try:
            # Log request context for circuit breaker
            self.circuit_breaker.log_request_context({
                'method': method,
                'url': url,
                'correlation_id': correlation_id
            })
            
            # Make the request with circuit breaker protection
            @self.circuit_breaker.protect
            def execute_request():
                response = self.session.request(
                    method=method,
                    url=url,
                    params=formatted_params,
                    data=request_data,
                    headers=request_headers,
                    timeout=request_timeout
                )
                
                # Check for rate limits
                rate_limit_info = handle_rate_limits(response)
                if rate_limit_info:
                    remaining = rate_limit_info.get('remaining', 'unknown')
                    reset_time = rate_limit_info.get('reset', 'unknown')
                    logger.debug(f"Rate limit info - Remaining: {remaining}, Reset: {reset_time}")
                    
                    # If we hit rate limit, raise exception for circuit breaker
                    if remaining == 0:
                        reset_seconds = rate_limit_info.get('reset_seconds', 60)
                        raise RateLimitError(f"Rate limit exceeded. Resets in {reset_seconds} seconds.",
                                           reset_seconds=reset_seconds)
                
                # Validate response
                is_valid, error_details = validate_response(response)
                if not is_valid:
                    self._handle_response_error(response)
                
                # Parse response
                return parse_response(response)
            
            # Execute the request
            return execute_request()
            
        except CircuitOpenError as e:
            logger.error(f"Circuit breaker open: {e}")
            raise
        except RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {e}")
            raise APIError(f"Rate limit exceeded: {e}", details={'reset_seconds': e.reset_seconds})
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            raise APIError(f"Request failed: {str(e)}")
    
    def clear_cache(self) -> None:
        """Clear the request cache"""
        clear_cache(self.make_request)
        logger.debug("Request handler cache cleared")

    def _validate_budget_id(self, budget_id: str) -> None:
        """
        Validate that the budget ID is the development budget ID.
        
        Args:
            budget_id: Budget ID to validate
            
        Raises:
            BudgetAccessError: If the budget ID is not the development budget ID
        """
        if budget_id != DEV_BUDGET_ID:
            logger.error(f"Attempted to access unauthorized budget: {budget_id}")
            raise BudgetAccessError(
                f"Access to budget {budget_id} is not allowed. Only the development budget can be accessed.",
                status_code=403
            ) 