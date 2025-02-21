import functools
import time
import logging
from typing import Optional, Callable, Any
import requests
from datetime import datetime, timedelta

class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors"""
    pass

class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit is open"""
    pass

class RateLimitError(CircuitBreakerError):
    """Raised when rate limit is exceeded"""
    pass

class CircuitBreaker:
    """
    Enhanced circuit breaker with rate limit handling and exponential backoff
    """
    def __init__(
        self,
        max_failures: int = 3,
        reset_timeout: int = 60,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        backoff_multiplier: float = 2.0,
        rate_limit_threshold: int = 5  # Track rate limits within this window
    ):
        """
        Initialize circuit breaker
        
        Args:
            max_failures (int): Maximum number of failures before opening circuit
            reset_timeout (int): Seconds to wait before attempting reset
            max_retries (int): Maximum number of retry attempts
            initial_backoff (float): Initial backoff time in seconds
            max_backoff (float): Maximum backoff time in seconds
            backoff_multiplier (float): Multiplier for exponential backoff
            rate_limit_threshold (int): Time window in minutes to track rate limits
        """
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_multiplier = backoff_multiplier
        self.rate_limit_threshold = rate_limit_threshold
        
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"
        self.logger = logging.getLogger(__name__)
        
        # Enhanced rate limiting state
        self.rate_limit_reset = None
        self.rate_limit_remaining = None
        self.rate_limit_history = []  # Track rate limit occurrences
        self.network_failures = 0  # Track consecutive network failures

    def _calculate_backoff(self, retry_count: int) -> float:
        """Calculate exponential backoff time"""
        backoff = self.initial_backoff * (self.backoff_multiplier ** retry_count)
        return min(backoff, self.max_backoff)

    def _handle_rate_limit_response(self, response: requests.Response):
        """
        Handle rate limit headers from response with enhanced tracking
        
        Args:
            response (requests.Response): Response to check for rate limits
        
        Raises:
            RateLimitError: If rate limit is exceeded
        """
        now = datetime.now()
        
        # Clean up old rate limit history
        self.rate_limit_history = [
            ts for ts in self.rate_limit_history 
            if (now - ts).total_seconds() < (self.rate_limit_threshold * 60)
        ]
        
        if response.status_code == 429:
            # Add current rate limit to history
            self.rate_limit_history.append(now)
            
            # Parse rate limit headers
            reset_time = response.headers.get('X-Rate-Limit-Reset')
            remaining = response.headers.get('X-Rate-Limit-Remaining', '0')
            
            if reset_time:
                self.rate_limit_reset = datetime.fromtimestamp(int(reset_time))
            else:
                # If no reset time provided, use default
                self.rate_limit_reset = now + timedelta(minutes=5)
            
            self.rate_limit_remaining = int(remaining)
            
            # Check if we're hitting rate limits too frequently
            if len(self.rate_limit_history) >= 3:  # 3 rate limits within threshold
                self.logger.error(
                    f"Hit rate limit {len(self.rate_limit_history)} times in "
                    f"{self.rate_limit_threshold} minutes. Consider reducing request frequency."
                )
            
            raise RateLimitError(
                f"Rate limit exceeded. Reset at {self.rate_limit_reset.isoformat()}. "
                f"Remaining requests: {self.rate_limit_remaining}"
            )

    def _should_retry(self, exception: Exception) -> bool:
        """
        Determine if the exception is retryable with enhanced network failure detection
        
        Args:
            exception (Exception): Exception to check
        
        Returns:
            bool: Whether the exception is retryable
        """
        if isinstance(exception, requests.RequestException):
            if isinstance(exception, requests.ConnectionError):
                self.network_failures += 1
                # If we've had too many network failures, don't retry
                if self.network_failures >= self.max_failures:
                    self.logger.error(
                        f"Too many consecutive network failures ({self.network_failures}). "
                        "Circuit will be opened."
                    )
                    return False
                return True
                
            if hasattr(exception, 'response'):
                status_code = exception.response.status_code
                
                # Don't retry client errors except rate limits
                if 400 <= status_code < 500 and status_code != 429:
                    return False
                    
                # Retry rate limits and server errors
                return status_code == 429 or (500 <= status_code < 600)
                
        return False

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator implementation with enhanced error handling
        
        Args:
            func (Callable): Function to wrap
            
        Returns:
            Callable: Wrapped function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if self.state == "open":
                if self.last_failure_time:
                    if time.time() - self.last_failure_time >= self.reset_timeout:
                        self.logger.info("Circuit reset timeout reached, moving to half-open")
                        self.state = "half-open"
                    else:
                        raise CircuitOpenError(
                            f"Circuit is open. Will reset after "
                            f"{self.reset_timeout - (time.time() - self.last_failure_time):.1f} seconds"
                        )
                
            # Check rate limit with enhanced logging
            if self.rate_limit_reset and datetime.now() < self.rate_limit_reset:
                wait_time = (self.rate_limit_reset - datetime.now()).total_seconds()
                self.logger.warning(
                    f"Rate limit in effect. Waiting {wait_time:.1f} seconds. "
                    f"Recent rate limits: {len(self.rate_limit_history)}"
                )
                time.sleep(wait_time)
            
            retry_count = 0
            last_exception = None
            
            while retry_count <= self.max_retries:
                try:
                    result = func(*args, **kwargs)
                    
                    # Success handling
                    if self.state == "half-open":
                        self.logger.info("Success in half-open state, closing circuit")
                        self.state = "closed"
                    
                    # Reset failure counters on success
                    self.failures = 0
                    self.network_failures = 0
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Enhanced error handling
                    if isinstance(e, requests.RequestException):
                        if hasattr(e, 'response'):
                            try:
                                self._handle_rate_limit_response(e.response)
                            except RateLimitError as rle:
                                self.logger.warning(str(rle))
                                # Wait for rate limit reset if this is not our last retry
                                if retry_count < self.max_retries:
                                    wait_time = (self.rate_limit_reset - datetime.now()).total_seconds()
                                    time.sleep(max(0, wait_time))
                                    continue
                        else:
                            # Log network-level errors
                            self.logger.error(
                                f"Network error: {str(e)}. "
                                f"Network failures: {self.network_failures}"
                            )
                    
                    # Increment failures
                    self.failures += 1
                    self.last_failure_time = time.time()
                    
                    # Check if we should retry
                    if not self._should_retry(e) or retry_count >= self.max_retries:
                        break
                    
                    # Calculate backoff time
                    backoff = self._calculate_backoff(retry_count)
                    self.logger.warning(
                        f"Attempt {retry_count + 1}/{self.max_retries + 1} failed. "
                        f"Retrying in {backoff:.1f} seconds. Error: {str(e)}"
                    )
                    time.sleep(backoff)
                    retry_count += 1
            
            # Check if we should open the circuit
            if self.failures >= self.max_failures:
                self.state = "open"
                self.logger.error(
                    f"Circuit breaker opened after {self.failures} failures. "
                    f"Network failures: {self.network_failures}. "
                    f"Rate limits: {len(self.rate_limit_history)}. "
                    f"Last error: {str(last_exception)}"
                )
            
            # Re-raise the last exception with context
            if isinstance(last_exception, requests.RequestException):
                if hasattr(last_exception, 'response'):
                    status_code = last_exception.response.status_code
                    error_msg = f"API error {status_code}: {last_exception.response.text}"
                else:
                    error_msg = f"Network error: {str(last_exception)}"
                raise CircuitBreakerError(error_msg) from last_exception
            raise last_exception
        
        return wrapper 