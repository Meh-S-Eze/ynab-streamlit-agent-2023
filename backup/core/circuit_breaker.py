import functools
import time
import logging
import uuid
from typing import Optional, Callable, Any, Dict
import requests
from datetime import datetime, timedelta
from tenacity import Retrying, stop_after_attempt, wait_exponential_jitter, RetryError

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
    Enhanced circuit breaker with correlation IDs and improved retry mechanism
    """
    def __init__(
        self,
        max_failures: int = 3,
        reset_timeout: int = 60,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        rate_limit_threshold: int = 5
    ):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.rate_limit_threshold = rate_limit_threshold
        
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"
        self.logger = logging.getLogger(__name__)
        
        # Enhanced rate limiting state
        self.rate_limit_reset = None
        self.rate_limit_remaining = None
        self.rate_limit_history = []
        self.network_failures = 0

    def _get_request_context(self, func: Callable, *args, **kwargs) -> Dict:
        """Get context information about the request for logging"""
        context = {
            'correlation_id': str(uuid.uuid4()),
            'function': func.__name__,
            'module': func.__module__,
        }
        
        # Extract API endpoint info if available in kwargs
        if 'url' in kwargs:
            context['url'] = kwargs['url']
        if 'method' in kwargs:
            context['method'] = kwargs['method']
            
        return context

    def _log_with_context(self, level: int, msg: str, context: Dict, error: Optional[Exception] = None):
        """Log message with request context"""
        log_data = {
            'correlation_id': context['correlation_id'],
            'function': context['function'],
            'module': context['module'],
            'message': msg
        }
        
        # Add API context if available
        if 'url' in context:
            log_data['url'] = context['url']
        if 'method' in context:
            log_data['method'] = context['method']
            
        # Add error details if present
        if error:
            log_data['error'] = str(error)
            if isinstance(error, requests.RequestException) and error.response:
                log_data['status_code'] = error.response.status_code
                log_data['response_text'] = error.response.text[:200]  # Truncate long responses
        
        # Add PagerDuty alert tag for critical errors
        if level >= logging.ERROR:
            log_data['alert'] = 'pagerduty'
        
        self.logger.log(level, str(log_data))

    def _handle_rate_limit_response(self, response: requests.Response, context: Dict):
        """Handle rate limit headers with enhanced logging"""
        now = datetime.now()
        
        # Clean up old rate limit history
        self.rate_limit_history = [
            ts for ts in self.rate_limit_history 
            if (now - ts).total_seconds() < (self.rate_limit_threshold * 60)
        ]
        
        if response.status_code == 429:
            self.rate_limit_history.append(now)
            
            reset_time = response.headers.get('X-Rate-Limit-Reset')
            remaining = response.headers.get('X-Rate-Limit-Remaining', '0')
            
            if reset_time:
                self.rate_limit_reset = datetime.fromtimestamp(int(reset_time))
            else:
                self.rate_limit_reset = now + timedelta(minutes=5)
            
            self.rate_limit_remaining = int(remaining)
            
            if len(self.rate_limit_history) >= 3:
                self._log_with_context(
                    logging.ERROR,
                    f"Hit rate limit {len(self.rate_limit_history)} times in {self.rate_limit_threshold} minutes",
                    context
                )
            
            raise RateLimitError(
                f"Rate limit exceeded. Reset at {self.rate_limit_reset.isoformat()}. "
                f"Remaining requests: {self.rate_limit_remaining}"
            )

    def __call__(self, func: Callable) -> Callable:
        """Decorator implementation with enhanced retry mechanism and logging"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            context = self._get_request_context(func, *args, **kwargs)
            
            if self.state == "open":
                if self.last_failure_time:
                    if time.time() - self.last_failure_time >= self.reset_timeout:
                        self._log_with_context(
                            logging.INFO,
                            "Circuit reset timeout reached, moving to half-open",
                            context
                        )
                        self.state = "half-open"
                    else:
                        raise CircuitOpenError(
                            f"Circuit is open. Will reset after "
                            f"{self.reset_timeout - (time.time() - self.last_failure_time):.1f} seconds"
                        )
            
            # Check rate limit
            if self.rate_limit_reset and datetime.now() < self.rate_limit_reset:
                wait_time = (self.rate_limit_reset - datetime.now()).total_seconds()
                self._log_with_context(
                    logging.WARNING,
                    f"Rate limit in effect. Waiting {wait_time:.1f} seconds",
                    context
                )
                time.sleep(wait_time)
            
            try:
                # Use tenacity for retries with exponential backoff and jitter
                for attempt in Retrying(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_exponential_jitter(initial=self.initial_backoff, max=self.max_backoff),
                    reraise=True
                ):
                    with attempt:
                        try:
                            result = func(*args, **kwargs)
                            
                            # Success handling
                            if self.state == "half-open":
                                self._log_with_context(
                                    logging.INFO,
                                    "Success in half-open state, closing circuit",
                                    context
                                )
                                self.state = "closed"
                            
                            # Reset failure counters
                            self.failures = 0
                            self.network_failures = 0
                            return result
                            
                        except Exception as e:
                            # Handle rate limits and network errors
                            if isinstance(e, requests.RequestException):
                                if hasattr(e, 'response'):
                                    try:
                                        self._handle_rate_limit_response(e.response, context)
                                    except RateLimitError:
                                        raise
                                else:
                                    self.network_failures += 1
                                    self._log_with_context(
                                        logging.ERROR,
                                        "Network error",
                                        context,
                                        error=e
                                    )
                            raise
                            
            except RetryError as e:
                # All retries failed
                self.failures += 1
                self.last_failure_time = time.time()
                
                if self.failures >= self.max_failures:
                    self.state = "open"
                    self._log_with_context(
                        logging.ERROR,
                        f"Circuit breaker opened after {self.failures} failures",
                        context,
                        error=e
                    )
                
                if isinstance(e.last_attempt.exception(), requests.RequestException):
                    error = e.last_attempt.exception()
                    if hasattr(error, 'response'):
                        error_msg = f"API error {error.response.status_code}: {error.response.text}"
                    else:
                        error_msg = f"Network error: {str(error)}"
                    raise CircuitBreakerError(error_msg) from error
                raise e.last_attempt.exception()
                
        return wrapper 