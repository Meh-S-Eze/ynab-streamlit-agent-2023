from functools import wraps
import logging

class CircuitBreaker:
    def __init__(self, max_failures=3):
        self.failures = 0
        self.max_failures = max_failures
        self.logger = logging.getLogger(__name__)

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.failures >= self.max_failures:
                self.logger.error("Service unavailable due to repeated failures")
                raise Exception("Service unavailable")
            try:
                result = func(*args, **kwargs)
                self.failures = 0
                return result
            except Exception as e:
                self.failures += 1
                self.logger.error(f"Operation failed: {e}")
                raise
        return wrapper 