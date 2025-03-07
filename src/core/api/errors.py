"""
API error classes for handling API exceptions.

This module provides a set of exception classes for handling API errors.
"""

from typing import Dict, Any, Optional


class APIError(Exception):
    """Base exception for API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class AuthenticationError(APIError):
    """Exception raised for authentication errors"""
    pass


class ResourceNotFoundError(APIError):
    """Exception raised when a resource is not found"""
    pass


class ValidationError(APIError):
    """Exception raised for validation errors"""
    pass


class ServerError(APIError):
    """Exception raised for server errors"""
    pass 