"""
API utilities for common request handling and error processing.

This module provides utilities for working with API requests and responses.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import requests
from datetime import datetime

# Setup logger
logger = logging.getLogger(__name__)


def parse_response(response: requests.Response) -> Dict[str, Any]:
    """
    Parse API response with error handling
    
    Args:
        response (requests.Response): Response object
        
    Returns:
        Dict[str, Any]: Parsed response data
        
    Raises:
        ValueError: If response cannot be parsed
    """
    try:
        if not response.text:
            return {}
            
        data = response.json()
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse response: {e}")
        logger.debug(f"Response text: {response.text[:500]}...")
        raise ValueError(f"Invalid JSON response: {e}")


def extract_error_details(response: requests.Response) -> Dict[str, Any]:
    """
    Extract error details from response
    
    Args:
        response (requests.Response): Response object
        
    Returns:
        Dict[str, Any]: Error details
    """
    try:
        if not response.text:
            return {
                "status_code": response.status_code,
                "message": "Empty response"
            }
            
        data = response.json()
        
        # Handle YNAB API error format
        if "error" in data:
            return {
                "status_code": response.status_code,
                "message": data["error"].get("message", "Unknown error"),
                "detail": data["error"].get("detail", ""),
                "id": data["error"].get("id", "")
            }
            
        # Generic error format
        return {
            "status_code": response.status_code,
            "message": "API error",
            "detail": data
        }
    except Exception as e:
        return {
            "status_code": response.status_code,
            "message": f"Failed to parse error: {str(e)}",
            "detail": response.text[:500] if response.text else ""
        }


def handle_rate_limits(response: requests.Response) -> Optional[Dict[str, Any]]:
    """
    Handle rate limit headers
    
    Args:
        response (requests.Response): Response object
        
    Returns:
        Optional[Dict[str, Any]]: Rate limit info if present
    """
    rate_limit_info = {}
    
    # Extract rate limit headers
    remaining = response.headers.get("X-Rate-Limit-Remaining")
    if remaining is not None:
        rate_limit_info["remaining"] = int(remaining)
        
    reset = response.headers.get("X-Rate-Limit-Reset")
    if reset is not None:
        try:
            reset_time = datetime.fromtimestamp(int(reset))
            rate_limit_info["reset_time"] = reset_time
        except (ValueError, TypeError):
            logger.warning(f"Invalid rate limit reset header: {reset}")
            
    # Return None if no rate limit info found
    return rate_limit_info if rate_limit_info else None


def build_query_params(params: Dict[str, Any]) -> Dict[str, str]:
    """
    Build query parameters with proper formatting
    
    Args:
        params (Dict[str, Any]): Raw parameters
        
    Returns:
        Dict[str, str]: Formatted parameters
    """
    formatted_params = {}
    
    for key, value in params.items():
        if value is None:
            continue
            
        if isinstance(value, bool):
            formatted_params[key] = str(value).lower()
        elif isinstance(value, (list, tuple)):
            formatted_params[key] = ",".join(str(item) for item in value)
        elif isinstance(value, datetime):
            formatted_params[key] = value.strftime("%Y-%m-%d")
        else:
            formatted_params[key] = str(value)
            
    return formatted_params


def validate_response(
    response: requests.Response, 
    expected_status_codes: List[int] = [200, 201, 204]
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Validate response status code and extract error details if needed
    
    Args:
        response (requests.Response): Response object
        expected_status_codes (List[int]): List of expected status codes
        
    Returns:
        Tuple[bool, Optional[Dict[str, Any]]]: (is_valid, error_details)
    """
    if response.status_code in expected_status_codes:
        return True, None
        
    error_details = extract_error_details(response)
    logger.error(f"API error: {error_details}")
    
    return False, error_details
