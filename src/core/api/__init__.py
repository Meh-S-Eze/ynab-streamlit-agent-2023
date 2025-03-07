"""
API clients for interacting with external services.

This module provides API clients for interacting with YNAB and other services.
"""

# Import error classes
from .errors import (
    APIError,
    AuthenticationError,
    ResourceNotFoundError,
    ValidationError,
    ServerError
)

# Import request handler
from .request_handler import RequestHandler

# Import base client
from .base_client import BaseAPIClient

# Import YNAB client
from .ynab_client import YNABClient, BudgetAccessError

# Import specialized API clients
from .budgets_api import BudgetsAPI
from .accounts_api import AccountsAPI
from .categories_api import CategoriesAPI
from .transactions_api import TransactionsAPI
from .payees_api import PayeesAPI

__all__ = [
    # Base classes
    'BaseAPIClient',
    'RequestHandler',
    
    # Error classes
    'APIError',
    'AuthenticationError',
    'ResourceNotFoundError',
    'ValidationError',
    'ServerError',
    'BudgetAccessError',
    
    # YNAB client
    'YNABClient',
    
    # Specialized API clients
    'BudgetsAPI',
    'AccountsAPI',
    'CategoriesAPI',
    'TransactionsAPI',
    'PayeesAPI'
]
