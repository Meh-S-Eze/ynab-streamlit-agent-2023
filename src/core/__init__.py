"""
Core module for YNAB integration.

This module provides the core functionality for interacting with the YNAB API,
including API clients, data models, and utility functions.
"""

# Import API clients
from .api import (
    YNABClient,
    APIError,
    AuthenticationError,
    ResourceNotFoundError,
    ValidationError,
    ServerError,
    BudgetAccessError
)

# Import models
from .models import (
    Budget,
    Account,
    Category,
    CategoryGroup,
    Transaction,
    Payee
)

# Import utilities
from .utils import (
    CircuitBreaker,
    cached_method,
    clear_cache,
    DateFormatter
)

# Import services
from .services import (
    CategoryService,
    TransactionService,
    BudgetService,
    AITaggingService
)

# Import prompts
from .prompts import (
    BasePrompt,
    TransactionCategoryPrompt,
    TransactionTaggingPrompt,
    CategoryMatchPrompt,
    CategoryHierarchyPrompt
)

# Import dependency injection container
from .container import Container

# Get development budget ID
import os
DEV_BUDGET_ID = os.environ.get('YNAB_BUDGET_DEV', '7c8d67c8-ed70-4ba8-a25e-931a2f294167')

# Create a default container instance
container = Container()

__all__ = [
    # API clients
    'YNABClient',
    
    # Error classes
    'APIError',
    'AuthenticationError',
    'ResourceNotFoundError',
    'ValidationError',
    'ServerError',
    'BudgetAccessError',
    
    # Models
    'Budget',
    'Account',
    'Category',
    'CategoryGroup',
    'Transaction',
    'Payee',
    
    # Utilities
    'CircuitBreaker',
    'cached_method',
    'clear_cache',
    'DateFormatter',
    
    # Services
    'CategoryService',
    'TransactionService',
    'BudgetService',
    'AITaggingService',
    
    # Prompts
    'BasePrompt',
    'TransactionCategoryPrompt',
    'TransactionTaggingPrompt',
    'CategoryMatchPrompt',
    'CategoryHierarchyPrompt',
    
    # Dependency Injection
    'Container',
    'container',
    
    # Constants
    'DEV_BUDGET_ID'
] 