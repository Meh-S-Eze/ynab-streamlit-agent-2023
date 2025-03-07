"""
Core data models for the YNAB integration.

This package contains Pydantic models for YNAB data structures.
"""

from .transaction import (
    Transaction, 
    TransactionCreate, 
    TransactionUpdate,
    TransactionAmount
)

from .category import (
    Category,
    CategoryGroup,
    CategoryUpdate,
    CategoryMatch
)

from .budget import (
    Budget,
    BudgetSummary,
    BudgetMonth,
    SpendingAnalysis
)

from .account import (
    Account,
    AccountSummary,
    AccountUpdate,
    AccountType
)

from .payee import (
    Payee,
    PayeeLocation,
    PayeeMatch,
    PayeeUpdate,
    PayeeCreate
)

__all__ = [
    # Transaction models
    'Transaction', 
    'TransactionCreate', 
    'TransactionUpdate',
    'TransactionAmount',
    
    # Category models
    'Category',
    'CategoryGroup',
    'CategoryUpdate',
    'CategoryMatch',
    
    # Budget models
    'Budget',
    'BudgetSummary',
    'BudgetMonth',
    'SpendingAnalysis',
    
    # Account models
    'Account',
    'AccountSummary',
    'AccountUpdate',
    'AccountType',
    
    # Payee models
    'Payee',
    'PayeeLocation',
    'PayeeMatch',
    'PayeeUpdate',
    'PayeeCreate'
]
