"""
YNAB API client for interacting with the YNAB API.

This module provides a client for interacting with the YNAB API,
with methods for accessing budgets, accounts, transactions, categories, and more.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union
from datetime import date

from .base_client import BaseAPIClient, APIError, ResourceNotFoundError, ValidationError
from ..models.budget import Budget
from ..models.account import Account
from ..models.category import Category, CategoryGroup
from ..models.transaction import Transaction
from ..models.payee import Payee
from ..utils.date_utils import DateFormatter

# Import specialized API clients
from .budgets_api import BudgetsAPI
from .accounts_api import AccountsAPI
from .categories_api import CategoriesAPI
from .transactions_api import TransactionsAPI
from .payees_api import PayeesAPI

# Setup logger
logger = logging.getLogger(__name__)

# Get the development budget ID from environment variables
DEV_BUDGET_ID = os.environ.get('YNAB_BUDGET_DEV', '7c8d67c8-ed70-4ba8-a25e-931a2f294167')


class BudgetAccessError(ValidationError):
    """Exception raised when attempting to access an unauthorized budget"""
    pass


class YNABClient(BaseAPIClient):
    """
    Client for interacting with the YNAB API.
    
    This client extends the BaseAPIClient with YNAB-specific methods for
    accessing budgets, accounts, transactions, categories, and more.
    """
    
    def __init__(self, api_key: str, **kwargs):
        """
        Initialize the YNAB API client.
        
        Args:
            api_key: YNAB API key
            **kwargs: Additional arguments to pass to BaseAPIClient
        """
        super().__init__(api_key, **kwargs)
        logger.debug("Initialized YNAB API client")
        
        # Cache for entity IDs to minimize API calls
        self._entity_cache = {
            'categories': {},
            'payees': {},
            'accounts': {}
        }
        
        # Initialize specialized API clients
        self.budgets_api = BudgetsAPI(self)
        self.accounts_api = AccountsAPI(self)
        self.categories_api = CategoriesAPI(self)
        self.transactions_api = TransactionsAPI(self)
        self.payees_api = PayeesAPI(self)
    
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
    
    def _clear_entity_cache(self, entity_type: Optional[str] = None) -> None:
        """
        Clear the entity cache.
        
        Args:
            entity_type: Type of entity to clear (categories, payees, accounts)
                         If None, clear all caches
        """
        if entity_type:
            if entity_type in self._entity_cache:
                self._entity_cache[entity_type] = {}
                logger.debug(f"Cleared {entity_type} cache")
        else:
            for key in self._entity_cache:
                self._entity_cache[key] = {}
            logger.debug("Cleared all entity caches")
    
    # Budget methods - delegate to budgets_api
    
    def get_budgets(self, include_accounts: bool = False) -> List[Budget]:
        """
        Get a list of budgets.
        
        Args:
            include_accounts: Whether to include account data
            
        Returns:
            List[Budget]: List of budgets
        """
        return self.budgets_api.get_budgets(include_accounts)
    
    def get_budget(self, budget_id: str) -> Budget:
        """
        Get a single budget by ID.
        
        Args:
            budget_id: Budget ID
            
        Returns:
            Budget: Budget object
            
        Raises:
            ResourceNotFoundError: If budget not found
            BudgetAccessError: If attempting to access an unauthorized budget
        """
        return self.budgets_api.get_budget(budget_id)
    
    def get_budget_settings(self, budget_id: str) -> Dict[str, Any]:
        """
        Get settings for a budget.
        
        Args:
            budget_id: Budget ID
            
        Returns:
            Dict: Budget settings
            
        Raises:
            BudgetAccessError: If attempting to access an unauthorized budget
        """
        return self.budgets_api.get_budget_settings(budget_id)
    
    # Account methods - delegate to accounts_api
    
    def get_accounts(self, budget_id: str) -> List[Account]:
        """
        Get a list of accounts for a budget.
        
        Args:
            budget_id: Budget ID
            
        Returns:
            List[Account]: List of accounts
            
        Raises:
            BudgetAccessError: If attempting to access an unauthorized budget
        """
        return self.accounts_api.get_accounts(budget_id)
    
    def get_account(self, budget_id: str, account_id: str) -> Account:
        """
        Get a single account by ID.
        
        Args:
            budget_id: Budget ID
            account_id: Account ID
            
        Returns:
            Account: Account object
            
        Raises:
            ResourceNotFoundError: If account not found
            BudgetAccessError: If attempting to access an unauthorized budget
        """
        return self.accounts_api.get_account(budget_id, account_id)
    
    def find_account_by_name(self, budget_id: str, account_name: str) -> Optional[Account]:
        """
        Find an account by name.
        
        Args:
            budget_id: Budget ID
            account_name: Account name to search for
            
        Returns:
            Optional[Account]: Account if found, None otherwise
            
        Raises:
            BudgetAccessError: If attempting to access an unauthorized budget
        """
        return self.accounts_api.find_account_by_name(budget_id, account_name)
    
    # Category methods - delegate to categories_api
    
    def get_categories(self, budget_id: str) -> List[CategoryGroup]:
        """
        Get a list of categories for a budget.
        
        Args:
            budget_id: Budget ID
            
        Returns:
            List[CategoryGroup]: List of category groups with categories
            
        Raises:
            BudgetAccessError: If attempting to access an unauthorized budget
        """
        return self.categories_api.get_categories(budget_id)
    
    def get_category(self, budget_id: str, category_id: str) -> Category:
        """
        Get a single category by ID.
        
        Args:
            budget_id: Budget ID
            category_id: Category ID
            
        Returns:
            Category: Category object
            
        Raises:
            ResourceNotFoundError: If category not found
            BudgetAccessError: If attempting to access an unauthorized budget
        """
        return self.categories_api.get_category(budget_id, category_id)
    
    def find_category_by_name(self, budget_id: str, category_name: str, 
                             group_name: Optional[str] = None) -> Optional[Category]:
        """
        Find a category by name.
        
        Args:
            budget_id: Budget ID
            category_name: Category name to search for
            group_name: Optional group name to narrow search
            
        Returns:
            Optional[Category]: Category if found, None otherwise
            
        Raises:
            BudgetAccessError: If attempting to access an unauthorized budget
        """
        return self.categories_api.find_category_by_name(budget_id, category_name, group_name)
    
    # Payee methods - delegate to payees_api
    
    def get_payees(self, budget_id: str) -> List[Payee]:
        """
        Get a list of payees for a budget.
        
        Args:
            budget_id: Budget ID
            
        Returns:
            List[Payee]: List of payees
            
        Raises:
            BudgetAccessError: If attempting to access an unauthorized budget
        """
        return self.payees_api.get_payees(budget_id)
    
    def get_payee(self, budget_id: str, payee_id: str) -> Payee:
        """
        Get a single payee by ID.
        
        Args:
            budget_id: Budget ID
            payee_id: Payee ID
            
        Returns:
            Payee: Payee object
            
        Raises:
            ResourceNotFoundError: If payee not found
            BudgetAccessError: If attempting to access an unauthorized budget
        """
        return self.payees_api.get_payee(budget_id, payee_id)
    
    def find_payee_by_name(self, budget_id: str, payee_name: str) -> Optional[Payee]:
        """
        Find a payee by name.
        
        Args:
            budget_id: Budget ID
            payee_name: Payee name to search for
            
        Returns:
            Optional[Payee]: Payee if found, None otherwise
            
        Raises:
            BudgetAccessError: If attempting to access an unauthorized budget
        """
        return self.payees_api.find_payee_by_name(budget_id, payee_name)
    
    # Transaction methods - delegate to transactions_api
    
    def get_transactions(self, 
                        budget_id: str, 
                        since_date: Optional[Union[str, date]] = None,
                        account_id: Optional[str] = None,
                        category_id: Optional[str] = None,
                        payee_id: Optional[str] = None) -> List[Transaction]:
        """
        Get a list of transactions for a budget.
        
        Args:
            budget_id: Budget ID
            since_date: Only return transactions on or after this date
            account_id: Filter by account ID
            category_id: Filter by category ID
            payee_id: Filter by payee ID
            
        Returns:
            List[Transaction]: List of transactions
            
        Raises:
            BudgetAccessError: If attempting to access an unauthorized budget
        """
        return self.transactions_api.get_transactions(
            budget_id, since_date, account_id, category_id, payee_id
        )
    
    def get_transaction(self, budget_id: str, transaction_id: str) -> Transaction:
        """
        Get a single transaction by ID.
        
        Args:
            budget_id: Budget ID
            transaction_id: Transaction ID
            
        Returns:
            Transaction: Transaction object
            
        Raises:
            ResourceNotFoundError: If transaction not found
            BudgetAccessError: If attempting to access an unauthorized budget
        """
        return self.transactions_api.get_transaction(budget_id, transaction_id)
    
    def create_transaction(self, budget_id: str, transaction: Transaction) -> Transaction:
        """
        Create a new transaction.
        
        Args:
            budget_id: Budget ID
            transaction: Transaction object
            
        Returns:
            Transaction: Created transaction with server-assigned ID
            
        Raises:
            BudgetAccessError: If attempting to access an unauthorized budget
        """
        return self.transactions_api.create_transaction(budget_id, transaction)
    
    def create_transactions(self, budget_id: str, transactions: List[Transaction]) -> List[Transaction]:
        """
        Create multiple transactions.
        
        Args:
            budget_id: Budget ID
            transactions: List of Transaction objects
            
        Returns:
            List[Transaction]: List of created transactions
            
        Raises:
            BudgetAccessError: If attempting to access an unauthorized budget
        """
        return self.transactions_api.create_transactions(budget_id, transactions)
    
    def update_transaction(self, budget_id: str, transaction_id: str, 
                          transaction_update: Dict[str, Any]) -> Transaction:
        """
        Update an existing transaction.
        
        Args:
            budget_id: Budget ID
            transaction_id: Transaction ID
            transaction_update: Dictionary with fields to update
            
        Returns:
            Transaction: Updated transaction
            
        Raises:
            BudgetAccessError: If attempting to access an unauthorized budget
        """
        return self.transactions_api.update_transaction(budget_id, transaction_id, transaction_update)
    
    def delete_transaction(self, budget_id: str, transaction_id: str) -> bool:
        """
        Delete a transaction.
        
        Args:
            budget_id: Budget ID
            transaction_id: Transaction ID
            
        Returns:
            bool: True if successful
            
        Raises:
            BudgetAccessError: If attempting to access an unauthorized budget
        """
        return self.transactions_api.delete_transaction(budget_id, transaction_id) 