"""
Transaction service for YNAB transactions.

This module provides functionality for working with YNAB transactions,
including transaction operations and categorization.
"""

import logging
from typing import List, Dict, Optional, Any, Union
from datetime import date

from ..models.transaction import Transaction, TransactionUpdate
from ..models.category import Category
from ..api.transactions_api import TransactionsAPI
from ..services.category_service import CategoryService

logger = logging.getLogger(__name__)

class TransactionService:
    """
    Service for working with YNAB transactions.
    
    This service provides methods for transaction operations and categorization.
    """
    
    def __init__(self, 
                transactions_api: TransactionsAPI,
                category_service: CategoryService):
        """
        Initialize the transaction service.
        
        Args:
            transactions_api: API client for transaction-related endpoints
            category_service: Service for category operations
        """
        self.transactions_api = transactions_api
        self.category_service = category_service
    
    def get_transactions(self, 
                       budget_id: str, 
                       since_date: Optional[Union[str, date]] = None,
                       account_id: Optional[str] = None,
                       category_id: Optional[str] = None,
                       payee_id: Optional[str] = None) -> List[Transaction]:
        """
        Get transactions for a budget.
        
        Args:
            budget_id: Budget ID
            since_date: Only return transactions on or after this date
            account_id: Filter by account ID
            category_id: Filter by category ID
            payee_id: Filter by payee ID
            
        Returns:
            List[Transaction]: List of transactions
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
        """
        return self.transactions_api.get_transaction(budget_id, transaction_id)
    
    def create_transaction(self, budget_id: str, transaction: Transaction) -> Transaction:
        """
        Create a new transaction.
        
        Args:
            budget_id: Budget ID
            transaction: Transaction to create
            
        Returns:
            Transaction: Created transaction
        """
        return self.transactions_api.create_transaction(budget_id, transaction)
    
    def create_transactions(self, budget_id: str, transactions: List[Transaction]) -> List[Transaction]:
        """
        Create multiple transactions.
        
        Args:
            budget_id: Budget ID
            transactions: List of transactions to create
            
        Returns:
            List[Transaction]: List of created transactions
        """
        return self.transactions_api.create_transactions(budget_id, transactions)
    
    def update_transaction(self, budget_id: str, transaction_id: str, 
                         transaction_update: Dict[str, Any]) -> Transaction:
        """
        Update a transaction.
        
        Args:
            budget_id: Budget ID
            transaction_id: Transaction ID
            transaction_update: Dictionary of fields to update
            
        Returns:
            Transaction: Updated transaction
        """
        return self.transactions_api.update_transaction(budget_id, transaction_id, transaction_update)
    
    def update_transaction_category(self, budget_id: str, transaction_id: str, 
                                  category_name: Optional[str] = None,
                                  category_id: Optional[str] = None) -> Transaction:
        """
        Update a transaction's category.
        
        Args:
            budget_id: Budget ID
            transaction_id: Transaction ID
            category_name: Category name (will be matched to ID)
            category_id: Category ID (direct assignment)
            
        Returns:
            Transaction: Updated transaction
        """
        # Get the current transaction
        current_transaction = self.get_transaction(budget_id, transaction_id)
        
        # If category_name is provided, find the category ID
        if category_name and not category_id:
            category_match = self.category_service.find_best_category_match(
                budget_id, category_name
            )
            
            if category_match.category:
                category_id = category_match.category.id
            else:
                logger.warning(f"No category match found for '{category_name}'")
                return current_transaction
        
        # Prepare the update
        update_data = {'category_id': category_id}
        
        # Update the transaction
        return self.update_transaction(budget_id, transaction_id, update_data)
    
    def bulk_update_categories(self, budget_id: str, 
                             updates: List[TransactionUpdate]) -> List[Transaction]:
        """
        Update categories for multiple transactions.
        
        Args:
            budget_id: Budget ID
            updates: List of transaction updates with category information
            
        Returns:
            List[Transaction]: List of updated transactions
        """
        updated_transactions = []
        
        for update in updates:
            transaction_id = update.transaction_id
            category_id = update.category_id
            
            # Update the transaction
            updated_transaction = self.update_transaction_category(
                budget_id, transaction_id, category_id=category_id
            )
            
            updated_transactions.append(updated_transaction)
        
        return updated_transactions
    
    def delete_transaction(self, budget_id: str, transaction_id: str) -> bool:
        """
        Delete a transaction.
        
        Args:
            budget_id: Budget ID
            transaction_id: Transaction ID
            
        Returns:
            bool: True if successful
        """
        return self.transactions_api.delete_transaction(budget_id, transaction_id)
    
    def search_transactions(self, budget_id: str, query_params: Dict[str, Any]) -> List[Transaction]:
        """
        Search for transactions based on query parameters.
        
        Args:
            budget_id: Budget ID
            query_params: Dictionary of query parameters
            
        Returns:
            List[Transaction]: List of matching transactions
        """
        # Extract parameters
        since_date = query_params.get('since_date')
        account_id = query_params.get('account_id')
        category_id = query_params.get('category_id')
        payee_id = query_params.get('payee_id')
        
        # Get transactions with filters
        return self.get_transactions(
            budget_id, since_date, account_id, category_id, payee_id
        )
