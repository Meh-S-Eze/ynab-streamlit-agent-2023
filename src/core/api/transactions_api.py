"""
Transactions API client for interacting with the YNAB API.

This module provides a client for accessing transaction-related endpoints in the YNAB API.
"""

import logging
from typing import Dict, List, Optional, Union, Any
from datetime import date

from ..models.transaction import Transaction
from .base_client import ResourceNotFoundError
from ..utils.date_utils import DateFormatter

logger = logging.getLogger(__name__)

class TransactionsAPI:
    """
    API client for transaction-related endpoints in the YNAB API.
    """
    
    def __init__(self, client):
        """
        Initialize the TransactionsAPI client.
        
        Args:
            client: The base client to use for API requests
        """
        self.client = client
    
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
        """
        # Validate budget ID
        self.client._validate_budget_id(budget_id)
        
        params = {}
        
        if since_date:
            params['since_date'] = DateFormatter.format_date(since_date)
            
        # Build endpoint based on filters
        if account_id:
            endpoint = f'budgets/{budget_id}/accounts/{account_id}/transactions'
        elif category_id:
            endpoint = f'budgets/{budget_id}/categories/{category_id}/transactions'
        elif payee_id:
            endpoint = f'budgets/{budget_id}/payees/{payee_id}/transactions'
        else:
            endpoint = f'budgets/{budget_id}/transactions'
            
        response = self.client.get(endpoint, params=params)
        transactions_data = response.get('data', {}).get('transactions', [])
        return [Transaction.from_api_response(tx_data) for tx_data in transactions_data]
    
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
        """
        # Validate budget ID
        self.client._validate_budget_id(budget_id)
        
        try:
            response = self.client.get(f'budgets/{budget_id}/transactions/{transaction_id}')
            transaction_data = response.get('data', {}).get('transaction', {})
            return Transaction.from_api_response(transaction_data)
        except ResourceNotFoundError:
            logger.error(f"Transaction not found: {transaction_id}")
            raise
    
    def create_transaction(self, budget_id: str, transaction: Transaction) -> Transaction:
        """
        Create a new transaction.
        
        Args:
            budget_id: Budget ID
            transaction: Transaction object
            
        Returns:
            Transaction: Created transaction with server-assigned ID
        """
        # Validate budget ID
        self.client._validate_budget_id(budget_id)
        
        data = {
            'transaction': transaction.to_api_dict()
        }
        
        response = self.client.post(f'budgets/{budget_id}/transactions', data=data)
        transaction_data = response.get('data', {}).get('transaction', {})
        return Transaction.from_api_response(transaction_data)
    
    def create_transactions(self, budget_id: str, transactions: List[Transaction]) -> List[Transaction]:
        """
        Create multiple transactions.
        
        Args:
            budget_id: Budget ID
            transactions: List of Transaction objects
            
        Returns:
            List[Transaction]: List of created transactions
        """
        # Validate budget ID
        self.client._validate_budget_id(budget_id)
        
        data = {
            'transactions': [tx.to_api_dict() for tx in transactions]
        }
        
        response = self.client.post(f'budgets/{budget_id}/transactions/bulk', data=data)
        transactions_data = response.get('data', {}).get('transactions', [])
        return [Transaction.from_api_response(tx_data) for tx_data in transactions_data]
    
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
        """
        # Validate budget ID
        self.client._validate_budget_id(budget_id)
        
        data = {
            'transaction': transaction_update
        }
        
        response = self.client.put(f'budgets/{budget_id}/transactions/{transaction_id}', data=data)
        transaction_data = response.get('data', {}).get('transaction', {})
        return Transaction.from_api_response(transaction_data)
    
    def delete_transaction(self, budget_id: str, transaction_id: str) -> bool:
        """
        Delete a transaction.
        
        Args:
            budget_id: Budget ID
            transaction_id: Transaction ID
            
        Returns:
            bool: True if successful
        """
        # Validate budget ID
        self.client._validate_budget_id(budget_id)
        
        try:
            self.client.delete(f'budgets/{budget_id}/transactions/{transaction_id}')
            return True
        except Exception as e:
            logger.error(f"Failed to delete transaction {transaction_id}: {e}")
            return False
