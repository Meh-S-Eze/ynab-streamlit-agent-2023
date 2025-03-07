"""
Accounts API client for interacting with the YNAB API.

This module provides a client for accessing account-related endpoints in the YNAB API.
"""

import logging
from typing import Dict, List, Optional

from ..models.account import Account
from .base_client import ResourceNotFoundError

logger = logging.getLogger(__name__)

class AccountsAPI:
    """
    API client for account-related endpoints in the YNAB API.
    """
    
    def __init__(self, client):
        """
        Initialize the AccountsAPI client.
        
        Args:
            client: The base client to use for API requests
        """
        self.client = client
    
    def get_accounts(self, budget_id: str) -> List[Account]:
        """
        Get a list of accounts for a budget.
        
        Args:
            budget_id: Budget ID
            
        Returns:
            List[Account]: List of accounts
        """
        # Validate budget ID
        self.client._validate_budget_id(budget_id)
        
        response = self.client.get(f'budgets/{budget_id}/accounts')
        accounts_data = response.get('data', {}).get('accounts', [])
        accounts = [Account.from_api(account_data) for account_data in accounts_data]
        
        # Cache account IDs
        for account in accounts:
            self.client._entity_cache['accounts'][account.name.lower()] = account.id
            
        return accounts
    
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
        """
        # Validate budget ID
        self.client._validate_budget_id(budget_id)
        
        try:
            response = self.client.get(f'budgets/{budget_id}/accounts/{account_id}')
            account_data = response.get('data', {}).get('account', {})
            account = Account.from_api(account_data)
            
            # Cache account ID
            self.client._entity_cache['accounts'][account.name.lower()] = account.id
            
            return account
        except ResourceNotFoundError:
            logger.error(f"Account not found: {account_id}")
            raise
    
    def find_account_by_name(self, budget_id: str, account_name: str) -> Optional[Account]:
        """
        Find an account by name.
        
        Args:
            budget_id: Budget ID
            account_name: Account name to search for
            
        Returns:
            Optional[Account]: Account if found, None otherwise
        """
        # Check cache first
        account_name_lower = account_name.lower()
        if account_name_lower in self.client._entity_cache['accounts']:
            account_id = self.client._entity_cache['accounts'][account_name_lower]
            try:
                return self.get_account(budget_id, account_id)
            except ResourceNotFoundError:
                # Remove from cache if not found
                del self.client._entity_cache['accounts'][account_name_lower]
        
        # If not in cache, get all accounts and search
        accounts = self.get_accounts(budget_id)
        for account in accounts:
            if account.name.lower() == account_name_lower:
                return account
        
        return None
