"""
Payees API client for interacting with the YNAB API.

This module provides a client for accessing payee-related endpoints in the YNAB API.
"""

import logging
from typing import Dict, List, Optional, Any

from ..models.payee import Payee
from .base_client import ResourceNotFoundError

logger = logging.getLogger(__name__)

class PayeesAPI:
    """
    API client for payee-related endpoints in the YNAB API.
    """
    
    def __init__(self, client):
        """
        Initialize the PayeesAPI client.
        
        Args:
            client: The base client to use for API requests
        """
        self.client = client
    
    def get_payees(self, budget_id: str) -> List[Payee]:
        """
        Get a list of payees for a budget.
        
        Args:
            budget_id: Budget ID
            
        Returns:
            List[Payee]: List of payees
        """
        # Validate budget ID
        self.client._validate_budget_id(budget_id)
        
        response = self.client.get(f'budgets/{budget_id}/payees')
        payees_data = response.get('data', {}).get('payees', [])
        payees = [Payee.from_api(payee_data) for payee_data in payees_data]
        
        # Cache payee IDs
        for payee in payees:
            self.client._entity_cache['payees'][payee.name.lower()] = payee.id
            
        return payees
    
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
        """
        # Validate budget ID
        self.client._validate_budget_id(budget_id)
        
        try:
            response = self.client.get(f'budgets/{budget_id}/payees/{payee_id}')
            payee_data = response.get('data', {}).get('payee', {})
            payee = Payee.from_api(payee_data)
            
            # Cache payee ID
            self.client._entity_cache['payees'][payee.name.lower()] = payee.id
            
            return payee
        except ResourceNotFoundError:
            logger.error(f"Payee not found: {payee_id}")
            raise
    
    def find_payee_by_name(self, budget_id: str, payee_name: str) -> Optional[Payee]:
        """
        Find a payee by name.
        
        Args:
            budget_id: Budget ID
            payee_name: Payee name to search for
            
        Returns:
            Optional[Payee]: Payee if found, None otherwise
        """
        # Check cache first
        payee_name_lower = payee_name.lower()
        if payee_name_lower in self.client._entity_cache['payees']:
            payee_id = self.client._entity_cache['payees'][payee_name_lower]
            try:
                return self.get_payee(budget_id, payee_id)
            except ResourceNotFoundError:
                # Remove from cache if not found
                del self.client._entity_cache['payees'][payee_name_lower]
        
        # If not in cache, get all payees and search
        payees = self.get_payees(budget_id)
        for payee in payees:
            if payee.name.lower() == payee_name_lower:
                return payee
        
        return None 