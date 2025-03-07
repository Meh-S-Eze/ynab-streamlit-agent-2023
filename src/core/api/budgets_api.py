"""
Budgets API client for interacting with the YNAB API.

This module provides a client for accessing budget-related endpoints in the YNAB API.
"""

import logging
import os
from typing import Dict, List, Any

from ..models.budget import Budget

logger = logging.getLogger(__name__)

# Get the development budget ID from environment variables
DEV_BUDGET_ID = os.environ.get('YNAB_BUDGET_DEV', '7c8d67c8-ed70-4ba8-a25e-931a2f294167')

class BudgetsAPI:
    """
    API client for budget-related endpoints in the YNAB API.
    """
    
    def __init__(self, client):
        """
        Initialize the BudgetsAPI client.
        
        Args:
            client: The base client to use for API requests
        """
        self.client = client
    
    def get_budgets(self, include_accounts: bool = False) -> List[Budget]:
        """
        Get a list of budgets.
        
        Args:
            include_accounts: Whether to include account data
            
        Returns:
            List[Budget]: List of budgets
        """
        params = {'include_accounts': str(include_accounts).lower()}
        response = self.client.get('budgets', params=params)
        
        budgets_data = response.get('data', {}).get('budgets', [])
        
        # Filter to only include the development budget
        filtered_budgets = [b for b in budgets_data if b.get('id') == DEV_BUDGET_ID]
        
        return [Budget.from_api(budget_data) for budget_data in filtered_budgets]
    
    def get_budget(self, budget_id: str) -> Budget:
        """
        Get a single budget by ID.
        
        Args:
            budget_id: Budget ID
            
        Returns:
            Budget: Budget object
            
        Raises:
            ResourceNotFoundError: If budget not found
        """
        # Validate budget ID
        self.client._validate_budget_id(budget_id)
        
        try:
            response = self.client.get(f'budgets/{budget_id}')
            budget_data = response.get('data', {}).get('budget', {})
            return Budget.from_api(budget_data)
        except Exception as e:
            logger.error(f"Budget not found: {budget_id}")
            raise
    
    def get_budget_settings(self, budget_id: str) -> Dict[str, Any]:
        """
        Get settings for a budget.
        
        Args:
            budget_id: Budget ID
            
        Returns:
            Dict: Budget settings
        """
        # Validate budget ID
        self.client._validate_budget_id(budget_id)
        
        response = self.client.get(f'budgets/{budget_id}/settings')
        return response.get('data', {}).get('settings', {})
