"""
Categories API client for interacting with the YNAB API.

This module provides a client for accessing category-related endpoints in the YNAB API.
"""

import logging
from typing import Dict, List, Optional

from ..models.category import Category, CategoryGroup
from .base_client import ResourceNotFoundError

logger = logging.getLogger(__name__)

class CategoriesAPI:
    """
    API client for category-related endpoints in the YNAB API.
    """
    
    def __init__(self, client):
        """
        Initialize the CategoriesAPI client.
        
        Args:
            client: The base client to use for API requests
        """
        self.client = client
    
    def get_categories(self, budget_id: str) -> List[CategoryGroup]:
        """
        Get a list of categories for a budget.
        
        Args:
            budget_id: Budget ID
            
        Returns:
            List[CategoryGroup]: List of category groups with categories
        """
        # Validate budget ID
        self.client._validate_budget_id(budget_id)
        
        response = self.client.get(f'budgets/{budget_id}/categories')
        groups_data = response.get('data', {}).get('category_groups', [])
        groups = [CategoryGroup.from_api(group_data) for group_data in groups_data]
        
        # Cache category IDs
        for group in groups:
            for category in group.categories:
                cache_key = f"{group.name.lower()}:{category.name.lower()}"
                self.client._entity_cache['categories'][cache_key] = category.id
                # Also cache by just category name for simpler lookups
                self.client._entity_cache['categories'][category.name.lower()] = category.id
                
        return groups
    
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
        """
        # Validate budget ID
        self.client._validate_budget_id(budget_id)
        
        try:
            response = self.client.get(f'budgets/{budget_id}/categories/{category_id}')
            category_data = response.get('data', {}).get('category', {})
            category = Category.from_api(category_data)
            
            # Cache category ID
            self.client._entity_cache['categories'][category.name.lower()] = category.id
            
            return category
        except ResourceNotFoundError:
            logger.error(f"Category not found: {category_id}")
            raise
    
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
        """
        # Check cache first
        category_name_lower = category_name.lower()
        cache_key = f"{group_name.lower()}:{category_name_lower}" if group_name else category_name_lower
        
        if cache_key in self.client._entity_cache['categories']:
            category_id = self.client._entity_cache['categories'][cache_key]
            try:
                return self.get_category(budget_id, category_id)
            except ResourceNotFoundError:
                # Remove from cache if not found
                del self.client._entity_cache['categories'][cache_key]
        
        # If not in cache, get all categories and search
        groups = self.get_categories(budget_id)
        
        for group in groups:
            if group_name and group.name.lower() != group_name.lower():
                continue
                
            for category in group.categories:
                if category.name.lower() == category_name_lower:
                    return category
        
        return None
