"""
Category service for YNAB categories.

This module provides functionality for working with YNAB categories,
including category matching, lookup, and hierarchy management.
"""

import logging
import re
from typing import List, Dict, Optional, Any, Tuple
from difflib import SequenceMatcher

from ..models.category import Category, CategoryGroup, CategoryMatch
from ..api.categories_api import CategoriesAPI
from ..utils.caching import cached_method

logger = logging.getLogger(__name__)

class CategoryService:
    """
    Service for working with YNAB categories.
    
    This service provides methods for category matching, lookup, and hierarchy management.
    """
    
    def __init__(self, categories_api: CategoriesAPI):
        """
        Initialize the category service.
        
        Args:
            categories_api: API client for category-related endpoints
        """
        self.categories_api = categories_api
        self._category_cache: Dict[str, Dict[str, Any]] = {}
    
    def clear_cache(self, budget_id: Optional[str] = None) -> None:
        """
        Clear the category cache.
        
        Args:
            budget_id: Budget ID to clear cache for, or None to clear all
        """
        if budget_id:
            if budget_id in self._category_cache:
                self._category_cache[budget_id] = {}
        else:
            self._category_cache = {}
        
        logger.debug(f"Cleared category cache for budget_id={budget_id or 'all'}")
    
    @cached_method(maxsize=10, ttl=300)  # Cache for 5 minutes
    def get_categories(self, budget_id: str) -> List[CategoryGroup]:
        """
        Get all categories for a budget.
        
        Args:
            budget_id: Budget ID
            
        Returns:
            List[CategoryGroup]: List of category groups with their categories
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
        """
        return self.categories_api.get_category(budget_id, category_id)
    
    def get_category_by_name(self, budget_id: str, category_name: str, 
                           group_name: Optional[str] = None) -> Optional[Category]:
        """
        Get a category by name.
        
        Args:
            budget_id: Budget ID
            category_name: Category name
            group_name: Optional group name to narrow search
            
        Returns:
            Optional[Category]: Category if found, None otherwise
        """
        return self.categories_api.find_category_by_name(budget_id, category_name, group_name)
    
    def get_category_hierarchy(self, budget_id: str) -> Dict[str, List[Category]]:
        """
        Get the category hierarchy for a budget.
        
        Args:
            budget_id: Budget ID
            
        Returns:
            Dict[str, List[Category]]: Dictionary mapping group names to lists of categories
        """
        category_groups = self.get_categories(budget_id)
        hierarchy = {}
        
        for group in category_groups:
            hierarchy[group.name] = group.categories
        
        return hierarchy
    
    def get_subcategories(self, budget_id: str, group_id: str) -> List[Category]:
        """
        Get subcategories for a category group.
        
        Args:
            budget_id: Budget ID
            group_id: Category group ID
            
        Returns:
            List[Category]: List of categories in the group
        """
        category_groups = self.get_categories(budget_id)
        
        for group in category_groups:
            if group.id == group_id:
                return group.categories
        
        return []
    
    def find_best_category_match(self, budget_id: str, category_name: str,
                               group_name: Optional[str] = None) -> CategoryMatch:
        """
        Find the best matching category for a given name.
        
        Args:
            budget_id: Budget ID
            category_name: Category name to match
            group_name: Optional group name to narrow search
            
        Returns:
            CategoryMatch: Best matching category with confidence score
        """
        if not category_name:
            return CategoryMatch(
                category=None,
                confidence=0.0,
                exact_match=False
            )
        
        # First try exact match
        exact_match = self.get_category_by_name(budget_id, category_name, group_name)
        if exact_match:
            return CategoryMatch(
                category=exact_match,
                confidence=1.0,
                exact_match=True
            )
        
        # If no exact match, try fuzzy matching
        category_groups = self.get_categories(budget_id)
        best_match = None
        best_score = 0.0
        
        # Normalize the input category name for better matching
        normalized_name = self._normalize_category_name(category_name)
        
        for group in category_groups:
            # Skip if group name is specified and doesn't match
            if group_name and group.name.lower() != group_name.lower():
                continue
                
            for category in group.categories:
                # Skip hidden categories
                if category.hidden:
                    continue
                
                # Normalize the category name
                normalized_category = self._normalize_category_name(category.name)
                
                # Calculate similarity score
                score = self._calculate_similarity(normalized_name, normalized_category)
                
                if score > best_score:
                    best_score = score
                    best_match = category
        
        # Require a minimum confidence threshold
        if best_score < 0.6:
            return CategoryMatch(
                category=None,
                confidence=best_score,
                exact_match=False
            )
        
        return CategoryMatch(
            category=best_match,
            confidence=best_score,
            exact_match=False
        )
    
    def _normalize_category_name(self, name: str) -> str:
        """
        Normalize a category name for better matching.
        
        Args:
            name: Category name to normalize
            
        Returns:
            str: Normalized category name
        """
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower()
        
        # Remove special characters
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if not str1 or not str2:
            return 0.0
        
        # Use SequenceMatcher for fuzzy matching
        return SequenceMatcher(None, str1, str2).ratio()
