"""
Category data models for YNAB integration.

This module contains Pydantic models for YNAB category data structures.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Annotated, Any


class CategoryUpdate(BaseModel):
    """Model for category update operations"""
    transaction_id: str
    category_id: str
    memo: Optional[str] = None


class Category(BaseModel):
    """Model for a YNAB category"""
    id: Annotated[str, Field(description="Category ID")]
    name: Annotated[str, Field(description="Category name")]
    group_id: Annotated[str, Field(description="Category group ID")]
    group_name: Annotated[Optional[str], Field(None, description="Category group name")]
    hidden: Annotated[bool, Field(default=False, description="Whether the category is hidden")]
    budgeted: Annotated[Optional[int], Field(None, description="Budgeted amount in milliunits")]
    activity: Annotated[Optional[int], Field(None, description="Activity amount in milliunits")]
    balance: Annotated[Optional[int], Field(None, description="Balance amount in milliunits")]
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any], group_name: Optional[str] = None) -> 'Category':
        """Create instance from YNAB API response"""
        return cls(
            id=data['id'],
            name=data['name'],
            group_id=data['category_group_id'],
            group_name=group_name,
            hidden=data.get('hidden', False),
            budgeted=data.get('budgeted'),
            activity=data.get('activity'),
            balance=data.get('balance')
        )
    
    @classmethod
    def from_api(cls, data: Dict[str, Any], group_name: Optional[str] = None) -> 'Category':
        """Alias for from_api_response for compatibility"""
        return cls.from_api_response(data, group_name)


class CategoryGroup(BaseModel):
    """Model for a YNAB category group"""
    id: Annotated[str, Field(description="Category group ID")]
    name: Annotated[str, Field(description="Category group name")]
    hidden: Annotated[bool, Field(default=False, description="Whether the group is hidden")]
    categories: Annotated[List[Category], Field(default_factory=list, description="Categories in this group")]
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'CategoryGroup':
        """Create instance from YNAB API response"""
        categories = []
        if 'categories' in data:
            categories = [
                Category.from_api_response(cat, data['name']) 
                for cat in data['categories']
            ]
            
        return cls(
            id=data['id'],
            name=data['name'],
            hidden=data.get('hidden', False),
            categories=categories
        )
    
    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> 'CategoryGroup':
        """Alias for from_api_response for compatibility"""
        return cls.from_api_response(data)


class CategoryMatch(BaseModel):
    """Model for category matching results"""
    category: Annotated[Optional[Category], Field(None, description="Matched category or None if no match")]
    confidence: Annotated[float, Field(ge=0, le=1, description="Match confidence score")]
    exact_match: Annotated[bool, Field(default=False, description="Whether this is an exact match")]
    
    @field_validator('confidence')
    def validate_confidence(cls, v):
        """Ensure confidence is between 0 and 1"""
        return max(0.0, min(1.0, float(v)))
