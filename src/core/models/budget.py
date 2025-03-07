"""
Budget data models for YNAB integration.

This module contains Pydantic models for YNAB budget data structures.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Annotated, Any
from datetime import date
from decimal import Decimal


class BudgetSummary(BaseModel):
    """Model for a YNAB budget summary"""
    id: Annotated[str, Field(description="Budget ID")]
    name: Annotated[str, Field(description="Budget name")]
    last_modified_on: Annotated[Optional[str], Field(None, description="Last modified timestamp")]
    currency_format: Annotated[Dict[str, Any], Field(description="Currency format settings")]
    date_format: Annotated[Dict[str, Any], Field(description="Date format settings")]
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'BudgetSummary':
        """Create instance from YNAB API response"""
        return cls(
            id=data['id'],
            name=data['name'],
            last_modified_on=data.get('last_modified_on'),
            currency_format=data.get('currency_format', {}),
            date_format=data.get('date_format', {})
        )


class Budget(BaseModel):
    """Model for a complete YNAB budget"""
    id: Annotated[str, Field(description="Budget ID")]
    name: Annotated[str, Field(description="Budget name")]
    last_modified_on: Annotated[Optional[str], Field(None, description="Last modified timestamp")]
    currency_format: Annotated[Dict[str, Any], Field(description="Currency format settings")]
    date_format: Annotated[Dict[str, Any], Field(description="Date format settings")]
    accounts: Annotated[List[Dict[str, Any]], Field(default_factory=list, description="Budget accounts")]
    categories: Annotated[List[Dict[str, Any]], Field(default_factory=list, description="Budget categories")]
    category_groups: Annotated[List[Dict[str, Any]], Field(default_factory=list, description="Budget category groups")]
    months: Annotated[List[Dict[str, Any]], Field(default_factory=list, description="Budget months")]
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'Budget':
        """Create instance from YNAB API response"""
        # Handle the case where the budget data is directly available
        # without being nested in a 'budget' field
        if 'id' in data:
            # This is a direct budget object
            return cls(
                id=data['id'],
                name=data['name'],
                last_modified_on=data.get('last_modified_on'),
                currency_format=data.get('currency_format', {}),
                date_format=data.get('date_format', {}),
                accounts=data.get('accounts', []),
                categories=data.get('categories', []),
                category_groups=data.get('category_groups', []),
                months=data.get('months', [])
            )
        else:
            # Fall back to the standard behavior
            budget_data = data.get('budget', {})
            return cls(
                id=budget_data['id'],
                name=budget_data['name'],
                last_modified_on=budget_data.get('last_modified_on'),
                currency_format=budget_data.get('currency_format', {}),
                date_format=budget_data.get('date_format', {}),
                accounts=budget_data.get('accounts', []),
                categories=budget_data.get('categories', []),
                category_groups=budget_data.get('category_groups', []),
                months=budget_data.get('months', [])
            )
    
    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> 'Budget':
        """Alias for from_api_response for compatibility"""
        return cls.from_api_response(data)


class SpendingAnalysis(BaseModel):
    """Model for spending analysis results"""
    start_date: Annotated[str, Field(description="Start date for analysis")]
    end_date: Annotated[str, Field(description="End date for analysis")]
    total_spent: Annotated[float, Field(description="Total spending amount")]
    category_breakdown: Annotated[Dict[str, float], Field(description="Spending by category")]
    unusual_transactions: Annotated[List[Dict[str, Any]], Field(description="Unusual transactions")]
    transaction_count: Annotated[int, Field(description="Total number of transactions analyzed")]


class BudgetMonth(BaseModel):
    """Model for a YNAB budget month"""
    month: Annotated[date, Field(description="Budget month")]
    income: Annotated[int, Field(description="Income for the month in milliunits")]
    budgeted: Annotated[int, Field(description="Budgeted amount for the month in milliunits")]
    activity: Annotated[int, Field(description="Activity amount for the month in milliunits")]
    to_be_budgeted: Annotated[int, Field(description="Amount to be budgeted in milliunits")]
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'BudgetMonth':
        """Create instance from YNAB API response"""
        return cls(
            month=date.fromisoformat(data['month']),
            income=data.get('income', 0),
            budgeted=data.get('budgeted', 0),
            activity=data.get('activity', 0),
            to_be_budgeted=data.get('to_be_budgeted', 0)
        )
