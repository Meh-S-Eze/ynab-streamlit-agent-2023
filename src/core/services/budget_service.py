"""
Budget service for YNAB budgets.

This module provides functionality for working with YNAB budgets,
including budget operations and analysis.
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import date, datetime, timedelta
from collections import defaultdict

from ..models.budget import Budget, BudgetSummary, SpendingAnalysis
from ..models.transaction import Transaction
from ..api.budgets_api import BudgetsAPI
from ..services.transaction_service import TransactionService
from ..utils.date_utils import DateFormatter

logger = logging.getLogger(__name__)

class BudgetService:
    """
    Service for working with YNAB budgets.
    
    This service provides methods for budget operations and analysis.
    """
    
    def __init__(self, 
                budgets_api: BudgetsAPI,
                transaction_service: TransactionService):
        """
        Initialize the budget service.
        
        Args:
            budgets_api: API client for budget-related endpoints
            transaction_service: Service for transaction operations
        """
        self.budgets_api = budgets_api
        self.transaction_service = transaction_service
    
    def get_budgets(self, include_accounts: bool = False) -> List[Budget]:
        """
        Get all budgets.
        
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
        """
        return self.budgets_api.get_budget(budget_id)
    
    def get_budget_settings(self, budget_id: str) -> Dict[str, Any]:
        """
        Get settings for a budget.
        
        Args:
            budget_id: Budget ID
            
        Returns:
            Dict[str, Any]: Budget settings
        """
        return self.budgets_api.get_budget_settings(budget_id)
    
    def analyze_spending(self, 
                       budget_id: str, 
                       start_date: Optional[date] = None,
                       end_date: Optional[date] = None) -> SpendingAnalysis:
        """
        Analyze spending for a budget.
        
        Args:
            budget_id: Budget ID
            start_date: Start date for analysis (defaults to 30 days ago)
            end_date: End date for analysis (defaults to today)
            
        Returns:
            SpendingAnalysis: Spending analysis results
        """
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.now().date()
        
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Format dates for API
        since_date = DateFormatter.format_date(start_date)
        
        # Get transactions for the date range
        transactions = self.transaction_service.get_transactions(
            budget_id, since_date=since_date
        )
        
        # Filter transactions to the date range
        filtered_transactions = [
            t for t in transactions 
            if DateFormatter.parse_date(t.date) <= end_date
        ]
        
        # Calculate total spent (outflows are negative in YNAB)
        total_spent = sum(
            abs(t.amount) / 1000 
            for t in filtered_transactions 
            if t.amount < 0 and not t.transfer_account_id
        )
        
        # Calculate category breakdown
        category_breakdown = defaultdict(float)
        for transaction in filtered_transactions:
            if transaction.amount < 0 and transaction.category_name and not transaction.transfer_account_id:
                category_breakdown[transaction.category_name] += abs(transaction.amount) / 1000
        
        # Sort categories by amount
        sorted_categories = dict(
            sorted(category_breakdown.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Find unusual transactions (outliers)
        # For simplicity, we'll define unusual as transactions larger than 2x the average
        outflow_amounts = [
            abs(t.amount) / 1000 
            for t in filtered_transactions 
            if t.amount < 0 and not t.transfer_account_id
        ]
        
        if outflow_amounts:
            average_amount = sum(outflow_amounts) / len(outflow_amounts)
            threshold = average_amount * 2
            
            unusual_transactions = [
                {
                    "date": t.date,
                    "payee": t.payee_name or "Unknown",
                    "category": t.category_name or "Uncategorized",
                    "amount": abs(t.amount) / 1000,
                    "memo": t.memo or ""
                }
                for t in filtered_transactions
                if t.amount < 0 and abs(t.amount) / 1000 > threshold and not t.transfer_account_id
            ]
        else:
            unusual_transactions = []
        
        # Create spending analysis
        return SpendingAnalysis(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            total_spent=total_spent,
            category_breakdown=sorted_categories,
            unusual_transactions=unusual_transactions,
            transaction_count=len(filtered_transactions)
        )
    
    def get_monthly_budget_data(self, 
                              budget_id: str, 
                              year: int, 
                              month: int) -> Dict[str, Any]:
        """
        Get budget data for a specific month.
        
        Args:
            budget_id: Budget ID
            year: Year
            month: Month (1-12)
            
        Returns:
            Dict[str, Any]: Monthly budget data
        """
        # This would typically call a YNAB API endpoint for monthly budget data
        # For now, we'll simulate it with transaction data
        
        # Create date range for the month
        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = date(year, month + 1, 1) - timedelta(days=1)
        
        # Get transactions for the month
        since_date = DateFormatter.format_date(start_date)
        transactions = self.transaction_service.get_transactions(
            budget_id, since_date=since_date
        )
        
        # Filter transactions to the month
        month_transactions = [
            t for t in transactions 
            if DateFormatter.parse_date(t.date) <= end_date
        ]
        
        # Calculate income and expenses
        income = sum(
            t.amount / 1000 
            for t in month_transactions 
            if t.amount > 0 and not t.transfer_account_id
        )
        
        expenses = sum(
            abs(t.amount) / 1000 
            for t in month_transactions 
            if t.amount < 0 and not t.transfer_account_id
        )
        
        # Calculate category spending
        category_spending = defaultdict(float)
        for transaction in month_transactions:
            if transaction.amount < 0 and transaction.category_name and not transaction.transfer_account_id:
                category_spending[transaction.category_name] += abs(transaction.amount) / 1000
        
        # Return monthly budget data
        return {
            "month": f"{year}-{month:02d}",
            "income": income,
            "expenses": expenses,
            "net": income - expenses,
            "category_spending": dict(category_spending),
            "transaction_count": len(month_transactions)
        }
