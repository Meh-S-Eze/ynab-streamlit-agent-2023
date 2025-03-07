"""
Natural Language Processing service for YNAB queries.

This module provides functionality for processing natural language queries
related to transactions, budgets, and other financial data.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import date, datetime, timedelta

from ..models.transaction import Transaction
from ..services.transaction_service import TransactionService
from ..services.category_service import CategoryService

logger = logging.getLogger(__name__)

class NLPService:
    """
    Service for processing natural language queries about financial data.
    
    This service interprets natural language queries and returns relevant data
    from the YNAB API based on the query intent and parameters.
    """
    
    def __init__(self,
                transaction_service: TransactionService,
                category_service: CategoryService):
        """
        Initialize the NLP service.
        
        Args:
            transaction_service: Service for transaction operations
            category_service: Service for category operations
        """
        self.transaction_service = transaction_service
        self.category_service = category_service
    
    def process_query(self, query: str, budget_id: str) -> Dict[str, Any]:
        """
        Process a natural language query about financial data.
        
        Args:
            query: Natural language query string
            budget_id: Budget ID to search within
            
        Returns:
            Dict[str, Any]: Results of the query processing
        """
        try:
            # For now, implement a simple keyword-based approach
            query_lower = query.lower()
            
            # Check for grocery-related queries
            if "grocery" in query_lower or "groceries" in query_lower:
                return self._process_grocery_query(query_lower, budget_id)
            
            # Check for spending-related queries
            elif "spending" in query_lower or "spent" in query_lower:
                return self._process_spending_query(query_lower, budget_id)
            
            # Check for category-related queries
            elif "categor" in query_lower:
                return self._process_category_query(query_lower, budget_id)
            
            # Check for time-based queries
            elif any(term in query_lower for term in ["last month", "this month", "last week", "this week"]):
                return self._process_time_based_query(query_lower, budget_id)
            
            # Default to a general transaction search
            else:
                return self._process_general_query(query_lower, budget_id)
                
        except Exception as e:
            logger.error(f"Error processing natural language query: {e}")
            return {
                "error": True,
                "summary": f"An error occurred while processing your query: {str(e)}"
            }
    
    def _process_grocery_query(self, query: str, budget_id: str) -> Dict[str, Any]:
        """Process queries related to grocery spending"""
        # Get all transactions
        transactions = self.transaction_service.get_transactions(budget_id)
        
        # Filter for grocery-related transactions
        grocery_transactions = [
            t for t in transactions 
            if (t.category_name and "grocery" in t.category_name.lower()) or
               (t.payee_name and any(store in t.payee_name.lower() for store in 
                                   ["grocery", "supermarket", "food", "market"]))
        ]
        
        # Format the results
        result = {
            "transactions": grocery_transactions,
            "summary": f"Found {len(grocery_transactions)} grocery transactions"
        }
        
        return result
    
    def _process_spending_query(self, query: str, budget_id: str) -> Dict[str, Any]:
        """Process queries related to overall spending"""
        # Get all transactions
        transactions = self.transaction_service.get_transactions(budget_id)
        
        # Filter for outflow transactions (expenses)
        expense_transactions = [t for t in transactions if t.amount < 0]
        
        # Calculate total spending (outflows are negative in YNAB)
        total_spent = sum(abs(t.amount) / 1000.0 for t in expense_transactions)
        
        # Format the results
        result = {
            "transactions": expense_transactions[:10],  # Limit to 10 for display
            "summary": f"Total spending: ${total_spent:.2f} across {len(expense_transactions)} transactions",
            "analysis": self._generate_spending_analysis(expense_transactions)
        }
        
        return result
    
    def _process_category_query(self, query: str, budget_id: str) -> Dict[str, Any]:
        """Process queries related to categories"""
        # Get all transactions
        transactions = self.transaction_service.get_transactions(budget_id)
        
        # Group transactions by category
        categories = {}
        for t in transactions:
            if t.category_name:
                if t.category_name not in categories:
                    categories[t.category_name] = []
                categories[t.category_name].append(t)
        
        # Format the results
        result = {
            "summary": f"Found transactions in {len(categories)} categories",
            "analysis": self._generate_category_analysis(categories)
        }
        
        return result
    
    def _process_time_based_query(self, query: str, budget_id: str) -> Dict[str, Any]:
        """Process queries with time-based filters"""
        # Determine date range based on query
        start_date, end_date = self._extract_date_range(query)
        
        # Get transactions within date range
        transactions = self.transaction_service.get_transactions(
            budget_id, 
            since_date=start_date
        )
        
        # Filter by end date if needed
        if end_date:
            transactions = [t for t in transactions if t.date <= end_date]
        
        # Format the results
        result = {
            "transactions": transactions[:10],  # Limit to 10 for display
            "summary": f"Found {len(transactions)} transactions from {start_date} to {end_date or 'now'}"
        }
        
        return result
    
    def _process_general_query(self, query: str, budget_id: str) -> Dict[str, Any]:
        """Process general transaction queries"""
        # Get all transactions
        transactions = self.transaction_service.get_transactions(budget_id)
        
        # Simple keyword matching
        keywords = [word for word in query.split() if len(word) > 3]
        
        if keywords:
            filtered_transactions = [
                t for t in transactions 
                if any(keyword in t.memo.lower() if t.memo else False or
                       keyword in t.payee_name.lower() if t.payee_name else False or
                       keyword in t.category_name.lower() if t.category_name else False
                       for keyword in keywords)
            ]
        else:
            filtered_transactions = transactions[:10]  # Just return recent transactions
        
        # Format the results
        result = {
            "transactions": filtered_transactions[:10],  # Limit to 10 for display
            "summary": f"Found {len(filtered_transactions)} transactions matching your query"
        }
        
        return result
    
    def _generate_spending_analysis(self, transactions: List[Transaction]) -> str:
        """Generate a simple analysis of spending patterns"""
        if not transactions:
            return "No transactions to analyze"
        
        # Group by category
        by_category = {}
        for t in transactions:
            category = t.category_name or "Uncategorized"
            if category not in by_category:
                by_category[category] = 0
            by_category[category] += abs(t.amount)
        
        # Find top categories
        top_categories = sorted(by_category.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Format analysis
        analysis = "Top spending categories: "
        for category, amount in top_categories:
            analysis += f"{category} (${amount/1000:.2f}), "
        
        return analysis.rstrip(", ")
    
    def _generate_category_analysis(self, categories: Dict[str, List[Transaction]]) -> str:
        """Generate a simple analysis of category spending"""
        if not categories:
            return "No categories to analyze"
        
        # Calculate total per category
        category_totals = {}
        for category, transactions in categories.items():
            category_totals[category] = sum(abs(t.amount) for t in transactions)
        
        # Find top categories
        top_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Format analysis
        analysis = "Top categories by spending: "
        for category, amount in top_categories:
            analysis += f"{category} (${amount/1000:.2f}), "
        
        return analysis.rstrip(", ")
    
    def _extract_date_range(self, query: str) -> tuple:
        """Extract date range from query text"""
        today = date.today()
        
        if "last month" in query:
            # First day of previous month
            first_day = date(today.year, today.month - 1 if today.month > 1 else 12, 1)
            # Last day of previous month
            if today.month > 1:
                last_day = date(today.year, today.month, 1) - timedelta(days=1)
            else:
                last_day = date(today.year - 1, 12, 31)
            return first_day, last_day
            
        elif "this month" in query:
            # First day of current month
            first_day = date(today.year, today.month, 1)
            return first_day, None
            
        elif "last week" in query:
            # 7 days ago
            first_day = today - timedelta(days=7)
            return first_day, today
            
        elif "this week" in query:
            # Start of current week (Monday)
            first_day = today - timedelta(days=today.weekday())
            return first_day, None
            
        else:
            # Default to last 30 days
            return today - timedelta(days=30), None 