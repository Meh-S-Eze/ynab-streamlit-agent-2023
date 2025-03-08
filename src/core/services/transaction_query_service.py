"""
Transaction query service for natural language processing.

This module provides functionality for processing natural language queries
related to transactions and returning relevant data.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import date, datetime, timedelta

from ..models.transaction import Transaction
from ..services.transaction_service import TransactionService
from ..services.category_service import CategoryService

logger = logging.getLogger(__name__)

class TransactionQueryService:
    """
    Service for processing natural language queries about transactions.
    
    This service interprets natural language queries and returns relevant transaction data
    from the YNAB API based on the query intent and parameters.
    """
    
    def __init__(self,
                transaction_service: TransactionService,
                category_service: CategoryService):
        """
        Initialize the transaction query service.
        
        Args:
            transaction_service: Service for transaction operations
            category_service: Service for category operations
        """
        self.transaction_service = transaction_service
        self.category_service = category_service
    
    def process_grocery_query(self, budget_id: str, query: str) -> Dict[str, Any]:
        """
        Process a query about grocery spending.

        Args:
            budget_id: The budget ID
            query: The natural language query

        Returns:
            A dictionary with the query results
        """
        # Add "grocery" to the query to ensure we filter for grocery-related transactions
        modified_query = query
        if "grocery" not in query.lower():
            modified_query = f"grocery {query}"
        
        return self.process_spending_query(budget_id, modified_query)
    
    def process_spending_query(self, budget_id: str, query: str) -> Dict[str, Any]:
        """
        Process a query about spending.

        Args:
            budget_id: The budget ID
            query: The natural language query

        Returns:
            A dictionary with the query results
        """
        try:
            # Extract time period from query
            time_period = self._extract_time_period(query)
            since_date = self._get_date_from_period(time_period)
            
            # Extract category or payee information
            category_name = self._extract_category(query)
            payee_name = self._extract_payee(query)
            
            # Get transactions
            transactions = self.transaction_service.get_transactions(
                budget_id=budget_id,
                since_date=since_date.strftime("%Y-%m-%d") if since_date else None
            )
            
            # Filter transactions based on category and/or payee
            filtered_transactions = []
            for transaction in transactions:
                if category_name and transaction.category_name and category_name.lower() in transaction.category_name.lower():
                    filtered_transactions.append(transaction)
                elif payee_name and transaction.payee_name and payee_name.lower() in transaction.payee_name.lower():
                    filtered_transactions.append(transaction)
                elif not category_name and not payee_name:
                    # If no specific category or payee was mentioned, include all transactions
                    filtered_transactions.append(transaction)
            
            # Calculate total spending
            total_spending = sum(t.amount for t in filtered_transactions if t.amount < 0)
            
            return {
                "transactions": filtered_transactions,
                "total_spending": abs(total_spending),
                "time_period": time_period,
                "category": category_name,
                "payee": payee_name,
                "summary": f"Found {len(filtered_transactions)} {'grocery ' if 'grocery' in query.lower() else ''}transactions"
            }
            
        except Exception as e:
            logger.error(f"Error processing spending query: {e}")
            return {
                "error": f"Error processing spending query: {e}",
                "transactions": [],
                "total_spending": 0,
                "summary": "Error processing query"
            }
    
    def process_category_query(self, budget_id: str, query: str) -> Dict[str, Any]:
        """
        Process a query about category spending.

        Args:
            budget_id: The budget ID
            query: The natural language query

        Returns:
            A dictionary with the query results
        """
        try:
            # Extract category from query
            category_name = self._extract_category(query)
            if not category_name:
                return {
                    "error": "No category found in query",
                    "transactions": [],
                    "summary": "No category found in query"
                }
            
            # Extract time period from query
            time_period = self._extract_time_period(query)
            since_date = self._get_date_from_period(time_period)
            
            # Get transactions
            transactions = self.transaction_service.get_transactions(
                budget_id=budget_id,
                since_date=since_date.strftime("%Y-%m-%d") if since_date else None
            )
            
            # Filter transactions based on category
            category_transactions = [
                t for t in transactions 
                if t.category_name and category_name.lower() in t.category_name.lower()
            ]
            
            # Calculate total spending in category
            total_spending = sum(t.amount for t in category_transactions if t.amount < 0)
            
            return {
                "transactions": category_transactions,
                "total_spending": abs(total_spending),
                "category": category_name,
                "time_period": time_period,
                "summary": f"Found {len(category_transactions)} transactions in category '{category_name}'"
            }
            
        except Exception as e:
            logger.error(f"Error processing category query: {e}")
            return {
                "error": f"Error processing category query: {e}",
                "transactions": [],
                "summary": "Error processing query"
            }
    
    def process_time_query(self, budget_id: str, query: str) -> Dict[str, Any]:
        """
        Process a query about transactions in a specific time period.

        Args:
            budget_id: The budget ID
            query: The natural language query

        Returns:
            A dictionary with the query results
        """
        try:
            # Extract time period from query
            time_period = self._extract_time_period(query)
            since_date = self._get_date_from_period(time_period)
            
            if not since_date:
                return {
                    "error": "No time period found in query",
                    "transactions": [],
                    "summary": "No time period found in query"
                }
            
            # Get transactions
            transactions = self.transaction_service.get_transactions(
                budget_id=budget_id,
                since_date=since_date.strftime("%Y-%m-%d")
            )
            
            return {
                "transactions": transactions,
                "time_period": time_period,
                "summary": f"Found {len(transactions)} transactions since {since_date.strftime('%Y-%m-%d')}"
            }
            
        except Exception as e:
            logger.error(f"Error processing time query: {e}")
            return {
                "error": f"Error processing time query: {e}",
                "transactions": [],
                "summary": "Error processing query"
            }
    
    def _extract_time_period(self, query: str) -> str:
        """
        Extract time period from query.

        Args:
            query: The natural language query

        Returns:
            The extracted time period
        """
        time_periods = [
            "today", "yesterday", "this week", "last week", 
            "this month", "last month", "this year", "last year",
            "past week", "past month", "past year", "last 7 days",
            "last 30 days", "last 90 days", "last 12 months"
        ]
        
        for period in time_periods:
            if period in query.lower():
                return period
        
        # Check for "since [date]" pattern
        since_match = re.search(r'since ([A-Za-z]+ \d+,? \d{4})', query, re.IGNORECASE)
        if since_match:
            return f"since {since_match.group(1)}"
        
        # Default to "last month" if no time period is found
        return "last month"
    
    def _get_date_from_period(self, time_period: str) -> Optional[datetime]:
        """
        Convert a time period string to a datetime object.

        Args:
            time_period: The time period string

        Returns:
            The corresponding datetime object
        """
        today = datetime.now()
        
        if time_period == "today":
            return today
        elif time_period == "yesterday":
            return today - timedelta(days=1)
        elif time_period in ["this week", "past week"]:
            return today - timedelta(days=7)
        elif time_period == "last week":
            return today - timedelta(days=14)
        elif time_period in ["this month", "past month"]:
            return today - timedelta(days=30)
        elif time_period == "last month":
            return today - timedelta(days=60)
        elif time_period in ["this year", "past year"]:
            return today - timedelta(days=365)
        elif time_period == "last year":
            return today - timedelta(days=730)
        elif time_period == "last 7 days":
            return today - timedelta(days=7)
        elif time_period == "last 30 days":
            return today - timedelta(days=30)
        elif time_period == "last 90 days":
            return today - timedelta(days=90)
        elif time_period == "last 12 months":
            return today - timedelta(days=365)
        elif time_period.startswith("since "):
            try:
                date_str = time_period[6:]
                return datetime.strptime(date_str, "%B %d, %Y")
            except ValueError:
                try:
                    return datetime.strptime(date_str, "%B %d %Y")
                except ValueError:
                    logger.error(f"Could not parse date: {date_str}")
                    return today - timedelta(days=30)  # Default to last 30 days
        
        # Default to last month
        return today - timedelta(days=30)
    
    def _extract_category(self, query: str) -> Optional[str]:
        """
        Extract category from query.

        Args:
            query: The natural language query

        Returns:
            The extracted category name
        """
        # Check for "category [name]" pattern
        category_match = re.search(r'category (\w+)', query, re.IGNORECASE)
        if category_match:
            return category_match.group(1)
        
        # Check for common categories in the query
        common_categories = [
            "groceries", "dining", "restaurants", "food", "entertainment", 
            "utilities", "rent", "mortgage", "transportation", "gas",
            "shopping", "clothing", "health", "medical", "education"
        ]
        
        for category in common_categories:
            if category in query.lower():
                return category
        
        return None
    
    def _extract_payee(self, query: str) -> Optional[str]:
        """
        Extract payee from query.

        Args:
            query: The natural language query

        Returns:
            The extracted payee name
        """
        # Check for "at [payee]" pattern
        payee_match = re.search(r'at ([A-Za-z0-9\s&\']+)', query, re.IGNORECASE)
        if payee_match:
            return payee_match.group(1).strip()
        
        # Check for "from [payee]" pattern
        payee_match = re.search(r'from ([A-Za-z0-9\s&\']+)', query, re.IGNORECASE)
        if payee_match:
            return payee_match.group(1).strip()
        
        # Check for "to [payee]" pattern
        payee_match = re.search(r'to ([A-Za-z0-9\s&\']+)', query, re.IGNORECASE)
        if payee_match:
            return payee_match.group(1).strip()
        
        return None 