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
from ..services.transaction_creation_service import TransactionCreationService
from ..services.transaction_query_service import TransactionQueryService
from ..services.transaction_cleanup_service import TransactionCleanupService

logger = logging.getLogger(__name__)

class NLPService:
    """
    Service for processing natural language queries related to financial data.
    This service delegates to specialized services based on the query intent.
    """

    def __init__(
        self, 
        transaction_service: TransactionService, 
        category_service: CategoryService,
        transaction_creation_service: TransactionCreationService,
        transaction_query_service: TransactionQueryService,
        transaction_cleanup_service: TransactionCleanupService
    ):
        """
        Initialize the NLPService.

        Args:
            transaction_service: Service for transaction operations
            category_service: Service for category operations
            transaction_creation_service: Service for transaction creation
            transaction_query_service: Service for transaction queries
            transaction_cleanup_service: Service for transaction cleanup
        """
        self.transaction_service = transaction_service
        self.category_service = category_service
        self.transaction_creation_service = transaction_creation_service
        self.transaction_query_service = transaction_query_service
        self.transaction_cleanup_service = transaction_cleanup_service
    
    def process_query(self, query: str, budget_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a natural language query.

        Args:
            query: The natural language query
            budget_id: The budget ID (optional)

        Returns:
            A dictionary with the query results
        """
        try:
            query_lower = query.lower()
            
            # Check for transaction creation
            if any(keyword in query_lower for keyword in ["create", "add", "new", "record"]) and any(keyword in query_lower for keyword in ["transaction", "expense", "purchase", "spending", "payment"]):
                return self.transaction_creation_service.process_transaction_creation(query, budget_id)
            
            # Check for transaction cleanup
            if any(keyword in query_lower for keyword in ["clean", "fix", "update", "correct"]) and any(keyword in query_lower for keyword in ["transaction", "payee", "memo"]):
                return self.transaction_cleanup_service.process_transaction_cleanup(query, budget_id)
            
            # Check for grocery-related queries
            if "grocery" in query_lower or "groceries" in query_lower:
                return self.transaction_query_service.process_grocery_query(budget_id, query)
            
            # Check for spending-related queries
            if any(keyword in query_lower for keyword in ["spend", "spent", "spending", "cost", "expense"]):
                return self.transaction_query_service.process_spending_query(budget_id, query)
            
            # Check for category-related queries
            if "category" in query_lower:
                return self.transaction_query_service.process_category_query(budget_id, query)
            
            # Check for time-based queries
            if any(period in query_lower for period in ["today", "yesterday", "week", "month", "year", "since"]):
                return self.transaction_query_service.process_time_query(budget_id, query)
            
            # Default to spending query
            return self.transaction_query_service.process_spending_query(budget_id, query)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "error": f"Error processing query: {e}",
                "transactions": [],
                "summary": "Error processing query"
            }
    
    def _process_transaction_creation(self, query: str, budget_id: str) -> Dict[str, Any]:
        """Process transaction creation queries"""
        from datetime import datetime, timedelta
        from decimal import Decimal
        import re
        from ..models.transaction import TransactionCreate, TransactionAmount
        
        try:
            # Extract amount - improved regex to capture dollar sign and full amount
            amount_match = re.search(r'\$\s*(\d+(?:\.\d+)?)', query)
            if not amount_match:
                return {"error": True, "summary": "Could not find a valid amount in your query. Please include a dollar amount like $10.50."}
            
            amount_str = amount_match.group(1)
            amount = Decimal(amount_str)
            
            # Extract payee directly with a more specific pattern
            payee_match = re.search(r'(?:at|from|to) ([A-Za-z0-9\s&\']+?)(?:\s+(?:for|in)\s+|\s+yesterday|\s+today|\s+tomorrow|\s+last week|$)', query)
            if not payee_match:
                return {"error": True, "summary": "Could not find a valid payee in your query. Please include 'at', 'from', or 'to' followed by the payee name."}
            
            payee_name = payee_match.group(1).strip()
            
            # Capitalize the first letter of each word in the payee name
            payee_name = ' '.join(word.capitalize() for word in payee_name.split())
            
            # Extract category information
            category_id = None
            category_name = None
            
            # Look for "for [category]" or "in [category]" patterns
            category_match = re.search(r'(?:for|in) ([A-Za-z0-9\s&]+)(?:category)?', query.lower())
            if category_match:
                potential_category = category_match.group(1).strip()
                
                # Try to match to an existing category
                category_match_result = self.category_service.find_best_category_match(budget_id, potential_category)
                
                if category_match_result.category:
                    category_id = category_match_result.category.id
                    category_name = category_match_result.category.name
                    logger.debug(f"Matched category: {category_name} (confidence: {category_match_result.confidence})")
            
            # Determine date
            date_obj = datetime.now().date()
            if "yesterday" in query.lower():
                date_obj = date_obj - timedelta(days=1)
            elif "tomorrow" in query.lower():
                date_obj = date_obj + timedelta(days=1)
            elif "last week" in query.lower():
                date_obj = date_obj - timedelta(days=7)
            
            # Get the first account
            accounts = self.transaction_service.transactions_api.client.accounts_api.get_accounts(budget_id)
            if not accounts:
                return {"error": True, "summary": "No accounts found in your budget"}
            
            account_id = accounts[0].id
            
            # Create transaction
            transaction_amount = TransactionAmount(amount=amount, is_outflow=True)
            
            # Format memo with AI-YY/MM/DD Created pattern
            current_date = datetime.now().strftime("%y/%m/%d")
            memo = f"AI-{current_date} Created"
            
            transaction = TransactionCreate(
                account_id=account_id,
                date=date_obj,
                amount=transaction_amount,
                payee_name=payee_name,
                category_id=category_id,
                memo=memo,
                cleared="uncleared",
                approved=False
            )
            
            # Create the transaction
            created_transaction = self.transaction_service.transactions_api.create_transaction(
                budget_id, transaction
            )
            
            # Include category in the success message if found
            category_info = f" in {category_name}" if category_name else ""
            
            return {
                "status": "success",
                "summary": f"Transaction created: ${amount} at {payee_name}{category_info} on {date_obj.strftime('%Y-%m-%d')}",
                "transaction": created_transaction
            }
            
        except Exception as e:
            logger.error(f"Error creating transaction: {e}")
            return {
                "error": True,
                "summary": f"Error creating transaction: {str(e)}"
            }
    
    def _process_transaction_cleanup(self, query: str, budget_id: str) -> Dict[str, Any]:
        """Process transaction cleanup queries to fix payee names and memo fields"""
        from datetime import datetime, timedelta
        import re
        
        try:
            # Extract date range for transactions to clean up
            start_date = None
            end_date = datetime.now().date()
            
            # Look for date range in query
            date_match = re.search(r'since ([A-Za-z]+ \d+,? \d{4})', query.lower())
            if date_match:
                date_str = date_match.group(1)
                try:
                    start_date = datetime.strptime(date_str, "%B %d, %Y").date()
                except ValueError:
                    try:
                        start_date = datetime.strptime(date_str, "%B %d %Y").date()
                    except ValueError:
                        logger.warning(f"Could not parse date: {date_str}")
            
            # If no specific date found, default to 30 days ago
            if not start_date:
                start_date = end_date - timedelta(days=30)
                
            # Get transactions in the date range
            transactions = self.transaction_service.get_transactions(
                budget_id, 
                since_date=start_date
            )
            
            # Filter for transactions that need cleanup
            transactions_to_fix = []
            
            for transaction in transactions:
                needs_fixing = False
                
                # Check for date terms in payee name
                payee_name = transaction.payee_name or ""
                if any(term in payee_name.lower() for term in ["yesterday", "today", "tomorrow", "last week"]):
                    needs_fixing = True
                
                # Check for old memo format
                memo = transaction.memo or ""
                if "natural language" in memo.lower() or "created via" in memo.lower():
                    needs_fixing = True
                
                if needs_fixing:
                    transactions_to_fix.append(transaction)
            
            # Fix the transactions
            fixed_count = 0
            for transaction in transactions_to_fix:
                try:
                    # Fix payee name
                    payee_name = transaction.payee_name or ""
                    original_payee = payee_name
                    
                    # Remove date terms
                    for term in ["yesterday", "today", "tomorrow", "last week"]:
                        if term in payee_name.lower():
                            payee_name = re.sub(r'\s*' + term + r'\s*', ' ', payee_name, flags=re.IGNORECASE).strip()
                    
                    # Fix memo
                    current_date = datetime.now().strftime("%y/%m/%d")
                    memo = f"AI-{current_date} Updated"
                    
                    # Only update if changes were made
                    if payee_name != original_payee or "natural language" in (transaction.memo or "").lower():
                        # Create update model
                        update = {
                            "payee_name": payee_name,
                            "memo": memo
                        }
                        
                        # Update the transaction
                        self.transaction_service.transactions_api.update_transaction(
                            budget_id, 
                            transaction.id, 
                            update
                        )
                        
                        fixed_count += 1
                except Exception as e:
                    logger.error(f"Error fixing transaction {transaction.id}: {e}")
            
            return {
                "status": "success",
                "summary": f"Cleaned up {fixed_count} transactions from {start_date} to {end_date}",
                "transactions_fixed": fixed_count,
                "total_transactions": len(transactions_to_fix)
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up transactions: {e}")
            return {
                "error": True,
                "summary": f"Error cleaning up transactions: {str(e)}"
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