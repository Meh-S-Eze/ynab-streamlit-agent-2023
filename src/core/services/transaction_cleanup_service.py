"""
Transaction cleanup service for natural language processing.

This module provides functionality for cleaning up transactions based on natural language queries.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from ..models.transaction import Transaction
from ..services.transaction_service import TransactionService

logger = logging.getLogger(__name__)

class TransactionCleanupService:
    """
    Service for cleaning up transactions based on natural language queries.
    
    This service interprets natural language queries and performs cleanup operations
    on transactions, such as fixing payee names and memo fields.
    """
    
    def __init__(self, transaction_service: TransactionService):
        """
        Initialize the transaction cleanup service.
        
        Args:
            transaction_service: Service for transaction operations
        """
        self.transaction_service = transaction_service
    
    def process_transaction_cleanup(self, query: str, budget_id: str) -> Dict[str, Any]:
        """
        Process transaction cleanup queries to fix payee names and memo fields.
        
        Args:
            query: Natural language query string
            budget_id: Budget ID to clean transactions in
            
        Returns:
            Dict[str, Any]: Results of the transaction cleanup
        """
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