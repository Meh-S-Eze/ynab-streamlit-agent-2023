"""
Transaction creation service for natural language processing.

This module provides functionality for creating transactions from natural language queries.
"""

import logging
import re
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal

from ..models.transaction import Transaction, TransactionCreate, TransactionAmount
from ..services.transaction_service import TransactionService
from ..services.category_service import CategoryService

logger = logging.getLogger(__name__)

class TransactionCreationService:
    """
    Service for creating transactions from natural language queries.
    
    This service interprets natural language queries and creates transactions
    in YNAB based on the extracted information.
    """
    
    def __init__(self,
                transaction_service: TransactionService,
                category_service: CategoryService):
        """
        Initialize the transaction creation service.
        
        Args:
            transaction_service: Service for transaction operations
            category_service: Service for category operations
        """
        self.transaction_service = transaction_service
        self.category_service = category_service
    
    def process_transaction_creation(self, query: str, budget_id: str) -> Dict[str, Any]:
        """
        Process transaction creation queries.
        
        Args:
            query: Natural language query string
            budget_id: Budget ID to create transaction in
            
        Returns:
            Dict[str, Any]: Results of the transaction creation
        """
        try:
            # Log the raw query for debugging
            logger.debug(f"Raw query for transaction creation: '{query}'")
            
            # First, clean the query by replacing escaped dollar signs with regular ones
            cleaned_query = query.replace('\\$', '$')
            logger.debug(f"Cleaned query: '{cleaned_query}'")
            
            # Extract amount with a simpler pattern
            amount_match = re.search(r'\$\s*(\d+(?:\.\d+)?)', cleaned_query)
            if not amount_match:
                # Try an alternative pattern without the dollar sign
                amount_match = re.search(r'(?:for|of)\s+(\d+(?:\.\d+)?)\s+(?:dollars|USD)?', cleaned_query, re.IGNORECASE)
                
            if not amount_match:
                return {"error": True, "summary": "Could not find a valid amount in your query. Please include a dollar amount like $10.50."}
            
            amount_str = amount_match.group(1)
            amount = Decimal(amount_str)
            
            # Log the extracted amount for debugging
            logger.debug(f"Extracted amount: {amount} from query: '{cleaned_query}'")
            
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