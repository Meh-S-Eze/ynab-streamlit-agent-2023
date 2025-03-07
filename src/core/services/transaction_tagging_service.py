"""
Transaction tagging service for YNAB transactions.

This module provides functionality for AI-based tagging of transactions,
including tag detection, application, and management.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

from ..models.transaction import Transaction, TransactionUpdate
from ..services.ai_tagging_service import AITaggingService

logger = logging.getLogger(__name__)

class TransactionTaggingService:
    """
    Service for AI-based tagging of transactions.
    
    This service provides methods for detecting, applying, and managing AI tags
    in transaction memos.
    """
    
    def __init__(self, ai_tagging_service: AITaggingService):
        """
        Initialize the transaction tagging service.
        
        Args:
            ai_tagging_service: Service for AI tagging operations
        """
        self.ai_tagging_service = ai_tagging_service
        
    def detect_ai_tag(self, transaction: Transaction) -> Optional[Dict[str, Any]]:
        """
        Detect if a transaction has an AI tag in its memo.
        
        Args:
            transaction: Transaction to check for AI tag
            
        Returns:
            Optional[Dict[str, Any]]: Detected tag data or None if no tag found
        """
        if not transaction.memo:
            return None
            
        return self.ai_tagging_service.detect_ai_tag(transaction.memo)
    
    def has_ai_tag(self, transaction: Transaction) -> bool:
        """
        Check if a transaction has an AI tag.
        
        Args:
            transaction: Transaction to check
            
        Returns:
            bool: True if transaction has an AI tag, False otherwise
        """
        return self.detect_ai_tag(transaction) is not None
    
    def apply_ai_tag(self, 
                   transaction: Transaction, 
                   action_type: str, 
                   action_data: Optional[Dict[str, Any]] = None) -> Transaction:
        """
        Apply an AI tag to a transaction.
        
        Args:
            transaction: Transaction to tag
            action_type: Type of action (e.g., 'categorize', 'split')
            action_data: Additional data for the tag
            
        Returns:
            Transaction: Updated transaction with AI tag
        """
        # Create a copy of the transaction to avoid modifying the original
        updated_transaction = Transaction(**transaction.dict())
        
        # Get the current memo or empty string if None
        memo = updated_transaction.memo or ""
        
        # Apply the AI tag
        tagged_memo = self.ai_tagging_service.apply_ai_tag(
            memo, action_type, action_data
        )
        
        # Update the transaction memo
        updated_transaction.memo = tagged_memo
        
        return updated_transaction
    
    def update_ai_tag(self, 
                    transaction: Transaction, 
                    action_type: str, 
                    action_data: Optional[Dict[str, Any]] = None) -> Transaction:
        """
        Update an existing AI tag or apply a new one if none exists.
        
        Args:
            transaction: Transaction to update
            action_type: Type of action (e.g., 'categorize', 'split')
            action_data: Additional data for the tag
            
        Returns:
            Transaction: Updated transaction with AI tag
        """
        # Create a copy of the transaction to avoid modifying the original
        updated_transaction = Transaction(**transaction.dict())
        
        # Get the current memo or empty string if None
        memo = updated_transaction.memo or ""
        
        # Update the AI tag
        tagged_memo = self.ai_tagging_service.update_ai_tag(
            memo, action_type, action_data
        )
        
        # Update the transaction memo
        updated_transaction.memo = tagged_memo
        
        return updated_transaction
    
    def remove_ai_tag(self, transaction: Transaction) -> Transaction:
        """
        Remove AI tag from a transaction.
        
        Args:
            transaction: Transaction to remove tag from
            
        Returns:
            Transaction: Updated transaction without AI tag
        """
        # Create a copy of the transaction to avoid modifying the original
        updated_transaction = Transaction(**transaction.dict())
        
        # Get the current memo or empty string if None
        memo = updated_transaction.memo or ""
        
        # Remove the AI tag
        clean_memo = self.ai_tagging_service.remove_ai_tag(memo)
        
        # Update the transaction memo
        updated_transaction.memo = clean_memo
        
        return updated_transaction
    
    def tag_transactions_batch(self, 
                             transactions: List[Transaction], 
                             action_type: str,
                             action_data_provider: Callable[[Transaction], Dict[str, Any]]) -> List[Transaction]:
        """
        Apply AI tags to a batch of transactions.
        
        Args:
            transactions: List of transactions to tag
            action_type: Type of action (e.g., 'categorize', 'split')
            action_data_provider: Function that returns action data for each transaction
            
        Returns:
            List[Transaction]: Updated transactions with AI tags
        """
        updated_transactions = []
        
        for transaction in transactions:
            try:
                # Get action data for this transaction
                action_data = action_data_provider(transaction)
                
                # Apply the tag
                tagged_transaction = self.apply_ai_tag(
                    transaction, action_type, action_data
                )
                
                updated_transactions.append(tagged_transaction)
                
            except Exception as e:
                logger.error(f"Error tagging transaction {transaction.id}: {e}")
                # Keep the original transaction if tagging fails
                updated_transactions.append(transaction)
        
        return updated_transactions 