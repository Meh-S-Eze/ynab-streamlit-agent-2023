"""
Transaction-specific prompt templates for YNAB AI integrations.

This module provides specialized prompt templates for transaction-related
operations, including categorization and AI tagging.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from ..models.category import Category, CategoryMatch
from ..models.transaction import Transaction, TransactionUpdate
from .base_prompts import BasePrompt, BasePromptDependencies, BasePromptResult

logger = logging.getLogger(__name__)

class TransactionCategoryPrompt(BasePrompt):
    """
    Prompt template for transaction categorization.
    
    This class provides a specialized prompt template for categorizing
    transactions based on their details.
    """
    
    def __init__(self, examples_path: Optional[str] = None):
        """
        Initialize a transaction categorization prompt template.
        
        Args:
            examples_path: Optional path to a JSON file containing examples
        """
        system_message = """
        You are an expert financial categorization assistant. Your task is to analyze transaction data 
        and determine the most appropriate budget category for each transaction.
        
        Focus on these key elements:
        - Transaction payee name
        - Transaction memo
        - Transaction amount
        - Transaction date
        
        Consider patterns in previous transactions and common financial categories.
        If a transaction is ambiguous, select the most probable category based on
        the provided information. 
        
        Be specific in your categorization and avoid generic categories when
        more specific ones would be appropriate. For example, prefer "Groceries"
        over "Food" if the transaction is clearly at a grocery store.
        """
        
        # Use provided examples path or default
        examples_path = examples_path or "src/core/prompts/examples/transaction_examples.json"
        
        super().__init__(system_message=system_message, examples_path=examples_path)
    
    def format_example(self, example: Dict[str, Any]) -> str:
        """
        Format a transaction categorization example.
        
        Args:
            example: Example transaction and category data
            
        Returns:
            str: Formatted example string
        """
        transaction = example.get("transaction", {})
        category = example.get("category", {})
        
        formatted = "Transaction Details:\n"
        formatted += f"- Payee: {transaction.get('payee_name', 'Unknown')}\n"
        formatted += f"- Amount: {transaction.get('amount', 0)}\n"
        formatted += f"- Date: {transaction.get('date', '')}\n"
        
        if "memo" in transaction and transaction["memo"]:
            formatted += f"- Memo: {transaction['memo']}\n"
        
        formatted += f"\nCorrect Category: {category.get('name', 'Uncategorized')}\n"
        formatted += f"Category Group: {category.get('group_name', 'None')}\n"
        
        if "explanation" in example:
            formatted += f"\nExplanation: {example['explanation']}\n"
        
        return formatted + "\n---\n"


class TransactionTaggingPrompt(BasePrompt):
    """
    Prompt template for AI tagging of transactions.
    
    This class provides a specialized prompt template for generating
    appropriate AI tags for transactions.
    """
    
    def __init__(self, examples_path: Optional[str] = None):
        """
        Initialize a transaction tagging prompt template.
        
        Args:
            examples_path: Optional path to a JSON file containing examples
        """
        system_message = """
        You are an expert financial transaction tagger. Your task is to analyze transaction data 
        and generate appropriate AI tags that describe actions taken on the transaction.
        
        AI tags follow the format: [AI {action_type} {date}]
        
        Valid action types include:
        - "created" - When a transaction is first created by AI
        - "modified" - When an existing transaction is modified by AI
        - "categorized" - When a transaction's category is changed by AI
        - "split" - When a transaction is split by AI
        - "merged" - When transactions are merged by AI
        
        Tags should always include the current date in YYYY-MM-DD format.
        
        Preserve existing AI tags when possible, updating only the action type and date
        if a new action is performed on a previously tagged transaction.
        """
        
        # Use provided examples path or default
        examples_path = examples_path or "src/core/prompts/examples/transaction_examples.json"
        
        super().__init__(system_message=system_message, examples_path=examples_path)
    
    def format_example(self, example: Dict[str, Any]) -> str:
        """
        Format a transaction tagging example.
        
        Args:
            example: Example transaction and tag data
            
        Returns:
            str: Formatted example string
        """
        transaction = example.get("transaction", {})
        action = example.get("action", "")
        tag = example.get("tag", "")
        
        formatted = "Transaction Details:\n"
        formatted += f"- Payee: {transaction.get('payee_name', 'Unknown')}\n"
        formatted += f"- Amount: {transaction.get('amount', 0)}\n"
        formatted += f"- Date: {transaction.get('date', '')}\n"
        
        if "memo" in transaction and transaction["memo"]:
            formatted += f"- Original Memo: {transaction['memo']}\n"
        
        formatted += f"\nAction Performed: {action}\n"
        formatted += f"Appropriate Tag: {tag}\n"
        
        if "explanation" in example:
            formatted += f"\nExplanation: {example['explanation']}\n"
        
        return formatted + "\n---\n"


class TransactionCategoryResult(BasePromptResult):
    """
    Result model for transaction categorization.
    """
    category_match: CategoryMatch = Field(
        description="The matched category with confidence score"
    )
    category_name: str = Field(
        description="The suggested category name"
    )
    group_name: Optional[str] = Field(
        None,
        description="The suggested category group name"
    )


# TODO: Fix dataclass inheritance issue
# @dataclass
# class TransactionCategoryDependencies(BasePromptDependencies):
#     """
#     Dependencies for transaction categorization.
#     """
#     transaction: Transaction
#     available_categories: List[Category]
#     prompt_template: BasePrompt
#     additional_context: Dict[str, Any] = field(default_factory=dict)
