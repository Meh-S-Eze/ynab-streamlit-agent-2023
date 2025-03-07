"""
Prompt management system for YNAB AI integrations.

This package provides a structured system for managing AI prompts,
including templates, examples, and result models.
"""

from .base_prompts import (
    BasePrompt,
    BasePromptDependencies,
    BasePromptResult
)

from .transaction_prompts import (
    TransactionCategoryPrompt,
    TransactionTaggingPrompt,
    TransactionCategoryResult,
    # TransactionCategoryDependencies  # Temporarily commented out
)

from .category_prompts import (
    CategoryMatchPrompt,
    CategoryHierarchyPrompt,
    CategoryMatchResult,
    # CategoryMatchDependencies  # Temporarily commented out
)

__all__ = [
    # Base classes
    'BasePrompt',
    'BasePromptDependencies',
    'BasePromptResult',
    
    # Transaction prompts
    'TransactionCategoryPrompt',
    'TransactionTaggingPrompt',
    'TransactionCategoryResult',
    # 'TransactionCategoryDependencies',  # Temporarily commented out
    
    # Category prompts
    'CategoryMatchPrompt',
    'CategoryHierarchyPrompt',
    'CategoryMatchResult',
    # 'CategoryMatchDependencies'  # Temporarily commented out
]
