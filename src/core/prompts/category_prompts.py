"""
Category-specific prompt templates for YNAB AI integrations.

This module provides specialized prompt templates for category-related
operations, including category matching and hierarchy management.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from ..models.category import Category, CategoryGroup, CategoryMatch
from .base_prompts import BasePrompt, BasePromptDependencies, BasePromptResult

logger = logging.getLogger(__name__)

class CategoryMatchPrompt(BasePrompt):
    """
    Prompt template for category matching.
    
    This class provides a specialized prompt template for matching
    a string to the most appropriate YNAB category.
    """
    
    def __init__(self, examples_path: Optional[str] = None):
        """
        Initialize a category matching prompt template.
        
        Args:
            examples_path: Optional path to a JSON file containing examples
        """
        system_message = """
        You are an expert at matching financial category names. Your task is to match an input string
        to the most appropriate category in a budget system.
        
        Consider these factors when matching:
        1. Exact matches should be prioritized
        2. Similar words or synonyms should be considered
        3. If a category group is specified, prioritize categories in that group
        4. Common abbreviations or shorthand should be recognized
        5. Assign a confidence score between 0 and 1, where:
           - 1.0 = Perfect/exact match
           - 0.8-0.99 = Very strong match (synonyms or minor variations)
           - 0.6-0.79 = Good match (related concepts)
           - 0.4-0.59 = Moderate match (somewhat related)
           - <0.4 = Weak match (minimal relation)
        
        If there is no reasonable match, respond with the most general applicable category
        and a low confidence score.
        """
        
        # Use provided examples path or default
        examples_path = examples_path or "src/core/prompts/examples/category_examples.json"
        
        super().__init__(system_message=system_message, examples_path=examples_path)
    
    def format_example(self, example: Dict[str, Any]) -> str:
        """
        Format a category matching example.
        
        Args:
            example: Example query and matching data
            
        Returns:
            str: Formatted example string
        """
        query = example.get("query", "")
        category = example.get("category", {})
        confidence = example.get("confidence", 0.0)
        
        formatted = f"Input string: {query}\n\n"
        formatted += "Available categories:\n"
        
        for group in example.get("category_groups", []):
            formatted += f"- {group['name']}:\n"
            for cat in group.get("categories", []):
                formatted += f"  - {cat['name']}\n"
        
        formatted += f"\nBest match: {category.get('name', 'None')}\n"
        formatted += f"Category group: {category.get('group_name', 'None')}\n"
        formatted += f"Confidence score: {confidence}\n"
        
        if "explanation" in example:
            formatted += f"\nExplanation: {example['explanation']}\n"
        
        return formatted + "\n---\n"


class CategoryHierarchyPrompt(BasePrompt):
    """
    Prompt template for category hierarchy operations.
    
    This class provides a specialized prompt template for working
    with category hierarchies and relationships.
    """
    
    def __init__(self, examples_path: Optional[str] = None):
        """
        Initialize a category hierarchy prompt template.
        
        Args:
            examples_path: Optional path to a JSON file containing examples
        """
        system_message = """
        You are an expert at analyzing financial category hierarchies. Your task is to help
        organize categories into logical groups and suggest improvements to category structures.
        
        Consider these principles when working with category hierarchies:
        
        1. Categories should be organized by purpose, not by payment method
        2. Similar spending types should be grouped together
        3. Categories should be specific enough to be meaningful but not too granular
        4. Most budgets benefit from these high-level groups:
           - Housing
           - Food
           - Transportation
           - Personal
           - Health
           - Entertainment
           - Debt
           - Savings/Investments
        5. Avoid having too many top-level groups (8-12 is usually sufficient)
        """
        
        # Use provided examples path or default
        examples_path = examples_path or "src/core/prompts/examples/category_examples.json"
        
        super().__init__(system_message=system_message, examples_path=examples_path)
    
    def format_example(self, example: Dict[str, Any]) -> str:
        """
        Format a category hierarchy example.
        
        Args:
            example: Example hierarchy data
            
        Returns:
            str: Formatted example string
        """
        scenario = example.get("scenario", "")
        current_hierarchy = example.get("current_hierarchy", [])
        suggested_changes = example.get("suggested_changes", [])
        
        formatted = f"Scenario: {scenario}\n\n"
        formatted += "Current Category Hierarchy:\n"
        
        for group in current_hierarchy:
            formatted += f"- {group['name']}:\n"
            for cat in group.get("categories", []):
                formatted += f"  - {cat['name']}\n"
        
        formatted += "\nSuggested Changes:\n"
        for change in suggested_changes:
            formatted += f"- {change}\n"
        
        if "explanation" in example:
            formatted += f"\nExplanation: {example['explanation']}\n"
        
        return formatted + "\n---\n"


class CategoryMatchResult(BasePromptResult):
    """
    Result model for category matching.
    """
    category_match: CategoryMatch = Field(
        description="The matched category with confidence score"
    )
    alternative_matches: List[CategoryMatch] = Field(
        default_factory=list,
        description="Alternative category matches with lower confidence"
    )


# TODO: Fix dataclass inheritance issue
# @dataclass
# class CategoryMatchDependencies(BasePromptDependencies):
#     """
#     Dependencies for category matching.
#     """
#     query: str
#     category_groups: List[CategoryGroup]
#     prompt_template: BasePrompt
#     preferred_group_name: Optional[str] = None
#     additional_context: Dict[str, Any] = field(default_factory=dict)
