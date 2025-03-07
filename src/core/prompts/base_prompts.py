"""
Base prompt management system for YNAB AI integrations.

This module provides base classes and utilities for managing AI prompts
using PydanticAI patterns. It implements structures for prompt templates,
example management, and prompt versioning.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

logger = logging.getLogger(__name__)

class BasePrompt:
    """
    Base class for all prompt templates with common functionality.
    
    This class provides shared methods for building prompts, loading examples,
    and formatting prompt elements. It serves as the foundation for specialized
    prompt templates.
    """
    
    def __init__(
        self,
        system_message: str,
        examples: Optional[List[Dict[str, Any]]] = None,
        examples_path: Optional[str] = None,
        version: str = "1.0"
    ):
        """
        Initialize a base prompt template.
        
        Args:
            system_message: The core system message for the prompt
            examples: Optional list of examples to include in the prompt
            examples_path: Optional path to a JSON file containing examples
            version: Version identifier for the prompt template
        """
        self.system_message = system_message
        self.version = version
        
        # Load examples from provided list or file
        if examples:
            self.examples = examples
        elif examples_path:
            self.examples = self.load_examples(examples_path)
        else:
            self.examples = []
    
    def build_prompt(self) -> str:
        """
        Build the complete prompt string from components.
        
        Returns:
            str: The formatted prompt string
        """
        # Start with system message
        prompt = f"{self.system_message}\n\n"
        
        # Add examples if available
        if self.examples:
            prompt += "Here are some examples:\n\n"
            for example in self.examples:
                prompt += self.format_example(example)
                prompt += "\n"
        
        return prompt
    
    def format_example(self, example: Dict[str, Any]) -> str:
        """
        Format a single example for inclusion in the prompt.
        
        Args:
            example: Example data dictionary
            
        Returns:
            str: Formatted example string
        """
        # Default implementation - should be overridden by subclasses
        return str(example)
    
    def load_examples(self, path: str) -> List[Dict[str, Any]]:
        """
        Load examples from a JSON file.
        
        Args:
            path: Path to the JSON file containing examples
            
        Returns:
            List[Dict[str, Any]]: List of example data dictionaries
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            # Handle both direct lists and {"examples": [...]} format
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "examples" in data:
                return data["examples"]
            else:
                logger.warning(f"Invalid examples format in {path}")
                return []
        except Exception as e:
            logger.error(f"Error loading examples from {path}: {str(e)}")
            return []
    
    def add_example(self, example: Dict[str, Any]) -> None:
        """
        Add a new example to the prompt template.
        
        Args:
            example: Example data dictionary to add
        """
        self.examples.append(example)
    
    def save_examples(self, path: str) -> bool:
        """
        Save current examples to a JSON file.
        
        Args:
            path: Path to save the examples JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(path, 'w') as f:
                json.dump({"examples": self.examples, "version": self.version}, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving examples to {path}: {str(e)}")
            return False


@dataclass
class BasePromptDependencies:
    """
    Base dependencies for prompt-based agents.
    
    This class defines the basic dependencies needed for agents that use
    prompts for generating responses.
    """
    additional_context: Dict[str, Any] = field(default_factory=dict)


class BasePromptResult(BaseModel):
    """
    Base result model for prompt-based agents.
    
    This class defines the basic structure for results returned by
    prompt-based agents.
    """
    success: bool = Field(
        description="Whether the operation was successful"
    )
    explanation: str = Field(
        description="Explanation of the result or error message"
    )
    confidence: float = Field(
        description="Confidence score for the result (0-1)",
        ge=0, 
        le=1
    )
