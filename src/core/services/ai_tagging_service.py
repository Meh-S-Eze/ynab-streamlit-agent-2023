"""
AI tagging service for transaction memos.

This module provides functionality for managing AI tags in transaction memos,
including detecting, applying, and updating tags.
"""

import re
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class AITaggingService:
    """
    Service for managing AI tags in transaction memos.
    
    This service provides methods for detecting, applying, and updating AI tags
    in transaction memos, ensuring consistent tagging across the application.
    """
    
    # AI tag format: [AI {action_type} {date}]
    TAG_PATTERN = r'\[AI ([^\]]+) (\d{4}-\d{2}-\d{2})\]'
    
    def __init__(self):
        """Initialize the AI tagging service."""
        logger.debug("Initializing AI tagging service")
    
    def has_ai_tag(self, memo: Optional[str]) -> bool:
        """
        Check if a memo contains an AI tag.
        
        Args:
            memo: Transaction memo to check
            
        Returns:
            bool: True if the memo contains an AI tag, False otherwise
        """
        if not memo:
            return False
        
        return bool(re.search(self.TAG_PATTERN, memo))
    
    def detect_ai_tag(self, memo: Optional[str]) -> Dict[str, Any]:
        """
        Detect and parse an AI tag in a memo.
        
        Args:
            memo: Transaction memo to check
            
        Returns:
            Dict[str, Any]: Dictionary containing tag information:
                - has_tag: Whether the memo contains an AI tag
                - action_type: The action type (if tag exists)
                - date: The date (if tag exists)
                - clean_memo: The memo without the AI tag
        """
        if not memo:
            return {
                "has_tag": False,
                "action_type": None,
                "date": None,
                "clean_memo": ""
            }
        
        match = re.search(self.TAG_PATTERN, memo)
        if not match:
            return {
                "has_tag": False,
                "action_type": None,
                "date": None,
                "clean_memo": memo
            }
        
        action_type = match.group(1)
        date_str = match.group(2)
        
        # Remove the tag from the memo
        clean_memo = re.sub(self.TAG_PATTERN, "", memo).strip()
        
        return {
            "has_tag": True,
            "action_type": action_type,
            "date": date_str,
            "clean_memo": clean_memo
        }
    
    def apply_ai_tag(self, memo: Optional[str], action_type: str) -> str:
        """
        Apply an AI tag to a memo.
        
        Args:
            memo: Transaction memo to tag
            action_type: Type of action (e.g., "created", "modified", "categorized")
            
        Returns:
            str: Memo with AI tag applied
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        ai_tag = f"[AI {action_type} {current_date}]"
        
        if not memo:
            return ai_tag
        
        # If there's already an AI tag, replace it
        if self.has_ai_tag(memo):
            return self.update_ai_tag(memo, action_type)
        
        # Otherwise, add the tag at the beginning
        return f"{ai_tag} {memo}"
    
    def update_ai_tag(self, memo: Optional[str], new_action: str) -> str:
        """
        Update an existing AI tag in a memo.
        
        Args:
            memo: Transaction memo with existing tag
            new_action: New action type
            
        Returns:
            str: Memo with updated AI tag
        """
        if not memo:
            return self.apply_ai_tag("", new_action)
        
        tag_info = self.detect_ai_tag(memo)
        
        if not tag_info["has_tag"]:
            return self.apply_ai_tag(memo, new_action)
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        new_tag = f"[AI {new_action} {current_date}]"
        
        # Replace the old tag with the new one
        return f"{new_tag} {tag_info['clean_memo']}"
    
    def get_clean_memo(self, memo: Optional[str]) -> str:
        """
        Get the memo without the AI tag.
        
        Args:
            memo: Transaction memo
            
        Returns:
            str: Memo without AI tag
        """
        if not memo:
            return ""
        
        tag_info = self.detect_ai_tag(memo)
        return tag_info["clean_memo"]
