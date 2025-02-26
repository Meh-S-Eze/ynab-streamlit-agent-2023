import google.generativeai as genai
from typing import List, Dict, Optional
from core.config import ConfigManager
from core.shared_models import ConfidenceResult, Transaction
from core.ai_client_factory import AIClientFactory, AIModelRole
import os

class GeminiSemanticMatcher:
    """
    Semantic matching using AI for transaction categorization
    
    Follows rules:
    - Semantic Matching
    - Contextual Understanding
    - Confidence Scoring
    """
    def __init__(
        self, 
        config_manager: Optional[ConfigManager] = None,
        ai_client_factory: Optional[AIClientFactory] = None
    ):
        self.config = config_manager or ConfigManager()
        
        # Use the AI client factory for model access
        # This provides automatic fallback to OpenAI if Gemini fails
        self.ai_client_factory = ai_client_factory or AIClientFactory()
    
    def categorize_with_semantic_matching(
        self, 
        transaction: Transaction, 
        existing_categories: List[Dict],
        transaction_history: List[Transaction] = None
    ) -> ConfidenceResult:
        """
        Use AI to semantically match transaction to categories
        
        Args:
            transaction: Transaction to categorize
            existing_categories: List of available categories
            transaction_history: Historical transactions for context
        
        Returns:
            ConfidenceResult with categorization details
        """
        # Prepare context-rich prompt for AI model
        context_prompt = f"""
        Categorize this financial transaction with high accuracy:

        Transaction Details:
        - Description: {transaction.payee}
        - Amount: ${transaction.amount}
        - Date: {transaction.date}

        Available Categories: {', '.join(cat['name'] for cat in existing_categories)}

        Context Considerations:
        {'Previous similar transactions: ' + ', '.join(
            f"{t.payee} ({t.category})" for t in transaction_history[:5]
        ) if transaction_history else 'No previous transaction context'}

        Provide:
        1. Most likely category
        2. Confidence score (0-1)
        3. Reasoning for categorization
        """

        try:
            # Generate categorization response using the AI client factory
            # This will automatically try the primary provider and fall back if needed
            response = self.ai_client_factory.generate_content(
                prompt=context_prompt,
                role=AIModelRole.REASONING,  # Use reasoning model for better categorization
                temperature=0.2,  # More deterministic
                max_tokens=1024
            )

            # Parse AI model response
            parsed_response = self._parse_ai_categorization(
                response.content,
                existing_categories,
                response.provider,
                response.model_name
            )

            return parsed_response

        except Exception as e:
            # Fallback mechanism
            return ConfidenceResult(
                category='Uncategorized',
                confidence=0.1,
                reasoning=f"Categorization failed: {str(e)}"
            )
    
    def _parse_ai_categorization(
        self, 
        response_text: str, 
        existing_categories: List[Dict],
        provider: str,
        model_name: str
    ) -> ConfidenceResult:
        """
        Parse AI model's categorization response
        
        Args:
            response_text: Raw text from AI model
            existing_categories: Available categories
            provider: The AI provider that generated the response
            model_name: The model name that generated the response
        
        Returns:
            Parsed ConfidenceResult
        """
        try:
            # Use AI to help parse its own response
            parsing_prompt = f"""
            Parse this categorization response:
            {response_text}

            Extract:
            1. Category name
            2. Confidence score (0-1)
            3. Reasoning

            Available Categories: {', '.join(cat['name'] for cat in existing_categories)}

            Return a JSON with keys: 
            - category
            - confidence
            - reasoning
            """

            parsing_response = self.ai_client_factory.generate_content(
                prompt=parsing_prompt,
                role=AIModelRole.REASONING,  # Using reasoning model for parsing
                temperature=0.1,  # Very deterministic
                max_tokens=512
            )

            # Additional parsing logic would go here
            # For now, we'll use a simplified approach
            return ConfidenceResult(
                category='Uncategorized',
                confidence=0.5,
                reasoning=f"Parsing implementation pending. Used {provider} model {model_name}"
            )

        except Exception as e:
            return ConfidenceResult(
                category='Uncategorized',
                confidence=0.1,
                reasoning=f"Parsing failed: {str(e)}"
            ) 