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
        context_prompt = f"""You are a YNAB (You Need A Budget) financial transaction categorization specialist. Your task is to assign the most appropriate budget category to a transaction based on its details and contextual information.

TRANSACTION TO CATEGORIZE:
Description: {transaction.payee}
Amount: ${transaction.amount}
Date: {transaction.date}

AVAILABLE BUDGET CATEGORIES:
{', '.join(cat['name'] for cat in existing_categories)}

HISTORICAL CONTEXT:
{f"Previous similar transactions (Payee → Category):\n" + '\n'.join(
    f"- {t.payee} → {t.category}" for t in transaction_history[:5]
) if transaction_history and len(transaction_history) > 0 else 'No previous transaction history available'}

CATEGORIZATION GUIDELINES:
1. Analyze the transaction description for keywords indicating the type of purchase
2. Consider the transaction amount as a contextual clue
3. Take into account any patterns from similar previous transactions
4. Focus on identifying the actual purpose of the transaction, not just the vendor
5. When uncertain between multiple categories, choose the most specific one

EXAMPLE CATEGORIZATIONS WITH REASONING:

Example 1 - Grocery Store:
Transaction: "KROGER #123" for $78.92 on 2023-05-12
{{
  "category": "Groceries",
  "confidence": 0.95,
  "reasoning": "Kroger is a well-known grocery chain, and the amount is typical for a regular grocery shopping trip",
  "alternative_categories": [
    {{
      "name": "Household Goods",
      "confidence": 0.45
    }},
    {{
      "name": "Personal Care",
      "confidence": 0.25
    }}
  ]
}}

Example 2 - Ambiguous Online Purchase:
Transaction: "AMAZON MARKETPLACE" for $35.67 on 2023-05-14
{{
  "category": "Shopping",
  "confidence": 0.65,
  "reasoning": "Amazon purchase with moderate confidence as it could be various items; amount suggests smaller household or personal items rather than major purchase",
  "alternative_categories": [
    {{
      "name": "Household Goods",
      "confidence": 0.55
    }},
    {{
      "name": "Entertainment",
      "confidence": 0.45
    }}
  ]
}}

Example 3 - Clear Utility Bill:
Transaction: "CITY POWER & LIGHT" for $145.33 on 2023-05-15
{{
  "category": "Utilities",
  "confidence": 0.98,
  "reasoning": "Description clearly indicates a utility company, and amount is in the typical range for a monthly utility bill",
  "alternative_categories": [
    {{
      "name": "Bills",
      "confidence": 0.60
    }}
  ]
}}

Example 4 - Low Confidence:
Transaction: "ACH TRANSFER 3887651" for $500.00 on 2023-05-16
{{
  "category": "Uncategorized",
  "confidence": 0.40,
  "reasoning": "Generic transfer description without clear purpose; amount is substantial but could match multiple categories; insufficient context to determine with high confidence",
  "alternative_categories": [
    {{
      "name": "Bills",
      "confidence": 0.35
    }},
    {{
      "name": "Transfer",
      "confidence": 0.35
    }},
    {{
      "name": "Rent",
      "confidence": 0.30
    }}
  ]
}}

PROVIDE A JSON RESPONSE WITH EXACTLY THESE FIELDS:
{{
  "category": "Most appropriate category name from the available list",
  "confidence": 0.85,  // Decimal between 0.0-1.0 indicating your confidence
  "reasoning": "Clear explanation of why this category is most appropriate",
  "alternative_categories": [
    {{
      "name": "Second best category",
      "confidence": 0.65
    }},
    {{
      "name": "Third best category",
      "confidence": 0.40
    }}
  ]
}}

CATEGORIZATION ACCURACY REQUIREMENTS:
- Use ONLY categories from the provided list
- Provide a confidence score that genuinely reflects certainty
- Give lower confidence scores (below 0.6) when genuinely uncertain
- Include 2-3 alternative categories when reasonable alternatives exist
- Ensure JSON is properly formatted with exact field names as shown
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