import google.generativeai as genai
from typing import List, Dict, Optional
from core.config import ConfigManager
from core.shared_models import ConfidenceResult, Transaction

class GeminiSemanticMatcher:
    """
    Semantic matching using Gemini AI for transaction categorization
    
    Follows rules:
    - Semantic Matching
    - Contextual Understanding
    - Confidence Scoring
    """
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        
        # Configure Gemini
        genai.configure(api_key=self.config.get('credentials.gemini.api_key'))
        self.model = genai.GenerativeModel('gemini-pro')
    
    def categorize_with_semantic_matching(
        self, 
        transaction: Transaction, 
        existing_categories: List[Dict],
        transaction_history: List[Transaction] = None
    ) -> ConfidenceResult:
        """
        Use Gemini to semantically match transaction to categories
        
        Args:
            transaction: Transaction to categorize
            existing_categories: List of available categories
            transaction_history: Historical transactions for context
        
        Returns:
            ConfidenceResult with categorization details
        """
        # Prepare context-rich prompt for Gemini
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
            # Generate categorization response
            response = self.model.generate_content(
                context_prompt,
                generation_config={
                    'temperature': 0.2,  # More deterministic
                    'max_output_tokens': 1024
                }
            )

            # Parse Gemini's response
            parsed_response = self._parse_gemini_categorization(
                response.text, 
                existing_categories
            )

            return parsed_response

        except Exception as e:
            # Fallback mechanism
            return ConfidenceResult(
                category='Uncategorized',
                confidence=0.1,
                reasoning=f"Categorization failed: {str(e)}"
            )
    
    def _parse_gemini_categorization(
        self, 
        response_text: str, 
        existing_categories: List[Dict]
    ) -> ConfidenceResult:
        """
        Parse Gemini's categorization response
        
        Args:
            response_text: Raw text from Gemini
            existing_categories: Available categories
        
        Returns:
            Parsed ConfidenceResult
        """
        try:
            # Use Gemini to help parse its own response
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

            parsing_response = self.model.generate_content(
                parsing_prompt,
                generation_config={
                    'temperature': 0.1,  # Very deterministic
                    'max_output_tokens': 512
                }
            )

            # Additional parsing logic would go here
            # For now, we'll use a simplified approach
            return ConfidenceResult(
                category='Uncategorized',
                confidence=0.5,
                reasoning="Parsing implementation pending"
            )

        except Exception as e:
            return ConfidenceResult(
                category='Uncategorized',
                confidence=0.1,
                reasoning=f"Parsing failed: {str(e)}"
            ) 