import google.generativeai as genai
from typing import Dict, List
from .config import ConfigManager
from .circuit_breaker import CircuitBreaker
from pydantic import BaseModel, ValidationError, Field

class SpendingAnalysis(BaseModel):
    total_spent: float = Field(..., description="Total amount spent")
    category_breakdown: Dict[str, float] = Field(..., description="Spending by category")
    unusual_transactions: List[Dict] = Field(default_factory=list, description="Transactions flagged as unusual")

class GeminiSpendingAnalyzer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or ConfigManager.get_gemini_key()
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    @CircuitBreaker()
    def analyze_transactions(self, transactions: List[Dict]) -> SpendingAnalysis:
        """Analyze transactions using Gemini AI with validation"""
        prompt = self._create_prompt(transactions)
        
        try:
            response = self.model.generate_content(prompt)
            parsed_response = self._parse_response(response.text)
            return SpendingAnalysis(**parsed_response)
        except ValidationError as e:
            raise ValueError(f"Invalid AI response: {e}")
    
    def _create_prompt(self, transactions: List[Dict]) -> str:
        """Create a structured prompt for Gemini"""
        return f"""Analyze the following financial transactions:
        {transactions}

        Provide a JSON response with:
        - total_spent: Total amount spent
        - category_breakdown: Spending by category
        - unusual_transactions: Any transactions that seem out of the ordinary

        Format the response as a valid JSON object."""
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse Gemini's text response into a dictionary"""
        # Implement robust JSON parsing with error handling
        import json
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Attempt to clean and parse the response
            cleaned_response = response_text.split('```json')[-1].split('```')[0].strip()
            return json.loads(cleaned_response) 