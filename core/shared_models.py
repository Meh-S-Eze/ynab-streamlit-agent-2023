from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class Transaction(BaseModel):
    """Standardized transaction model for both CLI and Streamlit"""
    id: str
    date: datetime
    amount: float
    category: str
    payee: str

class BudgetAnalysis(BaseModel):
    """Shared budget analysis model"""
    total_spent: float
    category_breakdown: Dict[str, float]
    unusual_transactions: List[Transaction]
    timestamp: datetime = Field(default_factory=datetime.now)

class ConfidenceResult(BaseModel):
    """
    Enhanced confidence result with more detailed information
    Follows confidence scoring rule
    """
    category: str
    confidence: float = Field(ge=0, le=1, description="Confidence score between 0 and 1")
    reasoning: Optional[str] = None
    transaction_ids: List[str] = []
    is_user_verified: bool = False
    
    def override_confidence(self, new_category: str, user_verified: bool = True):
        """
        Allow user to override categorization
        """
        self.category = new_category
        self.is_user_verified = user_verified
        self.confidence = 1.0  # User verification sets max confidence 