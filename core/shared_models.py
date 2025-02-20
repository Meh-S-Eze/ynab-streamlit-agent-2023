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