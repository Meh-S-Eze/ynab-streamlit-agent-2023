from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union
from datetime import date, datetime
from decimal import Decimal

class TransactionAmount(BaseModel):
    """Model for handling transaction amounts with milliunit conversion"""
    amount: Decimal = Field(..., description="Transaction amount in standard currency units")
    is_outflow: bool = Field(default=True, description="Whether this is an outflow (expense)")
    
    @validator('amount')
    def validate_amount(cls, v):
        """Ensure amount is a valid decimal"""
        if not isinstance(v, (int, float, Decimal, str)):
            raise ValueError("Amount must be a number")
        return Decimal(str(v))
    
    def to_milliunits(self) -> int:
        """Convert amount to milliunits for YNAB API"""
        milliunits = int(self.amount * 1000)
        return -milliunits if self.is_outflow else milliunits

class TransactionCreate(BaseModel):
    """Model for creating a new transaction"""
    account_id: str = Field(..., description="YNAB account ID")
    date: date = Field(..., description="Transaction date")
    amount: Union[TransactionAmount, int] = Field(..., description="Transaction amount")
    payee_name: Optional[str] = Field(None, description="Name of payee/merchant")
    memo: Optional[str] = Field(None, description="Transaction memo/note")
    cleared: str = Field(default="uncleared", description="Transaction cleared status")
    approved: bool = Field(default=False, description="Whether transaction is approved")
    category_id: Optional[str] = Field(None, description="YNAB category ID")
    category_name: Optional[str] = Field(None, description="Category name for lookup")
    
    @validator('date')
    def validate_date(cls, v):
        """Ensure date is not in the future"""
        if isinstance(v, str):
            try:
                v = datetime.strptime(v, '%Y-%m-%d').date()
            except ValueError:
                raise ValueError("Invalid date format. Use YYYY-MM-DD")
        
        if v > date.today():
            raise ValueError("Transaction date cannot be in the future")
        return v
    
    @validator('cleared')
    def validate_cleared(cls, v):
        """Validate cleared status"""
        valid_statuses = ['cleared', 'uncleared', 'reconciled']
        if v not in valid_statuses:
            raise ValueError(f"Invalid cleared status. Must be one of: {', '.join(valid_statuses)}")
        return v
    
    def to_api_format(self) -> Dict:
        """Convert to YNAB API format"""
        # Handle amount conversion
        if isinstance(self.amount, TransactionAmount):
            amount = self.amount.to_milliunits()
        else:
            # If amount is already in milliunits, use as is
            amount = self.amount
            
        return {
            'account_id': self.account_id,
            'date': self.date.isoformat(),
            'amount': amount,
            'payee_name': self.payee_name,
            'memo': self.memo,
            'cleared': self.cleared,
            'approved': self.approved,
            'category_id': self.category_id
        }

class TransactionUpdate(BaseModel):
    """Model for updating an existing transaction"""
    id: str = Field(..., description="Transaction ID")
    category_name: Optional[str] = Field(None, description="New category name")
    category_id: Optional[str] = Field(None, description="New category ID")
    memo: Optional[str] = Field(None, description="Updated memo")
    cleared: Optional[str] = Field(None, description="Updated cleared status")
    
    @validator('cleared')
    def validate_cleared(cls, v):
        """Validate cleared status if provided"""
        if v is not None:
            valid_statuses = ['cleared', 'uncleared', 'reconciled']
            if v not in valid_statuses:
                raise ValueError(f"Invalid cleared status. Must be one of: {', '.join(valid_statuses)}")
        return v

class ConfidenceResult(BaseModel):
    """Model for AI categorization results"""
    category: str = Field(..., description="Primary category assigned")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    reasoning: Optional[str] = Field(None, description="Explanation for category")
    transaction_ids: List[str] = Field(default_factory=list, description="Affected transaction IDs")
    alternative_categories: List[Dict[str, Union[str, float]]] = Field(
        default_factory=list,
        description="Alternative category suggestions"
    )
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Ensure confidence is between 0 and 1"""
        return max(0.0, min(1.0, float(v)))
    
    @validator('alternative_categories')
    def validate_alternatives(cls, v):
        """Validate alternative categories structure"""
        validated = []
        for alt in v:
            if isinstance(alt, dict) and 'name' in alt and 'confidence' in alt:
                validated.append({
                    'name': str(alt['name']),
                    'confidence': max(0.0, min(1.0, float(alt['confidence'])))
                })
        return validated 