from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Annotated
from datetime import date, datetime
from decimal import Decimal

class TransactionAmount(BaseModel):
    """Model for handling transaction amounts with milliunit conversion"""
    amount: Annotated[Decimal, Field(description="Transaction amount in standard currency units")]
    is_outflow: Annotated[bool, Field(default=True, description="Whether this is an outflow (expense)")]
    
    @validator('amount')
    def validate_amount(cls, v):
        """Ensure amount is a valid decimal"""
        if isinstance(v, dict):
            # Handle case where a dict is passed
            amount_value = v.get('amount')
            if amount_value is not None:
                return Decimal(str(amount_value))
            raise ValueError("Amount not found in dictionary")
        elif not isinstance(v, (int, float, Decimal, str)):
            raise ValueError("Amount must be a number")
        return Decimal(str(v))
    
    def to_milliunits(self) -> int:
        """Convert amount to milliunits for YNAB API"""
        milliunits = int(self.amount * 1000)
        return -milliunits if self.is_outflow else milliunits

class TransactionCreate(BaseModel):
    """Model for creating a new transaction"""
    account_id: Annotated[str, Field(description="YNAB account ID")]
    date: Annotated[date, Field(description="Transaction date")]
    amount: Annotated[Union[TransactionAmount, int, float, str, Decimal], Field(description="Transaction amount")]
    payee_name: Annotated[Optional[str], Field(None, description="Name of payee/merchant")]
    memo: Annotated[Optional[str], Field(None, description="Transaction memo/note")]
    cleared: Annotated[str, Field(default="uncleared", description="Transaction cleared status")]
    approved: Annotated[bool, Field(default=False, description="Whether transaction is approved")]
    category_id: Annotated[Optional[str], Field(None, description="YNAB category ID")]
    category_name: Annotated[Optional[str], Field(None, description="Category name for lookup")]
    
    @validator('amount')
    def validate_amount(cls, v):
        """Convert amount to TransactionAmount if needed"""
        if isinstance(v, TransactionAmount):
            return v
        elif isinstance(v, (int, float, str, Decimal)):
            # Convert to positive value and create TransactionAmount
            amount_value = abs(float(v))
            return TransactionAmount(
                amount=str(amount_value),
                is_outflow=float(v) < 0
            )
        elif isinstance(v, dict):
            # Handle dictionary input
            amount_value = v.get('amount')
            is_outflow = v.get('is_outflow', True)
            if amount_value is not None:
                return TransactionAmount(
                    amount=str(abs(float(amount_value))),
                    is_outflow=is_outflow
                )
        raise ValueError("Invalid amount format")
    
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
    id: Annotated[str, Field(description="Transaction ID")]
    category_name: Annotated[Optional[str], Field(None, description="New category name")]
    category_id: Annotated[Optional[str], Field(None, description="New category ID")]
    memo: Annotated[Optional[str], Field(None, description="Updated memo")]
    cleared: Annotated[Optional[str], Field(None, description="Updated cleared status")]
    
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
    category: Annotated[str, Field(description="Primary category assigned")]
    confidence: Annotated[float, Field(ge=0, le=1, description="Confidence score (0-1)")]
    reasoning: Annotated[Optional[str], Field(None, description="Explanation for category")]
    transaction_ids: Annotated[List[str], Field(default_factory=list, description="Affected transaction IDs")]
    alternative_categories: Annotated[
        List[Dict[str, Union[str, float]]], 
        Field(default_factory=list, description="Alternative category suggestions")
    ]
    
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

class Transaction(BaseModel):
    """Model for representing a YNAB transaction"""
    id: Annotated[str, Field(description="Transaction ID")]
    account_id: Annotated[str, Field(description="YNAB account ID")]
    date: Annotated[date, Field(description="Transaction date")]
    amount: Annotated[int, Field(description="Transaction amount in milliunits")]
    payee_name: Annotated[Optional[str], Field(None, description="Name of payee/merchant")]
    memo: Annotated[Optional[str], Field(None, description="Transaction memo/note")]
    cleared: Annotated[str, Field(description="Transaction cleared status")]
    approved: Annotated[bool, Field(description="Whether transaction is approved")]
    category_id: Annotated[Optional[str], Field(None, description="YNAB category ID")]
    category_name: Annotated[Optional[str], Field(None, description="Category name")]
    
    @validator('cleared')
    def validate_cleared(cls, v):
        """Validate cleared status"""
        valid_statuses = ['cleared', 'uncleared', 'reconciled']
        if v not in valid_statuses:
            raise ValueError(f"Invalid cleared status. Must be one of: {', '.join(valid_statuses)}")
        return v
    
    def to_update_model(self) -> TransactionUpdate:
        """Convert to update model"""
        return TransactionUpdate(
            id=self.id,
            category_id=self.category_id,
            category_name=self.category_name,
            memo=self.memo,
            cleared=self.cleared
        )
    
    @classmethod
    def from_api_response(cls, data: Dict) -> 'Transaction':
        """Create instance from YNAB API response"""
        return cls(
            id=data['id'],
            account_id=data['account_id'],
            date=datetime.strptime(data['date'], '%Y-%m-%d').date(),
            amount=data['amount'],
            payee_name=data.get('payee_name'),
            memo=data.get('memo'),
            cleared=data['cleared'],
            approved=data['approved'],
            category_id=data.get('category_id'),
            category_name=data.get('category_name')
        )

class SpendingAnalysis(BaseModel):
    """Model for spending analysis results"""
    total_spending: Annotated[Decimal, Field(description="Total spending amount")]
    category_breakdown: Annotated[Dict[str, Decimal], Field(description="Spending by category")]
    time_period: Annotated[str, Field(description="Analysis time period")]
    insights: Annotated[List[str], Field(description="Key spending insights")]
    recommendations: Annotated[List[str], Field(description="Budget recommendations")]
    
    @validator('total_spending')
    def validate_total(cls, v):
        """Ensure total is a valid decimal"""
        if not isinstance(v, (int, float, Decimal, str)):
            raise ValueError("Total spending must be a number")
        return Decimal(str(v))
    
    @validator('category_breakdown')
    def validate_breakdown(cls, v):
        """Validate category breakdown amounts"""
        validated = {}
        for category, amount in v.items():
            if not isinstance(amount, (int, float, Decimal, str)):
                raise ValueError(f"Invalid amount for category {category}")
            validated[str(category)] = Decimal(str(amount))
        return validated
    
    @validator('time_period')
    def validate_period(cls, v):
        """Validate time period format"""
        valid_periods = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']
        if v.lower() not in valid_periods:
            raise ValueError(f"Invalid time period. Must be one of: {', '.join(valid_periods)}")
        return v.lower() 