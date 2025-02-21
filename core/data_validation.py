from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime, date
from decimal import Decimal
from .shared_models import TransactionCreate, TransactionAmount, ConfidenceResult

class GeminiAnalysisResult(BaseModel):
    """Model for validating Gemini analysis results"""
    transaction_id: str = Field(..., description="ID of the analyzed transaction")
    category_name: str = Field(..., description="Suggested category name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    reasoning: Optional[str] = Field(None, description="Reasoning for the categorization")
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

class YNABUpdatePayload(BaseModel):
    """Model for validating YNAB update payloads"""
    transaction_id: str = Field(..., description="ID of the transaction to update")
    category_id: Optional[str] = Field(None, description="New category ID")
    category_name: Optional[str] = Field(None, description="New category name")
    memo: Optional[str] = Field(None, description="Updated memo")
    cleared: Optional[str] = Field(None, description="Updated cleared status")

    @validator('cleared')
    def validate_cleared(cls, v):
        """Validate cleared status"""
        if v is not None:
            valid_statuses = ['cleared', 'uncleared', 'reconciled']
            if v not in valid_statuses:
                raise ValueError(f"Invalid cleared status. Must be one of: {', '.join(valid_statuses)}")
        return v

class DataValidator:
    """
    Validates data flow between GeminiSpendingAnalyzer and YNABClient
    Implements data flow validation rule
    """
    @staticmethod
    def validate_gemini_analysis(analysis_results: List[Dict]) -> List[GeminiAnalysisResult]:
        """
        Validate Gemini analysis results before passing to YNAB
        
        Args:
            analysis_results (List[Dict]): Raw analysis results from Gemini
        
        Returns:
            List[GeminiAnalysisResult]: Validated analysis results
        
        Raises:
            ValueError: If validation fails
        """
        validated_results = []
        for result in analysis_results:
            try:
                validated = GeminiAnalysisResult(**result)
                validated_results.append(validated)
            except Exception as e:
                raise ValueError(f"Invalid analysis result: {str(e)}")
        return validated_results

    @staticmethod
    def validate_ynab_update(update_data: List[Dict]) -> List[YNABUpdatePayload]:
        """
        Validate YNAB update payload before sending to API
        
        Args:
            update_data (List[Dict]): Update data to validate
        
        Returns:
            List[YNABUpdatePayload]: Validated update payloads
        
        Raises:
            ValueError: If validation fails
        """
        validated_updates = []
        for update in update_data:
            try:
                validated = YNABUpdatePayload(**update)
                validated_updates.append(validated)
            except Exception as e:
                raise ValueError(f"Invalid update payload: {str(e)}")
        return validated_updates

    @staticmethod
    def validate_transaction_amount(amount: Union[int, float, str, TransactionAmount]) -> TransactionAmount:
        """
        Validate and convert transaction amount
        
        Args:
            amount: Amount to validate
        
        Returns:
            TransactionAmount: Validated amount
        
        Raises:
            ValueError: If amount is invalid
        """
        try:
            if isinstance(amount, TransactionAmount):
                return amount
            
            # Convert string or numeric to TransactionAmount
            if isinstance(amount, (int, float, str)):
                amount_value = float(amount)
                return TransactionAmount(
                    amount=abs(amount_value),
                    is_outflow=amount_value < 0
                )
            
            raise ValueError("Invalid amount type")
        except Exception as e:
            raise ValueError(f"Invalid amount: {str(e)}")

    @staticmethod
    def validate_transaction_date(transaction_date: Union[str, date, datetime]) -> date:
        """
        Validate transaction date
        
        Args:
            transaction_date: Date to validate
        
        Returns:
            date: Validated date
        
        Raises:
            ValueError: If date is invalid or in the future
        """
        try:
            if isinstance(transaction_date, datetime):
                validated_date = transaction_date.date()
            elif isinstance(transaction_date, date):
                validated_date = transaction_date
            elif isinstance(transaction_date, str):
                validated_date = datetime.strptime(transaction_date, '%Y-%m-%d').date()
            else:
                raise ValueError("Invalid date type")
            
            # Check for future date
            if validated_date > date.today():
                raise ValueError("Transaction date cannot be in the future")
            
            return validated_date
        except Exception as e:
            raise ValueError(f"Invalid date: {str(e)}")

    @staticmethod
    def prepare_ynab_payload(analysis_result: GeminiAnalysisResult) -> YNABUpdatePayload:
        """
        Convert Gemini analysis result to YNAB update payload
        
        Args:
            analysis_result (GeminiAnalysisResult): Validated analysis result
        
        Returns:
            YNABUpdatePayload: Prepared YNAB update payload
        """
        return YNABUpdatePayload(
            transaction_id=analysis_result.transaction_id,
            category_name=analysis_result.category_name
        ) 