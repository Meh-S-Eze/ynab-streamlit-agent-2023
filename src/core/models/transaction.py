"""
Transaction data models for YNAB integration.

This module contains Pydantic models for YNAB transaction data structures.
"""

from pydantic import (
    BaseModel, 
    Field, 
    field_validator,
    model_validator,
    ConfigDict,
    ValidationInfo,
    PrivateAttr
)
from typing import List, Dict, Optional, Union, Annotated, Literal, Any
from datetime import date, datetime
from decimal import Decimal, getcontext, InvalidOperation
from contextlib import contextmanager
import re
import logging

# Import utilities that will be moved to their own modules
from src.core.utils.caching import cached_method
from src.core.models.category import CategoryUpdate  # This will be implemented in category.py


class MilliunitConverter:
    """Handles precise decimal to milliunit conversion with safeguards"""
    
    def __init__(self, precision: int = 6):
        self.precision = precision
        self.max_milliunits = 999999999999  # Maximum YNAB milliunits
    
    @contextmanager
    def precise_context(self):
        """Context manager for precise decimal operations"""
        original_prec = getcontext().prec
        getcontext().prec = self.precision
        try:
            yield
        finally:
            getcontext().prec = original_prec
    
    def convert(self, value: Union[Decimal, float, str, int]) -> int:
        """
        Convert a decimal value to YNAB milliunits
        
        Args:
            value: Decimal value to convert
            
        Returns:
            int: Value in milliunits (1.00 -> 1000)
        """
        if value is None:
            return 0
            
        # Ensure we have a Decimal
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
            
        # Use precise context for calculation
        with self.precise_context():
            # Multiply by 1000 and round to nearest integer
            milliunits = int(value * 1000)
            
        # Validate against maximum
        if abs(milliunits) > self.max_milliunits:
            raise ValueError(f"Value {value} exceeds maximum YNAB milliunit limit")
            
        return milliunits


class AmountNormalizer:
    """Normalizes various amount input formats with defense-in-depth validation"""
    
    def normalize(self, raw_input: Any, context: Optional[Dict] = None) -> Decimal:
        """
        Normalize an amount input to a standard Decimal
        
        Args:
            raw_input: Input value to normalize
            context: Optional context with conversion rates
            
        Returns:
            Decimal: Normalized amount
        """
        if context is None:
            context = {}
            
        # Set default target currency
        if "target_currency" not in context:
            context["target_currency"] = "USD"
            
        # Apply validation chain
        value = self._sanitize_input(raw_input, context)
        
        # Apply bounds check
        value = self._apply_bounds_check(value)
        
        # Apply currency conversion if needed
        value = self._convert_currency(value, context)
        
        return value
        
    def _sanitize_input(self, value: Any, context: Optional[Dict]) -> Optional[Decimal]:
        """First layer of validation: sanitize input"""
        # Handle None values
        if value is None:
            return Decimal('0')
            
        # Try AI dict extraction first
        result = self._extract_from_ai_dict(value, context)
        if result is not None:
            return result
            
        # Try scientific notation conversion
        result = self._convert_scientific_notation(value, context)
        if result is not None:
            return result
            
        # Try currency string handling
        result = self._handle_currency_strings(value, context)
        if result is not None:
            return result
            
        # Basic numeric conversion as last resort
        return self._convert_basic_numeric(value, context)
    
    def _extract_from_ai_dict(self, value: Any, context: Optional[Dict]) -> Optional[Decimal]:
        """Extract amount from AI-generated dictionary"""
        try:
            # Check if this is an AI dict with value key
            if isinstance(value, dict) and "value" in value:
                # Extract the value
                extracted = value["value"]
                
                # Handle nested dictionaries
                if isinstance(extracted, dict) and "amount" in extracted:
                    extracted = extracted["amount"]
                
                # Convert to Decimal
                amount = Decimal(str(extracted))
                
                # Check for negative indicator
                is_negative = False
                if "is_outflow" in value and value["is_outflow"]:
                    is_negative = True
                elif "is_negative" in value and value["is_negative"]:
                    is_negative = True
                elif "type" in value and value["type"] in ["expense", "outflow"]:
                    is_negative = True
                    
                # Apply sign
                if is_negative:
                    amount = -abs(amount)
                    
                return amount
                
            # Check for amount field directly
            if isinstance(value, dict) and "amount" in value:
                extracted = value["amount"]
                return Decimal(str(extracted))
                
            return None
        except (TypeError, ValueError, InvalidOperation):
            return None
    
    def _convert_scientific_notation(self, value: Any, context: Optional[Dict]) -> Optional[Decimal]:
        """Handle scientific notation"""
        if isinstance(value, str) and ('e' in value.lower() or 'E' in value):
            try:
                return Decimal(value)
            except (InvalidOperation, ValueError):
                return None
        return None
    
    def _handle_currency_strings(self, value: Any, context: Optional[Dict]) -> Optional[Decimal]:
        """Handle currency strings with symbols"""
        if isinstance(value, str):
            # Remove currency symbols and commas
            cleaned = re.sub(r'[$€£¥,]', '', value)
            try:
                return Decimal(cleaned)
            except (InvalidOperation, ValueError):
                return None
        return None
    
    def _convert_basic_numeric(self, value: Any, context: Optional[Dict]) -> Decimal:
        """Final defense layer: Basic numeric conversion"""
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError) as e:
            raise ValueError(f"Cannot convert {type(value)} to Decimal: {str(e)}")
    
    def _apply_bounds_check(self, value: Decimal) -> Decimal:
        """Apply bounds checking to prevent overflow"""
        max_value = Decimal("999999999.999")  # Maximum YNAB amount
        if abs(value) > max_value:
            raise ValueError(f"Amount {value} exceeds maximum allowed value")
        return value

    def _convert_currency(self, value: Decimal, context: Dict) -> Decimal:
        """Currency conversion placeholder"""
        # TODO: Implement actual conversion rates
        if context["target_currency"] != "USD":
            raise NotImplementedError("Currency conversion not implemented")
        return value


class TransactionAmount(BaseModel):
    """Model for handling transaction amounts with comprehensive validation"""
    amount: Annotated[Decimal, Field(description="Transaction amount in standard currency units")]
    is_outflow: Annotated[bool, Field(default=True, description="Whether this is an outflow (expense)")]
    currency: Annotated[str, Field(default="USD", description="Currency code")]
    source_type: Literal["standard"] = "standard"
    
    _converter: MilliunitConverter = PrivateAttr(default_factory=MilliunitConverter)
    _normalizer: AmountNormalizer = PrivateAttr(default_factory=AmountNormalizer)
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    @field_validator("amount", mode="before")
    @classmethod
    def validate_amount(cls, value: Any, info: ValidationInfo) -> Decimal:
        """Comprehensive amount validation with context awareness"""
        try:
            # Handle None values
            if value is None:
                return Decimal('0')
                
            # Get validation context
            context = (info.context or {}).copy()
            
            # Get conversion rates from config (will be moved to config module)
            from core.config import ConfigManager
            context["conversion_rates"] = ConfigManager().get_conversion_rates()

            # Instantiate a new AmountNormalizer and use it
            normalizer = AmountNormalizer()
            normalized = normalizer.normalize(value, context)

            # Ensure positive value
            if normalized < 0:
                normalized = abs(normalized)

            return normalized
        except Exception as e:
            # Log the error but return a safe default value
            logging.error(f"Amount validation failed for {value}: {str(e)}")
            return Decimal('0')
    
    @field_validator("currency")
    def validate_currency(cls, v: str) -> str:
        """Validate currency code"""
        v = v.upper()
        # TODO: Add validation against ISO4217Currency enum
        return v
    
    def to_milliunits(self) -> int:
        """
        Convert amount to YNAB milliunits
        
        Returns:
            int: Amount in milliunits, negative for outflows
        """
        milliunits = self._converter.convert(self.amount)
        
        # Apply sign based on is_outflow
        if self.is_outflow:
            milliunits = -abs(milliunits)
        else:
            milliunits = abs(milliunits)
            
        return milliunits


class AmountFromAI(TransactionAmount):
    """Special transaction amount class for AI-generated data with additional validation"""
    source_type: Literal["gemini_v1"] = "gemini_v1"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    
    @field_validator("amount", mode="before")
    @classmethod
    def validate_ai_amount(cls, value: Any, info: ValidationInfo) -> Decimal:
        """Enhanced validation for AI-generated amounts"""
        try:
            # Get validation context
            context = (info.context or {}).copy()
            context["source"] = "gemini_ai"
            
            # Get conversion rates from config (will be moved to config module)
            from core.config import ConfigManager
            context["conversion_rates"] = ConfigManager().get_conversion_rates()
            
            # Use normalizer with AI context
            normalizer = AmountNormalizer()
            return normalizer.normalize(value, context)
        except Exception as e:
            logging.error(f"AI amount validation failed: {str(e)}")
            return Decimal('0')
    
    @field_validator('currency')
    def validate_ai_currency(cls, v):
        """Validate currency from AI"""
        v = v.upper() if v else "USD"
        return v


class TransactionCreate(BaseModel):
    """Model for creating a new transaction with discriminated unions"""
    account_id: Annotated[str, Field(description="YNAB account ID")]
    date: Annotated[date, Field(description="Transaction date")]
    amount: Union[AmountFromAI, TransactionAmount] = Field(description="Transaction amount")
    payee_name: Annotated[Optional[str], Field(None, description="Name of payee/merchant")]
    payee_id: Annotated[Optional[str], Field(None, description="YNAB payee ID")]
    memo: Annotated[Optional[str], Field(None, description="Transaction memo/note")]
    cleared: Annotated[str, Field(default="uncleared", description="Transaction cleared status")]
    approved: Annotated[bool, Field(default=False, description="Whether transaction is approved")]
    category_id: Annotated[Optional[str], Field(None, description="YNAB category ID")]
    category_name: Annotated[Optional[str], Field(None, description="Category name for lookup")]
    flag_name: Annotated[Optional[str], Field(None, description="Flag color: red, orange, yellow, green, blue, purple")]
    
    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        extra="forbid"
    )
    
    @model_validator(mode='before')
    def pre_validate_amount(cls, values: Dict) -> Dict:
        """Pre-process AI-generated amounts"""
        if isinstance(values.get('amount'), dict) and 'value' in values['amount']:
            values['amount'] = AmountFromAI(
                amount=Decimal(str(values['amount']['value'])),
                currency=values['amount'].get('currency', 'USD')
            )
        return values
    
    @classmethod
    def from_ai_response(cls, data: Dict[str, Any]) -> "TransactionCreate":
        """Create instance from AI response with enhanced validation"""
        try:
            # Process amount with AI context
            amount = AmountFromAI.model_validate(
                data.get("amount", {}),
                context={"source": "gemini_ai"}
            )
            
            # Create transaction with processed amount
            return cls.model_validate({
                **data,
                "amount": amount
            })
        except Exception as e:
            raise ValueError(f"Failed to create transaction from AI response: {str(e)}")

    def to_api_format(self) -> Dict[str, Any]:
        """
        Convert to YNAB API format with milliunits conversion
        
        Returns:
            Dict formatted for YNAB API
        """
        # Convert amount to milliunits
        milliunits = self.amount.to_milliunits()
        
        # Format date as ISO-8601 string
        date_str = self.date.isoformat() if isinstance(self.date, date) else str(self.date)
        
        # Create API payload
        payload = {
            "account_id": self.account_id,
            "date": date_str,
            "amount": milliunits,
            "payee_name": self.payee_name,
            "payee_id": self.payee_id,
            "memo": self.memo,
            "cleared": self.cleared,
            "approved": self.approved,
            "category_id": self.category_id,
            "flag_name": self.flag_name
        }
        
        # Remove None values
        return {k: v for k, v in payload.items() if v is not None}


class TransactionUpdate(BaseModel):
    """Model for updating an existing transaction"""
    id: Annotated[str, Field(description="Transaction ID")]
    category_name: Annotated[Optional[str], Field(None, description="New category name")]
    category_id: Annotated[Optional[str], Field(None, description="New category ID")]
    memo: Annotated[Optional[str], Field(None, description="Updated memo")]
    cleared: Annotated[Optional[str], Field(None, description="Updated cleared status")]
    
    @field_validator('cleared')
    def validate_cleared(cls, v):
        """Validate cleared status if provided"""
        if v is not None:
            valid_statuses = ['cleared', 'uncleared', 'reconciled']
            if v not in valid_statuses:
                raise ValueError(f"Invalid cleared status. Must be one of: {', '.join(valid_statuses)}")
        return v


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
    flag_name: Annotated[Optional[str], Field(None, description="Flag color: red, orange, yellow, green, blue, purple")]
    
    @field_validator('cleared')
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
            category_name=data.get('category_name'),
            flag_name=data.get('flag_name')
        )


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
    
    @field_validator('confidence')
    def validate_confidence(cls, v):
        """Ensure confidence is between 0 and 1"""
        return max(0.0, min(1.0, float(v)))
    
    @field_validator('alternative_categories')
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
