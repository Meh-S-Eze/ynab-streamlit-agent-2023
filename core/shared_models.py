from pydantic import (
    BaseModel, 
    Field, 
    field_validator,
    model_validator,
    Discriminator,
    Tag,
    ConfigDict,
    ValidationInfo,
    PrivateAttr
)
from typing import List, Dict, Optional, Union, Annotated, Literal, Any
from datetime import date, datetime
from decimal import Decimal, getcontext, InvalidOperation
from contextlib import contextmanager
import re
from enum import Enum
import logging
from core.config import ConfigManager

class ISO4217Currency(str, Enum):
    """ISO 4217 currency codes"""
    USD = "USD"  # US Dollar
    EUR = "EUR"  # Euro
    GBP = "GBP"  # British Pound
    CAD = "CAD"  # Canadian Dollar
    AUD = "AUD"  # Australian Dollar
    JPY = "JPY"  # Japanese Yen
    CNY = "CNY"  # Chinese Yuan
    INR = "INR"  # Indian Rupee
    NZD = "NZD"  # New Zealand Dollar
    CHF = "CHF"  # Swiss Franc

class AISource(BaseModel):
    """Model for tracking AI-generated data"""
    source_type: Literal["gemini_v1"] = "gemini_v1"

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
        """Convert to milliunits with comprehensive error checking"""
        with self.precise_context():
            try:
                # Normalize and validate input
                if isinstance(value, (float, int)):
                    value = str(value)
                decimal_value = Decimal(value).normalize()
                
                # Convert to milliunits
                milliunits = int(decimal_value * 1000)
                
                # Check bounds
                if abs(milliunits) > self.max_milliunits:
                    raise ValueError(f"Milliunit value {milliunits} exceeds maximum allowed")
                
                return milliunits
            except InvalidOperation as e:
                raise ValueError(f"Invalid decimal value: {str(e)}")
            except (OverflowError, ValueError) as e:
                raise ValueError(f"Conversion error: {str(e)}")

def safe_repr(value: Any) -> str:
    """Safely represent any value as a string for logging"""
    try:
        return repr(value)
    except Exception:
        return str(type(value))

class AmountNormalizer:
    """Normalizes various amount input formats with defense-in-depth validation"""
    
    def normalize(self, raw_input: Any, context: Optional[Dict] = None) -> Decimal:
        """
        Multi-layer normalization with context-aware processing
        
        Args:
            raw_input: Raw amount input
            context: Optional validation context
        """
        try:
            # Add currency conversion stub
            if context and context.get("currency_conversion"):
                return self._convert_currency(raw_input, context)
            
            validation_layers = [
                self._sanitize_input,
                self._extract_from_ai_dict,
                self._convert_scientific_notation,
                self._handle_currency_strings,
                self._convert_basic_numeric
            ]
            
            errors = []
            for layer in validation_layers:
                try:
                    result = layer(raw_input, context)
                    if result is not None:
                        return self._apply_bounds_check(result)
                except (ValueError, TypeError, AttributeError) as e:
                    errors.append(f"{layer.__name__}: {str(e)}")
                    continue
            
            raise ValueError(f"Amount validation failed at all layers: {'; '.join(errors)}")
        except ValueError as e:
            logging.error(f"Validation failed for input: {safe_repr(raw_input)}")
            raise
    
    def _sanitize_input(self, value: Any, context: Optional[Dict]) -> Optional[Decimal]:
        """Sanitize potentially dangerous input"""
        if isinstance(value, str):
            # Remove any non-numeric characters except .- 
            cleaned = re.sub(r'[^\d\.-]', '', value)
            return Decimal(cleaned) if cleaned else None
        return None
    
    def _extract_from_ai_dict(self, value: Any, context: Optional[Dict]) -> Optional[Decimal]:
        """Handle AI output with proper structure validation"""
        if isinstance(value, dict) and context.get("source") == "gemini_ai":
            if "amount" in value:
                # Validate currency if present
                currency = value.get("currency", "USD").upper()
                if currency not in ISO4217Currency.__members__:
                    raise ValueError(f"Invalid currency {currency} from AI")
                
                # Get conversion rate from context or config
                conversion_rate = context.get(
                    "conversion_rates", 
                    {"USD": 1.0}
                ).get(currency, 1.0)
                
                # Convert amount to base currency
                converted_amount = Decimal(str(value["amount"])) * Decimal(str(conversion_rate))
                
                return converted_amount
        return None
    
    def _convert_scientific_notation(self, value: Any, context: Optional[Dict]) -> Optional[Decimal]:
        """Third defense layer: Scientific notation handling"""
        if isinstance(value, str) and ('e' in value.lower() or 'E' in value):
            with getcontext() as ctx:
                ctx.prec = 10  # Increased precision for scientific notation
                return Decimal(value)
        return None
    
    def _handle_currency_strings(self, value: Any, context: Optional[Dict]) -> Optional[Decimal]:
        """Fourth defense layer: Currency string handling"""
        if isinstance(value, str):
            # Remove currency symbols, commas, and whitespace
            cleaned = re.sub(r'[^\d.-]', '', value)
            if cleaned:
                return Decimal(cleaned)
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
            # Get validation context
            context = (info.context or {}).copy()
            context["conversion_rates"] = ConfigManager().get_conversion_rates()

            # Instantiate a new AmountNormalizer and use it
            normalizer = AmountNormalizer()
            normalized = normalizer.normalize(value, context)

            # Ensure positive value
            if normalized < 0:
                normalized = abs(normalized)

            return normalized
        except Exception as e:
            raise ValueError(f"Amount validation failed: {str(e)}")
    
    @field_validator("currency")
    def validate_currency(cls, v: str) -> str:
        """Validate currency code"""
        v = v.upper()
        if v not in ISO4217Currency.__members__:
            raise ValueError(f"Invalid currency code: {v}")
        return v
    
    def to_milliunits(self) -> int:
        """Convert to milliunits with error handling"""
        try:
            # Create a converter instance since we're using PrivateAttr
            converter = MilliunitConverter()
            milliunits = converter.convert(self.amount)
            return -milliunits if self.is_outflow else milliunits
        except Exception as e:
            raise ValueError(f"Milliunit conversion failed: {str(e)}")

class AmountFromAI(TransactionAmount):
    """Special transaction amount class for AI-generated data with additional validation"""
    source_type: Literal["gemini_v1"] = "gemini_v1"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    
    @field_validator("amount", mode="before")
    @classmethod
    def validate_ai_amount(cls, value: Any, info: ValidationInfo) -> Decimal:
        """Enhanced validation for AI-generated amounts"""
        try:
            # Add AI-specific context
            context = (info.context or {}).copy()
            context["source"] = "gemini_ai"
            
            # Use parent class validator with context
            return super().validate_amount(value, info)
        except Exception as e:
            raise ValueError(f"AI amount validation failed: {str(e)}")
    
    @field_validator('currency')
    def validate_ai_currency(cls, v):
        if v not in ISO4217Currency.__members__:
            raise ValueError(f"Invalid AI-generated currency: {v}")
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

class SpendingAnalysis(BaseModel):
    """Model for spending analysis results"""
    total_spending: Annotated[Decimal, Field(description="Total spending amount")]
    category_breakdown: Annotated[Dict[str, Decimal], Field(description="Spending by category")]
    time_period: Annotated[str, Field(description="Analysis time period")]
    insights: Annotated[List[str], Field(description="Key spending insights")]
    recommendations: Annotated[List[str], Field(description="Budget recommendations")]
    
    @field_validator('total_spending')
    def validate_total(cls, v):
        """Ensure total is a valid decimal"""
        if not isinstance(v, (int, float, Decimal, str)):
            raise ValueError("Total spending must be a number")
        return Decimal(str(v))
    
    @field_validator('category_breakdown')
    def validate_breakdown(cls, v):
        """Validate category breakdown amounts"""
        validated = {}
        for category, amount in v.items():
            if not isinstance(amount, (int, float, Decimal, str)):
                raise ValueError(f"Invalid amount for category {category}")
            validated[str(category)] = Decimal(str(amount))
        return validated
    
    @field_validator('time_period')
    def validate_period(cls, v):
        """Validate time period format"""
        valid_periods = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']
        if v.lower() not in valid_periods:
            raise ValueError(f"Invalid time period. Must be one of: {', '.join(valid_periods)}")
        return v.lower() 