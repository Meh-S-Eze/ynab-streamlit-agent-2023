from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, Field, validator, constr
from datetime import datetime, date
from decimal import Decimal
import structlog
import logging.config
import re
import html
import uuid
from enum import Enum
from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path
import json
from .shared_models import TransactionCreate, TransactionAmount, ConfidenceResult
import os

# Configure structured logging
def setup_logging():
    """Configure structured logging with proper routing and formatting"""
    # Ensure log directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Base handlers configuration
    handlers = {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "console",
        },
        "json_file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/debug.json",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json",
        },
        "error_file": {
            "level": "ERROR",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/error.json",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json",
        }
    }
    
    # Add Slack handler if webhook URL is configured
    if os.getenv('SLACK_WEBHOOK_URL'):
        handlers["slack"] = {
            "level": "ERROR",
            "class": "core.logging_handlers.SlackHandler",
            "formatter": "json",
            "channel": "#alerts",
        }
    
    # Configure standard logging
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),
            },
            "console": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.dev.ConsoleRenderer(),
            },
        },
        "handlers": handlers,
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "json_file"],
                "level": "INFO",
            },
            "data_validation": {
                "handlers": ["json_file", "error_file"] + (["slack"] if "slack" in handlers else []),
                "level": "DEBUG",
                "propagate": False,
            },
        },
    })
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# Initialize logging configuration
setup_logging()

# Create structured logger
logger = structlog.get_logger("data_validation")

# Schema version for migration tracking
SCHEMA_VERSION = "1.0.0"

class ISO4217Currency(str, Enum):
    """Valid ISO 4217 currency codes"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"
    AUD = "AUD"
    # Add other currencies as needed

@dataclass
class ValidationMetrics:
    """Track validation error counts by type"""
    error_counts: Counter = field(default_factory=Counter)
    schema_version: str = "1.0"

    def increment_error(self, error_type: str) -> None:
        """Increment the count for a specific error type"""
        self.error_counts[error_type] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current validation metrics"""
        return {
            "error_counts": dict(self.error_counts),
            "schema_version": self.schema_version,
            "total_errors": sum(self.error_counts.values())
        }

# Global metrics tracker
validation_metrics = ValidationMetrics()

class IdempotencyKey(BaseModel):
    """Model for idempotency key validation"""
    key: str = Field(..., min_length=32, max_length=64)
    expiry: datetime
    
    @validator('key')
    def validate_key_format(cls, v):
        """Ensure key is a valid UUID or similar token"""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Invalid idempotency key format")
        return v

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
    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version for tracking")
    idempotency_key: Optional[str] = Field(None, description="Idempotency key for POST requests")

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

    @validator('category_name')
    def sanitize_category_name(cls, v):
        """Sanitize category name to prevent injection"""
        return html.escape(v.strip())

class YNABUpdatePayload(BaseModel):
    """Model for validating YNAB update payloads"""
    transaction_id: str = Field(..., description="ID of the transaction to update")
    category_id: Optional[str] = Field(None, description="New category ID")
    category_name: Optional[str] = Field(None, description="New category name")
    memo: Optional[constr(max_length=200)] = Field(None, description="Updated memo")
    cleared: Optional[str] = Field(None, description="Updated cleared status")
    currency: ISO4217Currency = Field(default=ISO4217Currency.USD, description="Transaction currency")
    amount: Decimal = Field(..., ge=Decimal('0.01'), description="Transaction amount")
    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version for tracking")
    idempotency_key: Optional[str] = Field(None, description="Idempotency key for POST requests")

    @validator('cleared')
    def validate_cleared(cls, v):
        """Validate cleared status"""
        if v is not None:
            valid_statuses = ['cleared', 'uncleared', 'reconciled']
            if v not in valid_statuses:
                validation_metrics.increment_error('invalid_cleared_status')
                raise ValueError(f"Invalid cleared status. Must be one of: {', '.join(valid_statuses)}")
        return v

    @validator('memo')
    def sanitize_memo(cls, v):
        """Sanitize memo field to prevent XSS"""
        if v:
            # Strip HTML tags and escape special characters
            v = re.sub(r'<[^>]+>', '', v)
            return html.escape(v.strip())
        return v

    @validator('idempotency_key')
    def validate_idempotency(cls, v):
        """Validate idempotency key if provided"""
        if v:
            try:
                IdempotencyKey(key=v, expiry=datetime.now())
            except ValueError as e:
                validation_metrics.increment_error('invalid_idempotency_key')
                raise ValueError(f"Invalid idempotency key: {str(e)}")
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
                # Add idempotency key for POST requests
                if 'idempotency_key' not in result:
                    result['idempotency_key'] = str(uuid.uuid4())
                
                validated = GeminiAnalysisResult(**result)
                validated_results.append(validated)
            except Exception as e:
                validation_metrics.increment_error('gemini_analysis_validation_error')
                # Log with structured data
                logger.warning(
                    "validation_error.gemini_analysis",
                    error=str(e),
                    error_type=type(e).__name__,
                    schema_version=SCHEMA_VERSION,
                    masked_data=DataValidator._mask_pii(result),
                    validation_type="gemini_analysis",
                    correlation_id=result.get('idempotency_key', 'unknown')
                )
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
                # Add idempotency key for POST requests
                if 'idempotency_key' not in update:
                    update['idempotency_key'] = str(uuid.uuid4())
                
                validated = YNABUpdatePayload(**update)
                validated_updates.append(validated)
            except Exception as e:
                validation_metrics.increment_error('ynab_update_validation_error')
                # Log with structured data
                logger.error(
                    "validation_error.ynab_update",
                    error=str(e),
                    error_type=type(e).__name__,
                    schema_version=SCHEMA_VERSION,
                    masked_data=DataValidator._mask_pii(update),
                    validation_type="ynab_update",
                    correlation_id=update.get('idempotency_key', 'unknown')
                )
                raise ValueError(f"Invalid update payload: {str(e)}")
        return validated_updates

    @staticmethod
    def _mask_pii(data: Dict) -> Dict:
        """Mask sensitive information in data for logging"""
        masked = data.copy()
        pii_fields = ['memo', 'payee_name', 'account_id']
        for field in pii_fields:
            if field in masked:
                value = str(masked[field])
                if len(value) > 4:
                    masked[field] = f"{value[:2]}...{value[-2:]}"
                else:
                    masked[field] = "****"
        return masked

    @staticmethod
    def get_validation_metrics() -> Dict[str, int]:
        """Get current validation error metrics"""
        metrics = validation_metrics.get_metrics()
        # Log metrics for monitoring
        logger.info(
            "validation_metrics",
            metrics=metrics,
            schema_version=SCHEMA_VERSION
        )
        return metrics

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
            category_name=analysis_result.category_name,
            idempotency_key=analysis_result.idempotency_key
        )

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