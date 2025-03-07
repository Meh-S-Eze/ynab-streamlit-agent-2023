"""
Account data models for YNAB integration.

This module contains Pydantic models for YNAB account data structures.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Annotated, Any
from decimal import Decimal
from enum import Enum


class AccountType(str, Enum):
    """Enum for YNAB account types"""
    CHECKING = "checking"
    SAVINGS = "savings"
    CASH = "cash"
    CREDIT_CARD = "creditCard"
    LINE_OF_CREDIT = "lineOfCredit"
    OTHER_ASSET = "otherAsset"
    OTHER_LIABILITY = "otherLiability"
    MORTGAGE = "mortgage"
    AUTO_LOAN = "autoLoan"
    INVESTMENT = "investment"
    LOAN = "loan"
    PAYPAL = "payPal"
    MERCHANT = "merchantAccount"


class Account(BaseModel):
    """Model for a YNAB account"""
    id: Annotated[str, Field(description="Account ID")]
    name: Annotated[str, Field(description="Account name")]
    type: Annotated[AccountType, Field(description="Account type")]
    on_budget: Annotated[bool, Field(description="Whether the account is on budget")]
    closed: Annotated[bool, Field(description="Whether the account is closed")]
    balance: Annotated[int, Field(description="Current balance in milliunits")]
    cleared_balance: Annotated[int, Field(description="Cleared balance in milliunits")]
    uncleared_balance: Annotated[int, Field(description="Uncleared balance in milliunits")]
    transfer_payee_id: Annotated[Optional[str], Field(None, description="Transfer payee ID")]
    deleted: Annotated[bool, Field(description="Whether the account is deleted")]
    note: Annotated[Optional[str], Field(None, description="Account note")]
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'Account':
        """Create instance from YNAB API response"""
        return cls(
            id=data['id'],
            name=data['name'],
            type=data['type'],
            on_budget=data['on_budget'],
            closed=data['closed'],
            balance=data['balance'],
            cleared_balance=data['cleared_balance'],
            uncleared_balance=data['uncleared_balance'],
            transfer_payee_id=data.get('transfer_payee_id'),
            deleted=data['deleted'],
            note=data.get('note')
        )
    
    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> 'Account':
        """Alias for from_api_response for compatibility"""
        return cls.from_api_response(data)
    
    def get_balance_decimal(self) -> Decimal:
        """Get the balance as a Decimal value"""
        return Decimal(self.balance) / 1000
    
    def get_cleared_balance_decimal(self) -> Decimal:
        """Get the cleared balance as a Decimal value"""
        return Decimal(self.cleared_balance) / 1000
    
    def get_uncleared_balance_decimal(self) -> Decimal:
        """Get the uncleared balance as a Decimal value"""
        return Decimal(self.uncleared_balance) / 1000


class AccountSummary(BaseModel):
    """Model for a YNAB account summary"""
    id: Annotated[str, Field(description="Account ID")]
    name: Annotated[str, Field(description="Account name")]
    type: Annotated[AccountType, Field(description="Account type")]
    on_budget: Annotated[bool, Field(description="Whether the account is on budget")]
    closed: Annotated[bool, Field(description="Whether the account is closed")]
    balance: Annotated[int, Field(description="Current balance in milliunits")]
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'AccountSummary':
        """Create instance from YNAB API response"""
        return cls(
            id=data['id'],
            name=data['name'],
            type=data['type'],
            on_budget=data['on_budget'],
            closed=data['closed'],
            balance=data['balance']
        )
    
    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> 'AccountSummary':
        """Alias for from_api_response for compatibility"""
        return cls.from_api_response(data)
    
    def get_balance_decimal(self) -> Decimal:
        """Get the balance as a Decimal value"""
        return Decimal(self.balance) / 1000


class AccountUpdate(BaseModel):
    """Model for updating a YNAB account"""
    name: Annotated[Optional[str], Field(None, description="Account name")]
    type: Annotated[Optional[AccountType], Field(None, description="Account type")]
    balance: Annotated[Optional[int], Field(None, description="Current balance in milliunits")]
    note: Annotated[Optional[str], Field(None, description="Account note")]
    
    @field_validator('balance')
    def validate_balance(cls, v):
        """Validate balance is a valid integer"""
        if v is not None and not isinstance(v, int):
            raise ValueError("Balance must be an integer in milliunits")
        return v
