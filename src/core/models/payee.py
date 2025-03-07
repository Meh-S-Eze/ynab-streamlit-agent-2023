"""
Payee data models for YNAB integration.

This module contains Pydantic models for YNAB payee data structures.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Annotated, Any


class Payee(BaseModel):
    """Model for a YNAB payee"""
    id: Annotated[str, Field(description="Payee ID")]
    name: Annotated[str, Field(description="Payee name")]
    transfer_account_id: Annotated[Optional[str], Field(None, description="Transfer account ID if this is a transfer payee")]
    deleted: Annotated[bool, Field(description="Whether the payee is deleted")]
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'Payee':
        """Create instance from YNAB API response"""
        return cls(
            id=data['id'],
            name=data['name'],
            transfer_account_id=data.get('transfer_account_id'),
            deleted=data.get('deleted', False)
        )
    
    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> 'Payee':
        """Alias for from_api_response for compatibility"""
        return cls.from_api_response(data)


class PayeeLocation(BaseModel):
    """Model for a YNAB payee location"""
    id: Annotated[str, Field(description="Payee location ID")]
    payee_id: Annotated[str, Field(description="Payee ID")]
    latitude: Annotated[Optional[str], Field(None, description="Latitude")]
    longitude: Annotated[Optional[str], Field(None, description="Longitude")]
    deleted: Annotated[bool, Field(default=False, description="Whether the payee location is deleted")]
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'PayeeLocation':
        """Create instance from YNAB API response"""
        return cls(
            id=data['id'],
            payee_id=data['payee_id'],
            latitude=data.get('latitude'),
            longitude=data.get('longitude'),
            deleted=data.get('deleted', False)
        )
    
    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> 'PayeeLocation':
        """Alias for from_api_response for compatibility"""
        return cls.from_api_response(data)


class PayeeMatch(BaseModel):
    """Model for payee matching results"""
    id: Annotated[str, Field(description="Payee ID")]
    name: Annotated[str, Field(description="Payee name")]
    score: Annotated[float, Field(description="Match score (0-1)")]
    
    @field_validator('score')
    def validate_score(cls, v):
        """Validate score is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError("Score must be between 0 and 1")
        return v


class PayeeUpdate(BaseModel):
    """Model for updating a YNAB payee"""
    name: Annotated[str, Field(description="Payee name")]
    
    @field_validator('name')
    def validate_name(cls, v):
        """Validate name is not empty"""
        if not v or not v.strip():
            raise ValueError("Payee name cannot be empty")
        return v.strip()


class PayeeCreate(BaseModel):
    """Model for creating a new YNAB payee"""
    name: Annotated[str, Field(description="Payee name")]
    
    @field_validator('name')
    def validate_name(cls, v):
        """Validate name is not empty"""
        if not v or not v.strip():
            raise ValueError("Payee name cannot be empty")
        return v.strip() 