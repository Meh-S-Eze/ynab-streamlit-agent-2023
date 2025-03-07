from typing import Optional
import os
from pydantic import BaseModel, Field
import logging

class APICredentials(BaseModel):
    """Model for API credentials"""
    ynab_token: str = Field(..., description="YNAB API token")
    ynab_budget_id: str = Field(..., description="YNAB budget ID")
    gemini_api_key: str = Field(..., description="Gemini API key")

class CredentialsManager:
    """
    Manages API credentials with proper dependency injection
    Implements the dependency injection for API credentials rule
    """
    def __init__(
        self,
        ynab_token: Optional[str] = None,
        ynab_budget_id: Optional[str] = None,
        gemini_api_key: Optional[str] = None
    ):
        """
        Initialize credentials manager
        
        Args:
            ynab_token (Optional[str]): YNAB API token
            ynab_budget_id (Optional[str]): YNAB budget ID
            gemini_api_key (Optional[str]): Gemini API key
        """
        self.logger = logging.getLogger(__name__)
        
        # Load credentials with priority:
        # 1. Constructor parameters
        # 2. Environment variables
        # 3. Raise error if not found
        self.credentials = APICredentials(
            ynab_token=ynab_token or os.getenv('YNAB_API_KEY'),
            ynab_budget_id=ynab_budget_id or os.getenv('YNAB_BUDGET_DEV'),
            gemini_api_key=gemini_api_key or os.getenv('GEMINI_API_KEY')
        )
        
        self.logger.info("Credentials manager initialized successfully")
    
    def get_ynab_token(self) -> str:
        """Get YNAB API token"""
        return self.credentials.ynab_token
    
    def get_ynab_budget_id(self) -> str:
        """Get YNAB budget ID"""
        return self.credentials.ynab_budget_id
    
    def get_gemini_api_key(self) -> str:
        """Get Gemini API key"""
        return self.credentials.gemini_api_key
    
    def update_credentials(
        self,
        ynab_token: Optional[str] = None,
        ynab_budget_id: Optional[str] = None,
        gemini_api_key: Optional[str] = None
    ):
        """
        Update credentials
        
        Args:
            ynab_token (Optional[str]): New YNAB API token
            ynab_budget_id (Optional[str]): New YNAB budget ID
            gemini_api_key (Optional[str]): New Gemini API key
        """
        if ynab_token:
            self.credentials.ynab_token = ynab_token
        if ynab_budget_id:
            self.credentials.ynab_budget_id = ynab_budget_id
        if gemini_api_key:
            self.credentials.gemini_api_key = gemini_api_key
        
        self.logger.info("Credentials updated successfully") 