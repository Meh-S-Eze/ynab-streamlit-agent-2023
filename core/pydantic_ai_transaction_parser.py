"""
Enhanced transaction parser using Pydantic AI for more robust parsing
"""
import os
import logging
from typing import Optional, Dict, Any, List, Union
from decimal import Decimal
from datetime import date, datetime
import re
import json
from functools import lru_cache
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from collections import deque
import asyncio

from pydantic import Field, BaseModel, ConfigDict
import pydantic_ai
from pydantic_ai import Agent, models
import google.generativeai as genai

from .shared_models import TransactionAmount, TransactionCreate
from .config import ConfigManager
from .ynab_client import YNABClient

# Configure logging
logger = logging.getLogger(__name__)
# Set logging level for this module
logger.setLevel(logging.DEBUG)

class TransactionInputModel(BaseModel):
    """Input model for transaction parsing"""
    query: str = Field(..., description="The natural language query describing a transaction")

class AmountData(BaseModel):
    """Model for parsed amount data"""
    value: float = Field(..., description="The numerical value of the transaction amount")
    is_outflow: bool = Field(True, description="Whether this is an expense (True) or income (False)")
    
    model_config = ConfigDict(
        extra="allow"
    )

class TransactionData(BaseModel):
    """
    Model for parsed transaction data with comprehensive validation
    """
    amount: AmountData = Field(..., description="The transaction amount details")
    payee_name: str = Field(..., description="The name of the merchant or payee")
    payee_id: Optional[str] = Field(None, description="The YNAB payee ID for existing payees")
    date: str = Field(..., description="Transaction date in ISO 8601 format (YYYY-MM-DD)")
    memo: Optional[str] = Field(None, description="Optional transaction description")
    category_name: Optional[str] = Field(None, description="Optional category name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the parsing (0-1)")
    
    model_config = ConfigDict(
        extra="allow"
    )

class PydanticAITransactionParser:
    """
    Transaction parser using Pydantic AI for more robust extraction from natural language
    """
    
    def __init__(self, payee_cache=None):
        """
        Initialize the transaction parser with AI model configuration
        
        Args:
            payee_cache: Optional cache of payees to use for payee lookup
        """
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize rate limiting
        self.request_times = deque(maxlen=10)
        
        # Initialize model config using environment variables
        self.reasoning_model = os.environ.get('GEMINI_REASONER_MODEL')
        if not self.reasoning_model:
            self.logger.warning("GEMINI_REASONER_MODEL not set in environment")
        
        # Get API key and region from environment
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            self.logger.warning("GOOGLE_API_KEY not found in environment.")
            
        # Initialize YNAB client for account and payee lookups
        from core.ynab_client import YNABClient
        self.ynab_client = YNABClient()
        
        # Initialize payee cache for lookups
        self._initialize_payee_cache()
        
        # Initialize Pydantic AI agent with Gemini backend
        pydantic_ai.settings.OPENAI_API_KEY = self.api_key  # Not used but required
        pydantic_ai.settings.GOOGLE_API_KEY = self.api_key
        
        # Cache for account IDs
        self._default_account_id = None
        
        # Initialize Gemini model with rate limiting
        self.model = genai.GenerativeModel(self.reasoning_model)
        self.last_api_call = 0
        self.min_delay = 2.0  # Minimum delay between API calls in seconds
        
        self.logger.info(f"Initialized PydanticAITransactionParser with model: {self.reasoning_model}")
        
    def _rate_limit(self):
        """Implement basic rate limiting"""
        now = time.time()
        elapsed = now - self.last_api_call
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_api_call = time.time()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=30))
    def _generate_with_retry(self, prompt: str) -> str:
        """Generate content with retry logic for API quota limits"""
        try:
            self._rate_limit()  # Apply rate limiting
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.0}
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                self.logger.warning("API quota limit hit, retrying after delay...")
                raise  # Let retry handle it
            raise ValueError(f"Failed to generate content: {str(e)}")

    @lru_cache(maxsize=1)
    def get_default_account_id(self):
        """
        Get the default account ID to use for transactions.
        This method is cached to avoid frequent API calls.
        
        Returns:
            str: The default account ID to use
        """
        try:
            # Get accounts from YNAB API
            accounts = self.ynab_client.get_accounts()
            
            # Use the first active account, or default to the first account if none are active
            default_account_id = None
            for account in accounts:
                if not account.get("closed", False):
                    default_account_id = account["id"]
                    logger.info(f"Using active account: {account.get('name')} ({default_account_id})")
                    break
            
            if not default_account_id and accounts:
                default_account_id = accounts[0]["id"]
                logger.info(f"Using first available account: {accounts[0].get('name')} ({default_account_id})")
            
            if not default_account_id:
                logger.warning("No available accounts found. Using budget_id as fallback.")
                default_account_id = self.ynab_client.budget_id
            
            return default_account_id
        except Exception as e:
            logger.error(f"Error getting default account ID: {e}")
            # Fallback to budget ID in case of error
            return self.ynab_client.budget_id
    
    async def parse_transaction(self, query: str) -> TransactionCreate:
        """
        Parse a natural language query into a structured transaction using Pydantic AI
        
        Args:
            query (str): Natural language query describing a transaction
            
        Returns:
            TransactionCreate: Structured transaction data
        """
        # Log using standard logging
        logger.info(f"Parsing transaction from query: {query}")
            
        try:
            # Create a Pydantic AI agent - using model name directly instead of the models.Gemini class
            agent = Agent(
                self.reasoning_model,  # Direct model name instead of models.Gemini
                system_prompt="""
                Extract transaction details from the user's input. Follow these rules:
                1. Extract the exact amount (numbers only)
                2. Determine if it's an expense (outflow) or income (inflow)
                3. Extract the payee name exactly as written
                4. Extract or infer the date (use today if unspecified)
                5. Extract any memo or additional notes
                6. Extract or infer the category name if present
                7. Assign a confidence score based on how clear the information is
                
                For dates, resolve relative terms like "yesterday" or "last week" to actual dates.
                For payee names, extract only the business or person name without additional words.
                Don't include words like "at", "from", "to", "for" in the payee name.
                Extract the clearest, most specific merchant or payee name possible.
                """,
                result_type=TransactionData
            )
            
            # Run the agent to extract structured data
            result = await agent.run(query)
            transaction_data = result.data
            
            # Look up payee ID if we have a payee name
            if transaction_data.payee_name and not transaction_data.payee_id:
                transaction_data.payee_id = self.get_payee_id(transaction_data.payee_name)
                if transaction_data.payee_id:
                    self.logger.info(f"Matched payee '{transaction_data.payee_name}' to existing payee ID: {transaction_data.payee_id}")
            
            # Convert to TransactionCreate format
            amount_value = transaction_data.amount.value
            is_outflow = transaction_data.amount.is_outflow
            
            # Format amount with correct sign
            amount = TransactionAmount(
                amount=Decimal(str(abs(amount_value))),
                is_outflow=is_outflow
            )
            
            # Convert date string to date object
            try:
                # First try to use DateFormatter from date_utils if available
                try:
                    from core.date_utils import DateFormatter
                    transaction_date = DateFormatter.parse_date(transaction_data.date)
                except (ImportError, AttributeError):
                    # Fallback to basic parsing if DateFormatter is not available
                    transaction_date = datetime.strptime(transaction_data.date, "%Y-%m-%d").date()
                    
                logger.debug(f"Parsed date: {transaction_date}")
            except Exception as e:
                # If date parsing fails, use today's date
                logger.warning(f"Date parsing failed: {e}. Using today's date")
                transaction_date = datetime.now().date()
            
            # Create the transaction object with required fields
            transaction = TransactionCreate(
                account_id=transaction_data.get('account_id') or self.get_default_account_id(),
                date=transaction_date,
                amount=amount,
                payee_name=transaction_data.payee_name if not transaction_data.payee_id else None,  # Only use payee_name if no payee_id
                payee_id=transaction_data.payee_id,
                memo=transaction_data.memo,
                category_name=transaction_data.category_name
            )
            
            logger.info(f"Successfully parsed transaction: {transaction}")
            
            return transaction
            
        except Exception as e:
            logger.error(f"Failed to parse transaction: {e}")
            raise ValueError(f"Could not parse transaction from: {query}. Error: {str(e)}")
    
    def parse_transaction_sync(self, query: str) -> TransactionCreate:
        """
        Synchronous version of parse_transaction
        
        Args:
            query (str): Natural language query describing a transaction
            
        Returns:
            TransactionCreate: Structured transaction data
            
        Raises:
            ValueError: If the transaction cannot be parsed by the AI model
        """
        # Get event loop or create one if doesn't exist
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the async function in the event loop
        try:
            result = loop.run_until_complete(self.parse_transaction(query))
            return result
        except Exception as e:
            # No fallback - per architecture principles, we focus on improving AI parsing
            self.logger.error(f"Error in AI transaction parsing: {str(e)}")
            # Return a more helpful error message
            error_msg = f"Unable to parse transaction using AI model: {str(e)}"
            self.logger.info(f"Raising error: {error_msg}")
            raise ValueError(error_msg)
    
    def _extract_json_from_text(self, text: str) -> str:
        """
        Extract JSON content from text, handling markdown code blocks
        
        Args:
            text (str): Raw text that may contain JSON
            
        Returns:
            str: Extracted JSON text
        """
        # Clean up the response text (remove markdown code blocks if present)
        if '```' in text:
            # First try to match ```json ... ``` format
            json_match = re.search(r'```(?:json)?\n(.*?)\n```', text, re.DOTALL)
            if json_match:
                return json_match.group(1).strip()
            
            # If that fails, try to match just the content between any ``` marks
            json_match = re.search(r'```(.*?)```', text, re.DOTALL)
            if json_match:
                return json_match.group(1).strip()
        
        # If no code blocks, return the original text stripped
        return text.strip() 

    def _initialize_payee_cache(self):
        """Initialize the payee cache from YNAB for lookups"""
        try:
            self.logger.info("Initializing payee cache")
            # This will trigger the YNAB client to build its internal payee cache
            self.ynab_client._initialize_payee_cache()
            self.logger.info("Payee cache initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize payee cache: {str(e)}")
    
    def get_payee_id(self, payee_name: str) -> Optional[str]:
        """
        Look up a payee ID by name from the YNAB cache
        
        Args:
            payee_name (str): Name of the payee to look up
            
        Returns:
            Optional[str]: Payee ID if found, None otherwise
        """
        if not payee_name:
            return None
            
        try:
            # Use the YNAB client's get_payee_id method
            payee_id = self.ynab_client.get_payee_id(payee_name)
            if payee_id:
                self.logger.debug(f"Found payee ID for '{payee_name}': {payee_id}")
            else:
                self.logger.debug(f"No payee ID found for '{payee_name}'")
            return payee_id
        except Exception as e:
            self.logger.error(f"Error looking up payee ID: {str(e)}")
            return None
            
    def get_payee_suggestions(self, partial_name: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Get suggestions for payees based on partial name match
        
        Args:
            partial_name (str): Partial payee name to match
            max_results (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, str]]: List of payee matches with id and name
        """
        try:
            if not partial_name or len(partial_name) < 2:
                return []
                
            # Get all payees from YNAB
            payees = self.ynab_client.get_payees()
            
            # Filter payees by partial name match (case insensitive)
            partial_name_lower = partial_name.lower()
            matches = [
                {"id": p["id"], "name": p["name"]}
                for p in payees
                if not p.get("deleted", False) 
                and partial_name_lower in p.get("name", "").lower()
            ]
            
            # Sort by closest match and limit results
            matches.sort(key=lambda p: p["name"].lower().find(partial_name_lower))
            return matches[:max_results]
            
        except Exception as e:
            self.logger.error(f"Error getting payee suggestions: {str(e)}")
            return [] 

    def gen_transaction(self, model, query):
        """
        Generate a transaction from a natural language query using Gemini model
        
        Args:
            model: Gemini model to use
            query: Natural language query to parse
            
        Returns:
            dict: Parsed transaction data
        """
        # Define system prompt for transaction parsing
        prompt = f"""You are a YNAB (You Need A Budget) transaction parser specializing in extracting structured financial data from natural language inputs. Your task is to accurately parse transaction details and format them as a valid JSON object.

USER TRANSACTION INPUT:
"{query}"

TRANSACTION PARSING RULES:

1. AMOUNT EXTRACTION:
   - Must be a positive number (e.g., 25.0)
   - Remove currency symbols ($ etc.) from final value
   - Convert words to numbers ("twenty five dollars" → 25.0)
   - Preserve exact decimal precision if specified
   - Extract the amount exactly as mentioned in the query

2. PAYEE/MERCHANT IDENTIFICATION:
   - Extract only the merchant/business name
   - For "at Target" → use "Target"
   - For "from Walmart" → use "Walmart"
   - For "paid Amazon" → use "Amazon"
   - Remove unnecessary details but keep full business name
   - Use the most specific merchant name possible (e.g., "Walmart Supercenter" instead of just "Walmart")
   - Extract the clearest possible payee name for matching with existing payees in YNAB

3. DATE FORMATTING:
   - Use YYYY-MM-DD format (ISO 8601)
   - Use today's date if no date is specified
   - Convert relative dates ("yesterday" → the actual date)
   - Convert month names to numbers ("January 5" → "2023-01-05")

4. TRANSACTION DIRECTION:
   - is_outflow=true for expenses (spending money)
     * Keywords: "spent", "paid", "bought", "purchased", "paid for"
   - is_outflow=false for income (receiving money)
     * Keywords: "received", "got paid", "earned", "deposited", "refunded"
   - Default to is_outflow=true if unclear

5. ADDITIONAL FIELDS:
   - memo: Brief description of what the transaction was for
   - category_name: Appropriate budget category (if determinable)
   - confidence: Your confidence in the parsing (0.0 to 1.0)

EXAMPLE TRANSACTION INPUTS AND OUTPUTS:

Example 1 - Basic Purchase:
Input: "I spent $45.99 at Target yesterday"
{
  "amount": {
    "value": 45.99,
    "is_outflow": true
  },
  "payee_name": "Target",
  "date": "2023-06-15",
  "memo": "Purchase at Target",
  "category_name": "Shopping",
  "confidence": 0.95
}

Example 2 - Income Transaction:
Input: "Received $1,250 salary deposit from Acme Corp"
{
  "amount": {
    "value": 1250.0,
    "is_outflow": false
  },
  "payee_name": "Acme Corp",
  "date": "2023-06-16",
  "memo": "Salary deposit",
  "category_name": "Income",
  "confidence": 0.98
}

Example 3 - Specific Date:
Input: "Paid electric bill $89.50 on April 5"
{
  "amount": {
    "value": 89.50,
    "is_outflow": true
  },
  "payee_name": "Electric Company",
  "date": "2023-04-05",
  "memo": "Electric bill payment",
  "category_name": "Utilities",
  "confidence": 0.90
}

Example 4 - Amount in Words:
Input: "Dinner at Mario's Restaurant cost twenty-five dollars"
{
  "amount": {
    "value": 25.0,
    "is_outflow": true
  },
  "payee_name": "Mario's Restaurant",
  "date": "2023-06-16",
  "memo": "Dinner expense",
  "category_name": "Dining Out",
  "confidence": 0.88
}

EXPECTED JSON RESPONSE FORMAT:
{
  "amount": {
    "value": 0.0,  // Numeric value as float
    "is_outflow": true  // Boolean: true for expenses, false for income
  },
  "payee_name": "Merchant Name",  // String
  "date": "YYYY-MM-DD",  // ISO 8601 formatted date as string
  "memo": "Transaction description",  // String
  "category_name": "Budget Category",  // String
  "confidence": 0.0  // Float between 0 and 1
}

IMPORTANT REQUIREMENTS:
- Return ONLY the valid JSON object with no additional text or explanation
- Ensure all required fields are included and properly formatted
- Use null for truly optional fields if information is not present or unclear
- Format all numeric values as actual numbers, not strings
- Do not include any Markdown formatting or code block markers
- Extract the clearest, most specific merchant name possible for matching with existing YNAB payees
""" 