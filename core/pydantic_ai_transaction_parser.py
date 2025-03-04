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
    
    def __init__(self):
        """Initialize the transaction parser with configuration"""
        self.config = ConfigManager()
        # Configure Gemini client with API key
        api_key = self.config.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        # Get model names from environment variables with defaults
        self.reasoning_model = self.config.get("GEMINI_REASONER_MODEL") or "gemini-1.5-pro"
        self.other_model = self.config.get("GEMINI_OTHER_MODEL") or "gemini-1.5-flash"
        
        # Initialize Pydantic AI agent with Gemini backend
        pydantic_ai.settings.OPENAI_API_KEY = api_key  # Not used but required
        pydantic_ai.settings.GOOGLE_API_KEY = api_key
        
        # Initialize YNABClient to get account_id
        self.ynab_client = YNABClient()
        
        # Cache for account IDs
        self._default_account_id = None
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
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
            # Create a Pydantic AI agent
            agent = Agent(
                system_message="""
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
                """,
                model=models.Gemini(model_name=self.reasoning_model)
            )
            
            # Create a message structure for the agent
            messages = [
                {"role": "user", "content": query}
            ]
            
            # Run the agent and extract structured data
            result = await agent.arun(
                messages,
                output_schema=TransactionData
            )
            
            # Convert to TransactionCreate format
            amount_value = result.amount.value
            is_outflow = result.amount.is_outflow
            
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
                    transaction_date = DateFormatter.parse_date(result.date)
                except (ImportError, AttributeError):
                    # Fallback to basic parsing if DateFormatter is not available
                    transaction_date = datetime.strptime(result.date, "%Y-%m-%d").date()
                    
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
                payee_name=result.payee_name,
                payee_id=result.payee_id,
                memo=result.memo,
                category_name=result.category_name
            )
            
            logger.info(f"Successfully parsed transaction: {transaction}")
            
            return transaction
            
        except Exception as e:
            logger.error(f"Failed to parse transaction: {e}")
            raise ValueError(f"Could not parse transaction from: {query}. Error: {str(e)}")
    
    def parse_transaction_sync(self, query: str) -> TransactionCreate:
        """
        Parse transaction data synchronously from a natural language query
        
        Args:
            query (str): Natural language query to parse
            
        Returns:
            TransactionCreate: Parsed transaction data
            
        Raises:
            ValueError: If transaction cannot be parsed
        """
        self.logger.debug(f"Parsing transaction from query: query='{query}'")
        
        # Generate prompt for Gemini
        prompt = f"""
You are a transaction parser. Extract transaction details from user input and return a JSON object.

RULES:
1. Amount must be a positive number (e.g. 25.0)
   - Remove currency symbols ($ etc)
   - Convert words to numbers (e.g. "twenty five" -> 25.0)
   - Must be exact amount as specified
2. Payee name must be exactly as written in the query
   - For "at Target" -> use "Target"
   - For "from Walmart" -> use "Walmart"
   - For "paid Amazon" -> use "Amazon"
3. Date should be in YYYY-MM-DD format
   - Use today if not specified
   - Convert relative dates (e.g. "yesterday" -> actual date)
4. Use is_outflow=true for expenses (spending money)
   - Examples: "spent", "paid", "bought", "purchased"
5. Use is_outflow=false for income (receiving money)
   - Examples: "received", "got paid", "earned", "deposited"

Examples:

Input: "Create a transaction for $25 at Target"
{{
    "amount": 25.0,
    "is_outflow": true,
    "date": "2024-03-19",
    "payee_name": "Target",
    "account_id": null,
    "category_name": null,
    "memo": null,
    "cleared": "cleared",
    "approved": true
}}

Input: "I got paid $1000 from my employer yesterday"
{{
    "amount": 1000.0,
    "is_outflow": false,
    "date": "2024-03-18",
    "payee_name": "employer",
    "account_id": null,
    "category_name": "Income",
    "memo": null,
    "cleared": "cleared",
    "approved": true
}}

Input: "Spent twenty five dollars and fifty cents at Walmart"
{{
    "amount": 25.50,
    "is_outflow": true,
    "date": "2024-03-19",
    "payee_name": "Walmart",
    "account_id": null,
    "category_name": null,
    "memo": null,
    "cleared": "cleared",
    "approved": true
}}

Input: "Add transaction for coffee at Starbucks $4.75"
{{
    "amount": 4.75,
    "is_outflow": true,
    "date": "2024-03-19",
    "payee_name": "Starbucks",
    "account_id": null,
    "category_name": "Dining Out",
    "memo": "Coffee",
    "cleared": "cleared",
    "approved": true
}}

Input: "Deposited a check for $500 from John"
{{
    "amount": 500.0,
    "is_outflow": false,
    "date": "2024-03-19",
    "payee_name": "John",
    "account_id": null,
    "category_name": "Income",
    "memo": "Check deposit",
    "cleared": "uncleared",
    "approved": true
}}

Now parse this query: "{query}"
Return ONLY the JSON object, no other text.
"""
        
        try:
            # Generate response from Gemini with retry logic
            response_text = self._generate_with_retry(prompt)
            self.logger.debug(f"Raw Gemini response: {response_text}")
            
            # Extract JSON from response (handle markdown code blocks if present)
            json_text = self._extract_json_from_text(response_text)
            self.logger.debug(f"Extracted JSON: {json_text}")
            
            # Parse JSON into dictionary
            try:
                transaction_data = json.loads(json_text)
                self.logger.debug(f"Parsed transaction data: {transaction_data}")
                
                # Validate required fields are present and not None
                required_fields = ['amount', 'payee_name', 'date']
                for field in required_fields:
                    if field not in transaction_data or transaction_data[field] is None:
                        raise ValueError(f"Missing or null required field: {field}")
                
                # Handle is_outflow flag
                is_outflow = transaction_data.get('is_outflow', True)
                self.logger.debug(f"Transaction direction: is_outflow={is_outflow}")
                
                # Create TransactionAmount object
                try:
                    amount_value = float(transaction_data['amount'])
                    if amount_value <= 0:
                        raise ValueError(f"Amount must be positive. Received: {amount_value}")
                    if amount_value > 1000000:  # Sanity check - no transactions over $1M
                        raise ValueError(f"Amount exceeds maximum allowed value of $1,000,000. Received: {amount_value}")
                        
                    # Convert to milliunits (multiply by 1000)
                    milliunit_amount = int(amount_value * 1000)
                    
                    amount = TransactionAmount(
                        amount=Decimal(str(milliunit_amount)),
                        is_outflow=is_outflow
                    )
                    self.logger.debug(f"Transaction amount: amount_milliunits={amount.amount}, is_outflow={amount.is_outflow}")
                except (TypeError, ValueError) as e:
                    self.logger.error(f"Failed to parse amount: {e}")
                    # Provide more specific error message
                    if 'amount' not in transaction_data:
                        raise ValueError("Amount is missing from transaction data")
                    elif transaction_data['amount'] is None:
                        raise ValueError("Amount cannot be null")
                    elif isinstance(transaction_data['amount'], str) and not transaction_data['amount'].strip():
                        raise ValueError("Amount cannot be empty")
                    else:
                        raise ValueError(f"Invalid amount value: {transaction_data.get('amount')}")
                
                # Parse transaction date
                date_str = transaction_data.get('date')
                if not date_str:
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    self.logger.debug(f"No date provided, using today: date={date_str}")
                else:
                    self.logger.debug(f"Using provided date: date={date_str}")
                
                # Convert date string to date object
                try:
                    # First try to use DateFormatter from date_utils if available
                    try:
                        from core.date_utils import DateFormatter
                        transaction_date = DateFormatter.parse_date(date_str)
                    except (ImportError, AttributeError):
                        # Fallback to basic parsing if DateFormatter is not available
                        transaction_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                        
                    self.logger.debug(f"Parsed date: {transaction_date}")
                except Exception as e:
                    # If date parsing fails, use today's date
                    self.logger.warning(f"Date parsing failed: {e}. Using today's date")
                    transaction_date = datetime.now().date()
                
                # Create TransactionCreate object with validated data
                transaction = TransactionCreate(
                    account_id=transaction_data.get('account_id') or self.get_default_account_id(),
                    date=transaction_date,
                    amount=amount,
                    payee_name=transaction_data['payee_name'],
                    payee_id=transaction_data.get('payee_id'),
                    category_name=transaction_data.get('category_name'),
                    memo=transaction_data.get('memo'),
                    cleared=transaction_data.get('cleared'),
                    approved=transaction_data.get('approved', True)
                )
                self.logger.debug(f"Created transaction object: account_id='{transaction.account_id}', date='{transaction.date}', "
                                f"payee_name='{transaction.payee_name}', category_name='{transaction.category_name}', "
                                f"amount={transaction.amount.amount}, is_outflow={transaction.amount.is_outflow}")
                
                return transaction
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON response: error='{str(e)}', json_text='{json_text}'")
                raise ValueError(f"Failed to parse transaction: Invalid JSON response from model: {str(e)}")
            except Exception as e:
                self.logger.error(f"Transaction parsing failed: error='{str(e)}', query='{query}'")
                raise ValueError(f"Failed to parse transaction: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Transaction parsing failed: error='{str(e)}', query='{query}'")
            raise ValueError(f"Failed to parse transaction: {str(e)}")

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