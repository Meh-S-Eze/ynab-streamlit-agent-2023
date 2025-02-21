import google.generativeai as genai
from typing import Dict, List, Optional, Protocol, runtime_checkable, Union
from .config import ConfigManager
from .circuit_breaker import CircuitBreaker
from .ynab_client import YNABClient
from pydantic import BaseModel, ValidationError, Field, validator
import logging
import json
import importlib
import sys
import os
import re
from datetime import date, datetime
from decimal import Decimal
import requests
from .shared_models import TransactionCreate, TransactionAmount
from .data_validation import DataValidator

@runtime_checkable
class AnalysisModule(Protocol):
    """
    Protocol for defining analysis modules with a standard interface
    Follows the abstract base classes for analysis modules rule
    """
    def analyze(self, data: List[Dict]) -> Dict:
        """
        Standard analysis method for all modules
        
        Args:
            data (List[Dict]): Input data to analyze
        
        Returns:
            Dict: Analysis results
        """
        ...

class ConfidenceResult(BaseModel):
    """
    Structured result with confidence scoring and alternative categories
    Implements AI-powered analysis with confidence scoring
    """
    category: str = Field(..., description="Primary category assigned to the transaction")
    confidence: float = Field(
        ge=0, 
        le=1, 
        description="Confidence score between 0 and 1"
    )
    reasoning: Optional[str] = Field(
        None, 
        description="Explanation for the category assignment"
    )
    transaction_ids: List[str] = Field(
        default_factory=list,
        description="List of transaction IDs this categorization applies to"
    )
    alternative_categories: List[Dict[str, Union[str, float]]] = Field(
        default_factory=list,
        description="List of alternative categories with their confidence scores"
    )
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Ensure confidence score is properly normalized"""
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
    
    class Config:
        """Pydantic model configuration"""
        arbitrary_types_allowed = True
        json_encoders = {
            Decimal: str
        }

class SpendingAnalysis(BaseModel):
    total_spent: float = Field(..., description="Total amount spent")
    category_breakdown: Dict[str, float] = Field(..., description="Spending by category")
    unusual_transactions: List[Dict] = Field(default_factory=list, description="Transactions flagged as unusual")

class GeminiHallucinationError(Exception):
    """Raised when Gemini's response is detected as a hallucination"""
    pass

class InvalidGeminiResponseError(Exception):
    """Raised when Gemini's response cannot be parsed or validated"""
    pass

class GeminiSpendingAnalyzer:
    """
    Enhanced spending analyzer with plugin support and confidence scoring
    """
    def __init__(self, config_manager: Optional[ConfigManager] = None, ynab_client: Optional[YNABClient] = None):
        """
        Initialize analyzer with dependency injection
        
        Args:
            config_manager (Optional[ConfigManager]): Configuration manager
            ynab_client (Optional[YNABClient]): YNAB client for API operations
        """
        self.config = config_manager or ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # Store YNAB client
        self.ynab_client = ynab_client
        
        # Plugin management
        self.analysis_modules: List[AnalysisModule] = []
        self._discover_plugins()
        
        # Configure Gemini model
        genai.configure(api_key=self.config.get('credentials.gemini.api_key'))
        self.model = genai.GenerativeModel('gemini-pro')

    def _discover_plugins(self):
        """
        Discover and load analysis modules dynamically
        Follows plugin-based expansion rule
        """
        try:
            # Simple plugin discovery in ai_modules directory
            plugin_dir = os.path.join(os.path.dirname(__file__), '..', 'ai_modules')
            if os.path.exists(plugin_dir):
                sys.path.insert(0, plugin_dir)
                
                for filename in os.listdir(plugin_dir):
                    if filename.endswith('_module.py'):
                        module_name = filename[:-3]  # Remove .py
                        try:
                            module = importlib.import_module(module_name)
                            for name, obj in module.__dict__.items():
                                if (isinstance(obj, type) and 
                                    issubclass(obj, AnalysisModule) and 
                                    obj is not AnalysisModule):
                                    self.register_analysis_module(obj())
                        except Exception as e:
                            self.logger.warning(f"Could not load plugin {module_name}: {e}")
        except Exception as e:
            self.logger.error(f"Plugin discovery failed: {e}")

    def register_analysis_module(self, module: AnalysisModule):
        """
        Register a new analysis module
        
        Args:
            module (AnalysisModule): Module to register
        """
        self.analysis_modules.append(module)
        self.logger.info(f"Registered analysis module: {module.__class__.__name__}")

    def ai_category_matcher(self, 
                             transactions: List[Dict], 
                             existing_categories: List[Dict]) -> List[Dict]:
        """
        Use AI to intelligently match transactions to categories
        
        Args:
            transactions (List[Dict]): List of transactions to categorize
            existing_categories (List[Dict]): List of existing YNAB categories
        
        Returns:
            List of transactions with AI-suggested categories
        """
        # Prepare prompt with context
        prompt = f"""
        You are an expert financial categorization assistant. 
        Your task is to match transactions to the most appropriate category.

        Existing Categories: {json.dumps(existing_categories)}

        For each transaction, provide:
        1. Best matching category name
        2. Confidence score (0-100%)
        3. Reasoning for the match

        Transaction Details Format:
        {{
            "id": "transaction_id",
            "description": "Transaction description",
            "amount": transaction_amount,
            "date": "transaction_date"
        }}

        Transactions: {json.dumps(transactions[:10])}  # Limit to first 10 for token management

        Response Format (JSON):
        [
            {{
                "transaction_id": "id",
                "suggested_category": "Category Name",
                "confidence": 85,
                "reasoning": "Detailed explanation of category match"
            }}
        ]
        """

        try:
            # Generate category matching recommendations
            response = self.model.generate_content(
                prompt, 
                generation_config={
                    'temperature': 0.2,  # More deterministic
                    'max_output_tokens': 1024
                }
            )

            # Parse and validate response
            try:
                parsed_response = json.loads(response.text)
                
                # Validate response structure
                if not isinstance(parsed_response, list):
                    raise ValueError("Response is not a list of category matches")
                
                return parsed_response
            
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse AI category matching response: {e}")
                return []

        except Exception as e:
            self.logger.error(f"AI Category Matching failed: {e}")
            return []

    def _validate_gemini_response(self, response_text: str, expected_fields: List[str]) -> Dict:
        """
        Validate Gemini's response for hallucinations and required fields
        
        Args:
            response_text (str): Raw response from Gemini
            expected_fields (List[str]): List of required fields
        
        Returns:
            Dict: Validated response data
            
        Raises:
            GeminiHallucinationError: If response appears to be a hallucination
            InvalidGeminiResponseError: If response cannot be parsed or is invalid
        """
        try:
            # Clean response text
            clean_text = response_text.strip()
            if '```' in clean_text:
                clean_text = re.search(r'```(?:json)?\n?(.*?)\n?```', clean_text, re.DOTALL).group(1)
            
            # Try to parse JSON
            try:
                parsed_data = json.loads(clean_text)
            except json.JSONDecodeError as e:
                raise InvalidGeminiResponseError(f"Failed to parse JSON response: {e}")
            
            # Check for hallucination indicators
            if isinstance(parsed_data, dict):
                # Check for nonsensical values
                for key, value in parsed_data.items():
                    if isinstance(value, (int, float)) and value > 1e6:  # Unreasonably large numbers
                        raise GeminiHallucinationError(f"Unreasonable value detected: {key}={value}")
                    if isinstance(value, str) and len(value) > 1000:  # Extremely long strings
                        raise GeminiHallucinationError(f"Unreasonably long string detected for {key}")
            
            # Validate required fields
            missing_fields = [field for field in expected_fields if field not in parsed_data]
            if missing_fields:
                raise InvalidGeminiResponseError(f"Missing required fields: {', '.join(missing_fields)}")
            
            return parsed_data
            
        except (AttributeError, TypeError) as e:
            raise InvalidGeminiResponseError(f"Invalid response structure: {e}")

    @CircuitBreaker()
    def categorize_transactions(self, transactions: List[Dict], ynab_categories: Optional[List[Dict]] = None) -> Dict:
        """
        Categorize transactions with enhanced error handling
        """
        self.logger.info(f"Categorizing {len(transactions)} transactions")
        
        try:
            # Generate categorization with enhanced context
            response = self.model.generate_content(
                self._create_categorization_prompt(transactions, ynab_categories),
                generation_config={
                    'temperature': 0.2,
                    'max_output_tokens': 2048
                }
            )
            
            # Validate response
            required_fields = ['transaction_id', 'category_name', 'confidence', 'reasoning']
            try:
                categorization_results = self._validate_gemini_response(
                    response.text,
                    required_fields
                )
            except GeminiHallucinationError as he:
                self.logger.error(f"Detected hallucination in Gemini response: {he}")
                # Retry with more strict parameters
                response = self.model.generate_content(
                    self._create_categorization_prompt(transactions, ynab_categories),
                    generation_config={
                        'temperature': 0.1,  # Lower temperature for more focused response
                        'max_output_tokens': 1024,  # Limit response size
                        'top_p': 0.7,  # More focused token selection
                        'top_k': 20  # Limit vocabulary
                    }
                )
                # Validate retry response
                categorization_results = self._validate_gemini_response(
                    response.text,
                    required_fields
                )
            
            # Validate results through DataValidator
            validated_results = DataValidator.validate_gemini_analysis(categorization_results)
            
            # Convert validated results to YNAB update payloads
            update_payloads = []
            for result in validated_results:
                update_payloads.append(
                    DataValidator.prepare_ynab_payload(result).dict()
                )
            
            return {
                'categorization_results': [result.dict() for result in validated_results],
                'update_payloads': update_payloads,
                'total_processed': len(validated_results),
                'high_confidence_count': sum(1 for r in validated_results if r.confidence >= 0.7),
                'low_confidence_count': sum(1 for r in validated_results if r.confidence < 0.4),
                'average_confidence': sum(r.confidence for r in validated_results) / len(validated_results) if validated_results else 0
            }
            
        except GeminiHallucinationError as he:
            self.logger.error(f"Gemini hallucination detected even after retry: {he}")
            return {
                'error': 'hallucination_detected',
                'message': str(he),
                'recommendation': 'Please review transactions manually'
            }
        except InvalidGeminiResponseError as ie:
            self.logger.error(f"Invalid Gemini response: {ie}")
            return {
                'error': 'invalid_response',
                'message': str(ie),
                'recommendation': 'Please try again with fewer transactions'
            }
        except Exception as e:
            self.logger.error(f"Categorization failed: {e}")
            raise

    def _generate_category_examples(self, category_name: str) -> List[str]:
        """Generate example transactions for a category to improve matching"""
        category_examples = {
            'Groceries': ['Walmart Grocery', 'Kroger', 'Whole Foods', 'Local Market'],
            'Eating Out': ['McDonalds', 'Local Restaurant', 'DoorDash', 'Food Delivery'],
            'Entertainment': ['Netflix', 'Movie Theater', 'Concert Tickets', 'Gaming'],
            'Transportation': ['Gas Station', 'Uber', 'Public Transit', 'Car Service'],
            'Bills': ['Electric Company', 'Water Bill', 'Internet Service', 'Phone Bill'],
            'Health & Wellness': ['Pharmacy', 'Gym Membership', 'Doctor Visit', 'Health Insurance'],
            'Hobbies': ['Craft Store', 'Hobby Shop', 'Sports Equipment', 'Music Store'],
            'Education': ['Tuition', 'Textbooks', 'Online Course', 'School Supplies'],
            'Vacation': ['Airline Tickets', 'Hotel Booking', 'Travel Agency', 'Resort'],
            'Home Maintenance': ['Home Depot', 'Plumber', 'Cleaning Service', 'Hardware Store']
        }
        return category_examples.get(category_name, ['No specific examples available'])

    def _create_categorization_prompt(self, 
                                      transactions: List[Dict], 
                                      existing_categories: Optional[List[Dict]] = None) -> str:
        """
        Create a structured prompt for transaction categorization
        
        Args:
            transactions (List[Dict]): Transactions to categorize
            existing_categories (List[Dict], optional): Existing category list
        
        Returns:
            Structured prompt string
        """
        # Safely handle existing categories
        existing_categories_str = json.dumps(existing_categories or [])
        
        return f"""You are an expert financial categorization AI. Your task is to categorize transactions with high precision.

CRITICAL RESPONSE REQUIREMENTS:
1. MUST respond ONLY in VALID JSON format
2. Create a JSON ARRAY of categorization objects
3. EACH object MUST have EXACTLY these keys:
   - "id": Transaction ID (string)
   - "category": Suggested category name (string)
   - "confidence": Decimal confidence score between 0 and 1 (float)
   - "reasoning": Brief explanation of categorization (string)

IMPORTANT GUIDELINES:
- Analyze transaction description, amount, and context
- Use existing categories as reference
- If unsure, use 'Uncategorized' with low confidence
- Confidence reflects categorization certainty
- Provide clear, concise reasoning

Existing Categories: {existing_categories_str}

Transactions to Categorize:
{json.dumps([
    {{
        "id": t.get('id', ''),
        "description": t.get('description', ''),
        "amount": t.get('amount', 0),
        "date": t.get('date', '')
    }} for t in transactions[:20]  # Limit to first 20 transactions
])}

STRICT EXAMPLE RESPONSE FORMAT:
[
    {{
        "id": "transaction_123",
        "category": "Groceries",
        "confidence": 0.85,
        "reasoning": "Purchased at Kroger, typical grocery store transaction"
    }},
    {{
        "id": "transaction_456",
        "category": "Dining Out",
        "confidence": 0.65,
        "reasoning": "Transaction at restaurant suggests eating out"
    }}
]

CRITICAL: Ensure VALID JSON. NO EXTRA TEXT. NO MARKDOWN.
"""

    def _parse_categorization_response(self, response_text: str) -> Dict:
        """
        Parse the AI-generated categorization response with robust error handling.
        
        Args:
            response_text (str): Raw text response from the AI model
        
        Returns:
            Dict containing parsed categorization results
        """
        try:
            # Remove any markdown code block formatting
            clean_text = response_text.strip('`').strip()
            
            # Try parsing as JSON with multiple strategies
            parsed_results = []
            
            # Strategy 1: Direct JSON parsing
            try:
                parsed_results = json.loads(clean_text)
            except json.JSONDecodeError:
                # Strategy 2: Extract JSON using regex with multiline support
                import re
                json_match = re.search(r'\[.*?\]', clean_text, re.DOTALL | re.MULTILINE)
                if json_match:
                    try:
                        parsed_results = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass
                
                # Strategy 3: Find JSON-like objects
                if not parsed_results:
                    json_objects = re.findall(r'\{[^{}]+\}', clean_text)
                    for obj_text in json_objects:
                        try:
                            parsed_results.append(json.loads(obj_text))
                        except json.JSONDecodeError:
                            continue
            
            # Validate parsed results
            if not isinstance(parsed_results, list):
                raise ValueError("Response must be a list of categorizations")
            
            # Normalize results
            categorization_results = {
                'categorization_results': [],
                'total_transactions': len(parsed_results)
            }
            
            for result in parsed_results:
                # More flexible key extraction
                transaction_id = (
                    result.get('transaction_id') or 
                    result.get('id') or 
                    result.get('transaction', {}).get('id') or 
                    'unknown'
                )
                
                # More flexible category extraction
                category = (
                    result.get('category') or 
                    result.get('suggested_category') or 
                    result.get('name') or 
                    'Uncategorized'
                )
                
                # More flexible confidence extraction
                confidence = (
                    result.get('confidence', 0.5) if isinstance(result.get('confidence'), (int, float)) 
                    else 0.5
                )
                
                # More flexible reasoning extraction
                reasoning = (
                    result.get('reasoning') or 
                    result.get('explanation') or 
                    'No reasoning provided'
                )
                
                categorization_results['categorization_results'].append({
                    'transaction_ids': [transaction_id],
                    'category': category,
                    'confidence': confidence,
                    'reasoning': reasoning
                })
            
            return categorization_results
        
        except Exception as e:
            self.logger.error(f"Failed to parse categorization response: {e}")
            return {
                'total_transactions': 0,
                'error': str(e),
                'fallback_strategy': 'manual_review_recommended'
            }

    @CircuitBreaker()
    def analyze_transactions(self, transactions: List[Dict]) -> SpendingAnalysis:
        """Analyze transactions using Gemini AI with validation"""
        prompt = self._create_prompt(transactions)
        
        try:
            response = self.model.generate_content(prompt)
            parsed_response = self._parse_response(response.text)
            return SpendingAnalysis(**parsed_response)
        except ValidationError as e:
            raise ValueError(f"Invalid AI response: {e}")
    
    def _create_prompt(self, transactions: List[Dict]) -> str:
        """Create a structured prompt for Gemini"""
        return f"""Analyze the following financial transactions:
        {transactions}

        Provide a JSON response with:
        - total_spent: Total amount spent
        - category_breakdown: Spending by category
        - unusual_transactions: Any transactions that seem out of the ordinary

        Format the response as a valid JSON object."""
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse Gemini's text response into a dictionary"""
        # Implement robust JSON parsing with error handling
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Attempt to clean and parse the response
            cleaned_response = response_text.split('```json')[-1].split('```')[0].strip()
            return json.loads(cleaned_response)

    @CircuitBreaker()
    def parse_transaction_creation(self, query: str) -> Dict:
        """Parse transaction details with enhanced error handling"""
        try:
            response = self.model.generate_content(
                self._create_transaction_prompt(query),
                generation_config={
                    'temperature': 0.1,
                    'max_output_tokens': 512
                }
            )
            
            # Validate response
            required_fields = ['amount', 'date', 'payee_name']
            try:
                parsed_details = self._validate_gemini_response(
                    response.text,
                    required_fields
                )
            except GeminiHallucinationError:
                # Retry with more constraints
                response = self.model.generate_content(
                    self._create_transaction_prompt(query),
                    generation_config={
                        'temperature': 0.05,
                        'max_output_tokens': 256,
                        'top_p': 0.5,
                        'top_k': 10
                    }
                )
                parsed_details = self._validate_gemini_response(
                    response.text,
                    required_fields
                )
            
            # Additional validation through models
            try:
                # Create TransactionAmount
                amount = TransactionAmount(
                    amount=abs(float(parsed_details['amount'])),
                    is_outflow=parsed_details.get('is_outflow', True)
                )
                
                # Create full transaction
                transaction = TransactionCreate(
                    amount=amount,
                    date=parsed_details['date'],
                    payee_name=parsed_details.get('payee_name'),
                    memo=parsed_details.get('memo'),
                    category_name=parsed_details.get('category_name'),
                    cleared='uncleared',
                    approved=False,
                    account_id=''
                )
                
                return transaction.dict(exclude_unset=True)
                
            except ValidationError as ve:
                raise InvalidGeminiResponseError(f"Invalid transaction data: {ve}")
            
        except GeminiHallucinationError as he:
            self.logger.error(f"Failed to parse transaction due to hallucination: {he}")
            return {
                'error': 'hallucination_detected',
                'message': str(he),
                'original_query': query
            }
        except InvalidGeminiResponseError as ie:
            self.logger.error(f"Invalid transaction parsing response: {ie}")
            return {
                'error': 'invalid_response',
                'message': str(ie),
                'original_query': query
            }
        except Exception as e:
            self.logger.error(f"Transaction parsing failed: {e}")
            raise

    @CircuitBreaker()
    def update_transaction_category(self, transaction_id: str, category_name: str, budget_id: Optional[str] = None) -> Dict:
        """
        Update a transaction's category using the YNAB client
        
        Args:
            transaction_id (str): ID of the transaction to update
            category_name (str): Name of the category to assign
            budget_id (Optional[str]): Budget ID. Uses default if not provided.
        
        Returns:
            Dict with update result
        """
        try:
            # Use YNABClient to update the transaction category
            return self.ynab_client.update_transaction_categories(
                budget_id=budget_id,
                transactions=[{
                    'id': transaction_id,
                    'category_name': category_name
                }]
            )
            
        except Exception as e:
            self.logger.error(f"Error updating transaction category: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def process_category_update_request(self, query: str, budget_id: Optional[str] = None) -> Dict:
        """
        Process a natural language request to update a transaction's category
        
        Args:
            query (str): Natural language query about category update
            budget_id (Optional[str]): Budget ID. Uses default if not provided.
        
        Returns:
            Dict with update result
        """
        prompt = """
        You are a financial transaction assistant. Extract category update details from this query:
        "{}"
        
        CRITICAL RULES:
        1. Extract the transaction identifier:
           - Look for transaction ID
           - Look for transaction amount (convert to float)
           - Look for transaction date (convert to YYYY-MM-DD)
           - Look for payee/merchant name
        2. Extract the target category:
           - Look for category name after "to", "as", "into"
           - Look for common category names (groceries, entertainment, etc.)
        
        Return ONLY a JSON object with these fields:
        {{
            "transaction_identifier": {{
                "amount": float,
                "date": "YYYY-MM-DD",
                "payee": "string"
            }},
            "category_name": "string"
        }}
        
        Example 1:
        Input: "Change the $25 Target transaction from February 15, 2025 to Groceries"
        Output:
        {{
            "transaction_identifier": {{
                "amount": 25.00,
                "date": "2025-02-15",
                "payee": "Target"
            }},
            "category_name": "Groceries"
        }}
        
        Example 2:
        Input: "Update the Target purchase for $25 on 2/15/25 to Groceries category"
        Output:
        {{
            "transaction_identifier": {{
                "amount": 25.00,
                "date": "2025-02-15",
                "payee": "Target"
            }},
            "category_name": "Groceries"
        }}
        
        IMPORTANT: Return ONLY the JSON object, no additional text or explanation.
        """.format(query)
        
        try:
            # Generate category update details
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.1,
                    'max_output_tokens': 512
                }
            )
            
            # Parse the response
            response_text = response.text.strip()
            
            # Log the raw response for debugging
            self.logger.debug(f"Raw response: {response_text}")
            
            # Remove any markdown code block formatting
            if '```' in response_text:
                response_text = re.search(r'```(?:json)?\n?(.*?)\n?```', response_text, re.DOTALL).group(1)
            
            # Clean up any remaining whitespace
            response_text = response_text.strip()
            
            # Log the cleaned response for debugging
            self.logger.debug(f"Cleaned response: {response_text}")
            
            try:
                update_details = json.loads(response_text)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON response: {e}")
                self.logger.error(f"Response text: {response_text}")
                raise ValueError(f"Invalid JSON response: {e}")
            
            # Validate update details
            identifier = update_details.get('transaction_identifier', {})
            if not identifier:
                raise ValueError("No transaction identifier found")
            
            # Validate required fields
            required_fields = ['amount', 'date', 'payee']
            missing_fields = [field for field in required_fields if not identifier.get(field)]
            if missing_fields:
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
            
            # Validate amount is a number
            try:
                identifier_amount = float(identifier['amount'])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid amount: {identifier.get('amount')}")
            
            # Validate date format
            try:
                datetime.strptime(identifier['date'], '%Y-%m-%d')
            except ValueError:
                raise ValueError(f"Invalid date format: {identifier.get('date')}")
            
            # Find the transaction based on the identifier
            transactions = self.ynab_client.get_transactions(budget_id)
            
            matching_transaction = None
            for transaction in transactions:
                # Convert YNAB amount from milliunits to dollars for comparison
                ynab_amount = abs(float(transaction['amount'])) / 1000
                
                # Check if transaction matches the identifier
                if (abs(ynab_amount - abs(identifier_amount)) < 0.01 and  # Allow small difference in amount
                    transaction['date'] == identifier['date'] and
                    transaction['payee_name'] == identifier['payee']):
                    matching_transaction = transaction
                    break
            
            if not matching_transaction:
                return {
                    'status': 'error',
                    'message': 'Could not find the specified transaction'
                }
            
            # Update the transaction category
            return self.update_transaction_category(
                transaction_id=matching_transaction['id'],
                category_name=update_details['category_name'],
                budget_id=budget_id
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process category update request: {e}")
            return {
                'status': 'error',
                'message': f'Failed to process request: {str(e)}'
            } 