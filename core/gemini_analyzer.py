import os
import re
import json
import logging
from typing import List, Dict, Optional, Union
from decimal import Decimal, InvalidOperation, getcontext, ROUND_HALF_UP
from datetime import datetime, date

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

from .config import ConfigManager
from .circuit_breaker import CircuitBreaker
from .ynab_client import YNABClient
from pydantic import BaseModel, ValidationError, Field, validator
import importlib
import sys
from .shared_models import (
    TransactionCreate, 
    TransactionAmount, 
    Transaction, 
    ConfidenceResult, 
    SpendingAnalysis,
    AmountFromAI,
    AISource
)
from .data_validation import DataValidator
from typing_extensions import runtime_checkable, Protocol

# Define CategoryUpdate locally if not available in shared_models
class CategoryUpdate(BaseModel):
    """Model for category update operations"""
    transaction_id: str
    category_id: str
    memo: Optional[str] = None

# Fallback class to maintain compatibility if ai_client_factory imports aren't available
try:
    from .ai_client_factory import AIClientFactory, AIProvider
except ImportError:
    AIClientFactory = None
    AIProvider = None

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

class MonetaryPrecision:
    def __init__(self):
        self.ctx = getcontext()
        self.ctx.prec = 6  # Set high precision
        self.ctx.rounding = ROUND_HALF_UP

    def parse_amount(self, input_str: str) -> Decimal:
        try:
            # Log the input for debugging
            print(f"Parsing amount: {input_str}")
            
            # Handle None or empty input
            if input_str is None or input_str == '':
                raise ValueError("Input cannot be None or empty")
            
            # Convert input to string if it's not already
            input_str = str(input_str).strip()
            
            # Strip currency symbols, handle comma-separated numbers
            cleaned_input = input_str.strip('$').replace(',', '')
            
            # Log the cleaned input
            print(f"Cleaned input: {cleaned_input}")
            
            return self.ctx.create_decimal(cleaned_input)
        except InvalidOperation as e:
            print(f"InvalidOperation error: {e}")
            raise ValueError(f"Invalid monetary format: {input_str}")
        except Exception as e:
            print(f"Unexpected error parsing amount: {e}")
            raise ValueError(f"Invalid monetary format: {input_str}")

class TransactionConfidenceScorer:
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.monetary_parser = MonetaryPrecision()

    def score_transaction(self, transaction: Dict) -> float:
        """
        Calculate confidence score for a transaction
        
        Scoring criteria:
        - Completeness of information
        - Consistency of data
        - Alignment with historical patterns
        """
        score = 1.0
        
        # Deduct points for missing or suspicious data
        if not transaction.get('payee'):
            score -= 0.2
        
        if not transaction.get('category'):
            score -= 0.1
        
        # Amount sanity check
        try:
            amount = self.monetary_parser.parse_amount(transaction.get('amount', 0))
            if amount <= 0 or amount > 10000:  # Arbitrary large transaction limit
                score -= 0.2
        except ValueError:
            score -= 0.3
        
        return max(0, min(score, 1.0))

    def is_transaction_reliable(self, transaction: Dict) -> bool:
        """Determine if transaction meets confidence threshold"""
        return self.score_transaction(transaction) >= self.threshold

class GeminiSpendingAnalyzer:
    """
    Enhanced spending analyzer with plugin support and confidence scoring
    """
    def __init__(
        self, 
        config_manager: Optional[ConfigManager] = None, 
        ynab_client: Optional[YNABClient] = None,
        ai_client_factory: Optional['AIClientFactory'] = None
    ):
        """
        Initialize Gemini Spending Analyzer with advanced configuration
        
        Args:
            config_manager (Optional[ConfigManager]): Configuration management
            ynab_client (Optional[YNABClient]): YNAB client for context
            ai_client_factory (Optional['AIClientFactory']): AI client factory for model selection
        """
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = config_manager or ConfigManager()
        
        # Store AI client factory
        self.ai_client_factory = ai_client_factory
        
        # Store YNAB client for context
        self.ynab_client = ynab_client
        
        # Initialize modules list
        self.analysis_modules = []
        
        # Configure some defaults in case we need to fallback to direct Gemini usage
        self.general_model_name = os.getenv('GEMINI_OTHER_MODEL', 'gemini-1.5-flash')
        self.reasoning_model_name = os.getenv('GEMINI_REASONER_MODEL', 'gemini-1.5-pro')
        
        # If we have a factory, we'll use it. Otherwise, we'll initialize Gemini directly
        if self.ai_client_factory is None:
            self._initialize_fallback_gemini()
        else:
            # Even with the factory, we still need to discover plugins
            self._discover_plugins()
        
    def _initialize_fallback_gemini(self):
        """Initialize direct Gemini models as fallback if no AI client factory is provided"""
        try:
            import google.generativeai as genai
            
            # Set up API key from environment or config
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("No Gemini API key found. Please set GEMINI_API_KEY.")
            
            genai.configure(api_key=api_key)
            
            # Initialize models with appropriate configurations
            # General model for quick tasks
            self.model = genai.GenerativeModel(
                self.general_model_name,  # Use environment variable with fallback
                generation_config=genai.types.GenerationConfig(
                    # Precise configuration for financial analysis
                    temperature=0.1,  # Low randomness for consistent results
                    top_p=0.9,  # Focus on most probable tokens
                    top_k=40,  # Consider top 40 tokens
                    max_output_tokens=2048,  # Sufficient for complex responses
                ),
                safety_settings={
                    # Strict safety settings for financial data
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                }
            )
            
            # Initialize reasoning model for complex tasks
            self.reasoning_model = genai.GenerativeModel(
                self.reasoning_model_name,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,  # Slightly more creative for reasoning
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=4096,  # Larger for more complex reasoning
                ),
                safety_settings={
                    # Same safety settings
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                }
            )
            self.logger.info(f"Initialized fallback Gemini models: {self.general_model_name} and {self.reasoning_model_name}")
            
            # Discover and register analysis plugins
            self._discover_plugins()
            
        except ImportError:
            self.logger.error("Google Generative AI package not found. Please install with: pip install google-generativeai")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini: {str(e)}")
            raise

    def _discover_plugins(self):
        """
        Discover and register analysis plugins from the plugins directory
        """
        try:
            # Initialize plugins dictionary if needed
            if not hasattr(self, 'plugins'):
                self.plugins = {}
                
            # Look for both traditional plugins and analysis modules
            self._discover_traditional_plugins()
            self._discover_analysis_modules()
                
        except Exception as e:
            self.logger.error(f"Error discovering plugins: {str(e)}")
            # Continue without plugins rather than failing completely
            self.plugins = {}
            
    def _discover_traditional_plugins(self):
        """Discover traditional plugins from the plugins directory"""
        try:
            # Define the plugins package path
            plugins_dir = os.path.join(os.path.dirname(__file__), 'plugins')
            plugins_package = 'core.plugins'
            
            # Check if the plugins directory exists
            if not os.path.exists(plugins_dir):
                self.logger.info(f"Plugins directory not found: {plugins_dir}")
                return
                
            # Find all potential plugin modules
            for filename in os.listdir(plugins_dir):
                if filename.endswith('.py') and not filename.startswith('_'):
                    module_name = filename[:-3]  # Remove .py extension
                    
                    try:
                        # Import the module
                        plugin_module = importlib.import_module(f"{plugins_package}.{module_name}")
                        
                        # Look for a register_plugin function
                        if hasattr(plugin_module, 'register_plugin'):
                            # Register the plugin
                            plugin_info = plugin_module.register_plugin(self)
                            
                            if plugin_info and isinstance(plugin_info, dict):
                                self.plugins[module_name] = plugin_info
                                self.logger.info(f"Registered plugin: {module_name}")
                            else:
                                self.logger.warning(f"Plugin {module_name} returned invalid registration info")
                        else:
                            self.logger.debug(f"Module {module_name} is not a valid plugin (no register_plugin function)")
                    except Exception as e:
                        self.logger.error(f"Error loading plugin {module_name}: {str(e)}")
                        
            self.logger.info(f"Loaded {len(self.plugins)} traditional plugins: {', '.join(self.plugins.keys()) if self.plugins else 'none'}")
        except Exception as e:
            self.logger.error(f"Error discovering traditional plugins: {str(e)}")
            
    def _discover_analysis_modules(self):
        """
        Discover and load analysis modules dynamically
        Follows plugin-based expansion rule
        """
        try:
            # Simple plugin discovery in ai_modules directory
            plugin_dir = os.path.join(os.path.dirname(__file__), '..', 'ai_modules')
            if os.path.exists(plugin_dir):
                sys.path.insert(0, plugin_dir)
                
                module_count = 0
                for filename in os.listdir(plugin_dir):
                    if filename.endswith('_module.py'):
                        module_name = filename[:-3]  # Remove .py
                        try:
                            module = importlib.import_module(module_name)
                            for name, obj in module.__dict__.items():
                                if (isinstance(obj, type) and 
                                    hasattr(obj, '_is_analysis_module') and 
                                    obj._is_analysis_module):
                                    module_instance = obj()
                                    self.register_analysis_module(module_instance)
                                    module_count += 1
                        except Exception as e:
                            self.logger.warning(f"Could not load analysis module {module_name}: {e}")
                            
                self.logger.info(f"Loaded {module_count} analysis modules")
        except Exception as e:
            self.logger.error(f"Analysis module discovery failed: {e}")
            # Continue even if analysis module discovery fails

    def _generate_with_model(self, prompt, use_reasoning=False, **kwargs):
        """Generate content using the appropriate model"""
        try:
            temperature = kwargs.get('temperature', 0.1 if not use_reasoning else 0.2)
            
            if self.ai_client_factory:
                # Use the AI client factory
                if use_reasoning:
                    response = self.ai_client_factory.generate_with_reasoning_model(prompt, temperature=temperature)
                else:
                    response = self.ai_client_factory.generate_with_general_model(prompt, temperature=temperature)
                
                # Extract content from AIClientResponse object
                return response.content if hasattr(response, 'content') else str(response)
            else:
                # Fallback to direct Gemini usage
                import google.generativeai as genai
                model = self.reasoning_model if use_reasoning else self.model
                response = model.generate_content(prompt)
                return response.text
        except Exception as e:
            self.logger.error(f"Error generating content: {str(e)}")
            raise

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
        self.logger.info(f"AI category matcher called with {len(transactions)} transactions and {len(existing_categories)} categories")
        
        # Organize categories by groups for better context
        category_groups = {}
        for cat in existing_categories:
            group = cat.get('group', 'Uncategorized')
            if group not in category_groups:
                category_groups[group] = []
            category_groups[group].append(cat)
        
        # Create formatted category list with hierarchical structure
        formatted_categories = []
        for group, cats in category_groups.items():
            formatted_categories.append(f"## {group}")
            for cat in cats:
                formatted_categories.append(f"- {cat.get('name')} (ID: {cat.get('id')})")
        
        # Example matches to guide the model
        example_matches = [
            {
                "transaction": {"description": "Grocery store purchase", "amount": 56.78},
                "match": {"category": "Groceries", "group": "Food", "confidence": 0.95, "reasoning": "Clear grocery store purchase"}
            },
            {
                "transaction": {"description": "Monthly Netflix subscription", "amount": 14.99},
                "match": {"category": "Streaming Services", "group": "Entertainment", "confidence": 0.98, "reasoning": "Clearly identified subscription service"}
            },
            {
                "transaction": {"description": "interest accrued", "amount": 12.34},
                "match": {"category": "Interest Income", "group": "Income", "confidence": 0.90, "reasoning": "Interest income is a type of income"}
            }
        ]
        
        # Prepare transactions batch for processing
        # Process in smaller batches if needed to avoid token limits
        batch_size = min(20, len(transactions))
        batches = [transactions[i:i+batch_size] for i in range(0, len(transactions), batch_size)]
        
        all_results = []
        
        for batch_idx, batch in enumerate(batches):
            self.logger.debug(f"Processing category matching batch {batch_idx+1}/{len(batches)} with {len(batch)} transactions")
            
            # Prepare prompt with context
            prompt = f"""
            You are an expert financial categorization assistant working with YNAB (You Need A Budget).
            Your task is to match transaction descriptions to the most appropriate category from the provided list.
            
            # Available Categories (organized by group)
            {'\n'.join(formatted_categories)}
            
            # Guidelines for Category Matching
            1. Match each transaction to the EXACT category name from the list above
            2. Consider the transaction description, amount, and other details
            3. Prioritize specific categories over general ones
            4. Look for keywords and context clues in the description
            5. If unsure, choose the best match and note lower confidence
            6. If no good match exists, recommend the most appropriate existing category
            7. IMPORTANT: Only use category names that exist in the provided list
            
            # Example Matches
            {json.dumps(example_matches, indent=2)}
            
            # Transactions to Categorize
            {json.dumps(batch, indent=2)}
            
            # Response Instructions
            Return a JSON array of objects with these fields:
            - transaction_id: ID of the transaction
            - suggested_category: EXACT name of the suggested category (must match one in the list)
            - confidence: Number between 0 and 1 indicating match confidence (0.9+ for high confidence)
            - reasoning: Brief explanation of why this category was chosen
            
            Example response format:
            ```json
            [
                {{
                    "transaction_id": "transaction-123",
                    "suggested_category": "Groceries",
                    "confidence": 0.95,
                    "reasoning": "Transaction description clearly indicates a grocery store purchase"
                }}
            ]
            ```
            
            IMPORTANT: Ensure your response contains valid JSON that can be parsed programmatically.
            Only suggest categories that exactly match ones from the provided list of available categories.
            """
            
            try:
                # Generate category matching recommendations
                response = self._generate_with_model(
                    prompt,
                    use_reasoning=False,
                    temperature=0.1
                )
                
                # Extract JSON content from response
                try:
                    if hasattr(response, 'text'):
                        response_text = response.text
                    else:
                        # Handle different response formats
                        response_text = str(response)
                    
                    # Use JSON extraction method
                    matches = self._extract_json_from_response(response_text)
                    
                    if matches:
                        # Validate each match
                        validated_matches = []
                        for match in matches:
                            if not isinstance(match, dict):
                                self.logger.warning(f"Skipping invalid match format: {match}")
                                continue
                                
                            # Ensure required fields are present
                            if 'transaction_id' not in match or 'suggested_category' not in match:
                                self.logger.warning(f"Skipping match missing required fields: {match}")
                                continue
                                
                            # Normalize confidence
                            if 'confidence' in match:
                                try:
                                    # Handle percentage values or decimal values
                                    conf_value = float(match['confidence'])
                                    if conf_value > 1:  # If provided as percentage
                                        conf_value = conf_value / 100
                                    match['confidence'] = max(0.0, min(1.0, conf_value))
                                except (ValueError, TypeError):
                                    match['confidence'] = 0.5  # Default confidence
                            else:
                                match['confidence'] = 0.5
                                
                            # Verify the suggested category exists
                            category_exists = False
                            for cat in existing_categories:
                                if (cat.get('name', '').lower() == match['suggested_category'].lower() or
                                    cat.get('full_name', '').lower() == match['suggested_category'].lower()):
                                    # Update to exact case match
                                    match['suggested_category'] = cat.get('name') or cat.get('full_name').split(':')[-1].strip()
                                    category_exists = True
                                    break
                                    
                            if not category_exists:
                                # If category doesn't exist, log warning but still include the match
                                self.logger.warning(f"Category '{match['suggested_category']}' not found in existing categories")
                                
                            validated_matches.append(match)
                            
                        all_results.extend(validated_matches)
                        self.logger.info(f"Successfully matched {len(validated_matches)} transactions in batch {batch_idx+1}")
                    else:
                        self.logger.warning(f"No valid matches found in batch {batch_idx+1}")
                
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse AI category matching response: {e}")
                    self.logger.debug(f"Raw response: {response_text}")
                
                except Exception as e:
                    self.logger.error(f"Error processing category matches: {str(e)}")
                    
            except Exception as e:
                self.logger.error(f"AI Category Matching failed for batch {batch_idx+1}: {str(e)}")
                
        self.logger.info(f"AI category matching complete. Matched {len(all_results)}/{len(transactions)} transactions")
        return all_results
    
    def _extract_json_from_response(self, response_text: str) -> List[Dict]:
        """
        Extract JSON data from a text response that might contain markdown or other formatting
        
        Args:
            response_text (str): Raw response text
            
        Returns:
            List[Dict]: Extracted JSON data or empty list if extraction fails
        """
        # Try to extract JSON from markdown code blocks first
        json_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(json_block_pattern, response_text)
        
        if matches:
            # Use the first code block that contains valid JSON
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
        
        # If no valid JSON in code blocks, try to find JSON objects directly
        try:
            # Look for array of objects
            array_pattern = r'\[\s*\{[\s\S]*\}\s*\]'
            array_match = re.search(array_pattern, response_text)
            if array_match:
                return json.loads(array_match.group(0))
            
            # Try parsing the entire response as JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            self.logger.warning("Could not extract JSON from response")
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
            
            # Log raw response for debugging
            self.logger.debug(f"Raw response text: {clean_text}")
            
            # Multiple strategies for JSON extraction
            json_extraction_patterns = [
                # Markdown code block with optional json hint
                r'```(?:json)?\n?(.*?)\n?```',
                # JSON between curly braces
                r'\{.*\}',
                # JSON-like text between first { and last }
                r'\{.*\}(?=\s*$)',
            ]
            
            parsed_data = None
            for pattern in json_extraction_patterns:
                try:
                    match = re.search(pattern, clean_text, re.DOTALL | re.MULTILINE)
                    if match:
                        candidate_text = match.group(0).strip()
                        
                        # Log candidate text for debugging
                        self.logger.debug(f"Candidate JSON text: {candidate_text}")
                        
                        try:
                            parsed_data = json.loads(candidate_text)
                            break  # Stop if successful parsing
                        except json.JSONDecodeError:
                            # Continue to next pattern if parsing fails
                            continue
                except Exception as e:
                    self.logger.warning(f"JSON extraction failed for pattern {pattern}: {e}")
            
            # If no valid JSON found
            if parsed_data is None:
                self.logger.error(f"Could not extract valid JSON from response: {clean_text}")
                raise InvalidGeminiResponseError("No valid JSON found in response")
            
            # Log parsed data
            self.logger.debug(f"Parsed data: {parsed_data}")
            
            # Check for hallucination indicators
            if isinstance(parsed_data, dict):
                # Check for nonsensical values
                for key, value in parsed_data.items():
                    if isinstance(value, (int, float)) and abs(value) > 1e6:  # Unreasonably large numbers
                        raise GeminiHallucinationError(f"Unreasonable value detected: {key}={value}")
                    if isinstance(value, str) and len(value) > 1000:  # Extremely long strings
                        raise GeminiHallucinationError(f"Unreasonably long string detected for {key}")
            
            # Validate required fields
            missing_fields = [field for field in expected_fields if field not in parsed_data]
            if missing_fields:
                raise InvalidGeminiResponseError(f"Missing required fields: {', '.join(missing_fields)}")
            
            return parsed_data
            
        except (AttributeError, TypeError) as e:
            self.logger.error(f"Response structure error: {e}")
            self.logger.error(f"Problematic response: {response_text}")
            raise InvalidGeminiResponseError(f"Invalid response structure: {e}")

    @CircuitBreaker()
    def categorize_transactions(self, transactions: List[Dict], ynab_categories: Optional[List[Dict]] = None) -> Dict:
        """
        Categorize transactions with enhanced error handling
        """
        self.logger.info(f"Categorizing {len(transactions)} transactions")
        
        try:
            # Generate categorization with enhanced context
            response = self._generate_with_model(
                self._create_categorization_prompt(transactions, ynab_categories),
                use_reasoning=True
            )
            
            # Validate response
            required_fields = ['transaction_id', 'category_name', 'confidence', 'reasoning']
            try:
                categorization_results = self._validate_gemini_response(
                    response,
                    required_fields
                )
            except GeminiHallucinationError as he:
                self.logger.error(f"Detected hallucination in Gemini response: {he}")
                # Retry with more strict parameters
                response = self._generate_with_model(
                    self._create_categorization_prompt(transactions, ynab_categories),
                    use_reasoning=True,
                    temperature=0.1
                )
                # Validate retry response
                categorization_results = self._validate_gemini_response(
                    response,
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
            response = self._generate_with_model(prompt, use_reasoning=True)
            parsed_response = self._parse_response(response)
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
            response = self._generate_with_model(
                self._create_transaction_prompt(query),
                use_reasoning=True
            )
            
            # Validate response
            required_fields = ['amount', 'date', 'payee_name']
            try:
                parsed_details = self._validate_gemini_response(
                    response,
                    required_fields
                )
            except GeminiHallucinationError:
                # Retry with more constraints
                response = self._generate_with_model(
                    self._create_transaction_prompt(query),
                    use_reasoning=True,
                    temperature=0.05
                )
                parsed_details = self._validate_gemini_response(
                    response,
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
            response = self._generate_with_model(
                prompt,
                use_reasoning=True,
                temperature=0.1
            )
            
            # Parse the response
            response_text = response.strip()
            
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

    def _create_transaction_prompt(self, query: str) -> str:
        """
        Create a structured prompt for transaction parsing with strict JSON output requirements
        
        Args:
            query (str): Natural language transaction description
        
        Returns:
            str: Prompt for Gemini to generate structured transaction data
        """
        # Extract amount using regex
        amount_match = re.search(r'\$?(\d+(?:\.\d{1,2})?)', query)
        amount = amount_match.group(1) if amount_match else '50.00'
        
        return f"""
Parse the following transaction description and return a STRICTLY FORMATTED JSON object:

Transaction Description: "{query}"

SPECIFIC PARSING REQUIREMENTS:
- EXTRACT EXACT AMOUNT: {amount}
- IDENTIFY PAYEE/MERCHANT
- USE TODAY'S DATE IF NO SPECIFIC DATE MENTIONED

JSON SCHEMA REQUIREMENTS:
- MUST be valid, parseable JSON
- MONETARY AMOUNT: {amount}
- Dates in YYYY-MM-DD format
- REQUIRED FIELDS: amount, payee_name, date
- OPTIONAL FIELDS: memo, category_name, is_outflow

EXAMPLE OUTPUT:
{{
    "amount": {amount},
    "payee_name": "Target",
    "date": "{date.today().strftime('%Y-%m-%d')}",
    "is_outflow": true,
    "memo": "Shopping trip",
    "category_name": "Shopping"
}}

PARSING INSTRUCTIONS:
1. Use the SPECIFIED AMOUNT: {amount}
2. Identify payee/merchant name from description
3. Use today's date
4. Determine if transaction is an expense (outflow)
5. Add a descriptive memo if possible
6. Suggest an appropriate category

IMPORTANT: 
- RETURN ONLY THE JSON
- NO MARKDOWN CODE BLOCKS
- NO ADDITIONAL TEXT
- ENSURE NUMERIC PRECISION
"""

    def parse_transaction(self, transaction_description: str) -> TransactionCreate:
        """
        Hybrid transaction parsing with reasoning chain and fallback mechanisms
        
        Args:
            transaction_description (str): Natural language transaction description
        
        Returns:
            TransactionCreate: Parsed and validated transaction
        """
        # Initialize confidence scorer and monetary parser
        confidence_scorer = TransactionConfidenceScorer()
        monetary_parser = MonetaryPrecision()
        
        # Logging setup
        self.logger.info(f"Parsing transaction: {transaction_description}")
        
        # Reasoning Chain Stage 1: AI-Powered Parsing
        try:
            # Generate structured prompt for AI parsing
            ai_parsing_prompt = f"""
            Parse the following transaction description with extreme precision:
            "{transaction_description}"
            
            Provide a JSON response with these REQUIRED fields:
            {{
                "amount": {{
                    "value": float,  # Absolute amount value
                    "is_outflow": bool  # Whether it's an expense
                }},
                "payee_name": str,  # Merchant or payee name
                "date": str,  # ISO 8601 date (YYYY-MM-DD)
                "memo": str,  # Optional description
                "confidence": float  # Confidence score (0-1)
            }}
            
            RULES:
            - Extract amount precisely
            - Determine if it's an expense or income
            - Use today's date if no date specified
            - Be extremely specific about parsing
            - Provide reasoning for each field
            """
            
            # Generate AI response
            ai_response = self._generate_with_model(
                ai_parsing_prompt,
                use_reasoning=False,
                temperature=0.2,
                max_output_tokens=256
            )
            
            # Parse AI response
            parsed_response = self._parse_ai_transaction_response(ai_response)
            
            # Validate confidence
            if parsed_response.get('confidence', 0) < 0.7:
                raise ValueError("Low confidence AI parsing")
            
            # Extract parsed data
            amount_data = parsed_response.get('amount', {})
            amount_value = amount_data.get('value')
            is_outflow = amount_data.get('is_outflow', True)
            
            # Fallback Stage 1: Regex Amount Extraction
            if not amount_value:
                try:
                    amount_value = monetary_parser.parse_amount(transaction_description)
                    is_outflow = amount_value < 0
                except Exception as e:
                    self.logger.warning(f"Regex amount parsing failed: {e}")
                    raise
            
            # Prepare transaction data
            transaction_data = {
                'account_id': os.getenv('DEFAULT_ACCOUNT_ID', ''),  # Use default account
                'date': parsed_response.get('date', str(date.today())),
                'amount': TransactionAmount(
                    amount=abs(float(amount_value)),
                    is_outflow=is_outflow
                ),
                'payee_name': parsed_response.get('payee_name', ''),
                'memo': parsed_response.get('memo', transaction_description)
            }
            
            # Create and validate transaction
            transaction = TransactionCreate(**transaction_data)
            
            # Log successful parsing
            self.logger.info(
                f"Successfully parsed transaction: {transaction.payee_name}, "
                f"Amount: {'$' if not transaction.amount.is_outflow else '-$'}{transaction.amount.amount}"
            )
            
            return transaction
        
        except Exception as primary_error:
            # Fallback Stage 2: Comprehensive Error Recovery
            self.logger.warning(f"Primary parsing failed: {primary_error}")
            
            try:
                # Attempt manual parsing with regex and heuristics
                amount = monetary_parser.parse_amount(transaction_description)
                is_outflow = amount < 0
                
                # Extract payee using simple heuristics
                payee_match = re.search(r'at\s+([A-Za-z\s]+)', transaction_description, re.IGNORECASE)
                payee_name = payee_match.group(1).strip() if payee_match else 'Unknown'
                
                # Create transaction with fallback data
                fallback_transaction = TransactionCreate(
                    account_id=os.getenv('DEFAULT_ACCOUNT_ID', ''),
                    date=date.today(),
                    amount=TransactionAmount(
                        amount=abs(float(amount)),
                        is_outflow=is_outflow
                    ),
                    payee_name=payee_name,
                    memo=transaction_description
                )
                
                # Log fallback parsing
                self.logger.warning(
                    f"Fallback parsing successful: {fallback_transaction.payee_name}, "
                    f"Amount: {'$' if not fallback_transaction.amount.is_outflow else '-$'}{fallback_transaction.amount.amount}"
                )
                
                return fallback_transaction
            
            except Exception as fallback_error:
                # Final error handling
                self.logger.error(f"Transaction parsing completely failed: {fallback_error}")
                raise ValueError(f"Could not parse transaction: {transaction_description}")
    
    def _parse_ai_transaction_response(self, response_text: str) -> Dict:
        """
        Parse and validate AI transaction response
        
        Args:
            response_text (str): Raw AI response text
        
        Returns:
            Dict: Parsed and validated transaction data
        """
        try:
            # Remove code block markers if present
            clean_text = response_text.strip('`').strip()
            
            # Try parsing as JSON
            try:
                parsed_data = json.loads(clean_text)
            except json.JSONDecodeError:
                # Attempt to extract JSON from text
                json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
                if json_match:
                    parsed_data = json.loads(json_match.group(0))
                else:
                    raise ValueError("No valid JSON found")
            
            # Validate required fields
            required_fields = ['amount', 'payee_name', 'date']
            for field in required_fields:
                if field not in parsed_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Normalize and validate data
            parsed_data['confidence'] = parsed_data.get('confidence', 0.5)
            parsed_data['memo'] = parsed_data.get('memo', '')
            
            return parsed_data
        
        except Exception as e:
            self.logger.error(f"AI response parsing failed: {e}")
            raise ValueError(f"Invalid AI transaction response: {str(e)}") 