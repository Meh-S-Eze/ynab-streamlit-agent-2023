from typing import Any, Dict, Callable, Optional, List
from functools import wraps
import time
import logging
from .agent_tool_tracker import tool_tracker
from .config import ConfigManager
from .ynab_client import YNABClient
from .gemini_analyzer import GeminiSpendingAnalyzer
from .pydantic_ai_transaction_parser import PydanticAITransactionParser
import json
import re
from datetime import date
import requests
import openai
import os

# Import AIClientFactory, with fallback if not available
try:
    from .ai_client_factory import AIClientFactory, AIProvider
except ImportError:
    AIClientFactory = None
    AIProvider = None

class BaseAgent:
    """
    Base class for AI agents with built-in tool tracking and configuration
    """
    def __init__(
        self, 
        name: str,
        config_manager: Optional[ConfigManager] = None,
        ynab_client: Optional[YNABClient] = None,
        gemini_analyzer: Optional[GeminiSpendingAnalyzer] = None,
        ai_client_factory: Optional['AIClientFactory'] = None
    ):
        """
        Initialize the agent with a name and dependencies
        
        Args:
            name (str): Unique identifier for the agent
            config_manager (ConfigManager, optional): Configuration management
            ynab_client (YNABClient, optional): YNAB API client
            gemini_analyzer (GeminiSpendingAnalyzer, optional): Gemini analyzer for AI operations
            ai_client_factory (AIClientFactory, optional): Factory for AI clients
        """
        self.name = name
        self.tool_tracker = tool_tracker
        self.logger = logging.getLogger(f"agent.{name}")
        
        # Initialize core services with dependency injection
        self.config = config_manager or ConfigManager()
        self.ynab_client = ynab_client or YNABClient()
        self.ai_client_factory = ai_client_factory
        
        # Initialize Gemini analyzer with dependencies if not provided
        if gemini_analyzer:
            self.gemini_analyzer = gemini_analyzer
        else:
            self.gemini_analyzer = GeminiSpendingAnalyzer(
                config_manager=self.config,
                ynab_client=self.ynab_client,
                ai_client_factory=self.ai_client_factory
            )
            
        # Initialize the Pydantic AI transaction parser
        self.transaction_parser = PydanticAITransactionParser()

    def use_tool(self, tool_func: Callable, *args, **kwargs):
        """
        Wrapper method to use a tool with automatic tracking
        
        Args:
            tool_func (Callable): The tool/function to be used
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool
        
        Returns:
            The result of the tool function
        """
        start_time = time.time()
        try:
            # Prepare input parameters for logging
            input_params = {
                'args': [str(arg) for arg in args],
                'kwargs': {k: str(v) for k, v in kwargs.items()}
            }
            
            # Execute the tool
            result = tool_func(*args, **kwargs)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log successful tool usage
            self.tool_tracker.log_tool_usage(
                agent_name=self.name,
                tool_name=tool_func.__name__,
                input_params=input_params,
                output=str(result),
                duration=duration,
                success=True
            )
            
            return result
        
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log failed tool usage
            self.tool_tracker.log_tool_usage(
                agent_name=self.name,
                tool_name=tool_func.__name__,
                input_params=input_params,
                success=False,
                duration=duration,
                error_message=str(e)
            )
            
            # Log the error
            self.logger.error(f"Tool {tool_func.__name__} failed: {e}", exc_info=True)
            
            raise

    def get_tool_usage_summary(self):
        """
        Get a summary of tool usage for this agent
        
        Returns:
            Dict: Summary of tool usage
        """
        return self.tool_tracker.get_agent_tool_summary(agent_name=self.name)

    def export_tool_usage(self, filename: str = None):
        """
        Export tool usage log
        
        Args:
            filename (str, optional): Custom filename for export
        """
        if filename is None:
            filename = f'{self.name}_tool_usage.json'
        self.tool_tracker.export_usage_log(filename)

    def analyze_budget(self):
        """
        Comprehensive budget analysis method
        
        Returns:
            Dict: Comprehensive budget analysis results
        """
        # Retrieve transactions
        transactions = self.use_tool(self.ynab_client.get_transactions)
        
        # Categorize transactions
        categorization = self.use_tool(
            self.gemini_analyzer.categorize_transactions, 
            transactions
        )
        
        # Generate insights
        insights = self.use_tool(
            self.gemini_analyzer.generate_spending_insights, 
            transactions, 
            categorization
        )
        
        return {
            'transactions': transactions,
            'categorization': categorization,
            'insights': insights
        }

    def auto_categorize_transactions(self, budget_id: Optional[str] = None):
        """
        Automatically categorize all transactions in a budget
        
        Args:
            budget_id (Optional[str]): Specific budget ID. Uses default if not provided.
        
        Returns:
            Dict with categorization and update results
        """
        try:
            # Retrieve budget context
            budget_context = self.get_budget_context(budget_id)
            
            transactions = budget_context['transactions']
            
            # Categorize transactions using Gemini analyzer
            categorization_result = self.use_tool(
                self.gemini_analyzer.categorize_transactions, 
                transactions
            )
            
            # Prepare transactions for update
            update_payload = []
            for transaction, result in zip(transactions, categorization_result.get('categorization_results', [])):
                if result.category:
                    update_payload.append({
                        'id': transaction['id'],
                        'category_name': result.category
                    })
            
            # Validate transactions before update
            validated_transactions = self.use_tool(
                self.ynab_client.validate_transaction_categories,
                update_payload
            )
            
            # Update transactions in YNAB
            update_result = self.use_tool(
                self.ynab_client.update_transaction_categories,
                budget_id=budget_context['budget_id'],
                transactions=validated_transactions
            )
            
            return {
                'budget_name': budget_context['budget_name'],
                'total_transactions': len(transactions),
                'categorization_result': categorization_result,
                'update_result': update_result,
                'categorized_transactions': validated_transactions
            }
        
        except Exception as e:
            self.logger.error(f"Auto-categorization failed: {e}", exc_info=True)
            return {
                'error': str(e),
                'total_transactions': 0,
                'categorization_result': {},
                'update_result': None,
                'categorized_transactions': []
            }

    def auto_categorize_uncategorized_transactions(self, budget_id: Optional[str] = None):
        """
        Automatically categorize all uncategorized transactions in a budget
        
        Args:
            budget_id (Optional[str]): Specific budget ID. Uses default if not provided.
        
        Returns:
            Dict with categorization and update results
        """
        try:
            # Retrieve budget context
            budget_context = self.get_budget_context(budget_id)
            
            # Get YNAB categories
            ynab_categories = self.ynab_client._get_budget_categories(budget_id)
            
            # Filter uncategorized transactions - check for null/empty category_id
            uncategorized_transactions = [
                t for t in budget_context['transactions'] 
                if not t.get('category_id') or t.get('category_id') == ''
            ]
            
            self.logger.info(f"Found {len(uncategorized_transactions)} uncategorized transactions")
            
            if not uncategorized_transactions:
                return {
                    'summary': 'No uncategorized transactions found.',
                    'total_transactions': len(budget_context['transactions']),
                    'uncategorized_count': 0,
                    'categorization_result': {}
                }
            
            # Categorize uncategorized transactions with YNAB categories
            categorization_result = self.use_tool(
                self.gemini_analyzer.categorize_transactions,
                transactions=uncategorized_transactions,
                ynab_categories=list(ynab_categories.values())
            )
            
            # Prepare transactions for update
            update_payload = []
            for transaction, result in zip(uncategorized_transactions, categorization_result.get('categorization_results', [])):
                if result.category:
                    update_payload.append({
                        'id': transaction['id'],
                        'category_name': result.category
                    })
            
            # Update transactions in YNAB
            update_result = self.use_tool(
                self.ynab_client.update_transaction_categories,
                budget_id=budget_context['budget_id'],
                transactions=update_payload
            )
            
            return {
                'summary': f"Categorized {len(update_payload)} uncategorized transactions in {budget_context['budget_name']} budget",
                'budget_name': budget_context['budget_name'],
                'total_transactions': len(budget_context['transactions']),
                'uncategorized_count': len(uncategorized_transactions),
                'categorized_count': len(update_payload),
                'categorization_result': categorization_result,
                'update_result': update_result
            }
        
        except Exception as e:
            self.logger.error(f"Auto-categorization of uncategorized transactions failed: {e}", exc_info=True)
            return {
                'error': str(e),
                'total_transactions': 0,
                'categorization_result': {},
                'update_result': None,
                'categorized_transactions': []
            }

    def get_budget_context(self, budget_id: Optional[str] = None):
        """
        Get a summary of the budget, including basic stats and recent transactions
        
        Args:
            budget_id (str, optional): Budget ID to analyze
            
        Returns:
            Dict[str, Any]: Summary of budget information
        """
        # If no budget_id specified, use the default
        if budget_id is None:
            budget_id = self.ynab_client.budget_id
            
        # Get budget info
        budget = self.ynab_client.get_budget_info()
        
        # Prepare response
        context = {
            "budget_name": budget.get("name", "Unknown"),
            "currency": budget.get("currency_format", {}).get("iso_code", "USD"),
            "date_format": budget.get("date_format", {"format": "YYYY-MM-DD"}),
            "last_month_activity": "Not available",
            "current_month_activity": "Not available",
            "accounts_summary": "Not available",
            "categories_summary": "Not available"
        }
        
        return context
        
    # AI-only parsing approach: We no longer use fallback methods for transaction parsing
    # as per architecture principles: "Never use fallback parsing when AI parsing fails - 
    # focus on improving AI parsing instead"

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query using the appropriate tools
        
        Args:
            query (str): User's natural language query
            
        Returns:
            Dict[str, Any]: Response containing results or error information
        """
        try:
            # Clean and normalize the query
            query = query.strip()
            self.logger.debug(f"Processing query: query='{query}'")
            
            # Initialize transaction parser
            parser = PydanticAITransactionParser()
            
            # Try to parse as a transaction creation request
            if any(phrase in query.lower() for phrase in ["create", "add", "new", "spent", "paid", "bought"]):
                self.logger.info("Detected transaction creation request")
                try:
                    # Parse the transaction
                    transaction = parser.parse_transaction_sync(query)
                    self.logger.debug(f"Parsed transaction: {transaction}")
                    
                    # Create the transaction
                    self.logger.info(f"Creating transaction for budget: budget_id='{self.ynab_client.budget_id}'")
                    result = self.ynab_client.create_transaction(transaction)
                    
                    if result and result.get("transaction"):
                        return {
                            "status": "success",
                            "summary": "Transaction successfully created",
                            "transaction": result["transaction"]
                        }
                    else:
                        raise ValueError("No transaction data in response")
                        
                except (TimeoutError, ConnectionError, requests.exceptions.RequestException) as e:
                    # No fallback for network errors - log a clear error message instead
                    self.logger.error(f"Network or API error: {str(e)}")
                    return {
                        "status": "error",
                        "summary": f"Network or API error: {str(e)}",
                        "details": "The AI service is currently unavailable. Please try again later."
                    }
                except ValueError as e:
                    # For data validation errors, return helpful error message
                    self.logger.error(f"Transaction validation failed: {str(e)}")
                    return {
                        "status": "error",
                        "summary": f"Invalid transaction format: {str(e)}",
                        "details": "Please provide a clearer transaction description with amount, payee, and date information."
                    }
                    
            # Get intent from Gemini
            intent_prompt = f"""You are a YNAB (You Need A Budget) financial query analyzer. Your task is to accurately identify what financial action the user wants to perform from their natural language query.

USER QUERY:
"{query}"

FINANCIAL INTENT CLASSIFICATION TASK:
Analyze the query to determine the specific financial operation the user intends to perform with YNAB.

POSSIBLE INTENTS:
1. "create_transaction" - User wants to record a new transaction
2. "update_category" - User wants to change a category for a transaction
3. "categorize_transactions" - User wants to assign categories to transactions
4. "get_balance" - User wants to check account balance or budget status
5. "show_transactions" - User wants to view or list transactions
6. "unknown" - Cannot determine a clear financial intent

ADDITIONAL DETAILS TO EXTRACT:
- For categorize_transactions: Should all transactions be categorized or only uncategorized ones?
- For show_transactions: Should only specific category transactions be shown?
- For update_category: Which transaction and what category is being referenced?

EXAMPLE INTENT ANALYSES:

Example 1 - Create Transaction:
Input: "I spent $45 at the Italian restaurant downtown yesterday"
{{
    "intent": "create_transaction",
    "description": "User wants to record a restaurant expense transaction",
    "details": {{
        "should_categorize_all": false,
        "specific_category": "Dining Out",
        "show_uncategorized": false,
        "show_category": null,
        "is_category_update": false
    }}
}}

Example 2 - Show Transactions:
Input: "Show me all my grocery expenses from last month"
{{
    "intent": "show_transactions",
    "description": "User wants to view grocery transactions from previous month",
    "details": {{
        "should_categorize_all": false,
        "specific_category": "Groceries",
        "show_uncategorized": false,
        "show_category": "Groceries",
        "is_category_update": false
    }}
}}

Example 3 - Update Category:
Input: "Change that Amazon purchase from yesterday to the Gifts category"
{{
    "intent": "update_category",
    "description": "User wants to recategorize an Amazon transaction to Gifts",
    "details": {{
        "should_categorize_all": false,
        "specific_category": "Gifts",
        "show_uncategorized": false,
        "show_category": null,
        "is_category_update": true
    }}
}}

Example 4 - Categorize Transactions:
Input: "Please categorize all the uncategorized transactions"
{{
    "intent": "categorize_transactions",
    "description": "User wants the system to automatically categorize all uncategorized transactions",
    "details": {{
        "should_categorize_all": false,
        "specific_category": null,
        "show_uncategorized": true,
        "show_category": null,
        "is_category_update": false
    }}
}}

Example 5 - Get Balance:
Input: "How much do I have left in my entertainment budget?"
{{
    "intent": "get_balance",
    "description": "User wants to check the remaining balance in entertainment category",
    "details": {{
        "should_categorize_all": false,
        "specific_category": "Entertainment",
        "show_uncategorized": false,
        "show_category": null,
        "is_category_update": false
    }}
}}

Example 6 - Ambiguous/Unknown:
Input: "YNAB is great for tracking expenses"
{{
    "intent": "unknown",
    "description": "User is making a general statement about YNAB, not requesting a specific action",
    "details": {{
        "should_categorize_all": false,
        "specific_category": null,
        "show_uncategorized": false,
        "show_category": null,
        "is_category_update": false
    }}
}}

RESPONSE FORMAT - EXACT JSON STRUCTURE:
{{
    "intent": "create_transaction",  // One of the six possible intents listed above
    "description": "User wants to add a new grocery purchase transaction",  // Brief explanation
    "details": {{
        "should_categorize_all": false,  // For categorization intents
        "specific_category": "Groceries",  // Category name if mentioned
        "show_uncategorized": false,  // For show transaction intents
        "show_category": null,  // Category filter for show transactions
        "is_category_update": false  // Whether this involves updating a transaction category
    }}
}}

RESPONSE REQUIREMENTS:
- Return ONLY valid JSON with no additional text
- Use null for missing or inapplicable fields
- Set boolean fields to true/false as appropriate
- Ensure your determination is based only on the actual query content
- Do not invent details not present in the original query
"""
            
            # Use _generate_with_model instead of directly accessing model
            intent_response = self.gemini_analyzer._generate_with_model(
                intent_prompt,
                use_reasoning=False,
                temperature=0.1
            )
            
            # Parse intent from response
            intent_text = intent_response
            self.logger.debug("Intent response: %r", intent_text)
            
            # Remove markdown code block if present
            if '```' in intent_text:
                # Extract content between ```json and ```
                match = re.search(r'```(?:json)?\n(.*?)\n```', intent_text, re.DOTALL)
                if match:
                    intent_text = match.group(1)
                else:
                    # Try extracting content between ``` and ```
                    match = re.search(r'```(.*?)```', intent_text, re.DOTALL)
                    if match:
                        intent_text = match.group(1)
            
            # Clean up and parse JSON
            intent_text = intent_text.strip()
            intent_data = json.loads(intent_text)
            
            # Process based on intent
            if intent_data['intent'] == 'create_transaction':
                # Try to parse as a transaction creation request using our new parser
                try:
                    # Use the new Pydantic AI transaction parser for more robust parsing
                    transaction = self.transaction_parser.parse_transaction_sync(query)
                    
                    # Convert date to ISO-8601 string for YNAB API
                    if isinstance(transaction.date, date):
                        date_str = transaction.date.isoformat()
                    else:
                        date_str = str(transaction.date)
                        
                    # Convert transaction to dictionary with proper amount handling
                    transaction_dict = {
                        'account_id': transaction.account_id,
                        'date': date_str,
                        'amount': float(transaction.amount.amount) if transaction.amount.is_outflow else -float(transaction.amount.amount),
                        'payee_name': transaction.payee_name,
                        'payee_id': transaction.payee_id,
                        'memo': transaction.memo,
                        'category_name': transaction.category_name,
                        'cleared': transaction.cleared,
                        'approved': transaction.approved,
                        'flag_name': 'purple'  # Purple flag for new AI-created transactions
                    }
                    
                    # Debug logging for transaction data
                    self.logger.debug(f"Transaction dictionary before creation: {transaction_dict}")
                    
                    # Create the transaction
                    result = self.ynab_client.create_transaction(transaction=transaction_dict)
                    
                    # Debug logging for response
                    self.logger.debug(f"Transaction creation response: {result}")
                    
                    if result['status'] == 'success':
                        return {
                            'summary': f"Successfully created transaction: {result['message']}",
                            'transaction': result['details']
                        }
                    else:
                        return {
                            'summary': f"Failed to create transaction: {result['message']}",
                            'error': result.get('details', {})
                        }
                except Exception as e:
                    self.logger.error(f"Transaction creation failed: {e}")
                    return {
                        'summary': f"Failed to create transaction: {str(e)}",
                        'error': {}
                    }
                    
            elif intent_data['intent'] == 'update_category':
                # Process category update request
                try:
                    result = self.gemini_analyzer.process_category_update_request(query)
                    
                    # No need to update the flag separately as our new method handles it
                    if result['status'] == 'success':
                        return {
                            'summary': 'Category updated successfully',
                            'transaction': result.get('transaction', {})
                        }
                    else:
                        return {
                            'summary': f"Failed to update category: {result['message']}",
                            'error': result.get('details', {})
                        }
                except Exception as e:
                    self.logger.error(f"Category update failed: {e}")
                    return {
                        'status': 'error',
                        'message': f'Failed to update category: {str(e)}'
                    }
                    
            elif intent_data['intent'] == 'show_transactions':
                # Get transactions
                budget_context = self.get_budget_context()
                transactions = budget_context['transactions']
                
                # Filter transactions based on details
                details = intent_data.get('details', {})
                if details.get('show_uncategorized', False):
                    transactions = [t for t in transactions if not t.get('category_id')]
                elif details.get('show_category'):
                    transactions = [t for t in transactions if t.get('category_name') == details['show_category']]
                
                # Format transaction summary
                transaction_summary = []
                for t in transactions:
                    summary = {
                        'id': t.get('id'),
                        'date': t.get('date'),
                        'amount': t.get('amount'),
                        'payee': t.get('payee_name'),
                        'category': t.get('category_name', 'Uncategorized')
                    }
                    transaction_summary.append(summary)
                
                return {
                    'summary': f"Found {len(transactions)} transactions",
                    'transactions': transaction_summary
                }
                
            elif intent_data['intent'] == 'categorize_transactions':
                details = intent_data.get('details', {})
                if details.get('should_categorize_all', True):
                    result = self.auto_categorize_transactions()
                else:
                    result = self.auto_categorize_uncategorized_transactions()
                
                # Enhance the result with more details
                if 'error' in result:
                    return {
                        'summary': f"Failed to categorize transactions: {result['error']}",
                        'error': result['error']
                    }
                else:
                    total_updated = result.get('update_result', {}).get('total_updated', 0)
                    unmatched = len(result.get('update_result', {}).get('unmatched_categories', []))
                    return {
                        'summary': f"Successfully categorized {total_updated} transactions. {unmatched} categories could not be matched.",
                        'details': {
                            'total_transactions': result.get('total_transactions', 0),
                            'categorized': total_updated,
                            'unmatched_categories': result.get('update_result', {}).get('unmatched_categories', [])
                        }
                    }
                
            elif intent_data['intent'] == 'get_balance':
                try:
                    # Use configured budget ID from environment
                    budget_id = self.ynab_client.budget_id
                    accounts = self.ynab_client.get_accounts(budget_id)
                    
                    # Get total budget balance and individual account balances
                    total_balance = 0
                    account_balances = []
                    
                    for account in accounts:
                        # Convert milliunits to actual currency
                        balance = account.get('balance', 0) / 1000.0
                        cleared_balance = account.get('cleared_balance', 0) / 1000.0
                        uncleared_balance = account.get('uncleared_balance', 0) / 1000.0
                        
                        # Add to total for non-closed accounts
                        if not account.get('closed', False):
                            total_balance += balance
                        
                        # Add to account balances list
                        account_balances.append({
                            'name': account.get('name', 'Unknown Account'),
                            'type': account.get('type', 'Unknown'),
                            'balance': f"${balance:.2f}",
                            'cleared_balance': f"${cleared_balance:.2f}",
                            'uncleared_balance': f"${uncleared_balance:.2f}",
                            'closed': account.get('closed', False)
                        })
                    
                    # Format total balance
                    formatted_total = f"${total_balance:.2f}"
                    
                    # Generate a text summary
                    summary = f"Current total balance: {formatted_total}\n\n"
                    for account in account_balances:
                        if not account['closed']:
                            summary += f"- {account['name']} ({account['type']}): {account['balance']}\n"
                    
                    # Return the results
                    return {
                        'summary': f"Current total balance: {formatted_total}",
                        'total_balance': formatted_total,
                        'accounts': account_balances,
                        'details': summary
                    }
                except Exception as e:
                    self.logger.error(f"Failed to get balance: {e}")
                    return {
                        'summary': f"Failed to retrieve balance: {str(e)}",
                        'error': {}
                    }
            
            else:
                return {
                    'summary': "I couldn't understand your request. Please try rephrasing it.",
                    'error': {}
                }
                
        except Exception as e:
            self.logger.error("Query processing failed: %s", str(e))
            return {
                'summary': f"Failed to process query: {str(e)}",
                'error': {}
            }

    def _parse_categorization_response(self, response_text: str) -> Dict:
        try:
            # Remove any markdown code block formatting
            clean_text = response_text.strip('`').strip()
            
            # Try parsing as JSON
            try:
                parsed_results = json.loads(clean_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, try extracting JSON from text
                json_match = re.search(r'\[.*\]', clean_text, re.DOTALL)
                if json_match:
                    parsed_results = json.loads(json_match.group(0))
                else:
                    raise ValueError("Could not extract JSON from response")
            
            # Validate parsed results
            if not isinstance(parsed_results, list):
                raise ValueError("Response must be a list of categorizations")
            
            # Normalize results
            normalized_results = {}
            for result in parsed_results:
                transaction_id = result.get('transaction_id', 'unknown')
                normalized_results[transaction_id] = {
                    'category': result.get('category', 'Uncategorized'),
                    'confidence': result.get('confidence', 0.5),
                    'reasoning': result.get('reasoning', 'No reasoning provided')
                }
            
            return {
                'categorizations': normalized_results,
                'total_categorized': len(normalized_results)
            }
        except Exception as e:
            # Error handling
            return {
                'error': 'Response parsing failed',
                'raw_response': response_text,
                'details': str(e)
            } 