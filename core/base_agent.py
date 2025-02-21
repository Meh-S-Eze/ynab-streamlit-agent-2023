from typing import Any, Dict, Callable, Optional, List
from functools import wraps
import time
import logging
from .agent_tool_tracker import tool_tracker
from .config import ConfigManager
from .ynab_client import YNABClient
from .gemini_analyzer import GeminiSpendingAnalyzer
import json
import re
from datetime import date
import requests

class BaseAgent:
    """
    Base class for AI agents with built-in tool tracking and configuration
    """
    def __init__(
        self, 
        name: str,
        config_manager: Optional[ConfigManager] = None,
        ynab_client: Optional[YNABClient] = None,
        gemini_analyzer: Optional[GeminiSpendingAnalyzer] = None
    ):
        """
        Initialize the agent with a name and dependencies
        
        Args:
            name (str): Unique identifier for the agent
            config_manager (ConfigManager, optional): Configuration management
            ynab_client (YNABClient, optional): YNAB API client
            gemini_analyzer (GeminiSpendingAnalyzer, optional): Gemini analyzer for AI operations
        """
        self.name = name
        self.tool_tracker = tool_tracker
        self.logger = logging.getLogger(f"agent.{name}")
        
        # Initialize core services with dependency injection
        self.config = config_manager or ConfigManager()
        self.ynab_client = ynab_client or YNABClient()
        
        # Initialize Gemini analyzer with dependencies if not provided
        if gemini_analyzer:
            self.gemini_analyzer = gemini_analyzer
        else:
            self.gemini_analyzer = GeminiSpendingAnalyzer(
                config_manager=self.config,
                ynab_client=self.ynab_client
            )

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
        Retrieve comprehensive budget context
        
        Args:
            budget_id (Optional[str]): Specific budget ID. Uses default if not provided.
        
        Returns:
            Dict with budget and transaction details
        """
        # Always use the configured budget ID from .env
        budget_id = budget_id or self.ynab_client.budget_id
        
        # Retrieve transactions for the budget
        transactions = self.use_tool(
            self.ynab_client.get_transactions, 
            budget_id=budget_id
        )
        
        # Get budget details through YNAB client
        budget = self.use_tool(
            self.ynab_client.get_budget_details,
            budget_id=budget_id
        )
        
        return {
            'budget_name': budget['name'],
            'budget_id': budget_id,
            'transactions': transactions
        }

    def process_query(self, query: str) -> Dict:
        """
        Process a natural language query and take appropriate action
        
        Args:
            query (str): Natural language query from the user
            
        Returns:
            Dict: Response with results or error information
        """
        self.logger.debug("Processing query: %r", query)
        
        try:
            # Get intent from Gemini
            intent_prompt = f"""
            Analyze this financial query and determine the intent:
            "{query}"
            
            Return ONLY a JSON object with these fields:
            {{
                "intent": "create_transaction" or "update_category" or "categorize_transactions" or "get_balance" or "show_transactions" or "unknown",
                "description": "Brief description of what the user wants to do",
                "details": {{
                    "should_categorize_all": true/false,  # Whether to categorize all transactions or just uncategorized ones
                    "specific_category": null or "category_name",  # If user wants to categorize to a specific category
                    "show_uncategorized": true/false,  # Whether to show only uncategorized transactions
                    "show_category": null or "category_name",  # If user wants to show transactions of a specific category
                    "is_category_update": true/false  # Whether this is a category update request
                }}
            }}
            """
            
            intent_response = self.gemini_analyzer.model.generate_content(
                intent_prompt,
                generation_config={
                    'temperature': 0.1
                }
            )
            
            # Parse intent from response
            intent_text = intent_response.text
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
                # Try to parse as a transaction creation request
                try:
                    transaction_details = self.gemini_analyzer.parse_transaction_creation(query)
                    if transaction_details:
                        # Get the first account from YNAB if not specified
                        if not transaction_details.get('account_id'):
                            accounts = self.ynab_client.get_accounts()
                            if accounts:
                                transaction_details['account_id'] = accounts[0]['id']
                        
                        # Create the transaction
                        result = self.ynab_client.create_transaction(
                            transaction=transaction_details
                        )
                        
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
                        'status': 'error',
                        'message': f'Failed to create transaction: {str(e)}'
                    }
                    
            elif intent_data['intent'] == 'update_category':
                # Process category update request
                try:
                    result = self.gemini_analyzer.process_category_update_request(query)
                    if result['status'] == 'success':
                        return {
                            'summary': 'Category updated successfully',
                            'transaction': result['transaction']
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
                balance = self.ynab_client.get_budget_balance(self.ynab_client.budget_id)
                return {
                    'summary': f"Current balance: {balance}",
                    'balance': balance
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