import requests
from typing import List, Dict, Optional
from functools import lru_cache
from .circuit_breaker import CircuitBreaker
from .config import ConfigManager
import logging
import os
from .shared_models import TransactionCreate, TransactionAmount
from .transaction_validator import TransactionValidator, DuplicateTransactionError, FutureDateError, InvalidTransactionError
from .data_validation import DataValidator
from .date_utils import DateFormatter

class YNABClient:
    def __init__(self, personal_token: str = None, budget_id: str = None):
        """
        Initialize YNAB client
        
        Args:
            personal_token (str, optional): YNAB API token. If not provided, uses YNAB_API_KEY from .env
            budget_id (str, optional): YNAB budget ID. If not provided, uses YNAB_BUDGET_ID from .env
        """
        self.logger = logging.getLogger(__name__)
        
        # Always try environment variables first
        env_token = os.getenv('YNAB_API_KEY')
        env_budget_id = os.getenv('YNAB_BUDGET_ID')
        
        # Use provided values as fallback
        self.personal_token = env_token or personal_token
        self.budget_id = env_budget_id or budget_id
        
        # Validate credentials
        if not self.personal_token:
            raise ValueError("No YNAB API token found. Please set YNAB_API_KEY in .env or provide a token.")
        if not self.budget_id:
            raise ValueError("No YNAB Budget ID found. Please set YNAB_BUDGET_ID in .env.")
        
        self.base_url = "https://api.ynab.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.personal_token}",
            "Content-Type": "application/json"
        }
    
    @CircuitBreaker(max_failures=3)
    def get_transactions(self, budget_id: Optional[str] = None, batch_size: int = 10) -> List[Dict]:
        """
        Retrieve transactions for a specific budget with batching
        
        Args:
            budget_id (str, optional): Specific budget ID. Uses default if not provided.
            batch_size (int, optional): Number of transactions to retrieve per batch. Defaults to 10.
        
        Returns:
            List of transactions
        """
        budget_id = budget_id or self.budget_id
        
        try:
            self.logger.debug(f"Retrieving transactions for budget {budget_id}")
            response = requests.get(
                f"{self.base_url}/budgets/{budget_id}/transactions", 
                headers=self.headers,
                params={
                    'per_page': batch_size  # Implement basic batching
                }
            )
            response.raise_for_status()
            transactions = response.json()["data"]["transactions"]
            
            # Log transaction details for debugging
            for transaction in transactions:
                self.logger.debug(
                    f"Transaction {transaction.get('id')}: "
                    f"category_id={transaction.get('category_id', 'None')}, "
                    f"payee={transaction.get('payee_name', 'None')}, "
                    f"amount={transaction.get('amount', 'None')}"
                )
            
            self.logger.info(f"Retrieved {len(transactions)} transactions")
            return transactions
        except requests.RequestException as e:
            self.logger.error(f"Failed to retrieve transactions: {e}")
            raise
    
    def validate_transactions(self, transactions: List[Dict]) -> List[Dict]:
        """
        Validate retrieved transactions
        
        Args:
            transactions (List[Dict]): List of transactions to validate
        
        Returns:
            List of validated transactions
        """
        validated_transactions = []
        for transaction in transactions:
            try:
                # Basic validation - add more complex validation as needed
                if all(key in transaction for key in ['id', 'date', 'amount', 'category_id']):
                    validated_transactions.append(transaction)
                else:
                    self.logger.warning(f"Invalid transaction structure: {transaction}")
            except Exception as e:
                self.logger.error(f"Transaction validation error: {e}")
        
        return validated_transactions

    def _get_budget_categories(self, budget_id: Optional[str] = None) -> Dict:
        """
        Retrieve and process categories for a specific budget
        
        Args:
            budget_id (str, optional): Specific budget ID. Uses default if not provided.
        
        Returns:
            Dictionary of processed categories
        """
        budget_id = budget_id or self.budget_id
        
        try:
            categories_response = requests.get(
                f"{self.base_url}/budgets/{budget_id}/categories", 
                headers=self.headers
            )
            categories_response.raise_for_status()
            category_groups = categories_response.json()['data']['category_groups']
            
            # Flatten categories and create a comprehensive mapping
            category_mapping = {}
            for group in category_groups:
                if group.get('hidden') or group.get('deleted'):
                    continue
                    
                for category in group.get('categories', []):
                    if category.get('hidden') or category.get('deleted'):
                        continue
                        
                    # Store full category name (with group) and simple name
                    full_name = f"{group['name']}: {category['name']}"
                    simple_name = category['name']
                    
                    # Add multiple variations for matching
                    variations = [
                        full_name.lower(),
                        simple_name.lower(),
                        simple_name.replace(' ', '').lower(),
                        simple_name.replace('&', 'and').lower(),
                        ''.join(c for c in simple_name if c.isalnum()).lower()
                    ]
                    
                    # Add all variations to mapping
                    for variation in variations:
                        category_mapping[variation] = {
                            'id': category['id'],
                            'name': simple_name,
                            'group': group['name']
                        }
            
            self.logger.debug(f"Retrieved {len(category_mapping)} category variations")
            return category_mapping
        
        except requests.RequestException as e:
            self.logger.error(f"Failed to retrieve budget categories: {e}")
            return {}

    def _find_best_category_match(self, category_name: str, category_mapping: Dict) -> Optional[str]:
        """
        Find the best matching category ID using multiple strategies with enhanced matching
        
        Args:
            category_name (str): Category name to match
            category_mapping (Dict): Mapping of category names to IDs
        
        Returns:
            Matched category ID or None
        """
        if not category_name:
            return None
            
        # Normalize input category name
        normalized_input = category_name.lower().strip()
        
        # Split into group and category if colon is present
        group_name = None
        if ':' in normalized_input:
            group_name, category_name = normalized_input.split(':', 1)
            group_name = group_name.strip()
            category_name = category_name.strip()
        else:
            category_name = normalized_input
        
        # Track match scores for each potential match
        matches = []
        
        for key, value in category_mapping.items():
            # Skip if group name is provided and doesn't match
            if group_name and value['group'].lower() != group_name:
                continue
            
            # Calculate match score based on multiple factors
            score = 0
            
            # Exact match gets highest score
            if key == normalized_input:
                score = 100
            
            # Group and category exact match
            elif group_name and value['group'].lower() == group_name and value['name'].lower() == category_name:
                score = 95
            
            # Category name exact match (ignoring group)
            elif value['name'].lower() == category_name:
                score = 90
            
            # Substring matches
            elif category_name in key:
                score = 80
            elif key in category_name:
                score = 75
            
            # Word-level matches
            input_words = set(category_name.split())
            key_words = set(key.split())
            common_words = input_words & key_words
            if common_words:
                # Score based on percentage of matching words
                word_match_score = len(common_words) / max(len(input_words), len(key_words)) * 70
                score = max(score, word_match_score)
            
            # Alphanumeric comparison
            input_alphanum = ''.join(c.lower() for c in category_name if c.isalnum())
            key_alphanum = ''.join(c.lower() for c in key if c.isalnum())
            if input_alphanum == key_alphanum:
                score = max(score, 85)
            
            # Add to matches if score is above threshold
            if score > 50:  # Minimum threshold for considering a match
                matches.append({
                    'id': value['id'],
                    'name': value['name'],
                    'group': value['group'],
                    'score': score
                })
        
        # Sort matches by score
        matches.sort(key=lambda x: (-x['score'], len(x['name'])))  # Higher score first, shorter name wins ties
        
        # Log match results for debugging
        if matches:
            best_match = matches[0]
            self.logger.debug(
                f"Category match for '{category_name}': "
                f"{best_match['group']}:{best_match['name']} "
                f"(score: {best_match['score']})"
            )
            if len(matches) > 1:
                self.logger.debug(
                    f"Alternative matches: "
                    f"{', '.join(f'{m['group']}:{m['name']} ({m['score']})' for m in matches[1:3])}"
                )
        else:
            self.logger.warning(f"No category match found for: {category_name}")
        
        return matches[0]['id'] if matches else None

    @CircuitBreaker(max_failures=3)
    def update_transaction_categories(self, budget_id: Optional[str] = None, transactions: List[Dict] = None):
        """
        Update transaction categories in YNAB with dynamic category mapping
        
        Args:
            budget_id (str, optional): Specific budget ID. Uses default if not provided.
            transactions (List[Dict]): List of transactions to update with new categories
        
        Returns:
            Response from YNAB API about the update
        """
        budget_id = budget_id or self.budget_id
        
        if not transactions:
            self.logger.warning("No transactions provided for category update")
            return None
        
        try:
            # Validate update payloads
            validated_updates = DataValidator.validate_ynab_update(transactions)
            
            # Retrieve existing categories from YNAB
            category_mapping = self._get_budget_categories(budget_id)
            
            if not category_mapping:
                self.logger.error("Failed to retrieve category mapping from YNAB")
                return {
                    'total_updated': 0,
                    'transactions': [],
                    'unmatched_categories': [],
                    'error': 'Failed to retrieve category mapping'
                }
            
            # Prepare transactions for YNAB API update
            update_payload = {
                "transactions": []
            }
            
            # Track unmatched categories for logging
            unmatched_categories = set()
            matched_categories = {}
            
            for update in validated_updates:
                category_name = update.category_name
                self.logger.debug(f"Processing category update for transaction {update.transaction_id}: {category_name}")
                
                # Try to find a matching category
                category_id = self._find_best_category_match(category_name, category_mapping)
                
                # If no match, log and skip
                if not category_id:
                    unmatched_categories.add(category_name)
                    self.logger.warning(f"No category match found for: {category_name}")
                    continue
                
                # Track successful matches
                matched_name = next((cat['name'] for cat in category_mapping.values() if cat['id'] == category_id), None)
                if matched_name:
                    matched_categories[category_name] = matched_name
                    self.logger.debug(f"Matched category {category_name} to {matched_name}")
                
                # Prepare transaction update
                update_payload['transactions'].append({
                    'id': update.transaction_id,
                    'category_id': category_id
                })
            
            # Log category matching results
            if matched_categories:
                self.logger.info(f"Category matches: {matched_categories}")
            if unmatched_categories:
                self.logger.warning(f"Unmatched categories: {unmatched_categories}")
            
            # Perform batch update
            if update_payload['transactions']:
                self.logger.info(f"Updating {len(update_payload['transactions'])} transactions with new categories")
                response = requests.patch(
                    f"{self.base_url}/budgets/{budget_id}/transactions",
                    headers=self.headers,
                    json=update_payload
                )
                response.raise_for_status()
                
                updated_transactions = response.json().get('data', {}).get('transactions', [])
                self.logger.info(f"Successfully updated {len(updated_transactions)} transaction categories")
                
                return {
                    'total_updated': len(updated_transactions),
                    'transactions': updated_transactions,
                    'unmatched_categories': list(unmatched_categories),
                    'category_matches': matched_categories
                }
            else:
                self.logger.warning("No transactions to update after category mapping")
                return {
                    'total_updated': 0,
                    'transactions': [],
                    'unmatched_categories': list(unmatched_categories),
                    'category_matches': matched_categories
                }
        
        except ValueError as ve:
            self.logger.error(f"Validation error: {ve}")
            raise
        except requests.RequestException as e:
            self.logger.error(f"Failed to update transaction categories: {e}")
            self.logger.error(f"Payload: {update_payload}")
            raise

    def validate_transaction_categories(self, transactions: List[Dict]) -> List[Dict]:
        """
        Validate transaction categories before updating
        
        Args:
            transactions (List[Dict]): Transactions to validate
        
        Returns:
            List of validated transactions
        """
        validated_transactions = []
        for transaction in transactions:
            # Basic validation checks
            if not all(key in transaction for key in ['id', 'category_name']):
                self.logger.warning(f"Invalid transaction for category update: {transaction}")
                continue
            
            # Additional validation logic can be added here
            validated_transactions.append(transaction)
        
        return validated_transactions

    def get_budget_context(self, budget_id: Optional[str] = None) -> Dict:
        """
        Retrieve comprehensive budget context
        
        Args:
            budget_id (Optional[str]): Specific budget ID. Uses default if not provided.
        
        Returns:
            Dict with budget and transaction details
        """
        # Retrieve budgets
        budgets = self.get_budgets()
        
        # Find the specified budget
        budget = next((b for b in budgets if b['id'] == (budget_id or self.budget_id)), None)
        
        if not budget:
            raise ValueError(f"Budget not found for ID: {budget_id or self.budget_id}")
        
        # Retrieve transactions for the budget
        transactions = self.get_transactions(budget_id=budget['id'])
        
        return {
            'budget_name': budget['name'],
            'budget_id': budget['id'],
            'transactions': transactions
        }

    def _format_date_for_api(self, input_date) -> str:
        """
        Format a date for the YNAB API using ISO-8601
        
        Args:
            input_date: Date to format
        
        Returns:
            str: ISO-8601 formatted date string
        """
        try:
            return DateFormatter.format_date(input_date)
        except ValueError as e:
            self.logger.error(f"Date formatting error: {e}")
            raise

    def create_transaction(self, budget_id: Optional[str] = None, transaction: Dict = None):
        """
        Create a transaction with comprehensive validation and error handling
        
        Args:
            budget_id (Optional[str]): Budget ID to create transaction in
            transaction (Dict): Transaction details
        
        Returns:
            Dict with transaction creation result
        """
        # Use default budget ID if not provided
        budget_id = budget_id or self.budget_id
        
        try:
            # Import validation dependencies
            from .shared_models import TransactionCreate, TransactionAmount
            from .transaction_validator import TransactionValidator, DuplicateTransactionError, FutureDateError, InvalidTransactionError
            
            # Format date to ISO-8601 if present
            if 'date' in transaction:
                transaction['date'] = self._format_date_for_api(transaction['date'])
            
            # Convert amount to TransactionAmount if not already
            if 'amount' in transaction and not isinstance(transaction['amount'], TransactionAmount):
                amount_value = transaction['amount']
                is_outflow = amount_value < 0 if isinstance(amount_value, (int, float)) else transaction.get('is_outflow', True)
                transaction['amount'] = TransactionAmount(
                    amount=abs(float(amount_value)) if isinstance(amount_value, (int, float)) else amount_value,
                    is_outflow=is_outflow
                )
            
            # Create transaction model
            transaction_model = TransactionCreate(**transaction)
            
            # Get existing transactions for duplicate checking
            existing_transactions = self.get_transactions(budget_id)
            
            # Validate transaction through pipeline
            validator = TransactionValidator()
            try:
                validated_transaction = validator.validate_transaction(
                    transaction_model,
                    existing_transactions
                )
            except DuplicateTransactionError as e:
                return {
                    'status': 'warning',
                    'message': str(e),
                    'details': {
                        'error_type': 'duplicate',
                        'transaction': transaction_model.dict()
                    }
                }
            except FutureDateError as e:
                return {
                    'status': 'error',
                    'message': str(e),
                    'details': {
                        'error_type': 'future_date',
                        'transaction': transaction_model.dict()
                    }
                }
            except InvalidTransactionError as e:
                return {
                    'status': 'error',
                    'message': str(e),
                    'details': {
                        'error_type': 'validation_error',
                        'transaction': transaction_model.dict()
                    }
                }
            
            # Convert to YNAB API format
            payload = {
                'transaction': validated_transaction.to_api_format()
            }
            
            # Remove None values from payload
            payload['transaction'] = {k: v for k, v in payload['transaction'].items() if v is not None}
            
            # Attempt transaction creation
            response = requests.post(
                f"{self.base_url}/budgets/{budget_id}/transactions", 
                headers=self.headers,
                json=payload
            )
            
            # Handle response
            if response.status_code == 201:
                created_transaction = response.json().get('data', {}).get('transaction', {})
                self.logger.info(f"Transaction created: {created_transaction.get('id')}")
                return {
                    'status': 'success',
                    'transaction_id': created_transaction.get('id'),
                    'message': 'Transaction successfully created',
                    'details': created_transaction
                }
            elif response.status_code == 409:
                self.logger.warning("Transaction creation conflict")
                return {
                    'status': 'conflict',
                    'message': 'Transaction may already exist',
                    'details': response.json()
                }
            else:
                self.logger.error(f"Transaction creation failed: {response.status_code}")
                return {
                    'status': 'error',
                    'code': response.status_code,
                    'message': response.text
                }
        
        except ValueError as ve:
            self.logger.error(f"Transaction validation error: {ve}")
            return {
                'status': 'error',
                'message': f'Validation error: {str(ve)}'
            }
        except Exception as e:
            self.logger.error(f"Transaction creation request failed: {e}")
            return {
                'status': 'error',
                'message': f'Network error: {str(e)}'
            }

    @CircuitBreaker(max_failures=3)
    def get_accounts(self, budget_id: Optional[str] = None) -> List[Dict]:
        """
        Get all accounts for a budget with enhanced error handling
        
        Args:
            budget_id (Optional[str]): Budget ID to get accounts for. Uses default if not provided.
        
        Returns:
            List of account dictionaries with structure:
            {
                'id': str,
                'name': str,
                'type': str,
                'on_budget': bool,
                'closed': bool,
                'balance': int,  # Balance in milliunits
                'cleared_balance': int,  # Cleared balance in milliunits
                'uncleared_balance': int,  # Uncleared balance in milliunits
                'transfer_payee_id': Optional[str],
                'deleted': bool
            }
        
        Raises:
            requests.RequestException: If the API request fails
            CircuitBreakerError: If too many failures occur
        """
        try:
            budget_id = budget_id or self.budget_id
            self.logger.debug(f"Retrieving accounts for budget {budget_id}")
            
            response = requests.get(
                f"{self.base_url}/budgets/{budget_id}/accounts",
                headers=self.headers
            )
            response.raise_for_status()
            
            accounts = response.json()['data']['accounts']
            
            # Log account details for debugging
            for account in accounts:
                self.logger.debug(
                    f"Account {account.get('id')}: "
                    f"name={account.get('name', 'None')}, "
                    f"type={account.get('type', 'None')}, "
                    f"balance={account.get('balance', 0)}"
                )
            
            self.logger.info(f"Retrieved {len(accounts)} accounts")
            return accounts
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to retrieve accounts: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"Response status: {e.response.status_code}")
                self.logger.error(f"Response body: {e.response.text}")
            raise

    def get_budget_balance(self, budget_id: Optional[str] = None) -> float:
        """
        Get the current balance for a budget
        
        Args:
            budget_id (Optional[str]): Budget ID to get balance for. Uses default if not provided.
        
        Returns:
            Current budget balance
        """
        budget_id = budget_id or self.budget_id
        response = requests.get(f"{self.base_url}/budgets/{budget_id}", headers=self.headers)
        response.raise_for_status()
        return response.json()['data']['budget']['balance']

    @CircuitBreaker(max_failures=3)
    def get_payees(self, budget_id: Optional[str] = None) -> List[Dict]:
        """
        Get all payees for a budget with enhanced error handling
        
        Args:
            budget_id (Optional[str]): Budget ID to get payees for. Uses default if not provided.
        
        Returns:
            List of payee dictionaries with structure:
            {
                'id': str,
                'name': str,
                'transfer_account_id': Optional[str],
                'deleted': bool
            }
        
        Raises:
            requests.RequestException: If the API request fails
            CircuitBreakerError: If too many failures occur
        """
        try:
            budget_id = budget_id or self.budget_id
            self.logger.debug(f"Retrieving payees for budget {budget_id}")
            
            response = requests.get(
                f"{self.base_url}/budgets/{budget_id}/payees",
                headers=self.headers
            )
            response.raise_for_status()
            
            payees = response.json()['data']['payees']
            
            # Log payee details for debugging
            active_payees = [p for p in payees if not p.get('deleted', False)]
            deleted_payees = [p for p in payees if p.get('deleted', False)]
            
            self.logger.debug(
                f"Retrieved {len(active_payees)} active payees and "
                f"{len(deleted_payees)} deleted payees"
            )
            
            for payee in active_payees:
                self.logger.debug(
                    f"Payee {payee.get('id')}: "
                    f"name={payee.get('name', 'None')}, "
                    f"transfer_account={payee.get('transfer_account_id', 'None')}"
                )
            
            self.logger.info(f"Retrieved {len(payees)} total payees")
            return payees
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to retrieve payees: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"Response status: {e.response.status_code}")
                self.logger.error(f"Response body: {e.response.text}")
            raise

    @CircuitBreaker(max_failures=3)
    def get_categories(self, budget_id: Optional[str] = None) -> List[Dict]:
        """
        Get all categories for a budget with enhanced error handling
        
        Args:
            budget_id (Optional[str]): Budget ID to get categories for. Uses default if not provided.
        
        Returns:
            List of category group dictionaries with nested categories:
            {
                'id': str,
                'name': str,
                'hidden': bool,
                'deleted': bool,
                'categories': [
                    {
                        'id': str,
                        'category_group_id': str,
                        'name': str,
                        'hidden': bool,
                        'deleted': bool,
                        'balance': int,  # Balance in milliunits
                        'goal_type': Optional[str],
                        'goal_target': Optional[int],
                        'goal_percentage_complete': Optional[int]
                    }
                ]
            }
        
        Raises:
            requests.RequestException: If the API request fails
            CircuitBreakerError: If too many failures occur
        """
        try:
            budget_id = budget_id or self.budget_id
            self.logger.debug(f"Retrieving categories for budget {budget_id}")
            
            response = requests.get(
                f"{self.base_url}/budgets/{budget_id}/categories",
                headers=self.headers
            )
            response.raise_for_status()
            
            category_groups = response.json()['data']['category_groups']
            
            # Log category details for debugging
            total_categories = 0
            active_categories = 0
            
            for group in category_groups:
                if group.get('deleted', False):
                    continue
                    
                group_categories = [
                    c for c in group.get('categories', [])
                    if not c.get('deleted', False)
                ]
                
                total_categories += len(group.get('categories', []))
                active_categories += len(group_categories)
                
                self.logger.debug(
                    f"Category Group {group.get('name')}: "
                    f"{len(group_categories)} active categories, "
                    f"hidden={group.get('hidden', False)}"
                )
                
                for category in group_categories:
                    self.logger.debug(
                        f"Category {category.get('id')}: "
                        f"name={category.get('name')}, "
                        f"balance={category.get('balance', 0)}, "
                        f"goal_type={category.get('goal_type', 'None')}"
                    )
            
            self.logger.info(
                f"Retrieved {len(category_groups)} category groups with "
                f"{active_categories} active categories out of {total_categories} total"
            )
            return category_groups
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to retrieve categories: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"Response status: {e.response.status_code}")
                self.logger.error(f"Response body: {e.response.text}")
            raise

    def get_category_by_id(self, category_id: str, budget_id: Optional[str] = None) -> Optional[Dict]:
        """
        Get a specific category by its ID with enhanced error handling
        
        Args:
            category_id (str): ID of the category to retrieve
            budget_id (Optional[str]): Budget ID to get category from. Uses default if not provided.
        
        Returns:
            Category dictionary or None if not found
        
        Raises:
            requests.RequestException: If the API request fails (except 404)
        """
        try:
            budget_id = budget_id or self.budget_id
            self.logger.debug(f"Retrieving category {category_id} from budget {budget_id}")
            
            response = requests.get(
                f"{self.base_url}/budgets/{budget_id}/categories/{category_id}",
                headers=self.headers
            )
            response.raise_for_status()
            
            category = response.json()['data']['category']
            self.logger.debug(
                f"Retrieved category: {category.get('name')}, "
                f"balance={category.get('balance', 0)}"
            )
            return category
            
        except requests.RequestException as e:
            if hasattr(e, 'response') and e.response.status_code == 404:
                self.logger.warning(f"Category {category_id} not found")
                return None
            self.logger.error(f"Failed to retrieve category: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"Response status: {e.response.status_code}")
                self.logger.error(f"Response body: {e.response.text}")
            raise 