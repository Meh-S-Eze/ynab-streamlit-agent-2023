import requests
from typing import List, Dict, Optional
from functools import lru_cache
from .circuit_breaker import CircuitBreaker
from .config import ConfigManager
import logging
import os

class YNABClient:
    def __init__(self, personal_token: str = None):
        """
        Initialize YNAB client
        
        Args:
            personal_token (str, optional): YNAB API token. If not provided, uses YNAB_API_KEY from .env
        """
        self.logger = logging.getLogger(__name__)
        
        # Prioritize passed personal_token, then .env, then config
        self.personal_token = (
            personal_token or 
            os.getenv('YNAB_API_KEY') or 
            ConfigManager.get('credentials.ynab.api_key')
        )
        
        # Validate personal token
        if not self.personal_token:
            raise ValueError("No YNAB API token found. Please set YNAB_API_KEY in .env or provide a token.")
        
        # Always use YNAB_BUDGET_ID from .env
        self.budget_id = os.getenv('YNAB_BUDGET_ID')
        
        # Validate budget ID
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
        Find the best matching category ID using multiple strategies
        
        Args:
            category_name (str): Category name to match
            category_mapping (Dict): Mapping of category names to IDs
        
        Returns:
            Matched category ID or None
        """
        if not category_name:
            return None
            
        # Enhanced matching strategies
        strategies = [
            lambda x: x.lower(),  # Exact lowercase match
            lambda x: x.replace(' ', '').lower(),  # Remove spaces
            lambda x: x.replace('&', 'and').lower(),  # Replace & with and
            lambda x: ''.join(c for c in x if c.isalnum()).lower(),  # Alphanumeric only
            lambda x: x.split()[0].lower(),  # First word match
            lambda x: x.split()[-1].lower(),  # Last word match
        ]
        
        # Try each strategy
        for strategy in strategies:
            processed_name = strategy(category_name)
            
            # Direct match
            if processed_name in category_mapping:
                return category_mapping[processed_name]['id']
            
            # Partial match
            for key, value in category_mapping.items():
                if processed_name in key or key in processed_name:
                    return value['id']
        
        self.logger.debug(f"No category match found for: {category_name}")
        return None

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
        
        for transaction in transactions:
            category_name = transaction.get('category_name', '')
            self.logger.debug(f"Processing category update for transaction {transaction.get('id')}: {category_name}")
            
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
                'id': transaction.get('id'),
                'category_id': category_id
            })
        
        # Log category matching results
        if matched_categories:
            self.logger.info(f"Category matches: {matched_categories}")
        if unmatched_categories:
            self.logger.warning(f"Unmatched categories: {unmatched_categories}")
        
        # Perform batch update
        try:
            # Only update if we have transactions to update
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
        
        # Validate input
        if not transaction:
            raise ValueError("Transaction details must be provided")
        
        # Validate transaction structure
        required_fields = ['account_id', 'date', 'amount']
        for field in required_fields:
            if field not in transaction:
                raise ValueError(f"Missing required field: {field}")
        
        # Milliunit conversion using Decimal for precise handling
        try:
            from decimal import Decimal, ROUND_HALF_UP
            
            # Handle string or numeric input
            amount_str = str(transaction['amount']).replace(',', '')
            
            # Convert amount to milliunits with precise rounding
            # Multiply by -1 if it's an outflow (negative amount)
            is_outflow = transaction.get('is_outflow', amount_str.startswith('-'))
            base_amount = abs(Decimal(amount_str))
            milliunits = int(
                base_amount.quantize(
                    Decimal('0.001'), 
                    rounding=ROUND_HALF_UP
                ) * 1000
            )
            
            # Apply negative sign for outflows
            transaction['amount'] = -milliunits if is_outflow else milliunits
            
        except Exception as e:
            self.logger.error(f"Milliunit conversion error: {e}")
            raise ValueError(f"Invalid amount format: {transaction['amount']}")
        
        # Prepare transaction payload with safe defaults
        payload = {
            'transaction': {
                'account_id': transaction['account_id'],
                'date': transaction['date'],
                'amount': transaction['amount'],
                'payee_name': transaction.get('payee_name', ''),
                'memo': transaction.get('memo', ''),
                'cleared': transaction.get('cleared', 'uncleared'),
                'approved': transaction.get('approved', False),
                'category_id': transaction.get('category_id')
            }
        }
        
        # Remove None values from payload
        payload['transaction'] = {k: v for k, v in payload['transaction'].items() if v is not None}
        
        # Attempt transaction creation
        try:
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
        
        except requests.RequestException as e:
            self.logger.error(f"Transaction creation request failed: {e}")
            return {
                'status': 'error',
                'message': f'Network error: {str(e)}'
            }

    def get_accounts(self, budget_id: Optional[str] = None) -> List[Dict]:
        """
        Get all accounts for a budget
        
        Args:
            budget_id (Optional[str]): Budget ID to get accounts for. Uses default if not provided.
        
        Returns:
            List of account dictionaries
        """
        budget_id = budget_id or self.budget_id
        response = requests.get(f"{self.base_url}/budgets/{budget_id}/accounts", headers=self.headers)
        response.raise_for_status()
        return response.json()['data']['accounts']

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