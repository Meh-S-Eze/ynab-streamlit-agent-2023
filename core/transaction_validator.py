from typing import List, Dict, Optional, Union
from datetime import datetime, date
import logging
from decimal import Decimal
from .shared_models import TransactionCreate, TransactionAmount

class DuplicateTransactionError(Exception):
    """Raised when a duplicate transaction is detected"""
    pass

class FutureDateError(Exception):
    """Raised when a transaction date is in the future"""
    pass

class InvalidTransactionError(Exception):
    """Raised when a transaction is invalid"""
    pass

class TransactionValidator:
    """
    Comprehensive transaction validation pipeline
    Implements validation rule for preventing duplicates and handling future dates
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_transaction(self, transaction: TransactionCreate, existing_transactions: List[Dict]) -> TransactionCreate:
        """
        Validate a single transaction through the complete pipeline
        
        Args:
            transaction (TransactionCreate): Transaction to validate
            existing_transactions (List[Dict]): List of existing transactions to check for duplicates
        
        Returns:
            Validated TransactionCreate object
        
        Raises:
            DuplicateTransactionError: If transaction appears to be a duplicate
            FutureDateError: If transaction date is in the future
            InvalidTransactionError: If transaction is invalid
        """
        try:
            # Check for future date
            if transaction.date > date.today():
                raise FutureDateError(f"Transaction date {transaction.date} is in the future")
            
            # Check for duplicates
            self._check_for_duplicates(transaction, existing_transactions)
            
            # Additional validation logic
            self._validate_amount(transaction.amount)
            self._validate_required_fields(transaction)
            
            return transaction
            
        except (DuplicateTransactionError, FutureDateError) as e:
            # Re-raise these specific errors
            raise
        except Exception as e:
            self.logger.error(f"Transaction validation failed: {e}")
            raise InvalidTransactionError(f"Transaction validation failed: {str(e)}")
    
    def validate_transactions(self, transactions: List[TransactionCreate], existing_transactions: List[Dict]) -> List[TransactionCreate]:
        """
        Validate a list of transactions
        
        Args:
            transactions (List[TransactionCreate]): Transactions to validate
            existing_transactions (List[Dict]): List of existing transactions to check for duplicates
        
        Returns:
            List of validated TransactionCreate objects
        """
        validated_transactions = []
        errors = []
        
        for transaction in transactions:
            try:
                validated = self.validate_transaction(transaction, existing_transactions)
                validated_transactions.append(validated)
            except Exception as e:
                errors.append({
                    'transaction': transaction.dict(),
                    'error': str(e)
                })
                self.logger.warning(f"Transaction validation failed: {e}")
        
        if errors:
            self.logger.warning(f"Some transactions failed validation: {errors}")
        
        return validated_transactions
    
    def _check_for_duplicates(self, transaction: TransactionCreate, existing_transactions: List[Dict]):
        """
        Check if a transaction appears to be a duplicate
        
        Args:
            transaction (TransactionCreate): Transaction to check
            existing_transactions (List[Dict]): List of existing transactions
        
        Raises:
            DuplicateTransactionError: If transaction appears to be a duplicate
        """
        # Convert transaction amount to milliunits for comparison
        if isinstance(transaction.amount, TransactionAmount):
            amount = transaction.amount.to_milliunits()
        else:
            amount = transaction.amount
        
        # Look for potential duplicates
        for existing in existing_transactions:
            # Check if transaction has the same:
            # 1. Date
            # 2. Amount (within small tolerance)
            # 3. Payee name (if provided)
            # 4. Within last 7 days (to avoid flagging regular payments)
            
            existing_date = datetime.strptime(existing['date'], '%Y-%m-%d').date()
            date_diff = abs((transaction.date - existing_date).days)
            
            if (date_diff <= 7 and  # Within 7 days
                abs(existing['amount'] - amount) < 1 and  # Same amount (within 0.001)
                (not transaction.payee_name or  # If payee provided, must match
                 transaction.payee_name.lower() == existing.get('payee_name', '').lower())):
                
                raise DuplicateTransactionError(
                    f"Potential duplicate transaction found: "
                    f"Date={existing['date']}, "
                    f"Amount={existing['amount']}, "
                    f"Payee={existing.get('payee_name')}"
                )
    
    def _validate_amount(self, amount: Union[TransactionAmount, int]):
        """
        Validate transaction amount
        
        Args:
            amount: Transaction amount to validate
        
        Raises:
            InvalidTransactionError: If amount is invalid
        """
        try:
            if isinstance(amount, TransactionAmount):
                # Ensure amount is not zero
                if amount.amount == 0:
                    raise InvalidTransactionError("Transaction amount cannot be zero")
                
                # Ensure amount is not too large
                if abs(amount.amount) > Decimal('999999999.999'):
                    raise InvalidTransactionError("Transaction amount is too large")
            else:
                # For raw milliunits
                if amount == 0:
                    raise InvalidTransactionError("Transaction amount cannot be zero")
                if abs(amount) > 999999999999:
                    raise InvalidTransactionError("Transaction amount is too large")
        except Exception as e:
            if not isinstance(e, InvalidTransactionError):
                raise InvalidTransactionError(f"Invalid amount: {str(e)}")
            raise
    
    def _validate_required_fields(self, transaction: TransactionCreate):
        """
        Validate required transaction fields
        
        Args:
            transaction (TransactionCreate): Transaction to validate
        
        Raises:
            InvalidTransactionError: If required fields are missing or invalid
        """
        # Account ID is required
        if not transaction.account_id:
            raise InvalidTransactionError("Account ID is required")
        
        # Date must be valid
        if not isinstance(transaction.date, date):
            raise InvalidTransactionError("Invalid date format")
        
        # If category_name is provided but category_id is not, this is an error
        if transaction.category_name and not transaction.category_id:
            self.logger.warning(
                f"Category name '{transaction.category_name}' provided but no category_id. "
                "This may indicate a category resolution failure."
            )
    
    def validate(self, transaction: TransactionCreate) -> TransactionCreate:
        """
        Validate a single transaction without checking for duplicates
        
        Args:
            transaction (TransactionCreate): Transaction to validate
            
        Returns:
            Validated TransactionCreate object
            
        Raises:
            FutureDateError: If transaction date is in the future
            InvalidTransactionError: If transaction is invalid
        """
        try:
            # Check for future date
            if transaction.date > date.today():
                raise FutureDateError(f"Transaction date {transaction.date} is in the future")
            
            # Validate amount
            self._validate_amount(transaction.amount)
            
            # Validate required fields
            self._validate_required_fields(transaction)
            
            return transaction
        except (FutureDateError, InvalidTransactionError):
            raise
        except Exception as e:
            self.logger.error(f"Unexpected validation error: {str(e)}")
            raise InvalidTransactionError(f"Validation failed: {str(e)}") 