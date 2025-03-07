#!/usr/bin/env python
import os
import sys
import logging
from decimal import Decimal
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path so we can import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables for testing
load_dotenv()

# Get model from environment, using the correct environment variable
GEMINI_MODEL = os.environ.get("GEMINI_REASONER_MODEL")

# Import after environment is loaded
from core.pydantic_ai_transaction_parser import PydanticAITransactionParser
from core.shared_models import TransactionCreate, TransactionAmount

def main():
    """Test the payee matching functionality"""
    # Setup logging
    setup_logging()
    
    # Load environment variables
    load_dotenv()
    
    logging.info(f"Using Gemini model: {GEMINI_MODEL}")
    
    # Print environment variables for debugging
    logger.info(f"YNAB_API_KEY: {'*' * 5 if os.getenv('YNAB_API_KEY') else 'Not set'}")
    logger.info(f"YNAB_BUDGET_DEV: {os.getenv('YNAB_BUDGET_DEV') or 'Not set'}")
    logger.info(f"GOOGLE_API_KEY: {'*' * 5 if os.getenv('GOOGLE_API_KEY') else 'Not set'}")
    
    try:
        # Initialize the transaction parser
        parser = PydanticAITransactionParser()
        
        # Log the initialization of the parser and payee cache
        logger.info("Transaction parser initialized")
        
        # Get available payees for reference
        payees = parser.ynab_client.get_payees()
        logger.info(f"Found {len(payees)} existing payees")
        
        # Print the first 5 payees for reference
        for i, payee in enumerate(payees[:5]):
            logger.info(f"Payee {i+1}: {payee.get('name')} (ID: {payee.get('id')})")
        
        # Test queries with different payee names
        test_queries = [
            "I spent $45 at Walmart yesterday",
            "Paid $32.50 for groceries at Kroger",
            "Bought a coffee at Starbucks for $5.75",
            "Dinner at Chipotle for $15.20",
            "Purchased gas at Shell for $42.30"
        ]
        
        # Process each test query
        for query in test_queries:
            logger.info(f"\nTesting query: {query}")
            
            # Parse the transaction
            transaction = parser.parse_transaction_sync(query)
            
            # Log the results
            logger.info(f"Payee Name: {transaction.payee_name}")
            logger.info(f"Payee ID: {transaction.payee_id}")
            logger.info(f"Amount: ${float(transaction.amount.amount)/1000} ({'outflow' if transaction.amount.is_outflow else 'inflow'})")
            logger.info(f"Date: {transaction.date}")
            logger.info(f"Memo: {transaction.memo}")
            logger.info(f"Category: {transaction.category_name}")
            
            # Check for payee matching
            if transaction.payee_id:
                logger.info(f"✅ Successfully matched to existing payee with ID: {transaction.payee_id}")
                
                # Find the payee name from the ID
                matching_payee = next((p for p in payees if p.get('id') == transaction.payee_id), None)
                if matching_payee:
                    logger.info(f"Matched to existing payee: {matching_payee.get('name')}")
            else:
                logger.info(f"❌ No matching payee found for: {transaction.payee_name}")
                
            # Try to look up payee by name directly
            payee_id = parser.get_payee_id(transaction.payee_name if transaction.payee_name else "")
            if payee_id:
                logger.info(f"Direct payee lookup found ID: {payee_id}")
            
            # Try to get suggestions based on partial name
            if transaction.payee_name:
                suggestions = parser.get_payee_suggestions(transaction.payee_name, max_results=3)
                if suggestions:
                    logger.info(f"Payee suggestions: {', '.join([s['name'] for s in suggestions])}")
    
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 