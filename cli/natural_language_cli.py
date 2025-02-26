import click
import logging
import sys
import os
import shlex
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.base_agent import BaseAgent
from core.container import Container

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@click.group()
def financial_assistant():
    """
    YNAB Financial Assistant - Natural Language Interface
    
    Process natural language queries for financial operations:
    - Create transactions
    - Update transaction categories
    - Show transaction history
    - Get budget information
    """
    pass

@financial_assistant.command()
@click.argument('query')
def process(query):
    """
    Process a natural language query.
    
    Examples:
    - Create a transaction for $50 at Walmart
    - Change the $25 Target transaction to Groceries
    - Show my recent transactions
    - What's my current balance?
    """
    try:
        # Initialize dependencies through container
        container = Container()
        agent = container.base_agent()
        
        # Process the query through the agent
        result = agent.process_query(query)
        
        # Format and display the result
        print("\n🤖 Financial Assistant Response:\n")
        if isinstance(result, dict):
            if result.get('status') == 'error':
                print(f"❌ {result.get('summary', 'Unknown error')}")
            elif 'summary' in result:
                print(f"✅ {result['summary']}")
                
                # Display transaction details if available
                if 'transaction' in result:
                    transaction = result['transaction']
                    print("\nTransaction Details:")
                    if isinstance(transaction, dict):
                        for key, value in transaction.items():
                            print(f"  {key}: {value}")
                
                # Display transactions list if available
                if 'transactions' in result and isinstance(result['transactions'], list):
                    transactions = result['transactions']
                    print(f"\nTransactions ({len(transactions)}):")
                    for idx, tx in enumerate(transactions[:10], 1):  # Show first 10
                        print(f"  {idx}. {tx.get('date', 'N/A')} | {tx.get('payee', 'N/A')} | ${float(tx.get('amount', 0))/1000:.2f} | {tx.get('category', 'Uncategorized')}")
                    if len(transactions) > 10:
                        print(f"  ... and {len(transactions) - 10} more transactions.")
                
                # Display detailed information if available
                if 'details' in result and isinstance(result['details'], str):
                    print("\nDetails:")
                    print(result['details'])
            else:
                print(result)
        else:
            print(result)
            
    except Exception as e:
        print("\n🤖 Financial Assistant Response:\n")
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    financial_assistant() 