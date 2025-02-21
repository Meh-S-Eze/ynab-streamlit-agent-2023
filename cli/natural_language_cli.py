import click
import logging
import sys
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
def process(query: str):
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
        agent = container.get(BaseAgent)
        
        # Process the query through the agent
        result = agent.process_query(query)
        
        # Format and display the result
        print("\n🤖 Financial Assistant Response:\n")
        if result.get('status') == 'error':
            logger.error(f"Error: {result.get('message', 'Unknown error')}")
            print(f"Error: {result}")
        else:
            print(result)
            
    except Exception as e:
        logger.error("Query processing failed: %s", str(e), exc_info=True)
        print("\n🤖 Financial Assistant Response:\n")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    financial_assistant() 