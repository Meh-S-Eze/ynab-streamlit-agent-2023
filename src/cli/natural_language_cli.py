import click
import logging
import time
import sys
import os
from typing import Dict, Any, List, Optional

# Add flexible import path handling
try:
    from core.container import Container  # When running from src directory
except ImportError:
    from src.core.container import Container  # When running from project root

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.command()
@click.argument('query')
@click.option('--budget-id', help='Budget ID to search within')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
def process_natural_language_query(query: str, budget_id: Optional[str] = None, verbose: bool = False):
    """
    Process natural language queries related to YNAB budgets and transactions.
    
    Examples:
        - "Show me my grocery spending for last month"
        - "Which categories did I overspend on?"
        - "Find transactions at Amazon over $50"
    """
    start_time = time.time()
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug(f"Processing query: {query}")
        logger.debug(f"Budget ID: {budget_id}")
    
    try:
        # Initialize services through the container
        nlp_service = Container.get_nlp_service()
        budget_service = Container.get_budget_service()
        
        # Get a default budget if none specified
        if not budget_id:
            budgets = Container.get_ynab_client().get_budgets()
            if budgets:
                budget_id = budgets[0].id
                logger.info(f"No budget specified, using: {budgets[0].name}")
            else:
                logger.error("No budgets found")
                click.echo("Error: No budgets found in your YNAB account")
                return
        
        # Process the natural language query
        logger.info(f"Processing query for budget {budget_id}: {query}")
        
        result = nlp_service.process_query(query, budget_id)
        
        # Output results
        if result:
            click.echo("\n" + "="*50)
            click.echo(f"Results for: {query}")
            click.echo("="*50)
            
            if 'transactions' in result:
                click.echo(f"\nFound {len(result['transactions'])} transactions:")
                for tx in result['transactions']:
                    click.echo(f"  {tx.date} | {tx.payee_name} | ${tx.amount/1000:.2f} | {tx.category_name}")
            
            if 'summary' in result:
                click.echo(f"\nSummary: {result['summary']}")
            
            if 'analysis' in result:
                click.echo(f"\nAnalysis: {result['analysis']}")
            
        else:
            click.echo("No results found for your query.")
        
        if verbose:
            elapsed_time = time.time() - start_time
            logger.debug(f"Query processing completed in {elapsed_time:.2f} seconds")
    
    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        click.echo(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    process_natural_language_query() 