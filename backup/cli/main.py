import click
from src.core.container import Container

@click.group()
def cli():
    """YNAB Financial Assistant CLI"""
    pass

@cli.command()
@click.option('--budget-id', help='Specific budget ID to analyze')
def analyze_budget(budget_id=None):
    """Analyze budget transactions via CLI"""
    # Initialize container and get services
    container = Container()
    ynab_client = container.get_ynab_client()
    budget_service = container.get_budget_service()
    
    # Get budgets if no specific budget provided
    if not budget_id:
        budgets = ynab_client.get_budgets()
        click.echo("Available Budgets:")
        for b in budgets:
            click.echo(f"{b['id']}: {b['name']}")
        return

    # Use the budget service to analyze the spending
    analysis = budget_service.get_spending_analysis(budget_id)
    
    click.echo("Budget Analysis:")
    click.echo(f"Total Spent: ${analysis.total_spent}")
    click.echo(f"Transaction Count: {analysis.transaction_count}")
    click.echo(f"Analysis Period: {analysis.start_date} to {analysis.end_date}")
    
    click.echo("\nCategory Breakdown:")
    for category, amount in analysis.category_breakdown.items():
        click.echo(f"  {category}: ${amount}")
        
    if analysis.unusual_transactions:
        click.echo("\nUnusual Transactions:")
        for tx in analysis.unusual_transactions:
            click.echo(f"  {tx.date} | {tx.payee_name} | ${tx.amount/1000:.2f} | {tx.category_name}")

if __name__ == '__main__':
    cli() 