import click
from core.ynab_client import YNABClient
from core.gemini_analyzer import GeminiSpendingAnalyzer
from core.state_manager import StateManager
from core.shared_models import BudgetAnalysis

@click.group()
def cli():
    """YNAB Financial Assistant CLI"""
    pass

@cli.command()
@click.option('--budget-id', help='Specific budget ID to analyze')
def analyze_budget(budget_id=None):
    """Analyze budget transactions via CLI"""
    client = YNABClient()
    
    # Get budgets if no specific budget provided
    if not budget_id:
        budgets = client.get_budgets()
        click.echo("Available Budgets:")
        for b in budgets:
            click.echo(f"{b['id']}: {b['name']}")
        return

    # Fetch transactions
    transactions = client.get_transactions(budget_id)
    
    # Use Gemini for analysis
    analyzer = GeminiSpendingAnalyzer()
    analysis = analyzer.analyze_transactions(transactions)
    
    # Save and display analysis
    budget_analysis = BudgetAnalysis(
        total_spent=analysis.total_spent,
        category_breakdown=analysis.category_breakdown,
        unusual_transactions=analysis.unusual_transactions
    )
    
    StateManager.update_state('recent_analysis', budget_analysis.dict())
    
    click.echo("Budget Analysis:")
    click.echo(f"Total Spent: ${budget_analysis.total_spent}")
    click.echo("Category Breakdown:")
    for category, amount in budget_analysis.category_breakdown.items():
        click.echo(f"  {category}: ${amount}")

if __name__ == '__main__':
    cli() 