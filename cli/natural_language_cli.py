import click
import google.generativeai as genai
from core.config import ConfigManager
from core.ynab_client import YNABClient
from core.gemini_analyzer import GeminiSpendingAnalyzer
from core.shared_models import BudgetAnalysis
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)

class NaturalLanguageAssistant:
    def __init__(self):
        # Get configuration
        self.config = ConfigManager()
        
        # Initialize YNAB client
        self.ynab_client = YNABClient()
        
        # Initialize Gemini analyzer
        self.gemini_analyzer = GeminiSpendingAnalyzer()
        
        # Configure Gemini model
        genai.configure(api_key=self.config.get('credentials.gemini.api_key'))
        self.model = genai.GenerativeModel('gemini-pro')

    def _get_budget_context(self):
        """Retrieve budget and transaction context"""
        budgets = self.ynab_client.get_budgets()
        
        # For simplicity, use the first budget
        budget = budgets[0]
        transactions = self.ynab_client.get_transactions(budget['id'])
        
        return {
            'budget_name': budget['name'],
            'budget_id': budget['id'],
            'transactions': transactions
        }

    def process_query(self, query: str):
        """
        Process natural language query about finances
        """
        # Get budget context
        budget_context = self._get_budget_context()
        
        # Create a comprehensive prompt
        prompt = f"""
        You are a financial assistant analyzing YNAB budget data.
        
        Budget Context:
        - Budget Name: {budget_context['budget_name']}
        - Total Transactions: {len(budget_context['transactions'])}
        
        User Query: {query}
        
        Provide a detailed, actionable response based on the budget data.
        Format your response as a JSON with these keys:
        - summary: A concise text summary
        - insights: List of key financial insights
        - recommendations: List of financial recommendations
        - data: Relevant numerical data
        """
        
        try:
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            # Parse the response
            try:
                parsed_response = json.loads(response.text)
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                parsed_response = {
                    'summary': response.text,
                    'insights': [],
                    'recommendations': [],
                    'data': {}
                }
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'summary': "I encountered an error processing your query.",
                'insights': [],
                'recommendations': [],
                'data': {}
            }

@click.command()
@click.option('--query', prompt='What would you like to know about your finances?', 
              help='Natural language query about your budget')
def financial_assistant(query):
    """
    YNAB Financial Assistant with Natural Language Processing
    """
    assistant = NaturalLanguageAssistant()
    
    # Process the query
    result = assistant.process_query(query)
    
    # Display results
    click.echo("\n🤖 Financial Assistant Response:\n")
    
    # Summary
    click.style("Summary:", fg='green', bold=True)
    click.echo(result.get('summary', 'No summary available'))
    
    # Insights
    if result.get('insights'):
        click.echo("\n💡 Key Insights:")
        for insight in result['insights']:
            click.echo(f"  - {insight}")
    
    # Recommendations
    if result.get('recommendations'):
        click.echo("\n📈 Recommendations:")
        for recommendation in result['recommendations']:
            click.echo(f"  - {recommendation}")
    
    # Additional Data
    if result.get('data'):
        click.echo("\n📊 Detailed Data:")
        click.echo(json.dumps(result['data'], indent=2))

if __name__ == '__main__':
    financial_assistant() 