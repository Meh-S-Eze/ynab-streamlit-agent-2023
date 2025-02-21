from typing import List, Dict
from core.gemini_analyzer import AnalysisModule

class SpendingPatternAnalyzer(AnalysisModule):
    """
    Sample plugin module for analyzing spending patterns
    """
    def analyze(self, transactions: List[Dict]) -> Dict:
        """
        Analyze spending patterns in transactions
        
        Args:
            transactions (List[Dict]): Transactions to analyze
        
        Returns:
            Dict with spending pattern insights
        """
        # Basic spending pattern analysis
        total_spending = sum(abs(float(t.get('amount', 0))) for t in transactions)
        
        # Identify top spending categories
        category_totals = {}
        for transaction in transactions:
            category = transaction.get('category_name', 'Uncategorized')
            amount = abs(float(transaction.get('amount', 0)))
            category_totals[category] = category_totals.get(category, 0) + amount
        
        # Sort categories by total spending
        sorted_categories = sorted(
            category_totals.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return {
            'spending_patterns': {
                'total_spending': total_spending,
                'top_categories': sorted_categories[:3],
                'category_breakdown': category_totals
            }
        } 