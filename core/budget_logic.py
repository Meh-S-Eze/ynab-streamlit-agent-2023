from typing import Protocol, List, Dict

class DataSource(Protocol):
    def get_transactions(self) -> List[Dict]:
        ...

class BudgetReporter:
    def __init__(self, data_source: DataSource):
        self.data_source = data_source
    
    def generate_spending_report(self) -> Dict:
        transactions = self.data_source.get_transactions()
        # Implement spending analysis logic
        return {
            "total_spent": sum(t['amount'] for t in transactions),
            "categories": self._categorize_spending(transactions)
        }
    
    def _categorize_spending(self, transactions: List[Dict]) -> Dict:
        # Implement category breakdown
        pass 