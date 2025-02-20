import requests
from typing import List, Dict
from functools import lru_cache
from .circuit_breaker import CircuitBreaker
from .config import ConfigManager

class YNABClient:
    def __init__(self, personal_token: str = None):
        self.personal_token = personal_token or ConfigManager.get_ynab_token()
        self.base_url = "https://api.ynab.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.personal_token}",
            "Content-Type": "application/json"
        }
    
    @CircuitBreaker()
    @lru_cache(maxsize=32)
    def get_budgets(self) -> List[Dict]:
        """Retrieve user's YNAB budgets with caching and circuit breaker"""
        response = requests.get(f"{self.base_url}/budgets", headers=self.headers)
        response.raise_for_status()
        return response.json()["data"]["budgets"]
    
    @CircuitBreaker()
    def get_transactions(self, budget_id: str) -> List[Dict]:
        """Retrieve transactions for a specific budget"""
        response = requests.get(f"{self.base_url}/budgets/{budget_id}/transactions", headers=self.headers)
        response.raise_for_status()
        return response.json()["data"]["transactions"] 