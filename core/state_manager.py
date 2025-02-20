from pathlib import Path
import pickle
import json
from typing import Dict, List, Optional

class StateManager:
    """
    Centralized state management for both CLI and Streamlit
    Follows CLI State Persistence and Shared Core Architecture rules
    """
    STATE_FILE = Path.home() / ".ynab_financial_assistant_state.json"

    @classmethod
    def save_state(cls, data: Dict):
        """Save state to persistent storage"""
        with open(cls.STATE_FILE, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load_state(cls) -> Dict:
        """Load state from persistent storage"""
        try:
            with open(cls.STATE_FILE, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                'budgets': [],
                'recent_transactions': [],
                'analysis_history': []
            }

    @classmethod
    def update_state(cls, key: str, value):
        """Update a specific state key"""
        state = cls.load_state()
        state[key] = value
        cls.save_state(state)

    @classmethod
    def get_state(cls, key: str, default=None):
        """Retrieve a specific state key"""
        state = cls.load_state()
        return state.get(key, default) 