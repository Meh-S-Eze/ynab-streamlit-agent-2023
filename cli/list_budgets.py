#!/usr/bin/env python
"""
Utility script to list all available YNAB budgets
"""
import sys
import os
import logging
from pathlib import Path

# Add parent directory to sys.path to import core modules
sys.path.append(str(Path(__file__).parent.parent))

from core.ynab_client import YNABClient
from core.config import ConfigManager

def list_budgets():
    """List all available YNAB budgets"""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize YNAB client
        client = YNABClient()
        
        # Get current budget ID for highlighting
        current_budget_id = os.getenv('YNAB_BUDGET_DEV')
        
        # Fetch budgets from API
        budgets = client.get_available_budgets()
        
        if not budgets:
            logger.info("No budgets found")
            return
        
        # Print budget information
        logger.info("\n=== Available YNAB Budgets ===")
        logger.info(f"Found {len(budgets)} budgets")
        logger.info("-" * 80)
        logger.info(f"{'Budget Name':<40} {'Budget ID':<40} {'Current':<10}")
        logger.info("-" * 80)
        
        for budget in budgets:
            budget_id = budget.get('id', 'N/A')
            is_current = budget_id == current_budget_id
            current_marker = "✓" if is_current else ""
            
            logger.info(f"{budget.get('name', 'Unknown'):<40} {budget_id:<40} {current_marker:<10}")
        
        logger.info("\nTo change the budget ID, update the YNAB_BUDGET_DEV in your .env file")
        logger.info("Example: YNAB_BUDGET_DEV=budget_id_here")
    
    except Exception as e:
        logger.error(f"Error retrieving budgets: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(list_budgets()) 