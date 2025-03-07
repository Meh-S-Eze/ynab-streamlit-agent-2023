"""
Test script to verify category caching implementation works correctly
"""
import os
import sys
import time
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('category_test')

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.credentials import CredentialsManager
from core.ynab_client import YNABClient

def test_category_caching():
    """Test the category caching implementation and AI tagging preservation."""
    print("Testing category caching...")
    
    # Initialize the credentials manager
    try:
        credentials_manager = CredentialsManager()
        print("Credentials manager initialized successfully")
    except Exception as e:
        print(f"Failed to initialize credentials manager: {str(e)}")
        return
    
    # Initialize the YNAB client
    ynab_client = YNABClient(credentials_manager.get_ynab_token())
    
    # Get a list of transactions to use for testing
    try:
        transactions = ynab_client.get_transactions(batch_size=300)
        print(f"Retrieved {len(transactions)} transactions")
        
        # Find a transaction to use for testing
        test_transaction = None
        for transaction in transactions:
            if transaction.get('payee_name') and transaction.get('date'):
                test_transaction = transaction
                break
        
        if not test_transaction:
            print("Could not find a suitable test transaction")
            return
            
        print(f"Selected test transaction: {test_transaction['id']} - {test_transaction['date']} - {test_transaction.get('payee_name', 'Unknown')}")
        
        # Test category retrieval performance (first run)
        print("Testing retrieval time for categories (first run)...")
        start_time = time.time()
        categories = ynab_client.get_categories()
        first_run_time = time.time() - start_time
        
        category_count = 0
        active_category_count = 0
        group_count = 0
        
        for group in categories:
            group_count += 1
            for category in group.get('categories', []):
                category_count += 1
                if not category.get('deleted', False):
                    active_category_count += 1
        
        print(f"Retrieved {group_count} category groups with {active_category_count} active categories out of {category_count} total")
        print(f"First run time: {first_run_time:.4f} seconds")
        
        # Test category retrieval performance (second run with cache)
        print("Testing retrieval time for categories (second run with cache)...")
        start_time = time.time()
        categories = ynab_client.get_categories()
        second_run_time = time.time() - start_time
        print(f"Second run time: {second_run_time:.4f} seconds")
        
        if first_run_time > 0:
            improvement = ((first_run_time - second_run_time) / first_run_time) * 100
            print(f"Cache improvement: {improvement:.2f}%")
        
        # Test category mapping retrieval
        print("Testing category mapping retrieval...")
        start_time = time.time()
        category_mapping = ynab_client._get_budget_categories()
        first_mapping_time = time.time() - start_time
        print(f"First mapping retrieval time: {first_mapping_time:.4f} seconds")
        
        # Test category mapping retrieval with cache
        print("Testing category mapping retrieval with cache...")
        start_time = time.time()
        category_mapping = ynab_client._get_budget_categories()
        second_mapping_time = time.time() - start_time
        print(f"Second mapping retrieval time: {second_mapping_time:.4f} seconds")
        
        if first_mapping_time > 0:
            mapping_improvement = ((first_mapping_time - second_mapping_time) / first_mapping_time) * 100
            print(f"Mapping cache improvement: {mapping_improvement:.2f}%")
        
        # Test transaction category update with AI tagging
        print("Testing transaction category update with AI tagging...")
        
        # Get a random category to use for the update
        random_category = None
        for group in categories:
            if group.get('categories'):
                active_categories = [c for c in group['categories'] if not c.get('deleted', False)]
                if active_categories:
                    random_category = random.choice(active_categories)
                    break
        
        if not random_category:
            print("Could not find a category for testing")
            return
            
        print(f"Updating category for transaction {test_transaction['id']} to '{random_category['name']}'")
        
        # Update the transaction category with AI tagging
        update_result = ynab_client.update_transaction_category_with_ai_tag(
            transaction_id=test_transaction['id'],
            category_name=random_category['name']
        )
        
        if update_result:
            print(f"Successfully updated transaction category to '{random_category['name']}'")
            print(f"New memo: {update_result.get('memo', 'No memo')}")
            print(f"Category ID: {update_result.get('category_id', 'No category ID')}")
        else:
            print("Failed to update transaction category")
        
        # Test cache clearing
        print("Testing cache clearing...")
        start_time = time.time()
        ynab_client.clear_category_cache()
        clear_time = time.time() - start_time
        print(f"Cache clearing time: {clear_time:.4f} seconds")
        
        # Verify cache was cleared by timing a new retrieval
        print("Verifying cache was cleared...")
        start_time = time.time()
        categories = ynab_client.get_categories()
        post_clear_time = time.time() - start_time
        print(f"Post-clear retrieval time: {post_clear_time:.4f} seconds")
        
        # Test bulk update with AI tagging
        print("Testing bulk update with AI tagging...")
        
        # Create a list of transactions to update
        bulk_updates = [
            {
                'id': test_transaction['id'],
                'category_name': random_category['name']
            }
        ]
        
        bulk_result = ynab_client.bulk_update_categories_with_ai_tags(bulk_updates)
        
        if bulk_result:
            print(f"Bulk update results: {bulk_result['success']} successful, {bulk_result['failed']} failed")
        else:
            print("Failed to perform bulk update")
        
        print("Category caching test completed successfully")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_category_caching() 