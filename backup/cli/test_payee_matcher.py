#!/usr/bin/env python
import os
import sys
import logging
import json
from typing import Optional, List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock transaction data class
class MockTransactionData:
    """Mock class for testing payee matching"""
    def __init__(self, payee_name, payee_id=None):
        self.payee_name = payee_name
        self.payee_id = payee_id

# Mock payee matcher class
class PayeeMatcher:
    """Simple class to demonstrate payee matching logic"""
    
    def __init__(self, payees):
        """Initialize with a list of payees"""
        self.payees = payees
        self._payee_cache = {
            payee['name'].lower(): payee['id'] 
            for payee in payees 
            if not payee.get('deleted', False) and payee.get('name')
        }
        logger.info(f"Initialized payee cache with {len(self._payee_cache)} entries")
    
    def get_payee_id(self, payee_name: str) -> Optional[str]:
        """Get payee ID from cache by name"""
        if not payee_name:
            return None
            
        # Case-insensitive lookup
        payee_id = self._payee_cache.get(payee_name.lower())
        
        if payee_id:
            logger.info(f"Found payee ID in cache: {payee_name} -> {payee_id}")
        else:
            logger.info(f"Payee not found in cache: {payee_name}")
            
        return payee_id
    
    def get_payee_suggestions(self, partial_name: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Get suggestions for payees based on partial name match"""
        if not partial_name or len(partial_name) < 2:
            return []
            
        # Filter payees by partial name match (case insensitive)
        partial_name_lower = partial_name.lower()
        matches = [
            {"id": p["id"], "name": p["name"]}
            for p in self.payees
            if not p.get("deleted", False) 
            and partial_name_lower in p.get("name", "").lower()
        ]
        
        # Sort by closest match and limit results
        matches.sort(key=lambda p: p["name"].lower().find(partial_name_lower))
        return matches[:max_results]
    
    def process_transaction(self, transaction_data: MockTransactionData) -> Dict:
        """Process a transaction to use payee_id when available"""
        # Look up payee ID if we have a payee name
        if transaction_data.payee_name and not transaction_data.payee_id:
            transaction_data.payee_id = self.get_payee_id(transaction_data.payee_name)
            
        # Prepare result
        result = {
            "payee_name": transaction_data.payee_name if not transaction_data.payee_id else None,
            "payee_id": transaction_data.payee_id,
            "matched": transaction_data.payee_id is not None
        }
        
        # If we have a payee ID, find the name
        if transaction_data.payee_id:
            matching_payee = next((p for p in self.payees if p.get('id') == transaction_data.payee_id), None)
            if matching_payee:
                result["matched_name"] = matching_payee.get('name')
        
        return result

def main():
    """Test the payee matching functionality"""
    try:
        # Create mock payee data
        mock_payees = [
            {"id": "payee-1", "name": "Walmart", "deleted": False},
            {"id": "payee-2", "name": "Target", "deleted": False},
            {"id": "payee-3", "name": "Amazon", "deleted": False},
            {"id": "payee-4", "name": "Kroger", "deleted": False},
            {"id": "payee-5", "name": "Costco", "deleted": False},
            {"id": "payee-6", "name": "Starbucks", "deleted": False},
            {"id": "payee-7", "name": "McDonald's", "deleted": False},
            {"id": "payee-8", "name": "Chipotle", "deleted": False},
            {"id": "payee-9", "name": "Walmart Neighborhood Market", "deleted": False},
            {"id": "payee-10", "name": "Old Payee", "deleted": True}
        ]
        
        # Initialize the payee matcher
        matcher = PayeeMatcher(mock_payees)
        
        # Test queries with different payee names
        test_queries = [
            "I spent $45 at Walmart yesterday",
            "Paid $32.50 for groceries at Kroger",
            "Bought a coffee at Starbucks for $5.75",
            "Dinner at Chipotle for $15.20",
            "Purchased gas at Shell for $42.30",  # Not in our list
            "Walmart Neighborhood Market groceries $65.75"
        ]
        
        # Process each test query
        for query in test_queries:
            logger.info(f"\nTesting query: {query}")
            
            # Extract payee name (simplified for demo)
            payee_name = None
            for known_payee in [p["name"] for p in mock_payees if not p.get("deleted", False)]:
                if known_payee.lower() in query.lower():
                    payee_name = known_payee
                    break
            
            # Use first word if no match (simplified)
            if not payee_name:
                payee_name = query.split()[0]
                
            logger.info(f"Extracted payee name: {payee_name}")
            
            # Create mock transaction data
            transaction = MockTransactionData(payee_name=payee_name)
            
            # Process the transaction
            result = matcher.process_transaction(transaction)
            
            # Log the results
            if result["matched"]:
                logger.info(f"✅ Successfully matched to existing payee: {result.get('matched_name')} (ID: {result['payee_id']})")
            else:
                logger.info(f"❌ No matching payee found for: {result['payee_name']}")
            
            # Try to get suggestions based on partial name
            if payee_name:
                suggestions = matcher.get_payee_suggestions(payee_name, max_results=3)
                if suggestions:
                    logger.info(f"Payee suggestions: {', '.join([s['name'] for s in suggestions])}")
    
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 