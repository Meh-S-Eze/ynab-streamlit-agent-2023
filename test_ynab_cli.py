#!/usr/bin/env python
"""
YNAB Natural Language CLI Test Script
This script tests the YNAB natural language interface with various command categories
and logs the results for analysis.
"""

import subprocess
import time
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'cli_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger('').addHandler(console)

# Define test cases by category
TEST_CASES = {
    "Transaction Creation": [
        "Create a transaction for $45.30 at Whole Foods yesterday",
        "Add a new purchase of $12.99 at Amazon for home supplies",
        "Record that I spent $8.50 on coffee this morning and categorize it as dining out"
    ],
    "Spending Analysis": [
        "How much did I spend at restaurants last month?",
        "What was my total grocery spending in April?",
        "Show me all transactions at Target over $50"
    ],
    "Budget Management": [
        "How much is left in my grocery budget this month?",
        "Am I over budget in any categories?",
        "How does my current spending compare to last month?"
    ],
    "Account Management": [
        "What's my current checking account balance?",
        "Show all transactions from my credit card this month",
        "Transfer $200 from savings to checking"
    ]
}

def run_cli_command(command):
    """Run a CLI command and return the output"""
    full_command = f"python cli.py \"{command}\""
    logging.info(f"Running: {full_command}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            full_command, 
            shell=True, 
            capture_output=True, 
            text=True
        )
        end_time = time.time()
        
        logging.debug(f"STDOUT: {result.stdout}")
        if result.stderr:
            logging.warning(f"STDERR: {result.stderr}")
        
        execution_time = end_time - start_time
        logging.info(f"Execution time: {execution_time:.2f} seconds")
        logging.info("-" * 80)
        
        return {
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "execution_time": execution_time
        }
    except Exception as e:
        logging.error(f"Error running command: {str(e)}")
        return {
            "command": command,
            "error": str(e)
        }

def main():
    """Run all test cases and log results"""
    total_tests = sum(len(cases) for cases in TEST_CASES.values())
    tests_run = 0
    tests_passed = 0
    
    logging.info("=" * 80)
    logging.info(f"YNAB Natural Language CLI Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 80)
    
    for category, commands in TEST_CASES.items():
        logging.info("\n" + "=" * 80)
        logging.info(f"Testing Category: {category}")
        logging.info("=" * 80)
        
        for command in commands:
            tests_run += 1
            logging.info(f"\nTest {tests_run}/{total_tests}: {command}")
            result = run_cli_command(command)
            
            # Simple success check - this should be made more sophisticated based on your actual CLI response format
            if result.get("returncode", 1) == 0 and not result.get("error"):
                tests_passed += 1
            
            # Add a small delay between tests to avoid overwhelming the API
            time.sleep(1)
    
    # Print summary
    logging.info("\n" + "=" * 80)
    logging.info(f"Test Summary: {tests_passed}/{tests_run} tests passed")
    logging.info("=" * 80)
    
    return tests_passed == tests_run

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 