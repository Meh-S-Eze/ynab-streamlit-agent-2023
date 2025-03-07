#!/bin/bash
# YNAB Natural Language CLI Test Script
# This script tests the YNAB CLI with various natural language commands

# Set up log file
LOG_FILE="cli_test_results_$(date +%Y%m%d_%H%M%S).log"
echo "===========================================================" | tee -a "$LOG_FILE"
echo "YNAB Natural Language CLI Test - $(date)" | tee -a "$LOG_FILE"
echo "===========================================================" | tee -a "$LOG_FILE"

# Run tests and log results
run_test() {
    echo -e "\nRunning: $1" | tee -a "$LOG_FILE"
    TIMEFORMAT=%R
    TIME=$( { time python cli.py "$1" > >(tee -a "$LOG_FILE") 2> >(tee -a "$LOG_FILE" >&2); } 2>&1 )
    echo "Execution time: $TIME seconds" | tee -a "$LOG_FILE"
    echo "-----------------------------------------------------------" | tee -a "$LOG_FILE"
}

# Transaction Creation Tests
echo -e "\n===========================================================" | tee -a "$LOG_FILE"
echo "Testing Category: Transaction Creation" | tee -a "$LOG_FILE"
echo "===========================================================" | tee -a "$LOG_FILE"

run_test "Create a transaction for $45.30 at Whole Foods yesterday"
run_test "Add a new purchase of $12.99 at Amazon for home supplies"
run_test "Record that I spent $8.50 on coffee this morning and categorize it as dining out"

# Spending Analysis Tests
echo -e "\n===========================================================" | tee -a "$LOG_FILE"
echo "Testing Category: Spending Analysis" | tee -a "$LOG_FILE"
echo "===========================================================" | tee -a "$LOG_FILE"

run_test "How much did I spend at restaurants last month?"
run_test "What was my total grocery spending in April?"
run_test "Show me all transactions at Target over $50"

# Budget Management Tests
echo -e "\n===========================================================" | tee -a "$LOG_FILE"
echo "Testing Category: Budget Management" | tee -a "$LOG_FILE"
echo "===========================================================" | tee -a "$LOG_FILE"

run_test "How much is left in my grocery budget this month?"
run_test "Am I over budget in any categories?"
run_test "How does my current spending compare to last month?"

# Account Management Tests
echo -e "\n===========================================================" | tee -a "$LOG_FILE"
echo "Testing Category: Account Management" | tee -a "$LOG_FILE"
echo "===========================================================" | tee -a "$LOG_FILE"

run_test "What's my current checking account balance?"
run_test "Show all transactions from my credit card this month"
run_test "Transfer $200 from savings to checking"

echo -e "\n===========================================================" | tee -a "$LOG_FILE"
echo "All tests completed. Results saved to $LOG_FILE" | tee -a "$LOG_FILE"
echo "===========================================================" | tee -a "$LOG_FILE" 