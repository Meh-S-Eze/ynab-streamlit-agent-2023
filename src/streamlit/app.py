import streamlit as st
import logging
from typing import Dict, Any, List, Optional

from src.core.container import Container

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main Streamlit application."""
    st.title("YNAB Financial Assistant")
    
    # Initialize Container services
    ynab_client = Container.get_ynab_client()
    budget_service = Container.get_budget_service()
    transaction_service = Container.get_transaction_service()
    
    # Initialize session state for storing app state
    if 'selected_budget' not in st.session_state:
        st.session_state.selected_budget = None
    if 'analysis_generated' not in st.session_state:
        st.session_state.analysis_generated = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("YNAB Connection")
        
        # Get budgets for selection
        try:
            budgets = ynab_client.get_budgets()
            budget_options = {b['name']: b['id'] for b in budgets}
            
            selected_budget_name = st.selectbox(
                "Select a Budget",
                options=list(budget_options.keys()),
                index=0 if budget_options else None,
            )
            
            if selected_budget_name:
                st.session_state.selected_budget = budget_options[selected_budget_name]
                st.success(f"Connected to budget: {selected_budget_name}")
            
            st.divider()
            
            # Display additional controls
            if st.session_state.selected_budget:
                if st.button("Generate Spending Analysis"):
                    with st.spinner("Analyzing your transactions..."):
                        st.session_state.analysis = budget_service.get_spending_analysis(
                            st.session_state.selected_budget
                        )
                        st.session_state.analysis_generated = True
                    st.success("Analysis complete!")
        
        except Exception as e:
            st.error(f"Error connecting to YNAB: {str(e)}")
            logger.exception("Error connecting to YNAB")
    
    # Main content area
    if not st.session_state.selected_budget:
        st.info("Please select a budget from the sidebar to get started.")
    else:
        st.header(f"Budget Analysis")
        
        # Natural language query input
        query = st.text_input("Ask a question about your finances:", placeholder="e.g., How much did I spend on groceries last month?")
        
        if query:
            with st.spinner("Processing your question..."):
                try:
                    result = transaction_service.process_natural_language_query(query, st.session_state.selected_budget)
                    
                    if result:
                        if 'summary' in result:
                            st.subheader("Summary")
                            st.write(result['summary'])
                        
                        if 'analysis' in result:
                            st.subheader("Analysis")
                            st.write(result['analysis'])
                        
                        if 'transactions' in result:
                            st.subheader(f"Transactions ({len(result['transactions'])})")
                            
                            # Convert transactions to a display table
                            if result['transactions']:
                                transactions_data = []
                                for tx in result['transactions']:
                                    transactions_data.append({
                                        "Date": tx.date,
                                        "Payee": tx.payee_name,
                                        "Amount": f"${tx.amount/1000:.2f}",
                                        "Category": tx.category_name,
                                    })
                                
                                st.table(transactions_data)
                            else:
                                st.info("No transactions found matching your query.")
                    else:
                        st.warning("No results found for your query.")
                
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    logger.exception(f"Error processing query: {e}")
        
        # Display spending analysis if generated
        if st.session_state.get('analysis_generated', False):
            analysis = st.session_state.analysis
            
            st.subheader("Spending Overview")
            st.metric("Total Spent", f"${analysis.total_spent}", delta=None)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Transaction Count", analysis.transaction_count)
            with col2:
                st.metric("Date Range", f"{analysis.start_date} to {analysis.end_date}")
            
            st.subheader("Category Breakdown")
            
            # Convert to a list for charting
            categories = list(analysis.category_breakdown.keys())
            amounts = list(analysis.category_breakdown.values())
            
            # Create a simple bar chart of spending by category
            if categories and amounts:
                chart_data = {"Category": categories, "Amount": amounts}
                st.bar_chart(chart_data, x="Category", y="Amount")
            
            # Display unusual transactions if any
            if analysis.unusual_transactions:
                st.subheader("Unusual Transactions")
                
                unusual_data = []
                for tx in analysis.unusual_transactions:
                    unusual_data.append({
                        "Date": tx.date,
                        "Payee": tx.payee_name,
                        "Amount": f"${tx.amount/1000:.2f}",
                        "Category": tx.category_name,
                    })
                
                st.table(unusual_data)

if __name__ == "__main__":
    main() 