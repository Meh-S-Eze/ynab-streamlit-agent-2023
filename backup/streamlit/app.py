import streamlit as st
from core.ynab_client import YNABClient
from core.gemini_analyzer import GeminiSpendingAnalyzer
from core.state_manager import StateManager
from core.shared_models import BudgetAnalysis

def main():
    st.title("YNAB Financial Assistant")
    
    # Load previous state
    previous_analysis = StateManager.get_state('recent_analysis')
    
    if previous_analysis:
        st.sidebar.write("Previous Analysis")
        st.sidebar.json(previous_analysis)
    
    # Token input
    token = st.text_input("Enter YNAB Personal Access Token")
    
    if token:
        client = YNABClient(token)
        
        # Budgets selection
        budgets = client.get_budgets()
        selected_budget = st.selectbox("Select Budget", [b['name'] for b in budgets])
        
        # Generate report
        if st.button("Analyze Budget"):
            budget_id = next(b['id'] for b in budgets if b['name'] == selected_budget)
            transactions = client.get_transactions(budget_id)
            
            analyzer = GeminiSpendingAnalyzer()
            analysis = analyzer.analyze_transactions(transactions)
            
            budget_analysis = BudgetAnalysis(
                total_spent=analysis.total_spent,
                category_breakdown=analysis.category_breakdown,
                unusual_transactions=analysis.unusual_transactions
            )
            
            # Save state
            StateManager.update_state('recent_analysis', budget_analysis.dict())
            
            # Display analysis
            st.write("### Spending Report")
            st.metric("Total Spent", f"${budget_analysis.total_spent:.2f}")
            
            st.write("#### Category Breakdown")
            st.bar_chart(budget_analysis.category_breakdown)
            
            st.write("#### Unusual Transactions")
            st.dataframe(budget_analysis.unusual_transactions)

if __name__ == "__main__":
    main() 