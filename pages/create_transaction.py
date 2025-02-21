import streamlit as st
from decimal import Decimal
from core.ynab_client import YNABClient
from datetime import date

def create_transaction_with_validation():
    """
    Streamlit function to create a transaction with comprehensive validation and error handling
    """
    st.title("YNAB Transaction Creator")
    
    # Initialize YNAB client
    try:
        ynab_client = YNABClient()
    except Exception as e:
        st.error(f"Failed to initialize YNAB client: {e}")
        return
    
    # Get available accounts
    try:
        budgets = ynab_client.get_budgets()
        if not budgets:
            st.error("No budgets found in your YNAB account")
            return
        
        # Default to the first budget if none selected
        default_budget = budgets[0]
        budget_names = {b['name']: b['id'] for b in budgets}
        selected_budget = st.selectbox(
            "Select Budget",
            options=list(budget_names.keys()),
            index=0
        )
        budget_id = budget_names[selected_budget]
        
    except Exception as e:
        st.error(f"Failed to retrieve budgets: {e}")
        return
    
    # Transaction input form
    with st.form("transaction_form"):
        st.write("Create New Transaction")
        
        # Input fields with validation
        account_id = st.text_input(
            "Account ID", 
            help="YNAB Account ID"
        )
        
        transaction_date = st.date_input(
            "Transaction Date",
            value=date.today(),
            help="Date of the transaction"
        )
        
        amount = st.number_input(
            "Amount",
            min_value=0.01,
            step=0.01,
            format="%.2f",
            help="Transaction amount (positive for inflow, negative for outflow)"
        )
        
        is_outflow = st.checkbox(
            "Is this an outflow (expense)?",
            value=True,
            help="Check for expenses, uncheck for income"
        )
        
        payee_name = st.text_input(
            "Payee Name",
            help="Name of the payee/merchant"
        )
        
        memo = st.text_input(
            "Memo",
            help="Optional transaction description"
        )
        
        # Transaction status
        cleared_status = st.selectbox(
            "Cleared Status",
            options=["uncleared", "cleared", "reconciled"],
            index=0
        )
        
        approved = st.checkbox(
            "Auto-approve transaction",
            value=False,
            help="Automatically approve the transaction in YNAB"
        )
        
        # Submit button
        submitted = st.form_submit_button("Create Transaction")
        
        if submitted:
            if not account_id:
                st.error("Account ID is required")
                return
            
            try:
                # Prepare transaction amount (negative for outflow)
                final_amount = -amount if is_outflow else amount
                
                # Prepare transaction payload
                transaction = {
                    'account_id': account_id,
                    'date': transaction_date.isoformat(),
                    'amount': final_amount,
                    'payee_name': payee_name,
                    'memo': memo,
                    'cleared': cleared_status,
                    'approved': approved
                }
                
                # Create transaction with comprehensive error handling
                with st.spinner("Processing transaction..."):
                    result = ynab_client.create_transaction(
                        budget_id=budget_id,
                        transaction=transaction
                    )
                
                # Handle different response statuses
                if result['status'] == 'success':
                    st.success(f"Transaction created successfully! ID: {result['transaction_id']}")
                    
                    # Show transaction details
                    st.json({
                        'Transaction ID': result['transaction_id'],
                        'Amount': f"{'$' if final_amount >= 0 else '-$'}{abs(final_amount):.2f}",
                        'Date': transaction_date.isoformat(),
                        'Payee': payee_name,
                        'Status': cleared_status.title()
                    })
                    
                elif result['status'] == 'warning':
                    st.warning(result['message'])
                    if st.button("Create Anyway"):
                        # Force create transaction
                        transaction['approved'] = True  # Auto-approve forced transactions
                        force_result = ynab_client.create_transaction(
                            budget_id=budget_id,
                            transaction=transaction
                        )
                        if force_result['status'] == 'success':
                            st.success(f"Transaction created! ID: {force_result['transaction_id']}")
                        else:
                            st.error(f"Failed to create transaction: {force_result['message']}")
                
                elif result['status'] == 'conflict':
                    st.warning("This transaction may be a duplicate.")
                    st.json(result['details'])
                    
                else:
                    st.error(f"Failed to create transaction: {result.get('message', 'Unknown error')}")
                    if 'code' in result:
                        st.error(f"Error code: {result['code']}")
            
            except ValueError as ve:
                st.error(f"Validation Error: {ve}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    create_transaction_with_validation() 