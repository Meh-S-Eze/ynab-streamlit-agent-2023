from flask import Flask, request, jsonify
import logging
import os
from src.core.container import Container
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/api/natural-language', methods=['POST'])
def process_natural_language():
    try:
        data = request.json
        query = data.get('query')
        budget_id = data.get('budget_id', os.getenv('YNAB_BUDGET_DEV'))
        
        if not query:
            return jsonify({"error": "Missing required parameter: query"}), 400
            
        # Initialize services through the container
        transaction_service = Container.get_transaction_service()
        
        # Process the natural language query
        logger.info(f"Processing query for budget {budget_id}: {query}")
        
        result = transaction_service.process_natural_language_query(query, budget_id)
        
        # Transform transaction objects to dictionaries for JSON serialization
        if 'transactions' in result:
            result['transactions'] = [
                {
                    'id': tx.id,
                    'date': str(tx.date),
                    'payee_name': tx.payee_name,
                    'amount': tx.amount / 1000.0,  # Convert milliunits to dollars
                    'category_name': tx.category_name,
                    'memo': tx.memo
                } for tx in result['transactions']
            ]
        
        return jsonify(result)
    
    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ai/generate', methods=['POST'])
def generate_ai_content():
    try:
        data = request.json
        prompt = data.get('prompt')
        
        if not prompt:
            return jsonify({"error": "Missing required parameter: prompt"}), 400
            
        # Get the AI client from the container
        ai_client = Container.get_ai_client()
        
        # Generate content
        response = ai_client.generate_content(prompt)
        
        return jsonify({
            "content": response.content,
            "provider": response.provider,
            "model_name": response.model_name
        })
    
    except Exception as e:
        logger.exception(f"Error generating AI content: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "AI API is running"})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 