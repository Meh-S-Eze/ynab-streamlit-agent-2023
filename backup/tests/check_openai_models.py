import os
from dotenv import load_dotenv
from openai import OpenAI
import sys

# Load environment variables from .env file
load_dotenv()

# Get API key and target models
api_key = os.getenv('OPENAI_API_KEY')
target_models = [os.getenv('OPENAI_MODEL'), os.getenv('OPENAI_REASONING_MODEL')]

print(f"Checking OpenAI models: {target_models}")
if api_key:
    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "****"
    print(f"Using API key format: {masked_key}")
else:
    print("Warning: No API key found in environment variables")
    sys.exit(1)

# Initialize the OpenAI client with the API key directly
try:
    client = OpenAI(
        api_key=api_key,
        # Uncomment the line below if you're using a specific organization
        # organization="YOUR_ORG_ID",
    )
    
    # Print OpenAI client version
    import openai
    print(f"OpenAI SDK Version: {openai.__version__}")
    
    # Attempt to list models
    print("Attempting to list available models...")
    models = client.models.list()
    
    # Check if the target models exist
    available_models = [model.id for model in models.data]
    print(f"Available models: {available_models[:5]} ... (and {len(available_models) - 5} more)")
    
    for model in target_models:
        if model in available_models:
            print(f"✅ Model '{model}' is available")
        else:
            print(f"❌ Model '{model}' was not found in the list of available models")
            
except Exception as e:
    print(f"Error connecting to OpenAI API: {str(e)}")
    # Print more detailed error information
    import traceback
    print("Detailed error information:")
    traceback.print_exc() 