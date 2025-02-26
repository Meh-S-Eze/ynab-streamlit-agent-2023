import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("No API key found in environment variables")
    exit(1)

# Mask key for display
masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "****"
print(f"Using API key: {masked_key}")

# Extract project ID from key if it's a project key
# Project keys typically start with "sk-proj-" followed by the project ID
project_id = None
if api_key.startswith('sk-proj-'):
    # This is a simplified extraction - may need adjustment
    parts = api_key.split('-')
    if len(parts) > 2:
        project_id = parts[2]
        print(f"Detected project key with project ID: {project_id[:4]}...")

# Initialize client with project configuration if applicable
if project_id:
    print("Initializing OpenAI client with project configuration...")
    client = OpenAI(
        api_key=api_key,
        project=project_id,  # Set project ID if available
        base_url="https://api.openai.com/v1",  # Ensure using correct base URL
    )
else:
    print("Initializing standard OpenAI client...")
    client = OpenAI(api_key=api_key)

# Try to list models
print("Attempting to list models...")
try:
    models = client.models.list()
    print(f"Successfully retrieved models! Total count: {len(models.data)}")
    print("First 5 models:")
    for model in models.data[:5]:
        print(f"- {model.id}")
        
    # Check for specific models
    target_models = [os.getenv('OPENAI_MODEL'), os.getenv('OPENAI_REASONING_MODEL')]
    print(f"\nChecking for configured models: {target_models}")
    
    for target in target_models:
        found = False
        for model in models.data:
            if model.id == target:
                print(f"✅ Model '{target}' is available")
                found = True
                break
        if not found:
            print(f"❌ Model '{target}' was not found")
            
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc() 