import os
import requests
from dotenv import load_dotenv

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

# API endpoint for listing models
url = "https://api.openai.com/v1/models"

# Set headers with authorization
headers = {
    "Authorization": f"Bearer {api_key}"
}

print("Sending direct HTTP request to OpenAI API...")
try:
    response = requests.get(url, headers=headers)
    
    # Check status code
    print(f"Response status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        models = data.get("data", [])
        print(f"Successfully retrieved {len(models)} models")
        print("First 5 models:")
        for model in models[:5]:
            print(f"- {model.get('id')}")
    else:
        print(f"Error response: {response.text}")
        
except Exception as e:
    print(f"Exception occurred: {str(e)}") 