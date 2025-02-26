import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Force reload environment variables from .env file
load_dotenv(override=True)

# Get API key from environment
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("Error: OPENAI_API_KEY not found in environment variables")
    sys.exit(1)

# Print masked API key for debugging (only showing beginning format)
masked_key = f"{api_key[:10]}...{api_key[-4:]}" if len(api_key) > 14 else "***masked***"
print(f"Using API key: {masked_key}")
print(f"API key length: {len(api_key)} characters")

# Get target models to check
target_models = []
openai_model = os.getenv('OPENAI_MODEL')
openai_reasoning_model = os.getenv('OPENAI_REASONING_MODEL')

if openai_model:
    target_models.append(openai_model)
    print(f"Checking model: {openai_model}")
if openai_reasoning_model:
    target_models.append(openai_reasoning_model)
    print(f"Checking model: {openai_reasoning_model}")

# Initialize OpenAI client
try:
    print("\nInitializing OpenAI client...")
    client = OpenAI(
        api_key=api_key,
        # Uncomment and set if you're using an organization ID
        # organization="your-organization-id",
    )
    
    # Print OpenAI client configuration (excluding sensitive info)
    print(f"Client base URL: {client.base_url}")
    print(f"Client default headers: {[k for k in client.default_headers.keys()]}")
    
    # Try listing models
    print("\nAttempting to list models...")
    models = client.models.list()
    
    # Check if models exist
    print("\nAvailable models:")
    available_models = []
    for model in models.data:
        available_models.append(model.id)
        print(f"- {model.id}")
    
    # Check if target models exist in available models
    print("\nTarget models check:")
    for model in target_models:
        if model in available_models:
            print(f"✅ {model} is available")
        else:
            print(f"❌ {model} is NOT available")
            # Check for similar models
            similar_models = [m for m in available_models if model.split('-')[0] in m]
            if similar_models:
                print(f"   Similar models you could use:")
                for sm in similar_models[:5]:  # Show up to 5 similar models
                    print(f"   - {sm}")
    
except Exception as e:
    print(f"\nError connecting to OpenAI API: {str(e)}")
    print("\nTroubleshooting tips:")
    print("1. Verify that your API key is correct and active in your OpenAI account")
    print("2. If using a project key (sk-proj-...), make sure it has the right permissions")
    print("3. Check your internet connection and firewall settings")
    print("4. Verify that the OpenAI API is not experiencing downtime")
    print("5. If using a proxy or VPN, try disabling it temporarily")
    
    # If it's an authentication error, provide more specific guidance
    if "authentication" in str(e).lower() or "api key" in str(e).lower():
        print("\nAPI Key specific troubleshooting:")
        print("- For project keys (sk-proj-...), make sure they're configured with the right permissions")
        print("- Try creating a new API key in your OpenAI dashboard")
        print("- Ensure there are no extra spaces or characters in your API key")
        print("- Check if your OpenAI account has billing configured properly") 