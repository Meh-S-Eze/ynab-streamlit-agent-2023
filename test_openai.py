from openai import OpenAI
import os

def test_openai_connection():
    try:
        # Initialize the client with just the API key
        client = OpenAI(
            api_key=os.environ.get('OPENAI_API_KEY')
        )
        
        # Try to create a simple chat completion as a test
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, are you available?"}]
        )
        print("Successfully connected to OpenAI API!")
        print("\nResponse:", response.choices[0].message.content)
            
    except Exception as e:
        print(f"Error connecting to OpenAI API: {str(e)}")
        print("\nDebug Information:")
        print(f"API Key (first 8 chars): {os.environ.get('OPENAI_API_KEY', 'NOT SET')[:8]}...")
        print("\nPlease check:")
        print("1. Verify your API key is correct")
        print("2. Ensure you have access to the GPT-4o models")
        print("3. Check if the model name is correct (currently using 'gpt-4o-mini')")

if __name__ == "__main__":
    test_openai_connection() 