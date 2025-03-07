import os

class OpenAIAPI:
    """
    Wrapper for OpenAI API interactions.
    """
    def __init__(self, model_name=None):
        """
        Initialize the OpenAI API.
        
        Args:
            model_name: Optional model name to use. If not provided, will use the GEMINI_OTHER_MODEL env var.
        """
        # Get model from environment if not provided
        self.model_name = model_name or os.environ.get("GEMINI_OTHER_MODEL")
        if not self.model_name:
            raise ValueError("GEMINI_OTHER_MODEL not set in environment and no model_name provided")
        
        # Initialize OpenAI
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) 