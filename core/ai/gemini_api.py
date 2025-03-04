import os

class GeminiAPI:
    """
    Wrapper for Gemini API interactions.
    """
    def __init__(self, model_name=None):
        """
        Initialize the Gemini API.
        
        Args:
            model_name: Optional model name to use. If not provided, will use the GEMINI_REASONER_MODEL env var.
        """
        # Get model from environment if not provided
        self.model_name = model_name or os.environ.get("GEMINI_REASONER_MODEL")
        if not self.model_name:
            raise ValueError("GEMINI_REASONER_MODEL not set in environment and no model_name provided")
        
        # Initialize Gemini
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        self.genai = genai
        self.model = genai.GenerativeModel(self.model_name) 