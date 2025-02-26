import os
from dotenv import load_dotenv
import sys
import logging
import json

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from core
from core.ai_client_factory import AIClientFactory, AIClientConfig, AIProvider, AIModelRole

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

def test_ai_factory():
    """Test the AI client factory with fallback functionality"""
    
    # Create configurations for testing
    configs = [
        # Auto configuration (uses Gemini if available, falls back to OpenAI)
        AIClientConfig(
            primary_provider=AIProvider.AUTO,
            fallback_provider=AIProvider.OPENAI,
            retry_on_failure=True
        ),
        # Gemini with OpenAI fallback
        AIClientConfig(
            primary_provider=AIProvider.GEMINI,
            fallback_provider=AIProvider.OPENAI,
            retry_on_failure=True
        ),
        # OpenAI only
        AIClientConfig(
            primary_provider=AIProvider.OPENAI,
            fallback_provider=AIProvider.GEMINI,
            retry_on_failure=False  # Disable fallback
        )
    ]
    
    # Test prompt
    prompt = "Summarize the benefits of having multiple AI providers with fallback capabilities in a financial application in 3 brief points."
    
    # Test each configuration
    for i, config in enumerate(configs):
        logger.info(f"Testing configuration {i+1}: {config.primary_provider} → {config.fallback_provider}")
        
        try:
            # Create factory with this config
            factory = AIClientFactory(config)
            
            # Test with general model
            logger.info("Testing with general model...")
            response = factory.generate_content(
                prompt=prompt,
                role=AIModelRole.GENERAL,
                temperature=0.7
            )
            
            logger.info(f"Response from {response.provider} model {response.model_name}:")
            logger.info(response.content)
            
            if hasattr(response, 'metadata') and response.metadata:
                logger.info(f"Metadata: {json.dumps(response.metadata, indent=2)}")
            
            # Test with reasoning model
            logger.info("Testing with reasoning model...")
            response = factory.generate_content(
                prompt=prompt,
                role=AIModelRole.REASONING,
                temperature=0.7
            )
            
            logger.info(f"Response from {response.provider} model {response.model_name}:")
            logger.info(response.content)
            
            if hasattr(response, 'metadata') and response.metadata:
                logger.info(f"Metadata: {json.dumps(response.metadata, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error with configuration {i+1}: {str(e)}")
    
    logger.info("AI Factory testing complete")

if __name__ == "__main__":
    logger.info("Starting AI Factory test")
    test_ai_factory() 