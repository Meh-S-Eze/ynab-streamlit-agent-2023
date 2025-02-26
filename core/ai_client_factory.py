import os
import logging
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
from pydantic import BaseModel, Field

# Ensure environment variables are loaded
load_dotenv(override=True)

logger = logging.getLogger(__name__)

class AIProvider(str, Enum):
    """Enum for supported AI providers"""
    GEMINI = "gemini"
    OPENAI = "openai"
    AUTO = "auto"  # Automatically selects the best available provider


class AIModelRole(str, Enum):
    """Enum for model roles"""
    GENERAL = "general"      # General-purpose model (default)
    REASONING = "reasoning"  # Advanced reasoning capabilities
    FAST = "fast"           # Optimized for speed
    VISION = "vision"       # For image analysis
    EMBEDDING = "embedding" # For embeddings/vectors


class AIClientResponse(BaseModel):
    """Standardized AI client response"""
    content: str = Field(..., description="The response content")
    provider: AIProvider = Field(..., description="The AI provider that generated the response")
    model_name: str = Field(..., description="The model name used")
    raw_response: Any = Field(None, description="The raw response object from the provider")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AIClientConfig(BaseModel):
    """Configuration for AI client factory"""
    primary_provider: AIProvider = Field(default=AIProvider.AUTO, description="Primary AI provider")
    fallback_provider: AIProvider = Field(default=AIProvider.OPENAI, description="Fallback AI provider")
    gemini_general_model: str = Field(default="gemini-1.5-flash", description="Gemini general model")
    gemini_reasoning_model: str = Field(default="gemini-1.5-pro", description="Gemini reasoning model")
    openai_general_model: str = Field(default="gpt-4o-mini", description="OpenAI general model")
    openai_reasoning_model: str = Field(default="o1-mini", description="OpenAI reasoning model")
    retry_on_failure: bool = Field(default=True, description="Whether to retry with fallback if primary fails")
    timeout_seconds: int = Field(default=30, description="Timeout for API requests in seconds")


class AIClientFactory:
    """
    Factory for creating AI clients with automatic fallbacks
    
    This class handles:
    1. Selection of the appropriate AI provider based on availability
    2. Graceful fallback from one provider to another if primary fails
    3. Consistent interface for interacting with different AI models
    4. Model selection based on the specific task requirements
    """
    
    def __init__(self, config: Optional[AIClientConfig] = None):
        """
        Initialize the AI client factory
        
        Args:
            config: Optional AI client configuration
        """
        self.config = config or AIClientConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize providers
        self._initialize_providers()
        
        # Check provider availability
        self._check_provider_availability()
    
    def _initialize_providers(self):
        """Initialize available AI providers"""
        self.providers_available = {
            AIProvider.GEMINI: False,
            AIProvider.OPENAI: False
        }
        
        self.clients = {}
        
        # Initialize Gemini if API key is available
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.providers_available[AIProvider.GEMINI] = True
                self.logger.info("Gemini API initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Gemini API: {str(e)}")
        
        # Initialize OpenAI if API key is available
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key:
            try:
                self.clients[AIProvider.OPENAI] = OpenAI(api_key=openai_api_key)
                self.providers_available[AIProvider.OPENAI] = True
                self.logger.info("OpenAI API initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI API: {str(e)}")
    
    def _check_provider_availability(self):
        """Check which providers are available and set the active provider"""
        if self.config.primary_provider == AIProvider.AUTO:
            # Auto-select based on availability
            if self.providers_available[AIProvider.GEMINI]:
                self.active_provider = AIProvider.GEMINI
            elif self.providers_available[AIProvider.OPENAI]:
                self.active_provider = AIProvider.OPENAI
            else:
                raise ValueError("No AI providers available. Check API keys and connections.")
        else:
            # Use specified primary provider if available
            if self.providers_available[self.config.primary_provider]:
                self.active_provider = self.config.primary_provider
            elif self.providers_available[self.config.fallback_provider]:
                self.logger.warning(
                    f"Primary provider {self.config.primary_provider} not available. "
                    f"Using fallback provider {self.config.fallback_provider}."
                )
                self.active_provider = self.config.fallback_provider
            else:
                raise ValueError("Neither primary nor fallback AI providers are available.")
        
        self.logger.info(f"Using {self.active_provider} as the active AI provider")
    
    def _get_model_name(self, provider: AIProvider, role: AIModelRole) -> str:
        """
        Get the appropriate model name for the provider and role
        
        Args:
            provider: AI provider
            role: Model role
        
        Returns:
            Model name
        """
        if provider == AIProvider.GEMINI:
            if role == AIModelRole.REASONING:
                return self.config.gemini_reasoning_model
            else:
                return self.config.gemini_general_model
        elif provider == AIProvider.OPENAI:
            if role == AIModelRole.REASONING:
                return self.config.openai_reasoning_model
            else:
                return self.config.openai_general_model
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def generate_content(
        self, 
        prompt: str, 
        role: AIModelRole = AIModelRole.GENERAL,
        provider: Optional[AIProvider] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AIClientResponse:
        """
        Generate content using the appropriate AI provider
        
        Args:
            prompt: The input prompt
            role: The model role
            provider: Optional override for the AI provider
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
        
        Returns:
            AIClientResponse with standardized response format
        """
        # Use specified provider or fall back to active provider
        use_provider = provider or self.active_provider
        
        try:
            # Try primary provider
            return self._generate_with_provider(
                use_provider, prompt, role, temperature, max_tokens
            )
        except Exception as e:
            self.logger.warning(f"Error with {use_provider}: {str(e)}")
            
            # Try fallback if enabled and different from primary
            if (self.config.retry_on_failure and 
                use_provider != self.config.fallback_provider and
                self.providers_available[self.config.fallback_provider]):
                
                self.logger.info(f"Falling back to {self.config.fallback_provider}")
                return self._generate_with_provider(
                    self.config.fallback_provider, prompt, role, temperature, max_tokens
                )
            else:
                # Re-raise if no fallback is available or fallback is disabled
                raise
    
    def _generate_with_provider(
        self, 
        provider: AIProvider, 
        prompt: str, 
        role: AIModelRole,
        temperature: float,
        max_tokens: Optional[int]
    ) -> AIClientResponse:
        """
        Generate content with a specific provider
        
        Args:
            provider: AI provider
            prompt: Input prompt
            role: Model role
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
        
        Returns:
            AIClientResponse with provider-specific response
        """
        model_name = self._get_model_name(provider, role)
        
        if provider == AIProvider.GEMINI:
            return self._generate_with_gemini(prompt, model_name, temperature, max_tokens)
        elif provider == AIProvider.OPENAI:
            return self._generate_with_openai(prompt, model_name, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _generate_with_gemini(
        self, 
        prompt: str, 
        model_name: str, 
        temperature: float,
        max_tokens: Optional[int]
    ) -> AIClientResponse:
        """
        Generate content with Gemini
        
        Args:
            prompt: Input prompt
            model_name: Gemini model name
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
        
        Returns:
            AIClientResponse with Gemini response
        """
        model = genai.GenerativeModel(model_name)
        
        generation_config = {
            'temperature': temperature,
        }
        
        if max_tokens:
            generation_config['max_output_tokens'] = max_tokens
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return AIClientResponse(
            content=response.text,
            provider=AIProvider.GEMINI,
            model_name=model_name,
            raw_response=response,
            metadata={
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        )
    
    def _generate_with_openai(
        self, 
        prompt: str, 
        model_name: str, 
        temperature: float,
        max_tokens: Optional[int]
    ) -> AIClientResponse:
        """
        Generate content with OpenAI
        
        Args:
            prompt: Input prompt
            model_name: OpenAI model name
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
        
        Returns:
            AIClientResponse with OpenAI response
        """
        client = self.clients[AIProvider.OPENAI]
        
        # Special handling for o1 models which only support default temperature
        if 'o1' in model_name.lower():
            # o1 models only support default temperature
            model_temperature = 1.0
            self.logger.info(f"Using default temperature (1.0) for {model_name} as it doesn't support custom temperature values")
        else:
            model_temperature = temperature
        
        params = {
            'model': model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': model_temperature,
        }
        
        if max_tokens:
            params['max_tokens'] = max_tokens
        
        response = client.chat.completions.create(**params)
        
        return AIClientResponse(
            content=response.choices[0].message.content,
            provider=AIProvider.OPENAI,
            model_name=model_name,
            raw_response=response,
            metadata={
                'temperature': model_temperature,
                'requested_temperature': temperature,
                'max_tokens': max_tokens,
                'finish_reason': response.choices[0].finish_reason,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
        )
    
    def generate_with_general_model(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> AIClientResponse:
        """
        Generate content using the general model
        
        Args:
            prompt: The input prompt
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
        
        Returns:
            AIClientResponse with standardized response format
        """
        return self.generate_content(
            prompt=prompt,
            role=AIModelRole.GENERAL,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def generate_with_reasoning_model(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> AIClientResponse:
        """
        Generate content using the reasoning model
        
        Args:
            prompt: The input prompt
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
        
        Returns:
            AIClientResponse with standardized response format
        """
        return self.generate_content(
            prompt=prompt,
            role=AIModelRole.REASONING,
            temperature=temperature,
            max_tokens=max_tokens
        ) 