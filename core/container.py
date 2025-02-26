from dependency_injector import containers, providers
from .config import ConfigManager
from .ynab_client import YNABClient
from .gemini_analyzer import GeminiSpendingAnalyzer
from .ai_modules.semantic_matcher import GeminiSemanticMatcher
from .ai_modules.spending_pattern_module import SpendingPatternAnalyzer
from .base_agent import BaseAgent
from .credentials import CredentialsManager
from .ai_client_factory import AIClientFactory, AIClientConfig, AIProvider

class Container(containers.DeclarativeContainer):
    """
    Centralized dependency injection container for the application
    Follows the dependency injection rule
    """
    # Core configuration
    config = providers.Singleton(ConfigManager)
    
    # Credentials management
    credentials_manager = providers.Singleton(
        CredentialsManager
    )
    
    # AI client factory with OpenAI fallback
    ai_client_factory = providers.Singleton(
        AIClientFactory,
        config=AIClientConfig(
            primary_provider=AIProvider.AUTO,
            fallback_provider=AIProvider.OPENAI,
            retry_on_failure=True
        )
    )
    
    # YNAB client with credentials dependency
    ynab_client = providers.Singleton(
        YNABClient,
        personal_token=credentials_manager.provided.get_ynab_token.call(),
        budget_id=credentials_manager.provided.get_ynab_budget_id.call()
    )
    
    # Gemini analyzer with dependencies
    gemini_analyzer = providers.Singleton(
        GeminiSpendingAnalyzer,
        config_manager=config,
        ynab_client=ynab_client,
        ai_client_factory=ai_client_factory
    )
    
    # Base agent with all dependencies
    base_agent = providers.Singleton(
        BaseAgent,
        name="CLIAgent",
        config_manager=config,
        ynab_client=ynab_client,
        gemini_analyzer=gemini_analyzer,
        ai_client_factory=ai_client_factory
    )
    
    semantic_matcher = providers.Factory(
        GeminiSemanticMatcher,
        config_manager=config,
        ai_client_factory=ai_client_factory
    )
    
    spending_pattern_analyzer = providers.Factory(
        SpendingPatternAnalyzer,
        config_manager=config,
        ai_client_factory=ai_client_factory
    ) 