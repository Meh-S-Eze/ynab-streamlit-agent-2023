from dependency_injector import containers, providers
from .config import ConfigManager
from .ynab_client import YNABClient
from .gemini_analyzer import GeminiSpendingAnalyzer
from .ai_modules.semantic_matcher import GeminiSemanticMatcher
from .ai_modules.spending_pattern_module import SpendingPatternAnalyzer
from .base_agent import BaseAgent

class Container(containers.DeclarativeContainer):
    """
    Centralized dependency injection container for the application
    Follows the dependency injection rule
    """
    # Core configuration
    config = providers.Singleton(ConfigManager)
    
    # YNAB client with config dependency
    ynab_client = providers.Singleton(
        YNABClient,
        config=config
    )
    
    # Gemini analyzer with dependencies
    gemini_analyzer = providers.Singleton(
        GeminiSpendingAnalyzer,
        config_manager=config,
        ynab_client=ynab_client
    )
    
    # Base agent with all dependencies
    base_agent = providers.Singleton(
        BaseAgent,
        name="CLIAgent",
        config_manager=config,
        ynab_client=ynab_client,
        gemini_analyzer=gemini_analyzer
    )
    
    semantic_matcher = providers.Factory(
        GeminiSemanticMatcher,
        config_manager=config
    )
    
    spending_pattern_analyzer = providers.Factory(
        SpendingPatternAnalyzer,
        config_manager=config
    ) 