"""
Dependency injection container for YNAB integration.

This module provides a centralized container for all dependencies in the YNAB
integration. It follows the dependency injection pattern to make components more
testable and maintainable.
"""

import os
import sys
from typing import Optional
from dependency_injector import containers, providers
import requests

# Add parent directory to path for proper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import configuration and credentials
from src.core.config import ConfigManager
from src.core.credentials import CredentialsManager
from src.core.ai_client_factory import AIClientFactory, AIClientConfig, AIProvider

# Import API clients
from .api import (
    YNABClient,
    BudgetsAPI,
    TransactionsAPI,
    CategoriesAPI,
    AccountsAPI,
    PayeesAPI,
    RequestHandler
)

# Import services
from .services import (
    CategoryService,
    TransactionService,
    BudgetService,
    AITaggingService,
    NLPService,
    TransactionTaggingService,
    TransactionCreationService,
    TransactionQueryService,
    TransactionCleanupService
)

# Import prompt classes
from .prompts import (
    BasePrompt,
    TransactionCategoryPrompt,
    TransactionTaggingPrompt,
    CategoryMatchPrompt,
    CategoryHierarchyPrompt
)

# Import utilities
from .utils import (
    CircuitBreaker,
    DateFormatter,
    cached_method,
    clear_cache
)


class Container(containers.DeclarativeContainer):
    """
    Centralized dependency injection container for the application.
    
    This container wires up all components of the YNAB integration system
    including API clients, services, and utilities.
    """
    # Core configuration
    config = providers.Singleton(ConfigManager)
    
    # Credentials management
    credentials_manager = providers.Singleton(
        CredentialsManager
    )
    
    # AI client factory with configuration
    ai_client_factory = providers.Singleton(
        AIClientFactory,
        config=AIClientConfig(
            primary_provider=AIProvider.AUTO,
            fallback_provider=AIProvider.OPENAI,
            retry_on_failure=True
        )
    )
    
    # Request handler with circuit breaker and error handling
    request_handler = providers.Singleton(
        RequestHandler,
        session=providers.Factory(
            requests.Session
        ),
        base_url="https://api.youneedabudget.com/v1",
        timeout=30,
        circuit_breaker=providers.Factory(
            CircuitBreaker,
            max_failures=5,
            reset_timeout=60
        )
    )

    # YNAB API clients
    ynab_client = providers.Singleton(
        YNABClient,
        api_key=credentials_manager.provided.get_ynab_token.call()
    )
    
    # Specialized API clients
    budgets_api = providers.Singleton(
        BudgetsAPI,
        client=ynab_client
    )
    
    transactions_api = providers.Singleton(
        TransactionsAPI,
        client=ynab_client
    )
    
    categories_api = providers.Singleton(
        CategoriesAPI,
        client=ynab_client
    )
    
    accounts_api = providers.Singleton(
        AccountsAPI,
        client=ynab_client
    )
    
    payees_api = providers.Singleton(
        PayeesAPI,
        client=ynab_client
    )
    
    # Prompt templates
    transaction_category_prompt = providers.Factory(
        TransactionCategoryPrompt,
        examples_path="src/core/prompts/examples/transaction_examples.json"
    )
    
    transaction_tagging_prompt = providers.Factory(
        TransactionTaggingPrompt,
        examples_path="src/core/prompts/examples/transaction_examples.json"
    )
    
    category_match_prompt = providers.Factory(
        CategoryMatchPrompt,
        examples_path="src/core/prompts/examples/category_examples.json"
    )
    
    category_hierarchy_prompt = providers.Factory(
        CategoryHierarchyPrompt,
        examples_path="src/core/prompts/examples/category_examples.json"
    )
    
    # AI tagging service
    ai_tagging_service = providers.Singleton(
        AITaggingService
    )
    
    # Service classes with dependencies
    category_service = providers.Singleton(
        CategoryService,
        categories_api=categories_api
    )
    
    # Transaction tagging service
    transaction_tagging_service = providers.Singleton(
        TransactionTaggingService,
        ai_tagging_service=ai_tagging_service
    )
    
    transaction_service = providers.Singleton(
        TransactionService,
        transactions_api=transactions_api,
        category_service=category_service
    )
    
    # NLP service for natural language processing
    def nlp_service(self) -> NLPService:
        """
        Get the NLP service.

        Returns:
            NLPService: The NLP service
        """
        if not hasattr(self, "_nlp_service"):
            self._nlp_service = NLPService(
                transaction_service=self.transaction_service(),
                category_service=self.category_service(),
                transaction_creation_service=self.transaction_creation_service(),
                transaction_query_service=self.transaction_query_service(),
                transaction_cleanup_service=self.transaction_cleanup_service()
            )
        return self._nlp_service
    
    # Specialized NLP services
    transaction_creation_service = providers.Singleton(
        TransactionCreationService,
        transaction_service=transaction_service,
        category_service=category_service
    )
    
    transaction_query_service = providers.Singleton(
        TransactionQueryService,
        transaction_service=transaction_service,
        category_service=category_service
    )
    
    transaction_cleanup_service = providers.Singleton(
        TransactionCleanupService,
        transaction_service=transaction_service
    )
    
    budget_service = providers.Singleton(
        BudgetService,
        budgets_api=budgets_api,
        transaction_service=transaction_service
    )

    # Define a method to get the default budget ID
    @classmethod
    def get_default_budget_id(cls) -> Optional[str]:
        """Get the default budget ID from environment variables."""
        return os.environ.get('YNAB_BUDGET_DEV', '7c8d67c8-ed70-4ba8-a25e-931a2f294167')
    
    # Helper methods for easy access to common services
    @classmethod
    def get_ynab_client(cls):
        """Get the YNABClient singleton instance."""
        return cls.ynab_client()
    
    @classmethod
    def get_category_service(cls):
        """Get the CategoryService singleton instance."""
        return cls.category_service()
    
    @classmethod
    def get_transaction_service(cls):
        """Get the TransactionService singleton instance."""
        return cls.transaction_service()
    
    @classmethod
    def get_budget_service(cls):
        """Get the BudgetService singleton instance."""
        return cls.budget_service()
    
    @classmethod
    def get_ai_tagging_service(cls):
        """Get the AITaggingService singleton instance."""
        return cls.ai_tagging_service()
    
    @classmethod
    def get_transaction_tagging_service(cls):
        """Get the transaction tagging service instance"""
        return cls.transaction_tagging_service()
    
    @classmethod
    def get_nlp_service(cls) -> NLPService:
        """
        Get the NLP service.

        Returns:
            NLPService: The NLP service
        """
        container = cls()
        return NLPService(
            transaction_service=container.transaction_service(),
            category_service=container.category_service(),
            transaction_creation_service=container.transaction_creation_service(),
            transaction_query_service=container.transaction_query_service(),
            transaction_cleanup_service=container.transaction_cleanup_service()
        )
    
    @classmethod
    def get_transaction_creation_service(cls):
        """Get the TransactionCreationService singleton instance."""
        return cls.transaction_creation_service()
    
    @classmethod
    def get_transaction_query_service(cls):
        """Get the TransactionQueryService singleton instance."""
        return cls.transaction_query_service()
    
    @classmethod
    def get_transaction_cleanup_service(cls):
        """Get the TransactionCleanupService singleton instance."""
        return cls.transaction_cleanup_service()
    
    @classmethod
    def wire_all_services(cls):
        """
        Initialize all services to ensure they are properly wired up.
        
        This is a convenience method to ensure all singleton services
        are instantiated and their dependencies resolved.
        """
        cls.get_ynab_client()
        cls.get_category_service()
        cls.get_transaction_service()
        cls.get_budget_service()
        cls.get_ai_tagging_service()
        cls.get_transaction_tagging_service()
        cls.get_nlp_service()
        cls.get_transaction_creation_service()
        cls.get_transaction_query_service()
        cls.get_transaction_cleanup_service()
        return True 