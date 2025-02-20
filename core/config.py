import os
from pathlib import Path
import yaml
from typing import Any, Dict, Optional
import logging
import structlog
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class ConfigManager:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._load_config()
        return cls._instance

    @classmethod
    def _load_config(cls):
        # Determine current environment
        environment = os.getenv('APP_ENV', 'development')

        # Load configuration from environment variables and .env
        cls._config = {
            'environment': {'current': environment},
            'credentials': {
                'ynab': {
                    'api_key': os.getenv('YNAB_API_KEY'),
                    'budget_id': os.getenv('YNAB_BUDGET_ID')
                },
                'gemini': {
                    'api_key': os.getenv('GEMINI_API_KEY')
                }
            },
            'features': {
                'ai_analysis': os.getenv('FEATURE_AI_ANALYSIS', 'false').lower() == 'true',
                'budget_forecasting': os.getenv('FEATURE_BUDGET_FORECASTING', 'false').lower() == 'true'
            },
            'circuit_breaker': {
                'max_failures': {
                    'ynab': int(os.getenv('CIRCUIT_BREAKER_YNAB_MAX_FAILURES', 3)),
                    'gemini': int(os.getenv('CIRCUIT_BREAKER_GEMINI_MAX_FAILURES', 5))
                }
            },
            'performance': {
                'caching': {
                    'budget_queries': {
                        'max_size': int(os.getenv('CACHE_BUDGET_QUERIES_MAX_SIZE', 32)),
                        'ttl_seconds': int(os.getenv('CACHE_BUDGET_QUERIES_TTL', 3600))
                    }
                }
            },
            'logging': {
                'level': os.getenv('LOG_LEVEL', 'INFO')
            }
        }

    @classmethod
    def get(cls, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieve configuration with environment-specific overrides
        """
        current_env = cls._config['environment']['current']
        
        # Split key for nested lookups
        keys = key.split('.')
        
        # Traverse config
        value = cls._config
        for k in keys:
            value = value.get(k, {})
        
        return value or default

    @classmethod
    def setup_logging(cls):
        """
        Configure logging based on environment
        """
        log_level = cls.get('logging.level', 'INFO')
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Optional: Use structlog for structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

# Initialize logging on import
ConfigManager().setup_logging() 