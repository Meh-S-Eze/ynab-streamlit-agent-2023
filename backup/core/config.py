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
    _logger = logging.getLogger(__name__)

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls.load_config()
        return cls._instance

    @classmethod
    def load_config(cls, config_path: Optional[str] = None):
        """
        Load configuration from multiple sources with priority:
        1. Explicitly passed config path
        2. .env file
        3. config.yaml in project root
        """
        load_dotenv()  # Load environment variables from .env file

        # Explicitly set YNAB credentials from environment variables
        cls._config['credentials'] = {
            'ynab': {
                'api_key': os.getenv('YNAB_API_KEY'),
                'budget_id': os.getenv('YNAB_BUDGET_DEV')
            }
        }

        # Default config paths
        default_paths = [
            config_path,
            os.path.join(os.getcwd(), 'config.yaml'),
            os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        ]

        for path in default_paths:
            if path and os.path.exists(path):
                try:
                    with open(path, 'r') as config_file:
                        cls._config.update(yaml.safe_load(config_file) or {})
                    cls._logger.info(f"Loaded configuration from {path}")
                    break
                except Exception as e:
                    cls._logger.warning(f"Failed to load config from {path}: {e}")

        # Overlay environment variables
        cls._overlay_env_vars()

        # Minimal debug logging
        cls._logger.debug("Configuration loaded successfully")

    @classmethod
    def _overlay_env_vars(cls):
        """
        Overlay environment variables onto configuration
        Supports nested configuration keys with dot notation
        Selectively processes only known configuration-related variables
        """
        safe_env_prefixes = ['YNAB_', 'GEMINI_', 'LOGGING_']
        
        for key, value in os.environ.items():
            try:
                # Only process environment variables with safe prefixes
                if any(key.startswith(prefix) for prefix in safe_env_prefixes):
                    # Convert environment variable to nested dict
                    keys = key.lower().split('_')
                    current = cls._config
                    for k in keys[:-1]:
                        current = current.setdefault(k, {})
                    current[keys[-1]] = value
            except Exception as e:
                # Silently ignore processing errors for non-critical env vars
                pass

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value using dot notation
        
        Args:
            key (str): Dot-separated configuration key
            default (Any, optional): Default value if key not found
        
        Returns:
            Configuration value or default
        """
        if not cls._config:
            cls.load_config()

        # Split key into nested dictionary traversal
        parts = key.split('.')
        value = cls._config
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part, {})
            else:
                return default

        return value if value != {} else default

    @classmethod
    def get_ynab_token(cls) -> Optional[str]:
        """
        Retrieve YNAB API token with fallback mechanisms
        
        Returns:
            YNAB API token or None
        """
        # Try configuration first
        token = cls.get('credentials.ynab.api_key')
        
        # Fallback to environment variable
        if not token:
            token = os.getenv('YNAB_API_TOKEN')
        
        if not token:
            cls._logger.error("No YNAB API token found")
        
        return token

    @classmethod
    def get_gemini_key(cls) -> Optional[str]:
        """
        Retrieve Gemini API token with fallback mechanisms
        
        Returns:
            Gemini API token or None
        """
        # Try configuration first
        token = cls.get('credentials.gemini.api_key')
        
        # Fallback to environment variable
        if not token:
            token = os.getenv('GEMINI_API_KEY')
        
        if not token:
            cls._logger.error("No Gemini API token found")
        
        return token

    @classmethod
    def get_available_budget_ids(cls) -> list:
        """
        Retrieve the list of available YNAB budget IDs from the environment
        
        Returns:
            list: List of budget IDs or empty list if none found
        """
        budget_ids_str = os.getenv('YNAB_AVAILABLE_BUDGETS', '')
        if not budget_ids_str:
            cls._logger.warning("No available budget IDs found in YNAB_AVAILABLE_BUDGETS environment variable")
            return []
            
        # Split the comma-separated string into a list
        budget_ids = [bid.strip() for bid in budget_ids_str.split(',') if bid.strip()]
        cls._logger.debug(f"Found {len(budget_ids)} available budget IDs")
        return budget_ids

    @classmethod
    def setup_logging(cls):
        """
        Configure logging based on environment
        Minimize verbosity and reduce noise
        """
        log_level = cls.get('logging.level', 'WARNING')
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(levelname)s: %(message)s',  # Simplified log format
            handlers=[
                logging.StreamHandler()  # Only console output
            ]
        )

        # Simplified structlog configuration
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Reduce logging for third-party libraries
        logging.getLogger('google.generativeai').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('httplib2').setLevel(logging.WARNING)

    def get_conversion_rates(self) -> Dict[str, float]:
        return self._config.get('currency_rates', {
            "USD": 1.0,
            "EUR": 0.93,
            "GBP": 0.80,
            # Add other rates
        })

# Initialize logging on import
ConfigManager().setup_logging() 