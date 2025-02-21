from datetime import datetime, date
from typing import Union
import logging

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

class DateFormatter:
    """
    Utility class for handling date formatting and validation for YNAB API
    Implements ISO-8601 date formatting rule
    """
    ISO_FORMAT = "%Y-%m-%d"
    ACCEPTED_FORMATS = [
        "%Y-%m-%d",    # ISO format: 2024-03-14
        "%m/%d/%Y",    # US format: 03/14/2024
        "%d/%m/%Y",    # UK format: 14/03/2024
        "%Y/%m/%d",    # Alternative ISO: 2024/03/14
        "%m-%d-%Y",    # US with dashes: 03-14-2024
        "%d-%m-%Y",    # UK with dashes: 14-03-2024
    ]
    
    # Use module-level logger instead of class-level
    logger = logger
    
    @staticmethod
    def parse_date(date_str: str) -> date:
        """
        Parse a date string in various formats and return a date object
        
        Args:
            date_str (str): Date string to parse
        
        Returns:
            date: Parsed date object
        
        Raises:
            ValueError: If date string cannot be parsed
        """
        if not isinstance(date_str, str):
            raise ValueError(f"Expected string but got {type(date_str)}")
            
        if not date_str:
            raise ValueError("Date string cannot be empty")
        
        # Try each format until one works
        errors = []
        for fmt in DateFormatter.ACCEPTED_FORMATS:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError as e:
                errors.append(f"Format {fmt}: {str(e)}")
                continue
        
        # Log the specific parsing errors for debugging
        DateFormatter.logger.debug(
            f"Failed to parse date '{date_str}' with all formats:\n" +
            "\n".join(errors)
        )
        
        raise ValueError(
            f"Date string '{date_str}' does not match any accepted format. "
            f"Please use YYYY-MM-DD format."
        )
    
    @staticmethod
    def format_date(input_date: Union[str, date, datetime]) -> str:
        """
        Format a date into ISO-8601 format (YYYY-MM-DD)
        
        Args:
            input_date: Date to format (string, date, or datetime)
        
        Returns:
            str: ISO-8601 formatted date string
        
        Raises:
            ValueError: If date cannot be formatted
        """
        try:
            if isinstance(input_date, str):
                parsed_date = DateFormatter.parse_date(input_date)
            elif isinstance(input_date, datetime):
                parsed_date = input_date.date()
            elif isinstance(input_date, date):
                parsed_date = input_date
            else:
                raise ValueError(f"Unsupported date type: {type(input_date)}")
            
            return parsed_date.strftime(DateFormatter.ISO_FORMAT)
            
        except Exception as e:
            DateFormatter.logger.error(f"Failed to format date: {e}")
            raise ValueError(f"Failed to format date: {str(e)}")
    
    @staticmethod
    def validate_date(date_str: str) -> bool:
        """
        Validate that a date string is in ISO-8601 format
        
        Args:
            date_str (str): Date string to validate
        
        Returns:
            bool: True if date is valid ISO-8601 format
        
        Raises:
            TypeError: If input is not a string
        """
        if not isinstance(date_str, str):
            raise TypeError(f"Expected string but got {type(date_str)}")
            
        try:
            datetime.strptime(date_str, DateFormatter.ISO_FORMAT)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def is_future_date(input_date: Union[str, date, datetime]) -> bool:
        """
        Check if a date is in the future
        
        Args:
            input_date: Date to check
        
        Returns:
            bool: True if date is in the future
        """
        try:
            if isinstance(input_date, str):
                parsed_date = DateFormatter.parse_date(input_date)
            elif isinstance(input_date, datetime):
                parsed_date = input_date.date()
            elif isinstance(input_date, date):
                parsed_date = input_date
            else:
                raise ValueError(f"Unsupported date type: {type(input_date)}")
            
            return parsed_date > date.today()
            
        except Exception as e:
            DateFormatter.logger.error(f"Failed to check future date: {e}")
            raise ValueError(f"Failed to check future date: {str(e)}") 