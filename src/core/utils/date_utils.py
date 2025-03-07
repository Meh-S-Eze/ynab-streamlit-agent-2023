"""
Date utilities for handling date formats and conversions.

This module provides utilities for working with dates in the YNAB API.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date, timedelta
import dateutil.parser
from dateutil.relativedelta import relativedelta

# Setup logger
logger = logging.getLogger(__name__)


class DateFormatter:
    """Utility class for date formatting and parsing"""
    
    @staticmethod
    def format_date(input_date: Union[str, datetime, date]) -> str:
        """
        Format a date for the YNAB API using ISO-8601
        
        Args:
            input_date: Date to format
        
        Returns:
            str: ISO-8601 formatted date string (YYYY-MM-DD)
            
        Raises:
            ValueError: If date cannot be parsed
        """
        try:
            if isinstance(input_date, str):
                # Try to parse the string as a date
                parsed_date = dateutil.parser.parse(input_date).date()
                return parsed_date.strftime('%Y-%m-%d')
            elif isinstance(input_date, datetime):
                return input_date.date().strftime('%Y-%m-%d')
            elif isinstance(input_date, date):
                return input_date.strftime('%Y-%m-%d')
            else:
                raise ValueError(f"Unsupported date type: {type(input_date)}")
        except Exception as e:
            logger.error(f"Date formatting error: {e}")
            raise ValueError(f"Invalid date format: {input_date}")
    
    @staticmethod
    def parse_date(date_str: str) -> date:
        """
        Parse a date string into a date object
        
        Args:
            date_str (str): Date string
            
        Returns:
            date: Parsed date
            
        Raises:
            ValueError: If date cannot be parsed
        """
        try:
            return dateutil.parser.parse(date_str).date()
        except Exception as e:
            logger.error(f"Date parsing error: {e}")
            raise ValueError(f"Invalid date string: {date_str}")
    
    @staticmethod
    def get_month_bounds(input_date: Union[str, datetime, date]) -> tuple:
        """
        Get the first and last day of the month for a given date
        
        Args:
            input_date: Date to get month bounds for
            
        Returns:
            tuple: (first_day, last_day) as date objects
            
        Raises:
            ValueError: If date cannot be parsed
        """
        try:
            if isinstance(input_date, str):
                dt = dateutil.parser.parse(input_date).date()
            elif isinstance(input_date, datetime):
                dt = input_date.date()
            elif isinstance(input_date, date):
                dt = input_date
            else:
                raise ValueError(f"Unsupported date type: {type(input_date)}")
                
            # First day of month
            first_day = date(dt.year, dt.month, 1)
            
            # Last day of month
            if dt.month == 12:
                last_day = date(dt.year + 1, 1, 1) - timedelta(days=1)
            else:
                last_day = date(dt.year, dt.month + 1, 1) - timedelta(days=1)
                
            return (first_day, last_day)
        except Exception as e:
            logger.error(f"Error getting month bounds: {e}")
            raise ValueError(f"Invalid date: {input_date}")
    
    @staticmethod
    def parse_relative_date(relative_date: str) -> date:
        """
        Parse a relative date string like 'last month', 'this year', etc.
        
        Args:
            relative_date (str): Relative date string
            
        Returns:
            date: Parsed date
            
        Raises:
            ValueError: If relative date cannot be parsed
        """
        today = date.today()
        relative_date = relative_date.lower().strip()
        
        # Handle 'today', 'yesterday', 'tomorrow'
        if relative_date == 'today':
            return today
        elif relative_date == 'yesterday':
            return today - timedelta(days=1)
        elif relative_date == 'tomorrow':
            return today + timedelta(days=1)
            
        # Handle 'last X', 'this X', 'next X'
        match = re.match(r'(last|this|next)\s+(day|week|month|year)', relative_date)
        if match:
            period_type = match.group(1)
            period_unit = match.group(2)
            
            if period_type == 'this':
                if period_unit == 'day':
                    return today
                elif period_unit == 'week':
                    # Start of current week (Monday)
                    return today - timedelta(days=today.weekday())
                elif period_unit == 'month':
                    return date(today.year, today.month, 1)
                elif period_unit == 'year':
                    return date(today.year, 1, 1)
            elif period_type == 'last':
                if period_unit == 'day':
                    return today - timedelta(days=1)
                elif period_unit == 'week':
                    # Start of previous week
                    return today - timedelta(days=today.weekday() + 7)
                elif period_unit == 'month':
                    # First day of previous month
                    if today.month == 1:
                        return date(today.year - 1, 12, 1)
                    else:
                        return date(today.year, today.month - 1, 1)
                elif period_unit == 'year':
                    return date(today.year - 1, 1, 1)
            elif period_type == 'next':
                if period_unit == 'day':
                    return today + timedelta(days=1)
                elif period_unit == 'week':
                    # Start of next week
                    return today + timedelta(days=7 - today.weekday())
                elif period_unit == 'month':
                    # First day of next month
                    if today.month == 12:
                        return date(today.year + 1, 1, 1)
                    else:
                        return date(today.year, today.month + 1, 1)
                elif period_unit == 'year':
                    return date(today.year + 1, 1, 1)
                    
        # Handle 'X days/weeks/months/years ago/from now'
        match = re.match(r'(\d+)\s+(day|week|month|year)s?\s+(ago|from now)', relative_date)
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            direction = match.group(3)
            
            if direction == 'ago':
                amount = -amount
                
            if unit == 'day':
                return today + timedelta(days=amount)
            elif unit == 'week':
                return today + timedelta(weeks=amount)
            elif unit == 'month':
                return today + relativedelta(months=amount)
            elif unit == 'year':
                return today + relativedelta(years=amount)
                
        # If we can't parse it as a relative date, try to parse it as an absolute date
        try:
            return dateutil.parser.parse(relative_date).date()
        except:
            raise ValueError(f"Could not parse relative date: {relative_date}")
            
    @staticmethod
    def get_date_range(start_date: Union[str, date, datetime], end_date: Union[str, date, datetime]) -> List[date]:
        """
        Get a list of dates in a range
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List[date]: List of dates in the range (inclusive)
            
        Raises:
            ValueError: If dates cannot be parsed
        """
        try:
            # Parse dates if they're strings
            if isinstance(start_date, str):
                start = DateFormatter.parse_date(start_date)
            elif isinstance(start_date, datetime):
                start = start_date.date()
            else:
                start = start_date
                
            if isinstance(end_date, str):
                end = DateFormatter.parse_date(end_date)
            elif isinstance(end_date, datetime):
                end = end_date.date()
            else:
                end = end_date
                
            # Generate date range
            date_list = []
            current = start
            while current <= end:
                date_list.append(current)
                current += timedelta(days=1)
                
            return date_list
        except Exception as e:
            logger.error(f"Error generating date range: {e}")
            raise ValueError(f"Invalid date range: {start_date} to {end_date}")
