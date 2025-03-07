import logging
import json
import requests
from typing import Optional
import os

class SlackHandler(logging.Handler):
    """
    Custom logging handler that sends log messages to Slack
    """
    def __init__(self, channel: str = "#alerts", webhook_url: Optional[str] = None):
        """
        Initialize Slack handler
        
        Args:
            channel (str): Slack channel to post to
            webhook_url (Optional[str]): Slack webhook URL. If not provided, uses SLACK_WEBHOOK_URL env var
        """
        super().__init__()
        self.channel = channel
        self.webhook_url = webhook_url or os.getenv('SLACK_WEBHOOK_URL')
        
        if not self.webhook_url:
            raise ValueError("Slack webhook URL not provided and SLACK_WEBHOOK_URL not set in environment")
    
    def emit(self, record: logging.LogRecord):
        """
        Send log message to Slack
        
        Args:
            record (logging.LogRecord): Log record to send
        """
        try:
            # Format the record
            msg = self.format(record)
            
            # Convert to dict if JSON string
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                data = {'message': msg}
            
            # Create Slack message
            slack_message = {
                'channel': self.channel,
                'username': 'YNAB Agent Logger',
                'icon_emoji': ':warning:' if record.levelno >= logging.WARNING else ':information_source:',
                'attachments': [{
                    'fallback': str(data.get('message', msg)),
                    'color': self._get_level_color(record.levelno),
                    'title': f"[{record.levelname}] {record.name}",
                    'text': json.dumps(data, indent=2),
                    'fields': [
                        {
                            'title': 'Module',
                            'value': record.module,
                            'short': True
                        },
                        {
                            'title': 'Function',
                            'value': record.funcName,
                            'short': True
                        }
                    ] if hasattr(record, 'module') and hasattr(record, 'funcName') else [],
                    'footer': f"Log ID: {getattr(record, 'correlation_id', 'N/A')}"
                }]
            }
            
            # Send to Slack
            response = requests.post(
                self.webhook_url,
                json=slack_message,
                timeout=5  # 5 second timeout
            )
            response.raise_for_status()
            
        except Exception as e:
            # Don't raise exceptions in logging handlers
            print(f"Failed to send log to Slack: {e}")
    
    def _get_level_color(self, level: int) -> str:
        """Get Slack message color based on log level"""
        if level >= logging.ERROR:
            return 'danger'  # Red
        elif level >= logging.WARNING:
            return 'warning'  # Yellow
        elif level >= logging.INFO:
            return 'good'  # Green
        return '#439FE0'  # Blue for debug 