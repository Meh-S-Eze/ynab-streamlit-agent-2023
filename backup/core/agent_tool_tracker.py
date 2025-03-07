import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List
from datetime import datetime
import json

@dataclass
class AgentToolUsage:
    """
    Dataclass to track detailed tool usage by agents
    """
    agent_name: str
    tool_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    input_params: Dict[str, Any] = field(default_factory=dict)
    output: Any = None
    duration: float = 0.0
    success: bool = True
    error_message: str = ''

    def to_dict(self):
        """Convert usage entry to a dictionary for serialization"""
        return {
            'agent_name': self.agent_name,
            'tool_name': self.tool_name,
            'timestamp': self.timestamp.isoformat(),
            'input_params': self.input_params,
            'success': self.success,
            'error_message': self.error_message
        }

class AgentToolTracker:
    """
    Centralized tool usage tracking with dependency injection support
    """
    _instance = None

    def __new__(cls):
        """Singleton implementation"""
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.usage_log = []
            cls._instance.logger = logging.getLogger('agent_tool_tracker')
        return cls._instance

    def log_tool_usage(self, agent_name: str, tool_name: str, 
                       input_params: Dict[str, Any], 
                       output: Any = None, 
                       success: bool = True, 
                       error_message: str = '',
                       duration: float = 0.0):
        """
        Log detailed information about tool usage by an agent
        
        Args:
            agent_name (str): Name of the agent using the tool
            tool_name (str): Name of the tool being used
            input_params (Dict): Parameters passed to the tool
            output (Any, optional): Result of the tool usage
            success (bool, optional): Whether the tool usage was successful
            error_message (str, optional): Error message if tool usage failed
            duration (float, optional): Duration of tool usage
        """
        usage_entry = AgentToolUsage(
            agent_name=agent_name,
            tool_name=tool_name,
            input_params=input_params,
            output=output,
            success=success,
            error_message=error_message
        )
        if duration > 0:
            usage_entry.duration = duration
        
        self.usage_log.append(usage_entry)
        
        # Logging for immediate visibility
        if success:
            self.logger.info(
                f"Agent '{agent_name}' used tool '{tool_name}' "
                f"with params: {json.dumps(input_params, default=str)} "
                f"(Duration: {duration:.4f}s)"
            )
        else:
            self.logger.error(
                f"Agent '{agent_name}' failed using tool '{tool_name}': "
                f"{error_message}"
            )

    def get_agent_tool_summary(self, agent_name: str = None):
        """
        Generate a summary of tool usage
        
        Args:
            agent_name (str, optional): Filter summary for a specific agent
        
        Returns:
            Dict: Summary of tool usage
        """
        filtered_log = [
            entry for entry in self.usage_log 
            if agent_name is None or entry.agent_name == agent_name
        ]

        summary = {
            'total_tool_calls': len(filtered_log),
            'successful_calls': sum(1 for entry in filtered_log if entry.success),
            'failed_calls': sum(1 for entry in filtered_log if not entry.success),
            'tool_breakdown': {}
        }

        for entry in filtered_log:
            if entry.tool_name not in summary['tool_breakdown']:
                summary['tool_breakdown'][entry.tool_name] = {
                    'total_calls': 0,
                    'successful_calls': 0,
                    'failed_calls': 0
                }
            
            summary['tool_breakdown'][entry.tool_name]['total_calls'] += 1
            if entry.success:
                summary['tool_breakdown'][entry.tool_name]['successful_calls'] += 1
            else:
                summary['tool_breakdown'][entry.tool_name]['failed_calls'] += 1

        return summary

    def export_usage_log(self, filename: str = 'agent_tool_usage.json'):
        """
        Export usage log to a JSON file
        
        Args:
            filename (str): Name of the file to export logs
        """
        with open(filename, 'w') as f:
            json.dump([entry.to_dict() for entry in self.usage_log], f, indent=2)

# Global tracker instance for easy access
tool_tracker = AgentToolTracker() 