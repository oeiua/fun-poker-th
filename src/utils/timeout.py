"""
Timeout handler for AI decision making.
"""

import logging
import signal
from typing import Any, Callable, TypeVar

T = TypeVar('T')

class TimeoutException(Exception):
    """Exception raised when a function times out."""
    pass


class TimeoutHandler:
    """
    Handler for implementing timeouts in function calls.
    
    This is especially useful for AI decision making to prevent
    excessive thinking time.
    """
    
    def __init__(self, timeout_seconds: float):
        """
        Initialize the timeout handler.
        
        Args:
            timeout_seconds: Timeout duration in seconds
        """
        self.timeout_seconds = timeout_seconds
    
    def _timeout_handler(self, signum: int, frame: Any) -> None:
        """
        Signal handler for timeouts.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        
        Raises:
            TimeoutException: Always raises this exception
        """
        raise TimeoutException("Function call timed out")
    
    def with_timeout(self, func: Callable[[], T], default_value: T = None) -> T:
        """
        Execute a function with a timeout.
        
        Args:
            func: Function to execute
            default_value: Value to return if the function times out
            
        Returns:
            Result of the function call, or default_value if it times out
        """
        # Set the timeout handler
        old_handler = signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(int(self.timeout_seconds))
        
        try:
            result = func()
            return result
        except TimeoutException:
            logging.warning(f"Function timed out after {self.timeout_seconds} seconds, using default value")
            return default_value
        finally:
            # Restore the old signal handler and cancel the alarm
            signal.signal(signal.SIGALRM, old_handler)
            signal.alarm(0)