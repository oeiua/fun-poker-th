"""
Timeout handler for AI decision making.
"""

import logging
import signal
import threading
import time
from typing import Any, Callable, TypeVar, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError

T = TypeVar('T')

class TimeoutException(Exception):
    """Exception raised when a function times out."""
    pass


class TimeoutHandler:
    """
    Handler for implementing timeouts in function calls.
    
    This is especially useful for AI decision making to prevent
    excessive thinking time.
    
    Note: Uses ThreadPoolExecutor for cross-platform compatibility in Python 3.12+
    """
    
    def __init__(self, timeout_seconds: float):
        """
        Initialize the timeout handler.
        
        Args:
            timeout_seconds: Timeout duration in seconds
        """
        self.timeout_seconds = timeout_seconds
    
    def with_timeout(self, func: Callable[[], T], default_value: Optional[T] = None) -> T:
        """
        Execute a function with a timeout using ThreadPoolExecutor.
        
        Args:
            func: Function to execute
            default_value: Value to return if the function times out
            
        Returns:
            Result of the function call, or default_value if it times out
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                # Wait for the result with a timeout
                result = future.result(timeout=self.timeout_seconds)
                return result
            except TimeoutError:
                logging.warning(f"Function timed out after {self.timeout_seconds} seconds, using default value")
                return default_value