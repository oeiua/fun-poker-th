"""
Logging functionality for the poker AI system.
"""
import os
import logging
import time
import sys
from typing import Optional, Dict, Any, List
import datetime

class Logger:
    """Custom logger for the poker AI system."""
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_format: Optional[str] = None
    ):
        """
        Initialize the logger.
        
        Args:
            log_file: Path to log file (if None, logs only to console)
            console_level: Logging level for console output
            file_level: Logging level for file output
            log_format: Format string for log messages
        """
        self.logger = logging.getLogger('poker_ai')
        self.logger.setLevel(logging.DEBUG)  # Capture all levels
        
        # Remove existing handlers if any
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Default format
        if log_format is None:
            log_format = '%(asctime)s [%(levelname)s] %(message)s'
        
        formatter = logging.Formatter(log_format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.log_file = log_file
        self.start_time = time.time()
        
        # Log initial message
        self.info(f"Logger initialized at {datetime.datetime.now()}")
    
    def debug(self, message: str) -> None:
        """
        Log a debug message.
        
        Args:
            message: Message to log
        """
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """
        Log an info message.
        
        Args:
            message: Message to log
        """
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """
        Log a warning message.
        
        Args:
            message: Message to log
        """
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """
        Log an error message.
        
        Args:
            message: Message to log
        """
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """
        Log a critical message.
        
        Args:
            message: Message to log
        """
        self.logger.critical(message)
    
    def log_dict(self, data: Dict[str, Any], prefix: str = "") -> None:
        """
        Log a dictionary with proper formatting.
        
        Args:
            data: Dictionary to log
            prefix: Prefix for log lines
        """
        self.info(f"{prefix}:")
        for key, value in data.items():
            if isinstance(value, dict):
                self.log_dict(value, prefix=f"{prefix}{key}.")
            else:
                self.info(f"{prefix}{key}: {value}")
    
    def log_section(self, title: str, char: str = "=") -> None:
        """
        Log a section header.
        
        Args:
            title: Section title
            char: Character to use for the separator line
        """
        separator = char * 50
        self.info(separator)
        self.info(title)
        self.info(separator)
    
    def log_elapsed_time(self, prefix: str = "Time elapsed") -> None:
        """
        Log elapsed time since logger initialization.
        
        Args:
            prefix: Prefix for the log message
        """
        elapsed = time.time() - self.start_time
        
        # Format elapsed time
        if elapsed < 60:
            time_str = f"{elapsed:.2f} seconds"
        elif elapsed < 3600:
            minutes = int(elapsed / 60)
            seconds = elapsed % 60
            time_str = f"{minutes} minutes {seconds:.2f} seconds"
        else:
            hours = int(elapsed / 3600)
            minutes = int((elapsed % 3600) / 60)
            seconds = elapsed % 60
            time_str = f"{hours} hours {minutes} minutes {seconds:.2f} seconds"
        
        self.info(f"{prefix}: {time_str}")
    
    def log_memory_usage(self, memory_info: Dict[str, Any]) -> None:
        """
        Log memory usage information.
        
        Args:
            memory_info: Dictionary with memory usage statistics
        """
        self.info("Memory Usage:")
        self.info(f"  RSS: {memory_info['rss'] / (1024 * 1024):.2f} MB")
        self.info(f"  VMS: {memory_info['vms'] / (1024 * 1024):.2f} MB")
        self.info(f"  System: {memory_info['percent']:.1f}% of {memory_info['total'] / (1024 * 1024 * 1024):.1f} GB")
        
        # Log GPU info if available
        if memory_info['gpu']:
            self.info("  GPU Memory:")
            for gpu_idx, gpu_data in memory_info['gpu'].items():
                self.info(f"    GPU {gpu_idx}:")
                self.info(f"      Allocated: {gpu_data['allocated']:.2f} MB")
                self.info(f"      Reserved: {gpu_data['reserved']:.2f} MB")
                self.info(f"      Max Allocated: {gpu_data['max_allocated']:.2f} MB")