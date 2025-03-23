#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Settings

This module handles configuration settings for the poker vision assistant.
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("PokerVision.Config.Settings")

# Default settings
DEFAULT_SETTINGS = {
    # Application settings
    "app": {
        "auto_analyze": False,
        "auto_analyze_interval": 3,  # seconds
        "show_debug_overlay": True,
        "log_level": "INFO",
        "max_history_size": 100,
        "save_screenshots": True,
        "screenshots_folder": "screenshots",
        "dark_mode": False,
        "target_window": None,  # Selected window for capture
        "last_window_index": -1,  # Last selected window index
        "debug_mode": True  # Save debug images for troubleshooting
    },
    
    # Vision settings
    "vision": {
        "card_detector": {
            "card_width": 60,
            "card_height": 80,
            "min_card_area": 2000,
            "confidence_threshold": 0.6
        },
        "ocr": {
            "confidence_threshold": 0.4,
            "enhanced_preprocessing": True
        }
    },
    
    # Poker strategy settings
    "strategy": {
        "aggression": 1.0,  # 0.0 to 2.0, where 1.0 is balanced
        "bluff_frequency": 0.3,  # 0.0 to 1.0
        "value_bet_threshold": 0.6,  # 0.0 to 1.0
        "cbet_frequency": 0.7,  # 0.0 to 1.0
        "simulation_iterations": 1000  # Number of Monte Carlo iterations
    },
    
    # UI settings
    "ui": {
        "main_window_size": [1280, 800],
        "splitter_sizes": [300, 980],
        "font_size": 10,
        "cards_display_suit_symbols": True,
        "statistics_view_default_tab": 0
    }
}

class Settings:
    """Class to handle application settings."""
    
    def __init__(self, settings_path: Optional[str] = None):
        """
        Initialize Settings.
        
        Args:
            settings_path: Path to the settings file, or None for default
        """
        # Set default path if not provided
        if settings_path is None:
            self.settings_path = os.path.join(
                str(Path.home()), 
                ".poker_vision", 
                "settings.json"
            )
        else:
            self.settings_path = settings_path
        
        # Initialize settings with defaults
        self.settings = DEFAULT_SETTINGS.copy()
        
        # Load settings from file
        self.load()
        
        logger.info("Settings initialized")
    
    def load(self) -> bool:
        """
        Load settings from file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(self.settings_path):
                with open(self.settings_path, 'r') as f:
                    loaded_settings = json.load(f)
                
                # Update settings with loaded values
                self._update_nested_dict(self.settings, loaded_settings)
                
                logger.info(f"Settings loaded from {self.settings_path}")
                return True
            else:
                logger.info(f"Settings file not found: {self.settings_path}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            return False
    
    def save(self) -> bool:
        """
        Save settings to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
            
            # Save settings to file
            with open(self.settings_path, 'w') as f:
                json.dump(self.settings, f, indent=4)
            
            logger.info(f"Settings saved to {self.settings_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return False
    
    def reset(self) -> None:
        """Reset settings to defaults."""
        self.settings = DEFAULT_SETTINGS.copy()
        logger.info("Settings reset to defaults")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value.
        
        Args:
            key: Setting key (can be nested with dots, e.g., "app.log_level")
            default: Default value if key not found
            
        Returns:
            Setting value or default
        """
        keys = key.split('.')
        
        # Navigate through nested dictionaries
        current = self.settings
        for k in keys:
            if k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a setting value.
        
        Args:
            key: Setting key (can be nested with dots, e.g., "app.log_level")
            value: Setting value
        """
        keys = key.split('.')
        
        # Navigate through nested dictionaries
        current = self.settings
        for i, k in enumerate(keys[:-1]):
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value
        current[keys[-1]] = value
        
        logger.debug(f"Setting {key} = {value}")
    
    def _update_nested_dict(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Update a nested dictionary recursively.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with updates
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively update nested dictionaries
                self._update_nested_dict(target[key], value)
            else:
                # Update or add the key
                target[key] = value

# Global settings instance
_settings_instance = None

def get_settings(settings_path: Optional[str] = None) -> Settings:
    """
    Get the global settings instance.
    
    Args:
        settings_path: Path to the settings file, or None for default
        
    Returns:
        Settings instance
    """
    global _settings_instance
    
    if _settings_instance is None:
        _settings_instance = Settings(settings_path)
    
    return _settings_instance

def load_settings(settings_path: Optional[str] = None) -> Settings:
    """
    Load settings from a file.
    
    Args:
        settings_path: Path to the settings file, or None for default
        
    Returns:
        Settings instance
    """
    settings = get_settings(settings_path)
    settings.load()
    return settings

def save_settings(settings: Optional[Settings] = None) -> bool:
    """
    Save settings to a file.
    
    Args:
        settings: Settings instance, or None for global instance
        
    Returns:
        True if successful, False otherwise
    """
    if settings is None:
        settings = get_settings()
    
    return settings.save()

def reset_settings() -> None:
    """Reset settings to defaults."""
    settings = get_settings()
    settings.reset()


# Test function
def test_settings():
    """Test the settings functionality."""
    import tempfile
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create a temporary settings file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
        temp_path = temp.name
    
    try:
        # Create settings instance with the temporary file
        settings = Settings(temp_path)
        
        # Test default values
        assert settings.get("app.log_level") == "INFO"
        assert settings.get("strategy.aggression") == 1.0
        assert settings.get("non_existent_key", "default") == "default"
        
        # Test setting values
        settings.set("app.log_level", "DEBUG")
        settings.set("strategy.aggression", 1.5)
        settings.set("new_key", "new_value")
        
        assert settings.get("app.log_level") == "DEBUG"
        assert settings.get("strategy.aggression") == 1.5
        assert settings.get("new_key") == "new_value"
        
        # Test saving settings
        assert settings.save()
        
        # Create a new settings instance to load the saved settings
        settings2 = Settings(temp_path)
        
        # Verify loaded settings
        assert settings2.get("app.log_level") == "DEBUG"
        assert settings2.get("strategy.aggression") == 1.5
        assert settings2.get("new_key") == "new_value"
        
        # Test resetting settings
        settings2.reset()
        assert settings2.get("app.log_level") == "INFO"
        assert settings2.get("strategy.aggression") == 1.0
        assert settings2.get("new_key", None) is None
        
        logger.info("Settings tests passed")
        
    finally:
        # Clean up
        try:
            os.unlink(temp_path)
        except:
            pass


if __name__ == "__main__":
    # Run test
    test_settings()