#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poker Vision Assistant - Main Application

This is the main entry point for the Poker Vision Assistant application.
It initializes all components and starts the UI.
"""

import sys
import logging
from pathlib import Path

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("poker_vision.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PokerVision")

# Import required modules
try:
    from ui.main_window import MainWindow
    from vision.screen_capture import ScreenCaptureManager
    from vision.game_state_extractor import GameStateExtractor
    from poker.hand_evaluator import HandEvaluator
    from vision.card_detector import CardDetector
    from poker.strategy import StrategyEngine
    from config.settings import load_settings, save_settings
except ImportError as e:
    logger.critical(f"Failed to import required modules: {e}")
    sys.exit(1)

# Check for required libraries
try:
    import cv2
    import numpy as np
    import easyocr
    from PyQt5 import QtWidgets
except ImportError as e:
    logger.critical(f"Required library not found: {e}")
    print(f"Error: Missing required library - {e}")
    print("Please install all required libraries with: pip install -r requirements.txt")
    sys.exit(1)

class PokerVisionAssistant:
    """Main application class that coordinates all components."""
    
    def __init__(self):
        """Initialize the Poker Vision Assistant."""
        logger.info("Initializing Poker Vision Assistant")
        
        # Load application settings
        self.settings = load_settings()
        
        # Initialize components
        self.screen_capture = ScreenCaptureManager()
        
        # Enable debug mode for the card detector
        debug_mode = self.settings.get("app.debug_mode", False)
        if debug_mode:
            logger.info("Debug mode enabled - debug images will be saved")
            
        # Initialize game state extractor with debug mode
        self.game_state_extractor = GameStateExtractor()
        
        # Pass debug mode to the card detector
        self.game_state_extractor.card_detector = CardDetector(debug_mode=debug_mode)
        
        self.hand_evaluator = HandEvaluator()
        self.strategy_engine = StrategyEngine()
        
        # Setup data structures for game state
        self.current_game_state = None
        self.game_history = []
        
    def start(self):
        """Start the application and show the main UI."""
        logger.info("Starting Poker Vision Assistant")
        
        # Create and start the UI
        app = QtWidgets.QApplication(sys.argv)
        self.main_window = MainWindow(self)
        self.main_window.show()
        
        # Connect signals and slots
        self.setup_connections()
        
        # Start the application event loop
        return app.exec_()
    
    def setup_connections(self):
        """Setup signal/slot connections between components."""
        # Connect UI actions to methods
        self.main_window.capture_button.clicked.connect(self.capture_and_analyze)
        self.main_window.settings_button.clicked.connect(self.show_settings)
        
    def capture_and_analyze(self):
        """Capture the current screen and analyze the poker game state."""
        logger.info("Capturing and analyzing screen")
        
        try:
            # Capture screen based on selected window
            if self.settings.get("app.target_window") is not None:
                # Capture specific window
                screenshot = self.screen_capture.capture_window(self.settings.get("app.target_window"))
            else:
                # Capture full screen as fallback
                logger.info("No target window selected, capturing full screen")
                screenshot = self.screen_capture.capture_screen()
            
            # Extract game state from the screenshot
            game_state = self.game_state_extractor.extract_game_state(screenshot)
            
            # Evaluate the current hand using the player_cards property
            hand_strength = self.hand_evaluator.evaluate_hand(
                game_state.player_cards, 
                game_state.community_cards
            )
            
            # Get decision recommendation
            decision = self.strategy_engine.get_decision(game_state, hand_strength)
            
            # Update UI with results
            self.main_window.update_analysis(screenshot, game_state, decision)
            
            # Log the game state
            self.game_history.append(game_state)
            self.current_game_state = game_state
            
            logger.info(f"Analysis complete. Recommendation: {decision}")
            
        except Exception as e:
            logger.error(f"Error during capture and analysis: {e}", exc_info=True)
            self.main_window.show_error(f"Analysis failed: {str(e)}")
    
    def show_settings(self):
        """Show the settings dialog."""
        self.main_window.show_settings_dialog()
    
    def save_current_settings(self):
        """Save current settings to disk."""
        save_settings(self.settings)
        logger.info("Settings saved")

def main():
    """Application entry point."""
    # Create and start the application
    app = PokerVisionAssistant()
    return app.start()

if __name__ == "__main__":
    sys.exit(main())