#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Game Analyzer View

This module implements the game analyzer view component for displaying
poker game analysis with overlays and visualizations.
"""

import logging
import os
import sys
import time
from typing import List, Dict, Tuple, Optional, Union, Any

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
                           QGroupBox, QScrollArea, QSizePolicy, QFrame)

from vision.game_state_extractor import GameState
from poker.constants import ActionType
from poker.strategy import Decision

logger = logging.getLogger("PokerVision.UI.GameAnalyzerView")

class GameAnalyzerView(QWidget):
    """Widget for displaying game analysis."""
    
    def __init__(self, parent=None):
        """
        Initialize the game analyzer view.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.screenshot = None
        self.game_state = None
        self.decision = None
        self.show_debug_overlay = True
        
        # Set up the UI
        self.setup_ui()
        
        logger.info("Game analyzer view initialized")
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main layout
        self.main_layout = QVBoxLayout(self)
        
        # Scroll area for screenshot
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        # Screenshot label
        self.screenshot_label = QLabel("No screenshot available")
        self.screenshot_label.setAlignment(Qt.AlignCenter)
        self.screenshot_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.screenshot_label.setMinimumSize(640, 480)
        self.screenshot_label.setFrameShape(QFrame.Box)
        
        # Add label to scroll area
        self.scroll_area.setWidget(self.screenshot_label)
        
        # Add scroll area to main layout
        self.main_layout.addWidget(self.scroll_area)
        
        # Game state panel
        self.game_state_group = QGroupBox("Game State")
        self.game_state_layout = QVBoxLayout()
        self.game_state_group.setLayout(self.game_state_layout)
        
        # Game info layout
        self.game_info_layout = QHBoxLayout()
        
        # Pot info
        self.pot_label = QLabel("Pot: $0.00")
        self.pot_label.setStyleSheet("font-weight: bold;")
        self.game_info_layout.addWidget(self.pot_label)
        
        # Blinds info
        self.blinds_label = QLabel("Blinds: $0.00/$0.00")
        self.blinds_label.setStyleSheet("font-weight: bold;")
        self.game_info_layout.addWidget(self.blinds_label)
        
        # Hand number info
        self.hand_label = QLabel("Hand #0")
        self.hand_label.setStyleSheet("font-weight: bold;")
        self.game_info_layout.addWidget(self.hand_label)
        
        # Add game info layout to game state layout
        self.game_state_layout.addLayout(self.game_info_layout)
        
        # Community cards
        self.community_cards_label = QLabel("Community Cards: ")
        self.game_state_layout.addWidget(self.community_cards_label)
        
        # Player cards
        self.player_cards_label = QLabel("Your Cards: ")
        self.game_state_layout.addWidget(self.player_cards_label)
        
        # Add game state group to main layout with fixed height
        self.game_state_group.setMaximumHeight(150)
        self.main_layout.addWidget(self.game_state_group)
    
    def update_analysis(self, screenshot: np.ndarray, game_state: GameState, decision: Decision):
        """
        Update the analysis display.
        
        Args:
            screenshot: Screenshot image
            game_state: Current game state
            decision: Recommended decision
        """
        logger.debug("Updating game analyzer view")
        
        # Save the data
        self.screenshot = screenshot
        self.game_state = game_state
        self.decision = decision
        
        # Update the display
        self.update_screenshot_display()
        self.update_game_state_display()
    
    def update_screenshot_display(self):
        """Update the screenshot display."""
        if self.screenshot is not None:
            # Create a copy of the screenshot
            display_img = self.screenshot.copy()
            
            # Add overlays if enabled
            if self.show_debug_overlay and self.game_state is not None:
                display_img = self.add_debug_overlay(display_img)
            
            # Convert to QImage and then QPixmap
            height, width, channel = display_img.shape
            bytes_per_line = 3 * width
            q_img = QtGui.QImage(display_img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            q_img = q_img.rgbSwapped()  # Convert BGR to RGB
            
            pixmap = QtGui.QPixmap.fromImage(q_img)
            
            # Scale pixmap if it's too large
            max_width = 1200
            max_height = 800
            
            if pixmap.width() > max_width or pixmap.height() > max_height:
                pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Set the pixmap
            self.screenshot_label.setPixmap(pixmap)
            self.screenshot_label.setMinimumSize(pixmap.width(), pixmap.height())
            
        else:
            # No screenshot available
            self.screenshot_label.setText("No screenshot available")
            self.screenshot_label.setPixmap(QtGui.QPixmap())
    
    def update_game_state_display(self):
        """Update the game state display."""
        if self.game_state is not None:
            # Update pot info
            self.pot_label.setText(f"Pot: ${self.game_state.pot:.2f} (Total: ${self.game_state.total_pot:.2f})")
            
            # Update blinds info
            self.blinds_label.setText(f"Blinds: ${self.game_state.small_blind:.2f}/${self.game_state.big_blind:.2f}")
            
            # Update hand number info
            self.hand_label.setText(f"Hand #{self.game_state.hand_number} (Games: {self.game_state.games_played})")
            
            # Update community cards
            community_cards_str = ", ".join(str(card) for card in self.game_state.community_cards)
            if not community_cards_str:
                community_cards_str = "None"
            self.community_cards_label.setText(f"Community Cards: {community_cards_str}")
            
            # Update player cards
            player = self.game_state.get_player(0)
            if player and player.cards:
                player_cards_str = ", ".join(str(card) for card in player.cards)
            else:
                player_cards_str = "Unknown"
            
            self.player_cards_label.setText(f"Your Cards: {player_cards_str}")
            
        else:
            # No game state available
            self.pot_label.setText("Pot: $0.00")
            self.blinds_label.setText("Blinds: $0.00/$0.00")
            self.hand_label.setText("Hand #0")
            self.community_cards_label.setText("Community Cards: None")
            self.player_cards_label.setText("Your Cards: Unknown")
    
    def add_debug_overlay(self, image: np.ndarray) -> np.ndarray:
        """
        Add debug overlay to the image.
        
        Args:
            image: Input image
            
        Returns:
            Image with debug overlay
        """
        result = image.copy()
        
        # Draw game state info
        cv2.putText(result, f"Hand #{self.game_state.hand_number}, Games: {self.game_state.games_played}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw pot and blinds
        cv2.putText(result, f"Pot: ${self.game_state.pot:.2f}, Total: ${self.game_state.total_pot:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(result, f"Blinds: ${self.game_state.small_blind:.2f}/${self.game_state.big_blind:.2f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw community cards
        community_cards_str = ", ".join(str(card) for card in self.game_state.community_cards)
        cv2.putText(result, f"Community Cards: {community_cards_str}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw player info and ROIs
        for player in self.game_state.players:
            # Get player position
            x, y, w, h = player.position
            
            # Draw rectangle around player
            color = (0, 255, 0) if player.is_active else (0, 0, 255)
            if player.player_id == self.game_state.current_player_id:
                color = (255, 255, 0)  # Yellow for current player
            if player.player_id == self.game_state.dealer_position:
                color = (255, 0, 255)  # Magenta for dealer
                
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Draw player info
            info_text = f"{player.name}: ${player.stack:.0f}"
            cv2.putText(result, info_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw player bet if any
            if player.bet > 0:
                bet_text = f"Bet: ${player.bet:.0f}"
                cv2.putText(result, bet_text, (x, y + h + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw player action if any
            if player.last_action != None:
                action_text = str(player.last_action)
                cv2.putText(result, action_text, (x, y + h + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw player cards if available
            if player.cards:
                cards_text = ", ".join(str(card) for card in player.cards)
                cv2.putText(result, cards_text, (x, y + h + 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw decision if available
        if self.decision:
            decision_text = f"Decision: {self.decision.action.name}"
            if self.decision.action in [ActionType.BET, ActionType.RAISE, ActionType.CALL, ActionType.ALL_IN]:
                decision_text += f" ${self.decision.amount:.2f}"
                
            cv2.putText(result, decision_text, (10, result.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw reasoning
            if len(self.decision.reasoning) > 60:
                reason_text = self.decision.reasoning[:57] + "..."
            else:
                reason_text = self.decision.reasoning
                
            cv2.putText(result, reason_text, (10, result.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result
    
    def set_debug_overlay(self, show_overlay: bool):
        """
        Set whether to show the debug overlay.
        
        Args:
            show_overlay: Whether to show the debug overlay
        """
        self.show_debug_overlay = show_overlay
        
        # Update the display
        self.update_screenshot_display()


def main():
    """Run the game analyzer view standalone (for testing)."""
    import sys
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create application
    app = QtWidgets.QApplication(sys.argv)
    
    # Create a simple window
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Game Analyzer View Test")
    window.resize(800, 600)
    
    # Create the game analyzer view
    analyzer_view = GameAnalyzerView()
    window.setCentralWidget(analyzer_view)
    
    # Show the window
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()