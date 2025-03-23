#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Window

This module implements the main window for the Poker Vision Assistant application.
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
from PyQt5.QtWidgets import (QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
                            QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter, 
                            QTabWidget, QStatusBar, QAction, QDialog, QFileDialog, 
                            QMessageBox, QDockWidget, QFrame, QGroupBox)

from vision.screen_capture import ScreenCaptureManager, WindowInfo
from vision.game_state_extractor import GameState
from poker.constants import ActionType
from poker.strategy import Decision
from .roi_editor import ROIEditorDialog
from .game_analyzer_view import GameAnalyzerView
from .statistics_view import StatisticsView

logger = logging.getLogger("PokerVision.UI.MainWindow")

class MainWindow(QMainWindow):
    """Main window for the Poker Vision Assistant application."""
    
    def __init__(self, app_controller):
        """
        Initialize the main window.
        
        Args:
            app_controller: The application controller object
        """
        super().__init__()
        
        self.app_controller = app_controller
        self.screen_capture = app_controller.screen_capture
        self.windows_list = []
        self.current_window = None
        self.last_analysis_time = 0
        
        # Set up the UI
        self.setup_ui()
        
        # Set up the signals
        self.setup_signals()
        
        # Set up the timer for auto-refresh
        self.analysis_timer = QTimer()
        self.analysis_timer.timeout.connect(self.auto_analyze)
        
        # Initialize window list and restore any saved window selection
        self.refresh_window_list()
        
        # Configure UI elements based on settings
        self.auto_analyze_button.setChecked(self.app_controller.settings.get("app.auto_analyze", False))
        self.auto_refresh_combo.setCurrentText(str(self.app_controller.settings.get("app.auto_analyze_interval", 3)))
        self.debug_overlay_check.setChecked(self.app_controller.settings.get("app.show_debug_overlay", True))
        
        # Apply settings to UI components
        if self.auto_analyze_button.isChecked():
            self.toggle_auto_analyze(True)
        
        logger.info("Main window initialized")
    
    def setup_ui(self):
        """Set up the user interface."""
        # Window properties
        self.setWindowTitle("Poker Vision Assistant")
        self.resize(1280, 800)
        
        # Central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create the toolbar
        self.setup_toolbar()
        
        # Create the status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Create the main splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.main_splitter)
        
        # Left panel - Control panel
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        self.main_splitter.addWidget(self.control_panel)
        
        # Right panel - Analysis view
        self.analysis_tabs = QTabWidget()
        self.main_splitter.addWidget(self.analysis_tabs)
        
        # Game analyzer view
        self.game_analyzer_view = GameAnalyzerView()
        self.analysis_tabs.addTab(self.game_analyzer_view, "Game Analysis")
        
        # Statistics view
        self.statistics_view = StatisticsView()
        self.analysis_tabs.addTab(self.statistics_view, "Statistics")
        
        # Set up the control panel
        self.setup_control_panel()
        
        # Set splitter sizes
        self.main_splitter.setSizes([300, 980])
        
        # Create dock widgets
        self.setup_dock_widgets()
        
        # Create menus
        self.setup_menus()
    
    def setup_toolbar(self):
        """Set up the toolbar."""
        self.toolbar = self.addToolBar("Main Toolbar")
        self.toolbar.setMovable(False)
        
        # Add Refresh Windows button
        self.refresh_windows_action = QAction(QtGui.QIcon("icons/refresh.png"), "Refresh Windows", self)
        self.refresh_windows_action.triggered.connect(self.refresh_window_list)
        self.toolbar.addAction(self.refresh_windows_action)
        
        # Add Capture button
        self.capture_button = QPushButton("Capture && Analyze")
        self.capture_button.setIcon(QtGui.QIcon("icons/capture.png"))
        self.toolbar.addWidget(self.capture_button)
        
        # Add Auto button
        self.auto_analyze_button = QPushButton("Auto Analyze")
        self.auto_analyze_button.setIcon(QtGui.QIcon("icons/auto.png"))
        self.auto_analyze_button.setCheckable(True)
        self.toolbar.addWidget(self.auto_analyze_button)
        
        # Add spacer
        spacer = QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)
        
        # Add Settings button
        self.settings_button = QPushButton("Settings")
        self.settings_button.setIcon(QtGui.QIcon("icons/settings.png"))
        self.toolbar.addWidget(self.settings_button)
    
    def setup_control_panel(self):
        """Set up the control panel."""
        # Window selection group
        self.window_group = QGroupBox("Window Selection")
        self.window_layout = QVBoxLayout()
        self.window_group.setLayout(self.window_layout)
        
        # Window combo box
        self.window_combo = QComboBox()
        self.window_layout.addWidget(self.window_combo)
        
        # Refresh windows button
        self.refresh_windows_button = QPushButton("Refresh")
        self.window_layout.addWidget(self.refresh_windows_button)
        
        # Add window group to control panel
        self.control_layout.addWidget(self.window_group)
        
        # Analysis options group
        self.analysis_group = QGroupBox("Analysis Options")
        self.analysis_layout = QVBoxLayout()
        self.analysis_group.setLayout(self.analysis_layout)
        
        # Auto refresh rate
        self.auto_refresh_layout = QHBoxLayout()
        self.auto_refresh_label = QLabel("Auto refresh rate (sec):")
        self.auto_refresh_combo = QComboBox()
        self.auto_refresh_combo.addItems(["1", "2", "3", "5", "10"])
        self.auto_refresh_combo.setCurrentIndex(2)  # Default to 3 seconds
        self.auto_refresh_layout.addWidget(self.auto_refresh_label)
        self.auto_refresh_layout.addWidget(self.auto_refresh_combo)
        self.analysis_layout.addLayout(self.auto_refresh_layout)
        
        # Show debug overlay
        self.debug_overlay_check = QtWidgets.QCheckBox("Show debug overlay")
        self.debug_overlay_check.setChecked(True)
        self.analysis_layout.addWidget(self.debug_overlay_check)
        
        # Add analysis group to control panel
        self.control_layout.addWidget(self.analysis_group)
        
        # Decision group
        self.decision_group = QGroupBox("Decision")
        self.decision_layout = QVBoxLayout()
        self.decision_group.setLayout(self.decision_layout)
        
        # Decision label
        self.decision_label = QLabel("No decision yet")
        self.decision_label.setAlignment(Qt.AlignCenter)
        self.decision_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        self.decision_layout.addWidget(self.decision_label)
        
        # Decision reasoning
        self.reasoning_label = QLabel("")
        self.reasoning_label.setAlignment(Qt.AlignCenter)
        self.reasoning_label.setWordWrap(True)
        self.decision_layout.addWidget(self.reasoning_label)
        
        # Hand strength
        self.hand_strength_label = QLabel("")
        self.hand_strength_label.setAlignment(Qt.AlignCenter)
        self.decision_layout.addWidget(self.hand_strength_label)
        
        # Equity label
        self.equity_label = QLabel("")
        self.equity_label.setAlignment(Qt.AlignCenter)
        self.decision_layout.addWidget(self.equity_label)
        
        # Add decision group to control panel
        self.control_layout.addWidget(self.decision_group)
        
        # Add a stretch to push everything up
        self.control_layout.addStretch()
    
    def setup_dock_widgets(self):
        """Set up the dock widgets."""
        # Log widget
        self.log_dock = QDockWidget("Log", self)
        self.log_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        
        self.log_widget = QWidget()
        self.log_layout = QVBoxLayout(self.log_widget)
        
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_layout.addWidget(self.log_text)
        
        self.log_dock.setWidget(self.log_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        
        # Hide by default
        self.log_dock.hide()
    
    def setup_menus(self):
        """Set up the application menus."""
        # Main menu bar
        self.menu_bar = self.menuBar()
        
        # File menu
        self.file_menu = self.menu_bar.addMenu("File")
        
        # Save screenshot action
        self.save_screenshot_action = QAction("Save Screenshot", self)
        self.save_screenshot_action.triggered.connect(self.save_screenshot)
        self.file_menu.addAction(self.save_screenshot_action)
        
        # Save game state action
        self.save_game_state_action = QAction("Save Game State", self)
        self.save_game_state_action.triggered.connect(self.save_game_state)
        self.file_menu.addAction(self.save_game_state_action)
        
        # Exit action
        self.exit_action = QAction("Exit", self)
        self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action)
        
        # Edit menu
        self.edit_menu = self.menu_bar.addMenu("Edit")
        
        # ROI editor action
        self.roi_editor_action = QAction("ROI Editor", self)
        self.roi_editor_action.triggered.connect(self.show_roi_editor)
        self.edit_menu.addAction(self.roi_editor_action)
        
        # Settings action
        self.settings_action = QAction("Settings", self)
        self.settings_action.triggered.connect(self.show_settings_dialog)
        self.edit_menu.addAction(self.settings_action)
        
        # View menu
        self.view_menu = self.menu_bar.addMenu("View")
        
        # Show log action
        self.show_log_action = QAction("Show Log", self)
        self.show_log_action.setCheckable(True)
        self.show_log_action.triggered.connect(self.toggle_log_view)
        self.view_menu.addAction(self.show_log_action)
        
        # Help menu
        self.help_menu = self.menu_bar.addMenu("Help")
        
        # About action
        self.about_action = QAction("About", self)
        self.about_action.triggered.connect(self.show_about_dialog)
        self.help_menu.addAction(self.about_action)
    
    def setup_signals(self):
        """Set up the signal connections."""
        # Connect window combo box
        self.window_combo.currentIndexChanged.connect(self.window_selected)
        
        # Connect refresh windows button
        self.refresh_windows_button.clicked.connect(self.refresh_window_list)
        
        # Connect capture button
        self.capture_button.clicked.connect(self.capture_and_analyze)
        
        # Connect auto analyze button
        self.auto_analyze_button.toggled.connect(self.toggle_auto_analyze)
        
        # Connect auto refresh combo
        self.auto_refresh_combo.currentIndexChanged.connect(self.update_auto_refresh_rate)
        
        # Connect debug overlay checkbox
        self.debug_overlay_check.stateChanged.connect(self.toggle_debug_overlay)
        
        # Connect settings button
        self.settings_button.clicked.connect(self.show_settings_dialog)
    
    def refresh_window_list(self):
        """Refresh the list of windows."""
        logger.debug("Refreshing window list")
        
        # Clear the combo box
        self.window_combo.clear()
        
        # Get the list of windows
        try:
            self.windows_list = self.screen_capture.get_window_list()
            
            # Add windows to combo box
            for window in self.windows_list:
                self.window_combo.addItem(window.title)
            
            # Add "Full Screen" option
            self.window_combo.addItem("Full Screen")
            
            # Restore previously selected window if possible
            last_window_index = self.app_controller.settings.get("app.last_window_index", -1)
            if last_window_index >= 0 and last_window_index <= self.window_combo.count():
                self.window_combo.setCurrentIndex(last_window_index)
                self.window_selected(last_window_index)
            
            logger.info(f"Found {len(self.windows_list)} windows")
            self.status_bar.showMessage(f"Found {len(self.windows_list)} windows")
            
        except Exception as e:
            logger.error(f"Error refreshing window list: {e}")
            self.status_bar.showMessage(f"Error refreshing window list: {e}")
    
    def window_selected(self, index):
        """
        Handle window selection.
        
        Args:
            index: Index of the selected window
        """
        if index < 0:
            return
        
        if index < len(self.windows_list):
            # Selected a specific window
            self.current_window = self.windows_list[index]
            logger.info(f"Selected window: {self.current_window.title}")
            self.status_bar.showMessage(f"Selected window: {self.current_window.title}")
            
            # Update the target window setting
            self.app_controller.settings.set("app.target_window", self.current_window)
            self.app_controller.settings.set("app.last_window_index", index)
            self.app_controller.save_current_settings()
        else:
            # Selected Full Screen
            self.current_window = None
            logger.info("Selected Full Screen")
            self.status_bar.showMessage("Selected Full Screen")
            
            # Update the target window setting
            self.app_controller.settings.set("app.target_window", None)
            self.app_controller.settings.set("app.last_window_index", index)
            self.app_controller.save_current_settings()
    
    def capture_and_analyze(self):
        """Capture and analyze the current window."""
        logger.debug("Capturing and analyzing")
        
        # Use the app controller to perform the analysis
        self.app_controller.capture_and_analyze()
        
        # Record the analysis time
        self.last_analysis_time = time.time()
    
    def toggle_auto_analyze(self, checked):
        """
        Toggle automatic analysis.
        
        Args:
            checked: Whether the auto analyze button is checked
        """
        if checked:
            # Start the timer
            refresh_rate = int(self.auto_refresh_combo.currentText())
            self.analysis_timer.start(refresh_rate * 1000)
            logger.info(f"Auto analyze enabled with refresh rate: {refresh_rate} seconds")
            self.status_bar.showMessage(f"Auto analyze enabled with refresh rate: {refresh_rate} seconds")
            
            # Update settings
            self.app_controller.settings.set("app.auto_analyze", True)
            self.app_controller.settings.set("app.auto_analyze_interval", refresh_rate)
            self.app_controller.save_current_settings()
        else:
            # Stop the timer
            self.analysis_timer.stop()
            logger.info("Auto analyze disabled")
            self.status_bar.showMessage("Auto analyze disabled")
            
            # Update settings
            self.app_controller.settings.set("app.auto_analyze", False)
            self.app_controller.save_current_settings()
    
    def update_auto_refresh_rate(self, index):
        """
        Update the auto refresh rate.
        
        Args:
            index: Index of the selected refresh rate
        """
        if self.auto_analyze_button.isChecked():
            # Update the timer interval
            refresh_rate = int(self.auto_refresh_combo.currentText())
            self.analysis_timer.setInterval(refresh_rate * 1000)
            logger.info(f"Auto refresh rate updated: {refresh_rate} seconds")
            self.status_bar.showMessage(f"Auto refresh rate updated: {refresh_rate} seconds")
            
            # Update settings
            self.app_controller.settings.set("app.auto_analyze_interval", refresh_rate)
            self.app_controller.save_current_settings()
    
    def auto_analyze(self):
        """Perform automatic analysis."""
        # Check if it's too soon since the last analysis
        current_time = time.time()
        min_interval = 1.0  # Minimum interval in seconds
        
        if current_time - self.last_analysis_time >= min_interval:
            self.capture_and_analyze()
    
    def toggle_debug_overlay(self, state):
        """
        Toggle debug overlay.
        
        Args:
            state: Checkbox state
        """
        show_overlay = state == Qt.Checked
        self.game_analyzer_view.set_debug_overlay(show_overlay)
        logger.info(f"Debug overlay: {'enabled' if show_overlay else 'disabled'}")
        
        # Update settings
        self.app_controller.settings.set("app.show_debug_overlay", show_overlay)
        self.app_controller.save_current_settings()
    
    def show_roi_editor(self):
        """Show the ROI editor dialog."""
        try:
            # Get a screenshot to use for the editor
            if self.current_window:
                screenshot = self.screen_capture.capture_window(self.current_window)
            else:
                screenshot = self.screen_capture.capture_screen()
            
            # Create and show the ROI editor dialog
            editor = ROIEditorDialog(self, screenshot, self.app_controller.game_state_extractor)
            editor.exec_()
            
        except Exception as e:
            logger.error(f"Error showing ROI editor: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open ROI editor: {str(e)}")
    
    def show_settings_dialog(self):
        """Show the settings dialog."""
        # TODO: Implement settings dialog
        QMessageBox.information(self, "Settings", "Settings dialog not yet implemented")
    
    def toggle_log_view(self, checked):
        """
        Toggle the log view.
        
        Args:
            checked: Whether the log view should be shown
        """
        if checked:
            self.log_dock.show()
        else:
            self.log_dock.hide()
    
    def save_screenshot(self):
        """Save the current screenshot to a file."""
        # Get the last screenshot
        screenshot = self.screen_capture.get_last_capture()
        
        if screenshot is None:
            QMessageBox.warning(self, "Save Screenshot", "No screenshot available")
            return
        
        # Ask for a file name
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "screenshot.png", "Images (*.png *.jpg)"
        )
        
        if file_path:
            success = self.screen_capture.save_screenshot(file_path, screenshot)
            
            if success:
                QMessageBox.information(self, "Save Screenshot", f"Screenshot saved to {file_path}")
            else:
                QMessageBox.critical(self, "Save Screenshot", "Failed to save screenshot")
    
    def save_game_state(self):
        """Save the current game state to a file."""
        # Check if we have a game state
        if self.app_controller.current_game_state is None:
            QMessageBox.warning(self, "Save Game State", "No game state available")
            return
        
        # Ask for a file name
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Game State", "game_state.json", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                import json
                
                # Convert game state to dictionary
                game_state = self.app_controller.current_game_state
                
                # Create a serializable representation
                data = {
                    "timestamp": game_state.timestamp,
                    "pot": game_state.pot,
                    "total_pot": game_state.total_pot,
                    "small_blind": game_state.small_blind,
                    "big_blind": game_state.big_blind,
                    "current_player_id": game_state.current_player_id,
                    "dealer_position": game_state.dealer_position,
                    "hand_number": game_state.hand_number,
                    "games_played": game_state.games_played,
                    "community_cards": [
                        {"value": card.value.name, "suit": card.suit.name}
                        for card in game_state.community_cards
                    ],
                    "players": [
                        {
                            "player_id": player.player_id,
                            "name": player.name,
                            "stack": player.stack,
                            "bet": player.bet,
                            "is_dealer": player.is_dealer,
                            "is_active": player.is_active,
                            "is_current": player.is_current,
                            "last_action": player.last_action.name,
                            "cards": [
                                {"value": card.value.name, "suit": card.suit.name}
                                for card in player.cards
                            ] if player.cards else []
                        }
                        for player in game_state.players
                    ]
                }
                
                # Save to file
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)
                
                QMessageBox.information(self, "Save Game State", f"Game state saved to {file_path}")
                
            except Exception as e:
                logger.error(f"Error saving game state: {e}")
                QMessageBox.critical(self, "Save Game State", f"Failed to save game state: {str(e)}")
    
    def show_about_dialog(self):
        """Show the about dialog."""
        QMessageBox.about(
            self, 
            "About Poker Vision Assistant",
            """<h1>Poker Vision Assistant</h1>
            <p>Version 1.0</p>
            <p>An advanced poker assistant that uses computer vision to analyze poker games in real time.</p>
            <p>&copy; 2023</p>"""
        )
    
    def update_analysis(self, screenshot: np.ndarray, game_state: GameState, decision: Decision):
        """
        Update the analysis display.
        
        Args:
            screenshot: Screenshot image
            game_state: Current game state
            decision: Recommended decision
        """
        # Update the game analyzer view
        self.game_analyzer_view.update_analysis(screenshot, game_state, decision)
        
        # Update the statistics view
        self.statistics_view.update_statistics(game_state)
        
        # Update the decision display
        self.update_decision_display(decision)
        
        # Update the status bar
        self.status_bar.showMessage(f"Analysis updated at {time.strftime('%H:%M:%S')}")
        
        # Log the analysis
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] Analysis updated")
        
        if decision:
            if decision.action in [ActionType.BET, ActionType.RAISE, ActionType.CALL, ActionType.ALL_IN]:
                self.log_text.append(f"  Decision: {decision.action.name} ${decision.amount:.2f}")
            else:
                self.log_text.append(f"  Decision: {decision.action.name}")
            
            self.log_text.append(f"  Reasoning: {decision.reasoning}")
            self.log_text.append("")
        
        # Scroll to the bottom of the log
        self.log_text.ensureCursorVisible()
    
    def update_decision_display(self, decision: Decision):
        """
        Update the decision display.
        
        Args:
            decision: Recommended decision
        """
        if not decision:
            self.decision_label.setText("No decision")
            self.reasoning_label.setText("")
            self.hand_strength_label.setText("")
            self.equity_label.setText("")
            return
        
        # Set decision text and color
        if decision.action == ActionType.FOLD:
            self.decision_label.setText("FOLD")
            self.decision_label.setStyleSheet("font-size: 20pt; font-weight: bold; color: red;")
        
        elif decision.action == ActionType.CHECK:
            self.decision_label.setText("CHECK")
            self.decision_label.setStyleSheet("font-size: 20pt; font-weight: bold; color: yellow;")
        
        elif decision.action == ActionType.CALL:
            self.decision_label.setText(f"CALL ${decision.amount:.2f}")
            self.decision_label.setStyleSheet("font-size: 20pt; font-weight: bold; color: yellow;")
        
        elif decision.action == ActionType.BET:
            self.decision_label.setText(f"BET ${decision.amount:.2f}")
            self.decision_label.setStyleSheet("font-size: 20pt; font-weight: bold; color: green;")
        
        elif decision.action == ActionType.RAISE:
            self.decision_label.setText(f"RAISE ${decision.amount:.2f}")
            self.decision_label.setStyleSheet("font-size: 20pt; font-weight: bold; color: green;")
        
        elif decision.action == ActionType.ALL_IN:
            self.decision_label.setText("ALL-IN")
            self.decision_label.setStyleSheet("font-size: 20pt; font-weight: bold; color: blue;")
        
        # Set reasoning text
        self.reasoning_label.setText(decision.reasoning)
        
        # Set hand strength and equity if available
        if hasattr(decision, 'hand_evaluation') and decision.hand_evaluation:
            self.hand_strength_label.setText(f"Hand: {decision.hand_evaluation.description}")
            self.equity_label.setText(f"Equity: {decision.hand_evaluation.equity:.1%}")
    
    def show_error(self, message: str):
        """
        Show an error message.
        
        Args:
            message: Error message
        """
        QMessageBox.critical(self, "Error", message)
        self.status_bar.showMessage(f"Error: {message}")
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] ERROR: {message}")
        self.log_text.ensureCursorVisible()


def main():
    """Run the main window standalone (for testing)."""
    import sys
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create application
    app = QtWidgets.QApplication(sys.argv)
    
    # Create a mock app controller
    class MockController:
        def __init__(self):
            self.screen_capture = ScreenCaptureManager()
            self.current_game_state = None
        
        def capture_and_analyze(self):
            logger.info("Mock capture and analyze")
    
    # Create and show the main window
    controller = MockController()
    window = MainWindow(controller)
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()