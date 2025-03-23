#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistics View

This module implements the statistics view component for displaying
poker game statistics and performance metrics.
"""

import logging
import os
import sys
import time
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import defaultdict, deque

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
                           QGridLayout, QGroupBox, QSplitter, QTabWidget,
                           QTableWidget, QTableWidgetItem, QHeaderView,
                           QFrame, QProgressBar)

from vision.game_state_extractor import GameState, PlayerState, PlayerAction
from poker.constants import ActionType, HandRank

logger = logging.getLogger("PokerVision.UI.StatisticsView")

class StatisticsView(QWidget):
    """Widget for displaying poker game statistics."""
    
    def __init__(self, parent=None):
        """
        Initialize the statistics view.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Game history
        self.game_history = []
        self.max_history_size = 100
        
        # Decision history
        self.decision_history = []
        
        # Performance metrics
        self.hands_played = 0
        self.wins = 0
        self.losses = 0
        self.big_blinds_won = 0
        
        # Action statistics
        self.action_stats = {
            ActionType.FOLD: 0,
            ActionType.CHECK: 0,
            ActionType.CALL: 0,
            ActionType.BET: 0,
            ActionType.RAISE: 0,
            ActionType.ALL_IN: 0
        }
        
        # Position statistics
        self.position_stats = defaultdict(lambda: {"hands": 0, "wins": 0, "bb_won": 0})
        
        # Hand statistics
        self.hand_stats = defaultdict(lambda: {"count": 0, "wins": 0, "bb_won": 0})
        
        # Set up the UI
        self.setup_ui()
        
        logger.info("Statistics view initialized")
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main layout
        self.main_layout = QVBoxLayout(self)
        
        # Create tabs for different statistics
        self.stats_tabs = QTabWidget()
        
        # General statistics tab
        self.general_tab = QWidget()
        self.general_layout = QVBoxLayout(self.general_tab)
        self.stats_tabs.addTab(self.general_tab, "General")
        
        # Position statistics tab
        self.position_tab = QWidget()
        self.position_layout = QVBoxLayout(self.position_tab)
        self.stats_tabs.addTab(self.position_tab, "Position")
        
        # Hand statistics tab
        self.hand_tab = QWidget()
        self.hand_layout = QVBoxLayout(self.hand_tab)
        self.stats_tabs.addTab(self.hand_tab, "Hands")
        
        # Decision history tab
        self.decision_tab = QWidget()
        self.decision_layout = QVBoxLayout(self.decision_tab)
        self.stats_tabs.addTab(self.decision_tab, "History")
        
        # Set up each tab
        self.setup_general_tab()
        self.setup_position_tab()
        self.setup_hand_tab()
        self.setup_decision_tab()
        
        # Add tabs to main layout
        self.main_layout.addWidget(self.stats_tabs)
    
    def setup_general_tab(self):
        """Set up the general statistics tab."""
        # Performance metrics group
        self.performance_group = QGroupBox("Performance Metrics")
        self.performance_layout = QGridLayout()
        self.performance_group.setLayout(self.performance_layout)
        
        # Hands played
        self.hands_played_label = QLabel("Hands Played:")
        self.hands_played_value = QLabel("0")
        self.hands_played_value.setStyleSheet("font-weight: bold;")
        self.performance_layout.addWidget(self.hands_played_label, 0, 0)
        self.performance_layout.addWidget(self.hands_played_value, 0, 1)
        
        # Win percentage
        self.win_pct_label = QLabel("Win Percentage:")
        self.win_pct_value = QLabel("0%")
        self.win_pct_value.setStyleSheet("font-weight: bold;")
        self.performance_layout.addWidget(self.win_pct_label, 1, 0)
        self.performance_layout.addWidget(self.win_pct_value, 1, 1)
        
        # BB/100
        self.bb_100_label = QLabel("BB/100:")
        self.bb_100_value = QLabel("0")
        self.bb_100_value.setStyleSheet("font-weight: bold;")
        self.performance_layout.addWidget(self.bb_100_label, 2, 0)
        self.performance_layout.addWidget(self.bb_100_value, 2, 1)
        
        # Add performance group to layout
        self.general_layout.addWidget(self.performance_group)
        
        # Action statistics group
        self.action_group = QGroupBox("Action Statistics")
        self.action_layout = QVBoxLayout()
        self.action_group.setLayout(self.action_layout)
        
        # Action statistics table
        self.action_table = QTableWidget(6, 2)
        self.action_table.setHorizontalHeaderLabels(["Action", "Count"])
        self.action_table.verticalHeader().setVisible(False)
        self.action_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Initialize action table rows
        self.action_table.setItem(0, 0, QTableWidgetItem("Fold"))
        self.action_table.setItem(1, 0, QTableWidgetItem("Check"))
        self.action_table.setItem(2, 0, QTableWidgetItem("Call"))
        self.action_table.setItem(3, 0, QTableWidgetItem("Bet"))
        self.action_table.setItem(4, 0, QTableWidgetItem("Raise"))
        self.action_table.setItem(5, 0, QTableWidgetItem("All-In"))
        
        # Set initial counts to 0
        for i in range(6):
            self.action_table.setItem(i, 1, QTableWidgetItem("0"))
        
        # Add table to layout
        self.action_layout.addWidget(self.action_table)
        
        # Add action group to layout
        self.general_layout.addWidget(self.action_group)
        
        # Add a stretch to push everything up
        self.general_layout.addStretch()
    
    def setup_position_tab(self):
        """Set up the position statistics tab."""
        # Position statistics group
        self.position_group = QGroupBox("Position Statistics")
        self.position_layout = QVBoxLayout()
        self.position_group.setLayout(self.position_layout)
        
        # Position statistics table
        self.position_table = QTableWidget(4, 4)
        self.position_table.setHorizontalHeaderLabels(["Position", "Hands", "Win %", "BB/100"])
        self.position_table.verticalHeader().setVisible(False)
        self.position_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Initialize position table rows
        self.position_table.setItem(0, 0, QTableWidgetItem("Early"))
        self.position_table.setItem(1, 0, QTableWidgetItem("Middle"))
        self.position_table.setItem(2, 0, QTableWidgetItem("Late"))
        self.position_table.setItem(3, 0, QTableWidgetItem("Blinds"))
        
        # Set initial values to 0
        for i in range(4):
            for j in range(1, 4):
                self.position_table.setItem(i, j, QTableWidgetItem("0"))
        
        # Add table to layout
        self.position_layout.addWidget(self.position_table)
        
        # Add position group to layout
        self.position_layout.addWidget(self.position_group)
        
        # VPIP by position group
        self.vpip_group = QGroupBox("VPIP by Position")
        self.vpip_layout = QVBoxLayout()
        self.vpip_group.setLayout(self.vpip_layout)
        
        # VPIP progress bars
        self.vpip_early_label = QLabel("Early:")
        self.vpip_early_bar = QProgressBar()
        self.vpip_early_bar.setRange(0, 100)
        self.vpip_early_bar.setValue(0)
        
        self.vpip_middle_label = QLabel("Middle:")
        self.vpip_middle_bar = QProgressBar()
        self.vpip_middle_bar.setRange(0, 100)
        self.vpip_middle_bar.setValue(0)
        
        self.vpip_late_label = QLabel("Late:")
        self.vpip_late_bar = QProgressBar()
        self.vpip_late_bar.setRange(0, 100)
        self.vpip_late_bar.setValue(0)
        
        self.vpip_blinds_label = QLabel("Blinds:")
        self.vpip_blinds_bar = QProgressBar()
        self.vpip_blinds_bar.setRange(0, 100)
        self.vpip_blinds_bar.setValue(0)
        
        # Add progress bars to layout
        vpip_grid = QGridLayout()
        vpip_grid.addWidget(self.vpip_early_label, 0, 0)
        vpip_grid.addWidget(self.vpip_early_bar, 0, 1)
        vpip_grid.addWidget(self.vpip_middle_label, 1, 0)
        vpip_grid.addWidget(self.vpip_middle_bar, 1, 1)
        vpip_grid.addWidget(self.vpip_late_label, 2, 0)
        vpip_grid.addWidget(self.vpip_late_bar, 2, 1)
        vpip_grid.addWidget(self.vpip_blinds_label, 3, 0)
        vpip_grid.addWidget(self.vpip_blinds_bar, 3, 1)
        
        self.vpip_layout.addLayout(vpip_grid)
        
        # Add VPIP group to layout
        self.position_layout.addWidget(self.vpip_group)
        
        # Add a stretch to push everything up
        self.position_layout.addStretch()
    
    def setup_hand_tab(self):
        """Set up the hand statistics tab."""
        # Hand statistics group
        self.hand_group = QGroupBox("Hand Statistics")
        self.hand_layout = QVBoxLayout()
        self.hand_group.setLayout(self.hand_layout)
        
        # Hand statistics table
        self.hand_table = QTableWidget(5, 4)
        self.hand_table.setHorizontalHeaderLabels(["Hand Type", "Count", "Win %", "BB/100"])
        self.hand_table.verticalHeader().setVisible(False)
        self.hand_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Initialize hand table rows
        self.hand_table.setItem(0, 0, QTableWidgetItem("Premium"))
        self.hand_table.setItem(1, 0, QTableWidgetItem("Strong"))
        self.hand_table.setItem(2, 0, QTableWidgetItem("Playable"))
        self.hand_table.setItem(3, 0, QTableWidgetItem("Marginal"))
        self.hand_table.setItem(4, 0, QTableWidgetItem("Weak"))
        
        # Set initial values to 0
        for i in range(5):
            for j in range(1, 4):
                self.hand_table.setItem(i, j, QTableWidgetItem("0"))
        
        # Add table to layout
        self.hand_layout.addWidget(self.hand_table)
        
        # Add hand group to layout
        self.hand_layout.addWidget(self.hand_group)
        
        # Best and worst hands group
        self.best_worst_group = QGroupBox("Best and Worst Hands")
        self.best_worst_layout = QGridLayout()
        self.best_worst_group.setLayout(self.best_worst_layout)
        
        # Best hands
        self.best_hands_label = QLabel("Best Hands:")
        self.best_hands_value = QLabel("None")
        self.best_worst_layout.addWidget(self.best_hands_label, 0, 0)
        self.best_worst_layout.addWidget(self.best_hands_value, 0, 1)
        
        # Worst hands
        self.worst_hands_label = QLabel("Worst Hands:")
        self.worst_hands_value = QLabel("None")
        self.best_worst_layout.addWidget(self.worst_hands_label, 1, 0)
        self.best_worst_layout.addWidget(self.worst_hands_value, 1, 1)
        
        # Add best-worst group to layout
        self.hand_layout.addWidget(self.best_worst_group)
        
        # Add a stretch to push everything up
        self.hand_layout.addStretch()
    
    def setup_decision_tab(self):
        """Set up the decision history tab."""
        # Decision history group
        self.history_group = QGroupBox("Decision History")
        self.history_layout = QVBoxLayout()
        self.history_group.setLayout(self.history_layout)
        
        # Decision history table
        self.history_table = QTableWidget(0, 4)
        self.history_table.setHorizontalHeaderLabels(["Hand", "Decision", "Amount", "Result"])
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Add table to layout
        self.history_layout.addWidget(self.history_table)
        
        # Add history group to layout
        self.decision_layout.addWidget(self.history_group)
    
    def update_statistics(self, game_state: GameState):
        """
        Update the statistics with the current game state.
        
        Args:
            game_state: Current game state
        """
        logger.debug("Updating statistics")
        
        # Add game state to history
        self.add_to_history(game_state)
        
        # Update performance metrics
        self.update_performance_metrics()
        
        # Update action statistics
        self.update_action_statistics()
        
        # Update position statistics
        self.update_position_statistics()
        
        # Update hand statistics
        self.update_hand_statistics()
        
        # Update decision history
        self.update_decision_history()
    
    def add_to_history(self, game_state: GameState):
        """
        Add a game state to the history.
        
        Args:
            game_state: Game state to add
        """
        # Check if this is a new hand
        if not self.game_history or game_state.hand_number != self.game_history[-1].hand_number:
            # Add to history
            self.game_history.append(game_state)
            
            # Limit history size
            if len(self.game_history) > self.max_history_size:
                self.game_history.pop(0)
            
            # Increment hands played
            self.hands_played += 1
    
    def update_performance_metrics(self):
        """Update performance metrics display."""
        # Update hands played
        self.hands_played_value.setText(str(self.hands_played))
        
        # Update win percentage
        if self.hands_played > 0:
            win_pct = (self.wins / self.hands_played) * 100
            self.win_pct_value.setText(f"{win_pct:.1f}%")
        else:
            self.win_pct_value.setText("0%")
        
        # Update BB/100
        if self.hands_played > 0:
            bb_100 = (self.big_blinds_won / self.hands_played) * 100
            self.bb_100_value.setText(f"{bb_100:.1f}")
            
            # Set color based on value
            if bb_100 > 0:
                self.bb_100_value.setStyleSheet("font-weight: bold; color: green;")
            elif bb_100 < 0:
                self.bb_100_value.setStyleSheet("font-weight: bold; color: red;")
            else:
                self.bb_100_value.setStyleSheet("font-weight: bold;")
        else:
            self.bb_100_value.setText("0")
            self.bb_100_value.setStyleSheet("font-weight: bold;")
    
    def update_action_statistics(self):
        """Update action statistics display."""
        # Update action counts
        for i, action_type in enumerate([
            ActionType.FOLD, ActionType.CHECK, ActionType.CALL,
            ActionType.BET, ActionType.RAISE, ActionType.ALL_IN
        ]):
            count = self.action_stats.get(action_type, 0)
            self.action_table.setItem(i, 1, QTableWidgetItem(str(count)))
    
    def update_position_statistics(self):
        """Update position statistics display."""
        # Update position table
        for i, position in enumerate(["Early", "Middle", "Late", "Blinds"]):
            stats = self.position_stats[position]
            
            # Update hands count
            self.position_table.setItem(i, 1, QTableWidgetItem(str(stats["hands"])))
            
            # Update win percentage
            if stats["hands"] > 0:
                win_pct = (stats["wins"] / stats["hands"]) * 100
                self.position_table.setItem(i, 2, QTableWidgetItem(f"{win_pct:.1f}%"))
            else:
                self.position_table.setItem(i, 2, QTableWidgetItem("0%"))
            
            # Update BB/100
            if stats["hands"] > 0:
                bb_100 = (stats["bb_won"] / stats["hands"]) * 100
                self.position_table.setItem(i, 3, QTableWidgetItem(f"{bb_100:.1f}"))
            else:
                self.position_table.setItem(i, 3, QTableWidgetItem("0"))
        
        # Update VPIP (Voluntarily Put money In Pot) by position
        vpip_early = self.calculate_vpip("Early")
        vpip_middle = self.calculate_vpip("Middle")
        vpip_late = self.calculate_vpip("Late")
        vpip_blinds = self.calculate_vpip("Blinds")
        
        self.vpip_early_bar.setValue(int(vpip_early * 100))
        self.vpip_middle_bar.setValue(int(vpip_middle * 100))
        self.vpip_late_bar.setValue(int(vpip_late * 100))
        self.vpip_blinds_bar.setValue(int(vpip_blinds * 100))
    
    def calculate_vpip(self, position: str) -> float:
        """
        Calculate VPIP for a given position.
        
        Args:
            position: Position name
            
        Returns:
            VPIP as a float (0.0 to 1.0)
        """
        # TODO: Implement VPIP calculation based on history
        # For now, return random values for demonstration
        import random
        return random.uniform(0.2, 0.5)
    
    def update_hand_statistics(self):
        """Update hand statistics display."""
        # Update hand table
        for i, hand_type in enumerate(["Premium", "Strong", "Playable", "Marginal", "Weak"]):
            stats = self.hand_stats[hand_type]
            
            # Update count
            self.hand_table.setItem(i, 1, QTableWidgetItem(str(stats["count"])))
            
            # Update win percentage
            if stats["count"] > 0:
                win_pct = (stats["wins"] / stats["count"]) * 100
                self.hand_table.setItem(i, 2, QTableWidgetItem(f"{win_pct:.1f}%"))
            else:
                self.hand_table.setItem(i, 2, QTableWidgetItem("0%"))
            
            # Update BB/100
            if stats["count"] > 0:
                bb_100 = (stats["bb_won"] / stats["count"]) * 100
                self.hand_table.setItem(i, 3, QTableWidgetItem(f"{bb_100:.1f}"))
            else:
                self.hand_table.setItem(i, 3, QTableWidgetItem("0"))
        
        # Update best and worst hands
        # TODO: Implement best and worst hands calculation
        self.best_hands_value.setText("AA, KK, AKs")
        self.worst_hands_value.setText("72o, 83o, 94o")
    
    def update_decision_history(self):
        """Update decision history display."""
        # Make sure the table has the correct number of rows
        self.history_table.setRowCount(len(self.decision_history))
        
        # Update table contents
        for i, decision in enumerate(self.decision_history):
            # Hand description
            self.history_table.setItem(i, 0, QTableWidgetItem(decision.get("hand", "")))
            
            # Decision type
            self.history_table.setItem(i, 1, QTableWidgetItem(decision.get("action", "")))
            
            # Amount
            amount = decision.get("amount", 0)
            if amount > 0:
                self.history_table.setItem(i, 2, QTableWidgetItem(f"${amount:.2f}"))
            else:
                self.history_table.setItem(i, 2, QTableWidgetItem(""))
            
            # Result
            result = decision.get("result", "")
            result_item = QTableWidgetItem(result)
            
            if result == "Win":
                result_item.setForeground(QtGui.QColor("green"))
            elif result == "Loss":
                result_item.setForeground(QtGui.QColor("red"))
            
            self.history_table.setItem(i, 3, result_item)
    
    def add_decision(self, decision_info: Dict[str, Any]):
        """
        Add a decision to the history.
        
        Args:
            decision_info: Decision information
        """
        # Add to history
        self.decision_history.append(decision_info)
        
        # Limit history size
        if len(self.decision_history) > self.max_history_size:
            self.decision_history.pop(0)
        
        # Update action stats
        action = decision_info.get("action_type")
        if action is not None:
            self.action_stats[action] = self.action_stats.get(action, 0) + 1
        
        # Update display
        self.update_decision_history()
        self.update_action_statistics()
    
    def record_hand_result(self, hand_type: str, position: str, result: str, bb_delta: float):
        """
        Record the result of a hand.
        
        Args:
            hand_type: Type of hand (Premium, Strong, etc.)
            position: Position (Early, Middle, Late, Blinds)
            result: Result (Win, Loss)
            bb_delta: Change in big blinds
        """
        # Update hand stats
        self.hand_stats[hand_type]["count"] += 1
        if result == "Win":
            self.hand_stats[hand_type]["wins"] += 1
        self.hand_stats[hand_type]["bb_won"] += bb_delta
        
        # Update position stats
        self.position_stats[position]["hands"] += 1
        if result == "Win":
            self.position_stats[position]["wins"] += 1
        self.position_stats[position]["bb_won"] += bb_delta
        
        # Update overall stats
        if result == "Win":
            self.wins += 1
        else:
            self.losses += 1
        self.big_blinds_won += bb_delta
        
        # Update displays
        self.update_performance_metrics()
        self.update_position_statistics()
        self.update_hand_statistics()


def main():
    """Run the statistics view standalone (for testing)."""
    import sys
    import random
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create application
    app = QtWidgets.QApplication(sys.argv)
    
    # Create a simple window
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Statistics View Test")
    window.resize(800, 600)
    
    # Create the statistics view
    stats_view = StatisticsView()
    window.setCentralWidget(stats_view)
    
    # Add some test data
    for i in range(20):
        # Add a decision
        decision_info = {
            "hand": f"Hand #{i+1}",
            "action": random.choice(["Fold", "Check", "Call", "Bet", "Raise"]),
            "action_type": random.choice([
                ActionType.FOLD, ActionType.CHECK, ActionType.CALL, 
                ActionType.BET, ActionType.RAISE
            ]),
            "amount": random.uniform(0, 10) if random.random() > 0.3 else 0,
            "result": random.choice(["Win", "Loss"])
        }
        
        stats_view.add_decision(decision_info)
        
        # Record hand result
        hand_type = random.choice(["Premium", "Strong", "Playable", "Marginal", "Weak"])
        position = random.choice(["Early", "Middle", "Late", "Blinds"])
        result = random.choice(["Win", "Loss"])
        bb_delta = random.uniform(-5, 10)
        
        stats_view.record_hand_result(hand_type, position, result, bb_delta)
    
    # Show the window
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()