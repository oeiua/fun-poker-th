#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Collector

This module provides functionality to collect and save data for training
and analysis purposes.
"""

import logging
import os
import json
import time
import csv
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import cv2

from vision.game_state_extractor import GameState, PlayerState
from vision.card_detector import Card
from poker.constants import ActionType
from poker.strategy import Decision

logger = logging.getLogger("PokerVision.Utils.DataCollector")

class DataCollector:
    """Class for collecting and saving training data."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the DataCollector.
        
        Args:
            data_dir: Directory to save data, or None for default
        """
        # Set default data directory if not provided
        if data_dir is None:
            self.data_dir = os.path.join(
                str(Path.home()), 
                ".poker_vision", 
                "training_data"
            )
        else:
            self.data_dir = data_dir
        
        # Create subdirectories
        self.screenshots_dir = os.path.join(self.data_dir, "screenshots")
        self.game_states_dir = os.path.join(self.data_dir, "game_states")
        self.card_samples_dir = os.path.join(self.data_dir, "card_samples")
        self.player_samples_dir = os.path.join(self.data_dir, "player_samples")
        self.decisions_dir = os.path.join(self.data_dir, "decisions")
        
        # Ensure directories exist
        for directory in [
            self.data_dir, 
            self.screenshots_dir, 
            self.game_states_dir,
            self.card_samples_dir,
            self.player_samples_dir,
            self.decisions_dir
        ]:
            os.makedirs(directory, exist_ok=True)
        
        # Session ID and game stats
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.collect_stats = {
            "screenshots": 0,
            "game_states": 0,
            "card_samples": 0,
            "player_samples": 0,
            "decisions": 0
        }
        
        # Counters and batch tracking
        self.last_game_state = None
        self.last_screenshot = None
        self.screenshot_counter = self._get_next_counter(self.screenshots_dir)
        self.game_state_counter = self._get_next_counter(self.game_states_dir)
        self.decisions_counter = self._get_next_counter(self.decisions_dir)
        
        # Initialize session log
        self.session_log_path = os.path.join(self.data_dir, f"session_{self.session_id}.log")
        self._init_session_log()
        
        logger.info(f"DataCollector initialized with data directory: {self.data_dir}")
    
    def _get_next_counter(self, directory: str) -> int:
        """
        Get the next available counter for a directory.
        
        Args:
            directory: Directory to scan
            
        Returns:
            Next available counter
        """
        try:
            # List files and find the highest number
            files = os.listdir(directory)
            counters = []
            
            for filename in files:
                try:
                    # Extract counter from filename (format: NNNNNN_*.*)
                    counter = int(filename.split('_')[0])
                    counters.append(counter)
                except (ValueError, IndexError):
                    continue
            
            # Return the next counter
            if counters:
                return max(counters) + 1
            else:
                return 1
            
        except Exception as e:
            logger.error(f"Error getting next counter: {e}")
            return 1
    
    def _init_session_log(self) -> None:
        """Initialize the session log file."""
        try:
            with open(self.session_log_path, 'w') as f:
                f.write(f"Poker Vision Assistant - Data Collection Session\n")
                f.write(f"Session ID: {self.session_id}\n")
                f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Data Directory: {self.data_dir}\n")
                f.write("\n=== Collection Log ===\n\n")
            
            logger.info(f"Session log initialized: {self.session_log_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize session log: {e}")
    
    def _log_collection(self, item_type: str, filename: str, info: Optional[str] = None) -> None:
        """
        Log a collection event to the session log.
        
        Args:
            item_type: Type of collected item
            filename: Filename of the collected item
            info: Additional information
        """
        try:
            with open(self.session_log_path, 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_line = f"[{timestamp}] {item_type}: {filename}"
                if info:
                    log_line += f" - {info}"
                f.write(log_line + "\n")
                
        except Exception as e:
            logger.error(f"Failed to log collection: {e}")
    
    def save_screenshot(self, screenshot: np.ndarray, 
                      prefix: str = "screenshot", 
                      with_timestamp: bool = True) -> Optional[str]:
        """
        Save a screenshot.
        
        Args:
            screenshot: Screenshot image
            prefix: Filename prefix
            with_timestamp: Whether to include timestamp in filename
            
        Returns:
            Path to the saved screenshot, or None if failed
        """
        try:
            # Generate filename
            counter_str = f"{self.screenshot_counter:06d}"
            if with_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{counter_str}_{prefix}_{timestamp}.png"
            else:
                filename = f"{counter_str}_{prefix}.png"
                
            # Save screenshot
            filepath = os.path.join(self.screenshots_dir, filename)
            success = cv2.imwrite(filepath, screenshot)
            
            if success:
                # Update counter and stats
                self.screenshot_counter += 1
                self.collect_stats["screenshots"] += 1
                
                # Save as last screenshot
                self.last_screenshot = screenshot.copy()
                
                # Log collection
                self._log_collection("Screenshot", filename)
                
                logger.debug(f"Screenshot saved: {filepath}")
                return filepath
            
            else:
                logger.error(f"Failed to save screenshot: {filepath}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving screenshot: {e}")
            return None
    
    def save_game_state(self, game_state: GameState,
                      screenshot: Optional[np.ndarray] = None) -> Optional[str]:
        """
        Save a game state.
        
        Args:
            game_state: Game state to save
            screenshot: Optional screenshot to save with the game state
            
        Returns:
            Path to the saved game state, or None if failed
        """
        try:
            # Generate filename
            counter_str = f"{self.game_state_counter:06d}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{counter_str}_game_state_{timestamp}.json"
            
            # Save game state
            filepath = os.path.join(self.game_states_dir, filename)
            
            # Convert game state to dictionary
            data = self._game_state_to_dict(game_state)
            
            # Add metadata
            data["metadata"] = {
                "session_id": self.session_id,
                "timestamp": timestamp,
                "counter": self.game_state_counter
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Update counter and stats
            self.game_state_counter += 1
            self.collect_stats["game_states"] += 1
            
            # Save as last game state
            self.last_game_state = game_state
            
            # Log collection
            hand_info = f"Hand #{game_state.hand_number}, Pot: ${game_state.pot:.2f}"
            self._log_collection("Game State", filename, hand_info)
            
            # Save screenshot if provided
            if screenshot is not None:
                screenshot_filename = f"{counter_str}_game_state_{timestamp}.png"
                screenshot_path = os.path.join(self.screenshots_dir, screenshot_filename)
                cv2.imwrite(screenshot_path, screenshot)
                
                # Update stats
                self.collect_stats["screenshots"] += 1
                
                # Log collection
                self._log_collection("Screenshot", screenshot_filename, hand_info)
            
            logger.debug(f"Game state saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving game state: {e}")
            return None
    
    def save_decision(self, decision: Decision, 
                    game_state: Optional[GameState] = None,
                    screenshot: Optional[np.ndarray] = None) -> Optional[str]:
        """
        Save a decision.
        
        Args:
            decision: Decision to save
            game_state: Optional game state
            screenshot: Optional screenshot
            
        Returns:
            Path to the saved decision, or None if failed
        """
        try:
            # Generate filename
            counter_str = f"{self.decisions_counter:06d}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{counter_str}_decision_{timestamp}.json"
            
            # Create decision data
            data = {
                "action": decision.action.name,
                "amount": float(decision.amount) if hasattr(decision, "amount") else 0.0,
                "confidence": float(decision.confidence) if hasattr(decision, "confidence") else 0.0,
                "reasoning": decision.reasoning if hasattr(decision, "reasoning") else "",
                "metadata": {
                    "session_id": self.session_id,
                    "timestamp": timestamp,
                    "counter": self.decisions_counter
                }
            }
            
            # Add game state if provided
            if game_state is not None:
                data["game_state"] = self._game_state_to_dict(game_state)
            
            # Save to file
            filepath = os.path.join(self.decisions_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Update counter and stats
            self.decisions_counter += 1
            self.collect_stats["decisions"] += 1
            
            # Log collection
            decision_info = f"{decision.action.name}"
            if hasattr(decision, "amount") and decision.amount > 0:
                decision_info += f" ${decision.amount:.2f}"
            self._log_collection("Decision", filename, decision_info)
            
            # Save screenshot if provided
            if screenshot is not None:
                screenshot_filename = f"{counter_str}_decision_{timestamp}.png"
                screenshot_path = os.path.join(self.screenshots_dir, screenshot_filename)
                cv2.imwrite(screenshot_path, screenshot)
                
                # Update stats
                self.collect_stats["screenshots"] += 1
                
                # Log collection
                self._log_collection("Screenshot", screenshot_filename, decision_info)
            
            logger.debug(f"Decision saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving decision: {e}")
            return None
    
    def save_card_sample(self, card: Card, image: np.ndarray, 
                       hand_id: Optional[int] = None) -> Optional[str]:
        """
        Save a card sample for training.
        
        Args:
            card: Card object
            image: Card image
            hand_id: Optional hand ID for tracking
            
        Returns:
            Path to the saved card sample, or None if failed
        """
        try:
            # Create subdirectory for this card (e.g., "ace_of_hearts")
            card_dir = os.path.join(
                self.card_samples_dir, 
                f"{card.value.name.lower()}_of_{card.suit.name.lower()}"
            )
            os.makedirs(card_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            counter = len(os.listdir(card_dir)) + 1
            
            if hand_id is not None:
                filename = f"{counter:06d}_hand{hand_id}_{timestamp}.png"
            else:
                filename = f"{counter:06d}_{timestamp}.png"
            
            # Save card image
            filepath = os.path.join(card_dir, filename)
            success = cv2.imwrite(filepath, image)
            
            if success:
                # Update stats
                self.collect_stats["card_samples"] += 1
                
                # Log collection
                card_info = f"{card.value.name} of {card.suit.name}"
                self._log_collection("Card Sample", filename, card_info)
                
                logger.debug(f"Card sample saved: {filepath}")
                return filepath
            
            else:
                logger.error(f"Failed to save card sample: {filepath}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving card sample: {e}")
            return None
    
    def save_player_sample(self, player_id: int, image: np.ndarray,
                         hand_id: Optional[int] = None) -> Optional[str]:
        """
        Save a player interface sample for training.
        
        Args:
            player_id: Player ID
            image: Player interface image
            hand_id: Optional hand ID for tracking
            
        Returns:
            Path to the saved player sample, or None if failed
        """
        try:
            # Create subdirectory for this player
            player_dir = os.path.join(self.player_samples_dir, f"player_{player_id}")
            os.makedirs(player_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            counter = len(os.listdir(player_dir)) + 1
            
            if hand_id is not None:
                filename = f"{counter:06d}_hand{hand_id}_{timestamp}.png"
            else:
                filename = f"{counter:06d}_{timestamp}.png"
            
            # Save player image
            filepath = os.path.join(player_dir, filename)
            success = cv2.imwrite(filepath, image)
            
            if success:
                # Update stats
                self.collect_stats["player_samples"] += 1
                
                # Log collection
                self._log_collection("Player Sample", filename, f"Player {player_id}")
                
                logger.debug(f"Player sample saved: {filepath}")
                return filepath
            
            else:
                logger.error(f"Failed to save player sample: {filepath}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving player sample: {e}")
            return None
    
    def extract_and_save_card_samples(self, game_state: GameState, 
                                    screenshot: np.ndarray) -> int:
        """
        Extract and save card samples from a screenshot.
        
        Args:
            game_state: Game state
            screenshot: Screenshot image
            
        Returns:
            Number of cards saved
        """
        # Count of saved cards
        saved_count = 0
        hand_id = game_state.hand_number
        
        try:
            # Extract and save community cards
            for i, card in enumerate(game_state.community_cards):
                if "community_cards" in game_state.game_state_extractor.rois:
                    roi = game_state.game_state_extractor.rois["community_cards"]
                    x, y, w, h = roi
                    
                    # Estimate individual card position (simple approximation)
                    card_width = w / max(len(game_state.community_cards), 1)
                    card_x = x + int(i * card_width)
                    card_y = y
                    card_w = int(card_width)
                    card_h = h
                    
                    # Crop card image
                    card_image = screenshot[card_y:card_y+card_h, card_x:card_x+card_w]
                    
                    # Save card sample
                    if card_image.size > 0:
                        self.save_card_sample(card, card_image, hand_id)
                        saved_count += 1
            
            # Extract and save player cards
            player = game_state.get_player(0)
            if player and player.cards:
                if "player_cards" in game_state.game_state_extractor.rois:
                    roi = game_state.game_state_extractor.rois["player_cards"]
                    x, y, w, h = roi
                    
                    # Estimate individual card position (simple approximation)
                    card_width = w / max(len(player.cards), 1)
                    
                    for i, card in enumerate(player.cards):
                        card_x = x + int(i * card_width)
                        card_y = y
                        card_w = int(card_width)
                        card_h = h
                        
                        # Crop card image
                        card_image = screenshot[card_y:card_y+card_h, card_x:card_x+card_w]
                        
                        # Save card sample
                        if card_image.size > 0:
                            self.save_card_sample(card, card_image, hand_id)
                            saved_count += 1
        
        except Exception as e:
            logger.error(f"Error extracting card samples: {e}")
        
        return saved_count
    
    def extract_and_save_player_samples(self, game_state: GameState, 
                                      screenshot: np.ndarray) -> int:
        """
        Extract and save player interface samples from a screenshot.
        
        Args:
            game_state: Game state
            screenshot: Screenshot image
            
        Returns:
            Number of player samples saved
        """
        # Count of saved player samples
        saved_count = 0
        hand_id = game_state.hand_number
        
        try:
            # Extract and save player interfaces
            if "player_positions" in game_state.game_state_extractor.rois:
                player_positions = game_state.game_state_extractor.rois["player_positions"]
                
                for i, position in enumerate(player_positions):
                    x, y, w, h = position
                    
                    # Crop player interface image
                    player_image = screenshot[y:y+h, x:x+w]
                    
                    # Save player sample
                    if player_image.size > 0:
                        self.save_player_sample(i, player_image, hand_id)
                        saved_count += 1
        
        except Exception as e:
            logger.error(f"Error extracting player samples: {e}")
        
        return saved_count
    
    def _game_state_to_dict(self, game_state: GameState) -> Dict[str, Any]:
        """
        Convert a game state to a dictionary for JSON serialization.
        
        Args:
            game_state: Game state to convert
            
        Returns:
            Dictionary representation
        """
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
                    "last_action": player.last_action.name if player.last_action else None,
                    "cards": [
                        {"value": card.value.name, "suit": card.suit.name}
                        for card in player.cards
                    ] if player.cards else []
                }
                for player in game_state.players
            ]
        }
        
        return data
    
    def export_session_data(self, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Export all session data to a zip file.
        
        Args:
            output_dir: Output directory, or None for current directory
            
        Returns:
            Path to the zip file, or None if failed
        """
        try:
            # Set default output directory if not provided
            if output_dir is None:
                output_dir = os.getcwd()
            
            # Generate zip filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"poker_vision_data_{self.session_id}_{timestamp}.zip"
            zip_path = os.path.join(output_dir, zip_filename)
            
            # Create the zip file
            shutil.make_archive(
                zip_path.replace(".zip", ""),  # remove .zip extension
                'zip',
                self.data_dir
            )
            
            logger.info(f"Session data exported to {zip_path}")
            return zip_path
            
        except Exception as e:
            logger.error(f"Error exporting session data: {e}")
            return None
    
    def export_stats(self, output_file: Optional[str] = None) -> Optional[str]:
        """
        Export collection statistics to a CSV file.
        
        Args:
            output_file: Output filename, or None for default
            
        Returns:
            Path to the CSV file, or None if failed
        """
        try:
            # Set default output file if not provided
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(
                    self.data_dir, 
                    f"stats_{self.session_id}_{timestamp}.csv"
                )
            
            # Add end time and duration to stats
            end_time = datetime.now()
            start_time = datetime.strptime(
                self.session_id, 
                "%Y%m%d_%H%M%S"
            )
            duration = (end_time - start_time).total_seconds()
            
            # Create stats data
            stats = {
                "session_id": self.session_id,
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": duration,
                "screenshots": self.collect_stats["screenshots"],
                "game_states": self.collect_stats["game_states"],
                "card_samples": self.collect_stats["card_samples"],
                "player_samples": self.collect_stats["player_samples"],
                "decisions": self.collect_stats["decisions"]
            }
            
            # Write to CSV
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=stats.keys())
                writer.writeheader()
                writer.writerow(stats)
            
            logger.info(f"Collection statistics exported to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting statistics: {e}")
            return None
    
    def collect_game_state(self, game_state: GameState, screenshot: np.ndarray,
                         decision: Optional[Decision] = None,
                         extract_samples: bool = True) -> Dict[str, Any]:
        """
        Collect and save a complete game state with screenshot and samples.
        
        Args:
            game_state: Game state to collect
            screenshot: Screenshot image
            decision: Optional decision
            extract_samples: Whether to extract and save card and player samples
            
        Returns:
            Dictionary of saved file paths
        """
        saved_files = {}
        
        # Save game state
        game_state_path = self.save_game_state(game_state)
        if game_state_path:
            saved_files["game_state"] = game_state_path
        
        # Save screenshot
        hand_num = f"hand{game_state.hand_number}"
        screenshot_path = self.save_screenshot(screenshot, prefix=hand_num)
        if screenshot_path:
            saved_files["screenshot"] = screenshot_path
        
        # Save decision if provided
        if decision:
            decision_path = self.save_decision(decision, game_state, screenshot)
            if decision_path:
                saved_files["decision"] = decision_path
        
        # Extract and save card and player samples
        if extract_samples:
            num_cards = self.extract_and_save_card_samples(game_state, screenshot)
            num_players = self.extract_and_save_player_samples(game_state, screenshot)
            
            saved_files["card_samples"] = num_cards
            saved_files["player_samples"] = num_players
        
        return saved_files
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary of statistics
        """
        # Add session information
        stats = self.collect_stats.copy()
        stats["session_id"] = self.session_id
        stats["data_dir"] = self.data_dir
        
        # Calculate session duration
        start_time = datetime.strptime(self.session_id, "%Y%m%d_%H%M%S")
        duration = (datetime.now() - start_time).total_seconds()
        stats["duration_seconds"] = duration
        
        return stats


# Test function
def test_data_collector():
    """Test the data collector functionality."""
    import tempfile
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create a temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a data collector
        collector = DataCollector(temp_dir)
        
        # Create a test image
        test_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "Test Image", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Test saving a screenshot
        screenshot_path = collector.save_screenshot(test_image)
        assert screenshot_path is not None
        
        # Create a mock game state
        class MockCard:
            def __init__(self, value_name, suit_name):
                self.value = type('obj', (object,), {'name': value_name})
                self.suit = type('obj', (object,), {'name': suit_name})
        
        class MockPlayer:
            def __init__(self, player_id):
                self.player_id = player_id
                self.name = f"Player {player_id}"
                self.stack = 100.0
                self.bet = 0.0
                self.is_dealer = False
                self.is_active = True
                self.is_current = False
                self.last_action = None
                self.cards = [MockCard("ACE", "HEARTS"), MockCard("KING", "SPADES")]
        
        class MockGameState:
            def __init__(self):
                self.timestamp = time.time()
                self.pot = 10.0
                self.total_pot = 10.0
                self.small_blind = 0.5
                self.big_blind = 1.0
                self.current_player_id = 0
                self.dealer_position = 1
                self.hand_number = 1
                self.games_played = 1
                self.community_cards = [
                    MockCard("QUEEN", "HEARTS"),
                    MockCard("JACK", "DIAMONDS"),
                    MockCard("TEN", "CLUBS")
                ]
                self.players = [MockPlayer(i) for i in range(3)]
                
                # Mock ROIs
                self.game_state_extractor = type('obj', (object,), {
                    'rois': {
                        "community_cards": [50, 50, 150, 100],
                        "player_cards": [200, 200, 100, 50],
                        "player_positions": [
                            [100, 100, 50, 50],
                            [200, 100, 50, 50],
                            [300, 100, 50, 50]
                        ]
                    }
                })
            
            def get_player(self, player_id):
                for player in self.players:
                    if player.player_id == player_id:
                        return player
                return None
        
        # Create a mock game state
        game_state = MockGameState()
        
        # Test saving game state
        game_state_path = collector.save_game_state(game_state)
        assert game_state_path is not None
        
        # Create a mock decision
        class MockDecision:
            def __init__(self):
                self.action = type('obj', (object,), {'name': 'CALL'})
                self.amount = 1.0
                self.confidence = 0.8
                self.reasoning = "Good pot odds"
        
        # Create a mock decision
        decision = MockDecision()
        
        # Test saving decision
        decision_path = collector.save_decision(decision, game_state)
        assert decision_path is not None
        
        # Test extracting and saving card samples
        num_cards = collector.extract_and_save_card_samples(game_state, test_image)
        assert num_cards > 0
        
        # Test extracting and saving player samples
        num_players = collector.extract_and_save_player_samples(game_state, test_image)
        assert num_players > 0
        
        # Test collecting game state
        saved_files = collector.collect_game_state(game_state, test_image, decision)
        assert len(saved_files) > 0
        
        # Test exporting stats
        stats_path = collector.export_stats()
        assert stats_path is not None
        
        # Test exporting session data
        zip_path = collector.export_session_data(temp_dir)
        if zip_path is not None:
            assert os.path.exists(zip_path)
        
        # Get stats
        stats = collector.get_stats()
        assert stats["screenshots"] > 0
        assert stats["game_states"] > 0
        assert stats["card_samples"] > 0
        assert stats["player_samples"] > 0
        assert stats["decisions"] > 0
        
        logger.info("Data collector tests passed")


if __name__ == "__main__":
    # Run test
    test_data_collector()