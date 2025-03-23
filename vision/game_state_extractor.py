#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Game State Extractor

This module extracts the complete game state from a poker game screenshot,
including cards, chips, player positions, and other game information.
"""

import logging
import numpy as np
import cv2
import json
import os
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from .card_detector import CardDetector, Card, CardSuit, CardValue
from .utils.image_processing import preprocess_image, apply_threshold
from .utils.ocr_utils import OCRHelper

logger = logging.getLogger("PokerVision.GameStateExtractor")

class PlayerAction(Enum):
    """Enumeration of possible player actions."""
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    BET = auto()
    RAISE = auto()
    ALL_IN = auto()
    SMALL_BLIND = auto()
    BIG_BLIND = auto()
    WAITING = auto()
    SITTING_OUT = auto()
    UNKNOWN = auto()
    
    def __str__(self) -> str:
        """String representation of player action."""
        if self == PlayerAction.FOLD:
            return "Fold"
        elif self == PlayerAction.CHECK:
            return "Check"
        elif self == PlayerAction.CALL:
            return "Call"
        elif self == PlayerAction.BET:
            return "Bet"
        elif self == PlayerAction.RAISE:
            return "Raise"
        elif self == PlayerAction.ALL_IN:
            return "All-In"
        elif self == PlayerAction.SMALL_BLIND:
            return "Small Blind"
        elif self == PlayerAction.BIG_BLIND:
            return "Big Blind"
        elif self == PlayerAction.WAITING:
            return "Waiting"
        elif self == PlayerAction.SITTING_OUT:
            return "Sitting Out"
        else:
            return "Unknown"


@dataclass
class PlayerState:
    """Class representing the state of a player."""
    player_id: int
    name: str = "Unknown"
    stack: float = 0.0
    bet: float = 0.0
    cards: List[Card] = field(default_factory=list)
    position: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, width, height
    is_dealer: bool = False
    is_active: bool = False
    is_current: bool = False
    last_action: PlayerAction = PlayerAction.UNKNOWN
    
    def __str__(self) -> str:
        """String representation of player state."""
        cards_str = ", ".join(str(card) for card in self.cards) if self.cards else "Hidden"
        return f"Player {self.player_id} ({self.name}): Stack=${self.stack:.2f}, Bet=${self.bet:.2f}, Cards=[{cards_str}], Action={self.last_action}"


@dataclass
class GameState:
    """Class representing the state of a poker game."""
    community_cards: List[Card] = field(default_factory=list)
    players: List[PlayerState] = field(default_factory=list)
    pot: float = 0.0
    total_pot: float = 0.0
    current_player_id: int = -1
    dealer_position: int = -1
    hand_number: int = 0
    games_played: int = 0
    small_blind: float = 0.0
    big_blind: float = 0.0
    timestamp: float = 0.0
    
    @property
    def player_cards(self) -> List[Card]:
        """
        Get the current player's cards.
        
        Returns:
            List of player's cards or empty list if not found
        """
        player = self.get_player(0)  # Assuming player ID 0 is the user
        if player and player.cards:
            return player.cards
        return []
    
    def get_player(self, player_id: int) -> Optional[PlayerState]:
        """
        Get a player by ID.
        
        Args:
            player_id: Player ID
            
        Returns:
            PlayerState object or None if not found
        """
        for player in self.players:
            if player.player_id == player_id:
                return player
        return None
    
    def get_current_player(self) -> Optional[PlayerState]:
        """
        Get the current player.
        
        Returns:
            PlayerState object or None if not found
        """
        if self.current_player_id != -1:
            return self.get_player(self.current_player_id)
        return None
    
    def get_dealer(self) -> Optional[PlayerState]:
        """
        Get the dealer.
        
        Returns:
            PlayerState object or None if not found
        """
        if self.dealer_position != -1:
            return self.get_player(self.dealer_position)
        return None
    
    def get_active_players(self) -> List[PlayerState]:
        """
        Get all active players.
        
        Returns:
            List of active PlayerState objects
        """
        return [player for player in self.players if player.is_active]
    
    def __str__(self) -> str:
        """String representation of game state."""
        community_cards_str = ", ".join(str(card) for card in self.community_cards) if self.community_cards else "None"
        players_str = "\n  ".join(str(player) for player in self.players)
        
        return (
            f"Game State (Hand #{self.hand_number}, Games Played: {self.games_played}):\n"
            f"  Pot: ${self.pot:.2f}, Total Pot: ${self.total_pot:.2f}\n"
            f"  Blinds: ${self.small_blind:.2f}/${self.big_blind:.2f}\n"
            f"  Community Cards: [{community_cards_str}]\n"
            f"  Players:\n  {players_str}"
        )


class GameStateExtractor:
    """Class for extracting the complete game state from a poker game screenshot."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the GameStateExtractor.
        
        Args:
            config_path: Path to the configuration file
        """
        logger.info("Initializing GameStateExtractor")
        
        # Set default config path if not provided
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "config", "roi_config.json")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize detectors
        self.card_detector = CardDetector()
        self.ocr_helper = OCRHelper()
        
        # Initialize ROIs
        self.rois = self.config.get("rois", {})
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        config = {}
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            
            # Validate and fix player positions if needed
            if "rois" in config and "player_positions" in config["rois"]:
                player_positions = config["rois"]["player_positions"]
                
                # Check if player positions is a list
                if not isinstance(player_positions, list):
                    logger.warning(f"player_positions is not a list, resetting to default")
                    config["rois"]["player_positions"] = self._create_default_config()["rois"]["player_positions"]
                else:
                    # Validate each position and fix if needed
                    for i, position in enumerate(player_positions):
                        if not isinstance(position, list) or len(position) != 4:
                            logger.warning(f"Invalid player position format for player {i}: {position}, using default")
                            # Replace with default position
                            default_positions = self._create_default_config()["rois"]["player_positions"]
                            if i < len(default_positions):
                                player_positions[i] = default_positions[i]
            
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}")
            config = self._create_default_config()
            
            # Save default configuration
            try:
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                logger.info(f"Saved default configuration to {config_path}")
            except Exception as e:
                logger.error(f"Failed to save default configuration: {e}")
        
        return config
    
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create a default configuration.
        
        Returns:
            Default configuration dictionary
        """
        # Default ROIs for PokerTH
        default_config = {
            "rois": {
                "community_cards": [300, 150, 400, 100],  # x, y, width, height
                "player_cards": [450, 400, 100, 80],
                "pot": [400, 100, 100, 30],
                "player_positions": [
                    [100, 400, 150, 100],  # Player 1 (user)
                    [50, 300, 150, 100],   # Player 2
                    [100, 200, 150, 100],  # Player 3
                    [300, 100, 150, 100],  # Player 4
                    [500, 100, 150, 100],  # Player 5
                    [700, 200, 150, 100],  # Player 6
                    [750, 300, 150, 100],  # Player 7
                    [700, 400, 150, 100],  # Player 8
                    [500, 450, 150, 100]   # Player 9
                ],
                "dealer_button": [0, 0, 0, 0],  # Will be detected automatically
                "hand_number": [10, 10, 100, 30],
                "games_played": [10, 40, 100, 30]
            },
            "extraction": {
                "chip_match_threshold": 0.6,
                "text_confidence_threshold": 0.5,
                "min_pot_amount": 1.0
            }
        }
        
        return default_config
    
    def extract_game_state(self, image: np.ndarray) -> GameState:
        """
        Extract the complete game state from a poker game screenshot.
        
        Args:
            image: Screenshot image
            
        Returns:
            GameState object
        """
        logger.info("Extracting game state")
        
        # Create a new game state
        game_state = GameState()
        game_state.timestamp = cv2.getTickCount() / cv2.getTickFrequency()
        
        try:
            # Extract community cards
            game_state.community_cards = self._extract_community_cards(image)
            
            # Extract pot
            game_state.pot, game_state.total_pot = self._extract_pot(image)
            
            # Extract players
            game_state.players = self._extract_players(image)
            
            # Extract dealer position
            game_state.dealer_position = self._extract_dealer_position(image)
            
            # Extract hand number and games played
            game_state.hand_number, game_state.games_played = self._extract_game_info(image)
            
            # Extract blinds
            game_state.small_blind, game_state.big_blind = self._extract_blinds(image, game_state.players)
            
            # Extract current player
            game_state.current_player_id = self._extract_current_player(image, game_state.players)
            
            logger.info("Game state extraction completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to extract game state: {e}", exc_info=True)
        
        return game_state
    
    def _extract_community_cards(self, image: np.ndarray) -> List[Card]:
        """
        Extract community cards from the image.
        
        Args:
            image: Screenshot image
            
        Returns:
            List of community cards
        """
        logger.debug("Extracting community cards")
        
        try:
            roi = self.rois.get("community_cards")
            if roi:
                return self.card_detector.detect_community_cards(image, roi)
            else:
                return self.card_detector.detect_community_cards(image)
        except Exception as e:
            logger.error(f"Failed to extract community cards: {e}", exc_info=True)
            return []
    
    def _extract_pot(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Extract pot amount from the image.
        
        Args:
            image: Screenshot image
            
        Returns:
            Tuple of (current pot, total pot)
        """
        logger.debug("Extracting pot")
        
        pot = 0.0
        total_pot = 0.0
        
        try:
            roi = self.rois.get("pot")
            if roi:
                x, y, w, h = roi
                pot_roi = image[y:y+h, x:x+w]
                
                # Preprocess the ROI for better OCR
                preprocessed = preprocess_image(pot_roi)
                
                # Perform OCR
                pot_text = self.ocr_helper.read_text(preprocessed)
                
                # Parse pot amount
                if pot_text:
                    # Look for patterns like "$100", "100", "Pot: $100", etc.
                    import re
                    pot_matches = re.findall(r'\$?\s*(\d+\.?\d*)', pot_text)
                    if pot_matches:
                        pot = float(pot_matches[0])
                        
                        # If there are two numbers, second one might be the total pot
                        if len(pot_matches) > 1:
                            total_pot = float(pot_matches[1])
                        else:
                            total_pot = pot
        except Exception as e:
            logger.error(f"Failed to extract pot: {e}", exc_info=True)
        
        return pot, total_pot
    
    def _extract_players(self, image: np.ndarray) -> List[PlayerState]:
        """
        Extract player information from the image.
        
        Args:
            image: Screenshot image
            
        Returns:
            List of PlayerState objects
        """
        logger.debug("Extracting players")
        
        players = []
        
        try:
            player_positions = self.rois.get("player_positions", [])
            
            # Make sure player_positions is a list
            if not isinstance(player_positions, list):
                logger.warning(f"player_positions is not a list: {type(player_positions)}")
                player_positions = []
            
            for i, position in enumerate(player_positions):
                try:
                    # Ensure position is a list or tuple with 4 elements
                    if not isinstance(position, (list, tuple)) or len(position) != 4:
                        logger.warning(f"Invalid player position format for player {i}: {position}")
                        continue
                        
                    # Extract player information
                    player = self._extract_player(image, position, i)
                    if player:
                        players.append(player)
                except Exception as e:
                    logger.error(f"Failed to extract player {i}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to extract players: {e}", exc_info=True)
        
        return players
    
    def _extract_player(self, image: np.ndarray, position: List[int], player_id: int) -> Optional[PlayerState]:
        """
        Extract information for a single player.
        
        Args:
            image: Screenshot image
            position: Player position (x, y, width, height)
            player_id: Player ID
            
        Returns:
            PlayerState object or None if player not present
        """
        x, y, w, h = position
        player_roi = image[y:y+h, x:x+w]
        
        # Check if player is present (empty seats might be detected)
        if self._is_player_present(player_roi):
            player = PlayerState(player_id=player_id, position=position)
            
            # Extract player name
            player.name = self._extract_player_name(player_roi)
            
            # Extract player stack
            player.stack = self._extract_player_stack(player_roi)
            
            # Extract player bet
            player.bet = self._extract_player_bet(player_roi)
            
            # Extract player cards if it's the user
            if player_id == 0:  # Assuming player_id 0 is the user
                player_cards_roi = self.rois.get("player_cards")
                if player_cards_roi:
                    x, y, w, h = player_cards_roi
                    cards_roi = image[y:y+h, x:x+w]
                    player.cards = self.card_detector.detect_player_cards(cards_roi)
            
            # Determine if player is active
            player.is_active = self._is_player_active(player_roi)
            
            # Extract last action
            player.last_action = self._extract_player_action(player_roi)
            
            return player
        
        return None
    
    def _is_player_present(self, player_roi: np.ndarray) -> bool:
        """
        Check if a player is present in the given ROI.
        
        Args:
            player_roi: Player ROI image
            
        Returns:
            True if player is present, False otherwise
        """
        # Convert to grayscale
        gray = cv2.cvtColor(player_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Check if there's enough content in the ROI
        white_pixels = cv2.countNonZero(thresh)
        total_pixels = player_roi.shape[0] * player_roi.shape[1]
        
        # If there's enough content, player is likely present
        return white_pixels / total_pixels > 0.05
    
    def _extract_player_name(self, player_roi: np.ndarray) -> str:
        """
        Extract player name from the player ROI.
        
        Args:
            player_roi: Player ROI image
            
        Returns:
            Player name
        """
        # Assume the top part of the player ROI contains the name
        height, width = player_roi.shape[:2]
        name_roi = player_roi[0:int(height/4), 0:width]
        
        # Preprocess the ROI for better OCR
        preprocessed = preprocess_image(name_roi)
        
        # Perform OCR
        name_text = self.ocr_helper.read_text(preprocessed)
        
        # Clean up the name
        if name_text:
            # Remove any non-alphanumeric characters except spaces
            import re
            name_text = re.sub(r'[^\w\s]', '', name_text).strip()
            
            return name_text
        
        return "Player"
    
    def _extract_player_stack(self, player_roi: np.ndarray) -> float:
        """
        Extract player stack from the player ROI.
        
        Args:
            player_roi: Player ROI image
            
        Returns:
            Player stack amount
        """
        # Assume the middle part of the player ROI contains the stack
        height, width = player_roi.shape[:2]
        stack_roi = player_roi[int(height/4):int(height/2), 0:width]
        
        # Preprocess the ROI for better OCR
        preprocessed = preprocess_image(stack_roi)
        
        # Perform OCR
        stack_text = self.ocr_helper.read_text(preprocessed)
        
        # Parse stack amount
        if stack_text:
            # Look for patterns like "$100", "100", "Stack: $100", etc.
            import re
            stack_matches = re.findall(r'\$?\s*(\d+\.?\d*)', stack_text)
            if stack_matches:
                return float(stack_matches[0])
        
        return 0.0
    
    def _extract_player_bet(self, player_roi: np.ndarray) -> float:
        """
        Extract player bet from the player ROI.
        
        Args:
            player_roi: Player ROI image
            
        Returns:
            Player bet amount
        """
        # Assume the bottom part of the player ROI contains the bet
        height, width = player_roi.shape[:2]
        bet_roi = player_roi[int(height/2):height, 0:width]
        
        # Preprocess the ROI for better OCR
        preprocessed = preprocess_image(bet_roi)
        
        # Perform OCR
        bet_text = self.ocr_helper.read_text(preprocessed)
        
        # Parse bet amount
        if bet_text:
            # Look for patterns like "$100", "100", "Bet: $100", etc.
            import re
            bet_matches = re.findall(r'\$?\s*(\d+\.?\d*)', bet_text)
            if bet_matches:
                return float(bet_matches[0])
        
        return 0.0
    
    def _is_player_active(self, player_roi: np.ndarray) -> bool:
        """
        Check if a player is active in the current hand.
        
        Args:
            player_roi: Player ROI image
            
        Returns:
            True if player is active, False otherwise
        """
        # Convert to HSV
        hsv = cv2.cvtColor(player_roi, cv2.COLOR_BGR2HSV)
        
        # Define a range for active player indicators (usually highlighted in some way)
        lower_highlight = np.array([20, 100, 100])  # Example: yellowish highlight
        upper_highlight = np.array([40, 255, 255])
        
        # Create a mask
        mask = cv2.inRange(hsv, lower_highlight, upper_highlight)
        
        # Check if there are highlighted pixels
        highlighted_pixels = cv2.countNonZero(mask)
        
        # If there are enough highlighted pixels, player is likely active
        return highlighted_pixels > 50
    
    def _extract_player_action(self, player_roi: np.ndarray) -> PlayerAction:
        """
        Extract the last action of a player.
        
        Args:
            player_roi: Player ROI image
            
        Returns:
            PlayerAction enum value
        """
        # Assume the bottom part of the player ROI contains the action
        height, width = player_roi.shape[:2]
        action_roi = player_roi[int(3*height/4):height, 0:width]
        
        # Preprocess the ROI for better OCR
        preprocessed = preprocess_image(action_roi)
        
        # Perform OCR
        action_text = self.ocr_helper.read_text(preprocessed)
        
        # Parse action
        if action_text:
            action_text = action_text.lower()
            
            if "fold" in action_text:
                return PlayerAction.FOLD
            elif "check" in action_text:
                return PlayerAction.CHECK
            elif "call" in action_text:
                return PlayerAction.CALL
            elif "bet" in action_text:
                return PlayerAction.BET
            elif "raise" in action_text:
                return PlayerAction.RAISE
            elif "all" in action_text and "in" in action_text:
                return PlayerAction.ALL_IN
            elif "small" in action_text and "blind" in action_text:
                return PlayerAction.SMALL_BLIND
            elif "big" in action_text and "blind" in action_text:
                return PlayerAction.BIG_BLIND
            elif "waiting" in action_text:
                return PlayerAction.WAITING
            elif "sitting" in action_text and "out" in action_text:
                return PlayerAction.SITTING_OUT
        
        return PlayerAction.UNKNOWN
    
    def _extract_dealer_position(self, image: np.ndarray) -> int:
        """
        Extract the dealer position.
        
        Args:
            image: Screenshot image
            
        Returns:
            Dealer player ID or -1 if not found
        """
        logger.debug("Extracting dealer position")
        
        try:
            # Try using the dealer button ROI if defined
            dealer_button_roi = self.rois.get("dealer_button")
            if dealer_button_roi and isinstance(dealer_button_roi, (list, tuple)) and len(dealer_button_roi) == 4:
                x, y, w, h = dealer_button_roi
                button_roi = image[y:y+h, x:x+w]
                
                # Detect the dealer button
                # This is typically a small circular button
                gray = cv2.cvtColor(button_roi, cv2.COLOR_BGR2GRAY)
                circles = cv2.HoughCircles(
                    gray, cv2.HOUGH_GRADIENT, 1, 20,
                    param1=50, param2=30, minRadius=10, maxRadius=30
                )
                
                if circles is not None:
                    # Button found
                    # Now determine which player it's closest to
                    button_x = x + int(circles[0][0][0])
                    button_y = y + int(circles[0][0][1])
                    
                    # Find the closest player
                    closest_player_id = -1
                    min_distance = float('inf')
                    
                    player_positions = self.rois.get("player_positions", [])
                    for i, position in enumerate(player_positions):
                        # Skip invalid positions
                        if not isinstance(position, (list, tuple)) or len(position) != 4:
                            continue
                            
                        player_x = position[0] + position[2]//2
                        player_y = position[1] + position[3]//2
                        
                        # Calculate distance
                        distance = np.sqrt((button_x - player_x)**2 + (button_y - player_y)**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_player_id = i
                    
                    return closest_player_id
            
            # If no dealer button ROI or button not found, try another approach
            # We can look for text like "D" or "Dealer" near player positions
            player_positions = self.rois.get("player_positions", [])
            for i, position in enumerate(player_positions):
                # Skip invalid positions
                if not isinstance(position, (list, tuple)) or len(position) != 4:
                    continue
                    
                x, y, w, h = position
                # Expand the ROI slightly
                expanded_x = max(0, x - 20)
                expanded_y = max(0, y - 20)
                expanded_w = min(image.shape[1] - expanded_x, w + 40)
                expanded_h = min(image.shape[0] - expanded_y, h + 40)
                
                expanded_roi = image[expanded_y:expanded_y+expanded_h, expanded_x:expanded_x+expanded_w]
                
                # Preprocess the ROI for better OCR
                preprocessed = preprocess_image(expanded_roi)
                
                # Perform OCR
                text = self.ocr_helper.read_text(preprocessed)
                
                # Check for dealer indicators
                if text and ("d" in text.lower() or "dealer" in text.lower()):
                    return i
        
        except Exception as e:
            logger.error(f"Failed to extract dealer position: {e}", exc_info=True)
        
        return -1
    
    def _extract_game_info(self, image: np.ndarray) -> Tuple[int, int]:
        """
        Extract hand number and games played.
        
        Args:
            image: Screenshot image
            
        Returns:
            Tuple of (hand number, games played)
        """
        logger.debug("Extracting game info")
        
        hand_number = 0
        games_played = 0
        
        try:
            # Extract hand number
            hand_number_roi = self.rois.get("hand_number")
            if hand_number_roi:
                x, y, w, h = hand_number_roi
                hand_number_img = image[y:y+h, x:x+w]
                
                # Preprocess the ROI for better OCR
                preprocessed = preprocess_image(hand_number_img)
                
                # Perform OCR
                hand_text = self.ocr_helper.read_text(preprocessed)
                
                # Parse hand number
                if hand_text:
                    import re
                    hand_matches = re.findall(r'hand\s*#?\s*(\d+)', hand_text.lower())
                    if hand_matches:
                        hand_number = int(hand_matches[0])
            
            # Extract games played
            games_played_roi = self.rois.get("games_played")
            if games_played_roi:
                x, y, w, h = games_played_roi
                games_played_img = image[y:y+h, x:x+w]
                
                # Preprocess the ROI for better OCR
                preprocessed = preprocess_image(games_played_img)
                
                # Perform OCR
                games_text = self.ocr_helper.read_text(preprocessed)
                
                # Parse games played
                if games_text:
                    import re
                    games_matches = re.findall(r'game[s]?\s*played\s*:?\s*(\d+)', games_text.lower())
                    if games_matches:
                        games_played = int(games_matches[0])
        
        except Exception as e:
            logger.error(f"Failed to extract game info: {e}", exc_info=True)
        
        return hand_number, games_played
    
    def _extract_blinds(self, image: np.ndarray, players: List[PlayerState]) -> Tuple[float, float]:
        """
        Extract small and big blind amounts.
        
        Args:
            image: Screenshot image
            players: List of player states
            
        Returns:
            Tuple of (small blind, big blind)
        """
        logger.debug("Extracting blinds")
        
        small_blind = 0.0
        big_blind = 0.0
        
        try:
            # Try to find players with small blind and big blind actions
            for player in players:
                if player.last_action == PlayerAction.SMALL_BLIND:
                    small_blind = player.bet
                elif player.last_action == PlayerAction.BIG_BLIND:
                    big_blind = player.bet
            
            # If we found small blind but not big blind, assume big blind is twice the small blind
            if small_blind > 0 and big_blind == 0:
                big_blind = small_blind * 2
            
            # If we found big blind but not small blind, assume small blind is half the big blind
            elif big_blind > 0 and small_blind == 0:
                small_blind = big_blind / 2
            
            # If we didn't find either, try to deduce from the bets
            elif small_blind == 0 and big_blind == 0:
                # Get all bets that might be blinds (small bets at the start of a hand)
                small_bets = sorted([player.bet for player in players if player.bet > 0])
                
                if len(small_bets) >= 2:
                    # Smallest bet is likely the small blind
                    small_blind = small_bets[0]
                    
                    # Second smallest bet is likely the big blind
                    big_blind = small_bets[1]
                
                elif len(small_bets) == 1:
                    # Only one bet, assume it's the big blind
                    big_blind = small_bets[0]
                    small_blind = big_blind / 2
        
        except Exception as e:
            logger.error(f"Failed to extract blinds: {e}", exc_info=True)
        
        return small_blind, big_blind
    
    def _extract_current_player(self, image: np.ndarray, players: List[PlayerState]) -> int:
        """
        Extract the current player ID.
        
        Args:
            image: Screenshot image
            players: List of player states
            
        Returns:
            Current player ID or -1 if not found
        """
        logger.debug("Extracting current player")
        
        try:
            # The current player is typically highlighted in some way
            player_positions = self.rois.get("player_positions", [])
            
            for i, position in enumerate(player_positions):
                # Skip invalid positions
                if not isinstance(position, (list, tuple)) or len(position) != 4:
                    continue
                    
                x, y, w, h = position
                player_roi = image[y:y+h, x:x+w]
                
                # Convert to HSV for better color detection
                hsv = cv2.cvtColor(player_roi, cv2.COLOR_BGR2HSV)
                
                # Define a range for the highlight color (typically yellow or green)
                lower_highlight = np.array([40, 100, 100])  # Green-yellow
                upper_highlight = np.array([80, 255, 255])
                
                # Create a mask
                mask = cv2.inRange(hsv, lower_highlight, upper_highlight)
                
                # Check if there are highlighted pixels
                highlighted_pixels = cv2.countNonZero(mask)
                
                # If there are enough highlighted pixels, this is likely the current player
                if highlighted_pixels > 100:
                    return i
            
            # If no player is highlighted, check if any player has a "thinking" animation
            # This would require additional processing for motion detection
            # For simplicity, we'll skip this for now
            
        except Exception as e:
            logger.error(f"Failed to extract current player: {e}", exc_info=True)
        
        return -1
    
    def draw_game_state(self, image: np.ndarray, game_state: GameState) -> np.ndarray:
        """
        Draw the extracted game state on the image.
        
        Args:
            image: Screenshot image
            game_state: Extracted game state
            
        Returns:
            Image with drawn game state
        """
        result = image.copy()
        
        # Draw game info
        cv2.putText(result, f"Hand #{game_state.hand_number}, Games: {game_state.games_played}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw pot
        cv2.putText(result, f"Pot: ${game_state.pot:.2f}, Total: ${game_state.total_pot:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw blinds
        cv2.putText(result, f"Blinds: ${game_state.small_blind:.2f}/${game_state.big_blind:.2f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw community cards
        community_cards_str = ", ".join(str(card) for card in game_state.community_cards)
        cv2.putText(result, f"Community Cards: {community_cards_str}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw player info
        for player in game_state.players:
            # Get player position
            x, y, w, h = player.position
            
            # Draw rectangle around player
            color = (0, 255, 0) if player.is_active else (0, 0, 255)
            if player.player_id == game_state.current_player_id:
                color = (255, 255, 0)  # Yellow for current player
            if player.is_dealer:
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
            if player.last_action != PlayerAction.UNKNOWN:
                action_text = str(player.last_action)
                cv2.putText(result, action_text, (x, y + h + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw player cards if available
            if player.cards:
                cards_text = ", ".join(str(card) for card in player.cards)
                cv2.putText(result, cards_text, (x, y + h + 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    
    def update_roi_config(self, roi_name: str, roi_value: List[int]) -> bool:
        """
        Update a ROI configuration.
        
        Args:
            roi_name: Name of the ROI
            roi_value: New ROI value (x, y, width, height)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if "rois" not in self.config:
                self.config["rois"] = {}
            
            # Special handling for player_positions
            if roi_name == "player_positions":
                # Ensure roi_value is a list of lists
                if not isinstance(roi_value, list):
                    logger.error(f"Invalid player_positions format: {roi_value}")
                    return False
                
                # Validate each position
                for i, position in enumerate(roi_value):
                    if not isinstance(position, (list, tuple)) or len(position) != 4:
                        logger.error(f"Invalid player position format at index {i}: {position}")
                        return False
            else:
                # For other ROIs, ensure it's a list of 4 integers
                if not isinstance(roi_value, (list, tuple)) or len(roi_value) != 4:
                    logger.error(f"Invalid ROI format for {roi_name}: {roi_value}")
                    return False
                
            # Update the ROI
            self.config["rois"][roi_name] = roi_value
            self.rois = self.config["rois"]
            
            # Save configuration
            config_path = os.path.join(os.path.dirname(__file__), "..", "config", "roi_config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
                
            logger.info(f"Updated ROI configuration for {roi_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update ROI configuration: {e}", exc_info=True)
            return False


# Helper module imports
from .utils.image_processing import preprocess_image, apply_threshold
from .utils.ocr_utils import OCRHelper


# Test function
def test_game_state_extractor():
    """Test the game state extractor functionality."""
    import time
    import os
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Initialize game state extractor
    extractor = GameStateExtractor()
    
    # Load test image
    test_image_path = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "test_poker_game.png")
    if os.path.exists(test_image_path):
        # Load image
        image = cv2.imread(test_image_path)
        
        # Extract game state
        start_time = time.time()
        game_state = extractor.extract_game_state(image)
        end_time = time.time()
        
        print(f"Extracted game state in {end_time - start_time:.3f} seconds:")
        print(game_state)
        
        # Draw game state
        result = extractor.draw_game_state(image, game_state)
        
        # Save result
        cv2.imwrite("game_state.png", result)
        print(f"Saved result to game_state.png")
    else:
        print(f"Test image not found: {test_image_path}")


if __name__ == "__main__":
    # Run test
    test_game_state_extractor()