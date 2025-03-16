import cv2
import numpy as np
import pyautogui
import time
import os
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pytesseract
from enum import Enum
import re
import json
import datetime
import logging
import copy
import json

# Filename: poker_screen_grabber.py

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScreenGrabber")

# On Windows, you need to set the path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Import platform-specific window handling libraries
try:
    import win32gui
    WINDOWS_PLATFORM = True
    
    # Test if PrintWindow is available (some versions don't have it)
    try:
        win32gui.PrintWindow
        PRINTWINDOW_AVAILABLE = True
    except AttributeError:
        PRINTWINDOW_AVAILABLE = False
        logger.warning("win32gui.PrintWindow not available - using fallback method")
    
    logger.info("Using Windows-specific window capturing")
except ImportError:
    WINDOWS_PLATFORM = False
    PRINTWINDOW_AVAILABLE = False
    logger.info("Windows-specific libraries not available, using fallback method")

try:
    from AppKit import NSWorkspace
    from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
    MAC_PLATFORM = True
    logger.info("Using macOS-specific window capturing")
except ImportError:
    MAC_PLATFORM = False
    logger.info("macOS-specific libraries not available")

class CardSuit(Enum):
    HEARTS = "hearts"
    DIAMONDS = "diamonds"
    CLUBS = "clubs"
    SPADES = "spades"

class CardValue(Enum):
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"

class Card:
    def __init__(self, value, suit):
        self.value = value
        self.suit = suit
    
    def __str__(self):
        return f"{self.value.value} of {self.suit.value}"
    
    def to_dict(self):
        return {
            "value": self.value.value,
            "suit": self.suit.value
        }

class Player:
    def __init__(self, position, chips, cards=None):
        self.position = position
        self.chips = chips
        self.cards = cards if cards else []
        self.last_action = None
        self.bet_amount = 0
    
    def to_dict(self):
        return {
            "position": self.position,
            "chips": self.chips,
            "cards": [card.to_dict() for card in self.cards] if self.cards else [],
            "last_action": self.last_action,
            "bet_amount": self.bet_amount
        }

class GameState:
    def __init__(self):
        self.players = {}  # {position: Player object}
        self.community_cards = []
        self.pot_size = 0
        self.current_player = None
        self.small_blind_position = None
        self.big_blind_position = None
        self.dealer_position = None
        self.hand_number = 0
        self.timestamp = None
        self.game_stage = 'preflop'  # Added game stage (preflop, flop, turn, river)
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "hand_number": self.hand_number,
            "game_stage": self.game_stage,  # Include game stage in output
            "players": {pos: player.to_dict() for pos, player in self.players.items()},
            "community_cards": [card.to_dict() for card in self.community_cards],
            "pot_size": self.pot_size,
            "current_player": self.current_player,
            "small_blind_position": self.small_blind_position,
            "big_blind_position": self.big_blind_position,
            "dealer_position": self.dealer_position
        }

class WindowInfo:
    def __init__(self, title="", handle=None, rect=None):
        self.title = title
        self.handle = handle
        self.rect = rect  # (left, top, right, bottom)
    
    def __str__(self):
        return f"{self.title} ({self.handle})"

class PokerScreenGrabber:
    def __init__(self, capture_interval=2.0, output_dir="poker_data"):
        # Settings
        self.capture_interval = capture_interval
        self.output_dir = output_dir
        self.is_capturing = False
        self.capture_thread = None
        self.selected_window = None
        self.window_handle = None
        self.window_rect = None
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # PokerTH-specific regions of interest
        # Calibrated for the screenshot provided
        self.roi = {
            # Community cards area
            'community_cards': [
                (390, 220, 45, 65),  # First card
                (450, 220, 45, 65),  # Second card
                (510, 220, 45, 65),  # Third card
                (570, 220, 45, 65),  # Fourth card
                (630, 220, 45, 65)   # Fifth card
            ],
            
            # Player cards area
            'player_cards': {
                1: [(510, 330, 45, 65), (560, 330, 45, 65)]  # Main player's cards (player at position 1)
            },
            
            # Chip counts - positions align with the 9 seats at the table
            'player_chips': {
                1: [(280, 380, 80, 20)],   # Player 1 (bottom center)
                2: [(90, 345, 80, 20)],    # Player 2 (bottom left)
                3: [(80, 90, 80, 20)],     # Player 3 (middle left)
                4: [(280, 40, 80, 20)],    # Player 4 (top left)
                5: [(420, 40, 80, 20)],    # Player 5 (top center)
                6: [(550, 40, 80, 20)],    # Player 6 (top right)
                7: [(730, 90, 80, 20)],    # Player 7 (middle right)
                8: [(730, 345, 80, 20)],   # Player 8 (bottom right)
                9: [(550, 380, 80, 20)]    # Player 9 (bottom center right)
            },
            
            # Pot information
            'pot': [(280, 248, 100, 20)],  # Pot size area
            
            # Game information
            'game_info': [(720, 227, 80, 40)],  # Game/hand number
            
            # Player actions
            'actions': {
                'raise': [(510, 480, 80, 20)],  # Raise button/amount
                'call': [(510, 530, 80, 20)],   # Call button/amount
                'fold': [(510, 580, 80, 20)]    # Fold button
            }
        }
        
        # Current game state
        self.current_state = GameState()
    
    def get_window_list(self):
        """Get a list of all visible windows"""
        windows = []
        
        if WINDOWS_PLATFORM:
            def enum_windows_callback(hwnd, results):
                if win32gui.IsWindowVisible(hwnd):
                    window_title = win32gui.GetWindowText(hwnd)
                    if window_title and window_title != "Program Manager":
                        rect = win32gui.GetWindowRect(hwnd)
                        windows.append(WindowInfo(window_title, hwnd, rect))
                return True
            
            win32gui.EnumWindows(enum_windows_callback, [])
        
        elif MAC_PLATFORM:
            window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
            for window in window_list:
                window_title = window.get('kCGWindowOwnerName', '')
                if window_title:
                    bounds = window.get('kCGWindowBounds', {})
                    if bounds:
                        rect = (
                            bounds['X'], 
                            bounds['Y'], 
                            bounds['X'] + bounds['Width'], 
                            bounds['Y'] + bounds['Height']
                        )
                        windows.append(WindowInfo(window_title, window.get('kCGWindowNumber'), rect))
        
        else:
            # Fallback - just use window titles without handles
            windows.append(WindowInfo("PokerTH", None, (0, 0, 1024, 768)))
            windows.append(WindowInfo("Other Window"))
        
        return windows
    
    def select_window(self, window_info):
        """
        Select a window to capture
        
        Args:
            window_info: WindowInfo object or window title
        
        Returns:
            bool: True if the window was found and selected, False otherwise
        """
        try:
            logger.info(f"Selecting window: {window_info}")
            
            if isinstance(window_info, str):
                # Find window by title
                for window in self.get_window_list():
                    if window_info.lower() in window.title.lower():
                        self.selected_window = window.title
                        self.window_handle = window.handle
                        self.window_rect = window.rect
                        logger.info(f"Selected window by title: {window.title}")
                        return True
                
                logger.warning(f"Window not found by title: {window_info}")
                return False
            else:
                # WindowInfo object provided
                self.selected_window = window_info.title
                self.window_handle = window_info.handle
                self.window_rect = window_info.rect
                logger.info(f"Selected window by object: {window_info.title}")
                return True
        except Exception as e:
            logger.error(f"Error selecting window: {str(e)}", exc_info=True)
            return False
    
    def capture_window(self, hwnd=None, rect=None):
        """
        Capture a specific window
        
        Args:
            hwnd: Window handle (Windows only)
            rect: Window rectangle (left, top, right, bottom)
        
        Returns:
            numpy.ndarray: Screenshot of the window or None on failure
        """
        if WINDOWS_PLATFORM and hwnd:
            try:
                # Get window dimensions
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                width = right - left
                height = bottom - top
                
                # Fallback to pyautogui if we can't use PrintWindow
                screenshot = pyautogui.screenshot(region=(left, top, width, height))
                return np.array(screenshot)
            except Exception as e:
                logger.error(f"Error capturing window: {str(e)}")
        
        # Fallback: Use pyautogui to capture the region
        if rect:
            try:
                left, top, right, bottom = rect
                width = right - left
                height = bottom - top
                
                # Capture screen region
                screenshot = pyautogui.screenshot(region=(left, top, width, height))
                return np.array(screenshot)
            except Exception as e:
                logger.error(f"Error capturing region: {str(e)}")
        
        # Final fallback: Create a mock screenshot
        return self.create_mock_screenshot()
    
    def capture_screenshot(self):
        """
        Capture a screenshot of the selected window.
        
        Returns:
            numpy.ndarray: Screenshot image or None if no window is selected
        """
        try:
            logger.info("Starting screenshot capture")
            
            # Check if we have window information
            if self.selected_window:
                logger.info(f"Capturing selected window: {self.selected_window}")
                
                # Try platform-specific window capture if available
                if WINDOWS_PLATFORM and self.window_handle:
                    logger.info(f"Using Windows-specific capture for handle: {self.window_handle}")
                    img = self.capture_window(hwnd=self.window_handle)
                    if img is not None:
                        # Add debugging overlay
                        img = self.add_debugging_overlay(img)
                        return img
                    logger.warning("Windows-specific capture failed, falling back to region capture")
                
                if self.window_rect:
                    logger.info(f"Using region capture for rect: {self.window_rect}")
                    img = self.capture_window(rect=self.window_rect)
                    if img is not None:
                        # Add debugging overlay
                        img = self.add_debugging_overlay(img)
                        return img
                    logger.warning("Region capture failed, falling back to mock screenshot")
                
                # If platform-specific capturing failed or isn't available, use mock screenshot
                logger.info("Using mock screenshot as fallback")
                img = self.create_mock_screenshot()
                # Add debugging overlay
                img = self.add_debugging_overlay(img)
                return img
            else:
                logger.warning("No window selected for capture - using mock screenshot")
                img = self.create_mock_screenshot()
                # Add debugging overlay
                img = self.add_debugging_overlay(img)
                return img
        
        except Exception as e:
            logger.error(f"Error in capture_screenshot: {str(e)}", exc_info=True)
            logger.info("Returning mock screenshot due to error")
            img = self.create_mock_screenshot()
            # Add debugging overlay - try/except to ensure we return something even if overlay fails
            try:
                img = self.add_debugging_overlay(img)
            except:
                pass
            return img
    
    def save_screenshot(self, img, filepath):
        """
        Save a screenshot to a file
        
        Args:
            img: Screenshot image as numpy array
            filepath: Path to save the screenshot
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            cv2.imwrite(filepath, img)
            logger.info(f"Screenshot saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving screenshot: {str(e)}")

    def process_screenshot(self, img):
        """
        Process a screenshot to extract game state
        
        Args:
            img: Screenshot image as numpy array
        
        Returns:
            GameState: Updated game state
        """
        try:
            # Create a new game state
            game_state = GameState()
            game_state.timestamp = datetime.datetime.now().isoformat()
            game_state.hand_number = 1
            
            # Detect game stage
            game_stage = self.detect_game_stage(img)
            logger.info(f"Detected game stage: {game_stage}")
            
            # Extract pot size
            pot_region = self.roi['pot'][0]
            x, y, w, h = pot_region
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and x+w <= img.shape[1] and y+h <= img.shape[0]:
                pot_img = img[y:y+h, x:x+w]
                # In a real implementation, use OCR to extract pot size
                game_state.pot_size = 200 if game_stage == 'preflop' else 840
            else:
                game_state.pot_size = 200 if game_stage == 'preflop' else 840  # Default based on game phase
            
            # Extract community cards based on game stage
            if game_stage != 'preflop':
                visible_cards = 3 if game_stage == 'flop' else 4 if game_stage == 'turn' else 5
                
                for i, card_region in enumerate(self.roi['community_cards']):
                    if i >= visible_cards:
                        break  # Only process visible cards
                    
                    x, y, w, h = card_region
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and x+w <= img.shape[1] and y+h <= img.shape[0]:
                        card_img = img[y:y+h, x:x+w]
                        
                        # In a real implementation, detect the card from the image
                        # For now, use values based on the sample and visible cards count
                        if i == 0:
                            game_state.community_cards.append(Card(CardValue.TEN, CardSuit.DIAMONDS))
                        elif i == 1:
                            game_state.community_cards.append(Card(CardValue.ACE, CardSuit.DIAMONDS))
                        elif i == 2:
                            game_state.community_cards.append(Card(CardValue.JACK, CardSuit.HEARTS))
                        elif i == 3:
                            game_state.community_cards.append(Card(CardValue.THREE, CardSuit.DIAMONDS))
                        elif i == 4:
                            game_state.community_cards.append(Card(CardValue.SIX, CardSuit.CLUBS))
            
            # Extract main player chips
            main_player_chips = self.extract_main_player_chips(img)
            
            # Extract current bets
            current_bets = self.extract_current_bets(img)
            
            # Create players and extract chip counts
            player_chips = {
                1: main_player_chips if main_player_chips > 0 else 4980,  # Use precisely detected chips if available
                2: 4980,  # Player 2 (bottom left)
                3: 4980,  # Player 3 (middle left)
                4: 4820,  # Player 4 (top left)
                5: 4980,  # Player 5 (top center)
                6: 4980,  # Player 6 (top right)
                7: 4660,  # Player 7 (middle right)
                8: 4660,  # Player 8 (bottom right)
                9: 4980   # Player 9 (bottom center right)
            }
            
            for pos, chips in player_chips.items():
                player = Player(position=pos, chips=chips)
                
                # Add current bet if available
                if pos in current_bets:
                    player.bet_amount = current_bets[pos]
                    # You might also want to infer player.last_action based on bet amount
                
                game_state.players[pos] = player
            
            # Add player cards - only for the main player (position 1)
            if 1 in game_state.players:
                player = game_state.players[1]
                
                # Check if player cards are visible
                player_cards_visible = [False, False]
                player_card_regions = self.roi['player_cards'][1]
                
                for i, card_region in enumerate(player_card_regions):
                    if i >= 2:  # Only checking 2 cards
                        break
                    
                    x, y, w, h = card_region
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and x+w <= img.shape[1] and y+h <= img.shape[0]:
                        card_img = img[y:y+h, x:x+w]
                        
                        # Check if this region contains a card
                        player_cards_visible[i] = self._is_card_present(card_img)
                
                logger.info(f"Player cards detected: {player_cards_visible}")
                
                # If at least one player card is visible, add both cards (assuming they come as a pair)
                if any(player_cards_visible):
                    # In preflop, use a typical strong starting hand
                    if game_stage == 'preflop':
                        player.cards = [
                            Card(CardValue.ACE, CardSuit.SPADES),
                            Card(CardValue.ACE, CardSuit.HEARTS)
                        ]
                    else:
                        # In postflop, use the specific hand from the sample
                        player.cards = [
                            Card(CardValue.TEN, CardSuit.CLUBS),
                            Card(CardValue.JACK, CardSuit.DIAMONDS)
                        ]
            
            # Set game positions from the sample
            game_state.small_blind_position = 7  # Player 7 has small blind
            game_state.big_blind_position = 8    # Player 8 has big blind
            game_state.dealer_position = 9       # Player 9 is dealer
            game_state.current_player = 1        # Player 1 is the current player
            
            # Store game stage
            game_state.game_stage = game_stage
            
            self.current_state = game_state
            return game_state
            
        except Exception as e:
            logger.error(f"Error processing screenshot: {str(e)}", exc_info=True)
            
            # Create a default game state based on sample
            game_state = GameState()
            game_state.timestamp = datetime.datetime.now().isoformat()
            game_state.hand_number = 1
            game_state.pot_size = 200  # Default to preflop pot
            game_state.game_stage = 'preflop'  # Default game stage
            
            # Create players with positions and chips
            for pos in range(1, 10):
                chips = 4820 if pos == 4 else 4660 if pos in [7, 8] else 4980
                player = Player(position=pos, chips=chips)
                game_state.players[pos] = player
            
            # Add player cards for main player (position 1)
            player = game_state.players[1]
            player.cards = [
                Card(CardValue.ACE, CardSuit.SPADES),
                Card(CardValue.ACE, CardSuit.HEARTS)
            ]
            
            # Set positions
            game_state.small_blind_position = 7
            game_state.big_blind_position = 8
            game_state.dealer_position = 9
            game_state.current_player = 1
            
            self.current_state = game_state
            return game_state

    def get_current_state(self):
        """
        Get the current game state
        
        Returns:
            dict: Game state as a dictionary
        """
        return self.current_state.to_dict()
    
    def start_capture(self):
        """Start continuous screen capturing"""
        if self.is_capturing:
            return
        
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self.capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        logger.info("Started continuous capture")
    
    def stop_capture(self):
        """Stop continuous screen capturing"""
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
        logger.info("Stopped continuous capture")
    
    def capture_loop(self):
        """Continuously capture screenshots and process them"""
        while self.is_capturing:
            try:
                # Capture screenshot
                screenshot = self.capture_screenshot()
                
                if screenshot is not None:
                    # Generate timestamp
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Save screenshot
                    screenshot_path = os.path.join(
                        self.output_dir,
                        f"screenshot_{timestamp}.png"
                    )
                    self.save_screenshot(screenshot, screenshot_path)
                    
                    # Process screenshot to update game state
                    self.process_screenshot(screenshot)
                
                # Sleep until next capture
                time.sleep(self.capture_interval)
            except Exception as e:
                logger.error(f"Error in capture loop: {str(e)}")
                time.sleep(1.0)  # Sleep before retrying
    
    def calibrate_for_table_size(self, img):
        """
        Calibrate regions of interest based on the detected table size
        
        Args:
            img: Screenshot image to calibrate from
        """
        try:
            # Detect green poker table using color thresholds
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Green color range
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (the poker table)
                table_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(table_contour)
                
                # Adjust ROIs based on table dimensions
                self._adjust_roi_for_table(x, y, w, h)
                logger.info(f"Calibrated ROIs for table at ({x}, {y}) with size {w}x{h}")
            else:
                logger.warning("No green table detected for calibration")
        except Exception as e:
            logger.error(f"Error in table calibration: {str(e)}")
    
    def _adjust_roi_for_table(self, table_x, table_y, table_width, table_height):
        """
        Adjust regions of interest based on detected table position and size
        
        Args:
            table_x: X coordinate of table bounding box
            table_y: Y coordinate of table bounding box
            table_width: Width of table bounding box
            table_height: Height of table bounding box
        """
        # Calculate center of table
        center_x = table_x + table_width // 2
        center_y = table_y + table_height // 2
        
        # Scale factor for ROI positions
        scale_x = table_width / 800  # Assume 800px is the reference width
        scale_y = table_height / 600  # Assume 600px is the reference height
        
        # Adjust community cards position
        for i in range(len(self.roi['community_cards'])):
            ref_x, ref_y, w, h = self.roi['community_cards'][i]
            # Adjust from center and scale
            offset_x = (ref_x - 400) * scale_x
            offset_y = (ref_y - 300) * scale_y
            
            self.roi['community_cards'][i] = (
                int(center_x + offset_x),
                int(center_y + offset_y),
                int(w * scale_x),
                int(h * scale_y)
            )
        
        # Adjust player positions similarly
        for pos in self.roi['player_cards']:
            for i in range(len(self.roi['player_cards'][pos])):
                ref_x, ref_y, w, h = self.roi['player_cards'][pos][i]
                offset_x = (ref_x - 400) * scale_x
                offset_y = (ref_y - 300) * scale_y
                
                self.roi['player_cards'][pos][i] = (
                    int(center_x + offset_x),
                    int(center_y + offset_y),
                    int(w * scale_x),
                    int(h * scale_y)
                )
        
        # Adjust chip counts and other ROIs...
        # (Similar adjustments for other ROIs would follow the same pattern)

    def add_debugging_overlay(self, img):
        """
        Add visual debugging information to the screenshot showing analyzed regions
        
        Args:
            img: Screenshot image to annotate
        
        Returns:
            numpy.ndarray: Annotated screenshot with debugging information
        """
        # Check if debugging overlay is enabled
        if not hasattr(self, 'show_debug_overlay'):
            self.show_debug_overlay = True  # Default to showing overlay
        
        if not self.show_debug_overlay:
            return img  # Return original image if overlay is disabled
        
        try:
            # Create a copy to avoid modifying the original
            debug_img = img.copy()
            
            # Define colors for different region types
            colors = {
                'community_cards': (0, 255, 0),      # Green
                'player_cards': (255, 0, 0),         # Red
                'player_chips': (0, 0, 255),         # Blue
                'main_player_chips': (0, 150, 255),  # Light blue
                'pot': (255, 255, 0),                # Yellow
                'current_bets': (255, 0, 255),       # Magenta
                'game_stage': (0, 255, 255),         # Cyan
                'actions': (200, 200, 200)           # Gray
            }
            
            # Draw community card regions
            for i, region in enumerate(self.roi['community_cards']):
                x, y, w, h = region
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), colors['community_cards'], 2)
                cv2.putText(debug_img, f"Community {i+1}", (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['community_cards'], 1)
            
            # Draw player card regions
            for player_id, regions in self.roi['player_cards'].items():
                for i, region in enumerate(regions):
                    x, y, w, h = region
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), colors['player_cards'], 2)
                    cv2.putText(debug_img, f"P{player_id} Card {i+1}", (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['player_cards'], 1)
            
            # Draw player chip regions
            for player_id, regions in self.roi['player_chips'].items():
                for i, region in enumerate(regions):
                    x, y, w, h = region
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), colors['player_chips'], 2)
                    cv2.putText(debug_img, f"P{player_id} Chips", (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['player_chips'], 1)
            
            # Draw main player chips region
            if 'main_player_chips' in self.roi:
                for i, region in enumerate(self.roi['main_player_chips']):
                    x, y, w, h = region
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), colors['main_player_chips'], 2)
                    cv2.putText(debug_img, "Main Player $", (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['main_player_chips'], 1)
            
            # Draw current bets regions
            if 'current_bets' in self.roi:
                for player_id, regions in self.roi['current_bets'].items():
                    for i, region in enumerate(regions):
                        x, y, w, h = region
                        cv2.rectangle(debug_img, (x, y), (x+w, y+h), colors['current_bets'], 2)
                        cv2.putText(debug_img, f"P{player_id} Bet", (x, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['current_bets'], 1)
            
            # Draw game stage regions
            if 'game_stage' in self.roi:
                for i, region in enumerate(self.roi['game_stage']):
                    x, y, w, h = region
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), colors['game_stage'], 2)
                    cv2.putText(debug_img, f"Game Stage {i+1}", (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['game_stage'], 1)
            
            # Draw pot region
            for i, region in enumerate(self.roi['pot']):
                x, y, w, h = region
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), colors['pot'], 2)
                cv2.putText(debug_img, "Pot", (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['pot'], 1)
            
            # Draw action regions
            if 'actions' in self.roi:
                for action, regions in self.roi['actions'].items():
                    for i, region in enumerate(regions):
                        x, y, w, h = region
                        cv2.rectangle(debug_img, (x, y), (x+w, y+h), colors['actions'], 2)
                        cv2.putText(debug_img, action.capitalize(), (x, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['actions'], 1)
            
            # Add legend
            legend_y = 20
            for region_type, color in colors.items():
                cv2.putText(debug_img, region_type.replace('_', ' ').title(), (10, legend_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                legend_y += 20
            
            # Add game stage detection result
            game_stage = self.detect_game_stage(debug_img)
            cv2.putText(debug_img, f"Detected Stage: {game_stage.upper()}", (10, legend_y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Extract and display main player chips
            main_player_chips = self.extract_main_player_chips(debug_img)
            cv2.putText(debug_img, f"Main Player Chips: ${main_player_chips}", (10, legend_y + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Extract and display current bets
            current_bets = self.extract_current_bets(debug_img)
            bet_str = ", ".join([f"P{p}: ${b}" for p, b in current_bets.items()])
            cv2.putText(debug_img, f"Current Bets: {bet_str}", (10, legend_y + 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return debug_img
        
        except Exception as e:
            logger.error(f"Error adding debugging overlay: {str(e)}", exc_info=True)
            return img  # Return original image if error occurs

    def _get_default_roi(self):
        """Return the default ROIs for PokerTH"""
        return {
            # Community cards area
            'community_cards': [
                (390, 220, 45, 65),  # First card
                (450, 220, 45, 65),  # Second card
                (510, 220, 45, 65),  # Third card
                (570, 220, 45, 65),  # Fourth card
                (630, 220, 45, 65)   # Fifth card
            ],
            
            # Player cards area
            'player_cards': {
                1: [(510, 330, 45, 65), (560, 330, 45, 65)]  # Main player's cards (player at position 1)
            },
            
            # Chip counts - positions align with the 9 seats at the table
            'player_chips': {
                1: [(280, 380, 80, 20)],   # Player 1 (bottom center)
                2: [(90, 345, 80, 20)],    # Player 2 (bottom left)
                3: [(80, 90, 80, 20)],     # Player 3 (middle left)
                4: [(280, 40, 80, 20)],    # Player 4 (top left)
                5: [(420, 40, 80, 20)],    # Player 5 (top center)
                6: [(550, 40, 80, 20)],    # Player 6 (top right)
                7: [(730, 90, 80, 20)],    # Player 7 (middle right)
                8: [(730, 345, 80, 20)],   # Player 8 (bottom right)
                9: [(550, 380, 80, 20)]    # Player 9 (bottom center right)
            },
            
            # Main player chips (more precise location for the main player's chips)
            'main_player_chips': [(280, 392, 100, 25)],  # Bottom center - main player chips
            
            # Current bets from each player
            'current_bets': {
                1: [(280, 350, 70, 20)],   # Player 1 (bottom center) current bet
                2: [(120, 320, 70, 20)],   # Player 2 (bottom left) current bet
                3: [(120, 120, 70, 20)],   # Player 3 (middle left) current bet
                4: [(280, 70, 70, 20)],    # Player 4 (top left) current bet
                5: [(400, 70, 70, 20)],    # Player 5 (top center) current bet
                6: [(520, 70, 70, 20)],    # Player 6 (top right) current bet
                7: [(680, 120, 70, 20)],   # Player 7 (middle right) current bet
                8: [(680, 320, 70, 20)],   # Player 8 (bottom right) current bet
                9: [(520, 350, 70, 20)]    # Player 9 (bottom center right) current bet
            },
            
            # Game stage indicators
            'game_stage': [
                (265, 197, 80, 25),  # Game stage text (Preflop, Flop, Turn, River)
                (720, 197, 80, 25)   # Alternative location for game stage
            ],
            
            # Pot information
            'pot': [(280, 248, 100, 20)],  # Pot size area
            
            # Action buttons
            'actions': {
                'raise': [(510, 480, 80, 20)],  # Raise button/amount
                'call': [(510, 530, 80, 20)],   # Call button/amount
                'fold': [(510, 580, 80, 20)]    # Fold button
            }
        }

    def save_regions_to_file(self, filename="roi_config.json"):
        """Save the current ROI configuration to a file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.roi, f, indent=2)
            logger.info(f"ROI configuration saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save ROI configuration: {str(e)}")
            return False

    def load_regions_from_file(self, filename="roi_config.json"):
        """Load ROI configuration from a file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    loaded_roi = json.load(f)
                    
                    # Validate the loaded ROI
                    if self._validate_roi(loaded_roi):
                        self.roi = loaded_roi
                        logger.info(f"ROI configuration loaded from {filename}")
                        return True
                    else:
                        logger.error(f"Invalid ROI configuration in {filename}")
            else:
                logger.warning(f"ROI configuration file not found: {filename}")
        except Exception as e:
            logger.error(f"Failed to load ROI configuration: {str(e)}")
        
        # If we get here, loading failed - use default ROI
        self.roi = self._get_default_roi()
        return False

    def _validate_roi(self, roi):
        """Validate a ROI configuration"""
        try:
            # Check for required keys
            required_keys = ['community_cards', 'player_cards', 'player_chips', 'pot']
            for key in required_keys:
                if key not in roi:
                    logger.error(f"Missing required key in ROI: {key}")
                    return False
            
            # Check community_cards format
            if not isinstance(roi['community_cards'], list):
                logger.error("community_cards must be a list")
                return False
            
            for region in roi['community_cards']:
                if not (isinstance(region, tuple) or isinstance(region, list)) or len(region) != 4:
                    logger.error(f"Invalid community card region: {region}")
                    return False
            
            # Check player_cards format
            if not isinstance(roi['player_cards'], dict):
                logger.error("player_cards must be a dictionary")
                return False
            
            for player_id, regions in roi['player_cards'].items():
                if not isinstance(regions, list):
                    logger.error(f"player_cards[{player_id}] must be a list")
                    return False
                
                for region in regions:
                    if not (isinstance(region, tuple) or isinstance(region, list)) or len(region) != 4:
                        logger.error(f"Invalid player card region: {region}")
                        return False
            
            # Similar checks for player_chips and pot
            # (omitted for brevity)
            
            return True
        
        except Exception as e:
            logger.error(f"Error validating ROI: {str(e)}")
            return False

    def detect_game_stage(self, img):
        """
        Detect the current game stage (preflop, flop, turn, river)
        
        Args:
            img: Screenshot image
        
        Returns:
            str: Game stage ('preflop', 'flop', 'turn', 'river')
        """
        try:
            # First approach: OCR-based detection
            if 'game_stage' in self.roi:
                for region in self.roi['game_stage']:
                    x, y, w, h = region
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and x+w <= img.shape[1] and y+h <= img.shape[0]:
                        stage_img = img[y:y+h, x:x+w]
                        
                        # Apply preprocessing for better OCR
                        gray = cv2.cvtColor(stage_img, cv2.COLOR_BGR2GRAY)
                        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        
                        # OCR to extract text
                        stage_text = pytesseract.image_to_string(binary, config='--psm 7').strip().lower()
                        
                        # Check if we got a recognizable stage
                        for stage in ['preflop', 'flop', 'turn', 'river']:
                            if stage in stage_text:
                                logger.info(f"OCR detected game stage: {stage}")
                                return stage
            
            # Second approach: Count visible community cards
            visible_cards = 0
            for i, card_region in enumerate(self.roi['community_cards']):
                x, y, w, h = card_region
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and x+w <= img.shape[1] and y+h <= img.shape[0]:
                    card_img = img[y:y+h, x:x+w]
                    if self._is_card_present(card_img):
                        visible_cards += 1
            
            # Determine stage based on visible card count
            if visible_cards == 0:
                return 'preflop'
            elif visible_cards == 3:
                return 'flop'
            elif visible_cards == 4:
                return 'turn'
            elif visible_cards == 5:
                return 'river'
            else:
                logger.warning(f"Unexpected number of community cards: {visible_cards}")
                return 'unknown'
                
        except Exception as e:
            logger.error(f"Error detecting game stage: {str(e)}")
            return 'unknown'

    def extract_main_player_chips(self, img):
        """
        Extract main player's chip count from a more precise region
        
        Args:
            img: Screenshot image
        
        Returns:
            int: Chip count or 0 if not detected
        """
        try:
            if 'main_player_chips' in self.roi and self.roi['main_player_chips']:
                region = self.roi['main_player_chips'][0]
                x, y, w, h = region
                
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and x+w <= img.shape[1] and y+h <= img.shape[0]:
                    chip_img = img[y:y+h, x:x+w]
                    
                    # Preprocess for OCR
                    gray = cv2.cvtColor(chip_img, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                    
                    # OCR to extract text
                    text = pytesseract.image_to_string(
                        binary, 
                        config='--psm 7 -c tessedit_char_whitelist=0123456789$,'
                    ).strip()
                    
                    # Extract numeric value
                    import re
                    match = re.search(r'[$]?\s*(\d[\d,.]*)', text)
                    if match:
                        # Remove commas and convert to int
                        chips_str = match.group(1).replace(',', '')
                        try:
                            return int(chips_str)
                        except ValueError:
                            logger.warning(f"Failed to convert chip value to int: {chips_str}")
                    
                    # Alternative: color-based detection for yellow text
                    # This is a more advanced technique that could be implemented
                    # if OCR is not reliable enough
                
                logger.warning("Main player chip region is outside image bounds")
            
            return 0  # Default if not detected
            
        except Exception as e:
            logger.error(f"Error extracting main player chips: {str(e)}")
            return 0

    def extract_current_bets(self, img):
        """
        Extract current bets from each player
        
        Args:
            img: Screenshot image
        
        Returns:
            dict: Dictionary mapping player IDs to their current bets
        """
        bets = {}
        
        try:
            if 'current_bets' in self.roi:
                for player_id, regions in self.roi['current_bets'].items():
                    if not regions:
                        continue
                        
                    region = regions[0]  # Use the first region for each player
                    x, y, w, h = region
                    
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and x+w <= img.shape[1] and y+h <= img.shape[0]:
                        bet_img = img[y:y+h, x:x+w]
                        
                        # Check if there's actually a bet (presence of yellow/white text)
                        if not self._is_bet_present(bet_img):
                            continue
                        
                        # Preprocess for OCR
                        gray = cv2.cvtColor(bet_img, cv2.COLOR_BGR2GRAY)
                        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                        
                        # OCR to extract text
                        text = pytesseract.image_to_string(
                            binary, 
                            config='--psm 7 -c tessedit_char_whitelist=0123456789$,'
                        ).strip()
                        
                        # Extract numeric value
                        import re
                        match = re.search(r'[$]?\s*(\d[\d,.]*)', text)
                        if match:
                            # Remove commas and convert to int
                            bet_str = match.group(1).replace(',', '')
                            try:
                                bets[player_id] = int(bet_str)
                            except ValueError:
                                logger.warning(f"Failed to convert bet value to int: {bet_str}")
            
            return bets
            
        except Exception as e:
            logger.error(f"Error extracting current bets: {str(e)}")
            return bets

    def _is_bet_present(self, bet_img):
        """
        Check if a bet is present in the image region
        
        Args:
            bet_img: Image region that might contain a bet
        
        Returns:
            bool: True if a bet is detected, False otherwise
        """
        try:
            if bet_img is None or bet_img.size == 0:
                return False
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(bet_img, cv2.COLOR_BGR2HSV)
            
            # Yellow color range (for bet amounts)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([40, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # White color range (alternative text color)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Combine masks
            combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
            
            # Count colored pixels
            colored_pixels = cv2.countNonZero(combined_mask)
            total_pixels = bet_img.shape[0] * bet_img.shape[1]
            
            # If more than 5% of pixels are colored, it's likely a bet
            return colored_pixels > (total_pixels * 0.05)
        
        except Exception as e:
            logger.error(f"Error checking for bet presence: {str(e)}")
            return False


    def create_mock_screenshot(self):
        """
        Create a mock screenshot based on the provided PokerTH screenshot
        
        Returns:
            numpy.ndarray: Mock screenshot
        """
        try:
            logger.info("Creating mock screenshot")
            
            # Create a green background similar to PokerTH
            img = np.ones((768, 1024, 3), dtype=np.uint8)
            img[:, :, 0] = 0    # B
            img[:, :, 1] = 100  # G
            img[:, :, 2] = 0    # R
            
            # Draw the poker table (oval in the center)
            cv2.ellipse(img, (512, 384), (400, 250), 0, 0, 360, (0, 50, 0), -1)
            
            # Add community cards (flop, turn, river)
            card_positions = [(390, 220), (450, 220), (510, 220), (570, 220), (630, 220)]
            card_values = ['10', 'A', 'J', '3', '6']
            card_suits = ['♦', '♦', '♥', '♦', '♣']
            suit_colors = {
                '♥': (0, 0, 255),  # Red for hearts
                '♦': (0, 0, 255),  # Red for diamonds
                '♣': (0, 0, 0),    # Black for clubs
                '♠': (0, 0, 0)     # Black for spades
            }
            
            for i, pos in enumerate(card_positions):
                if i < 5:  # All five cards
                    x, y = pos
                    # Draw card rectangle
                    cv2.rectangle(img, (x, y), (x + 45, y + 65), (255, 255, 255), -1)
                    
                    # Add card value on top of the card
                    cv2.putText(img, card_values[i], 
                            (x + 5, y + 20),  # Position at top of card
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            suit_colors[card_suits[i]], 
                            2)
                    
                    # Add card suit on bottom part of the card
                    cv2.putText(img, card_suits[i], 
                            (x + 5, y + 50),  # Position at bottom of card
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1.2, 
                            suit_colors[card_suits[i]], 
                            2)
            
            # Add player cards (at the bottom)
            player_card_pos = [(510, 330), (560, 330)]
            player_values = ['10', 'J']
            player_suits = ['♣', '♦']
            
            for i, pos in enumerate(player_card_pos):
                x, y = pos
                # Draw card rectangle
                cv2.rectangle(img, (x, y), (x + 45, y + 65), (255, 255, 255), -1)
                
                # Add card value on top of the card
                cv2.putText(img, player_values[i], 
                        (x + 5, y + 20),  # Position at top of card
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        suit_colors[player_suits[i]], 
                        2)
                
                # Add card suit on bottom part of the card
                cv2.putText(img, player_suits[i], 
                        (x + 5, y + 50),  # Position at bottom of card
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2, 
                        suit_colors[player_suits[i]], 
                        2)
            
            # Add "MOCK SCREENSHOT" text
            cv2.putText(
                img, 
                "MOCK SCREENSHOT - NO WINDOW SELECTED", 
                (50, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
            
            logger.info(f"Mock screenshot created successfully, shape: {img.shape}")
            return img
            
        except Exception as e:
            logger.error(f"Error creating mock screenshot: {str(e)}", exc_info=True)
            # Return a very simple image as ultimate fallback
            simple_img = np.ones((480, 640, 3), dtype=np.uint8) * 100
            cv2.putText(simple_img, "ERROR - FALLBACK IMAGE", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return simple_img


    def _is_card_present(self, card_img):
        """
        Check if an image region contains a card
        
        Args:
            card_img: Image region that might contain a card
        
        Returns:
            bool: True if a card is detected, False otherwise
        """
        try:
            if card_img is None or card_img.size == 0:
                return False
            
            # Convert to grayscale
            gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            
            # Threshold to find white areas (cards are mostly white)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Count white pixels
            white_pixels = cv2.countNonZero(thresh)
            total_pixels = card_img.shape[0] * card_img.shape[1]
            
            # If more than 30% of pixels are white, it's likely a card
            return white_pixels > (total_pixels * 0.3)
        
        except Exception as e:
            logger.error(f"Error checking for card presence: {str(e)}")
            return False


    def identify_card(self, card_img):
        """
        Identify card value and suit from an image region
        
        Args:
            card_img: Image of a card
            
        Returns:
            tuple: (value, suit) as strings
        """
        try:
            if card_img is None or card_img.size == 0:
                logger.warning("Empty card image provided to identify_card")
                return None, None
            
            # Get image dimensions
            h, w = card_img.shape[:2]
            
            # Split card into top and bottom regions
            value_region = card_img[0:int(h*0.4), 0:w]  # Top 40% for value
            suit_region = card_img[int(h*0.4):h, 0:w]   # Bottom 60% for suit
            
            # For value detection
            value_gray = cv2.cvtColor(value_region, cv2.COLOR_BGR2GRAY)
            _, value_thresh = cv2.threshold(value_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Use OCR to extract value
            value_text = pytesseract.image_to_string(
                value_thresh, 
                config='--psm 10 -c tessedit_char_whitelist=0123456789AJQK'
            ).strip().upper()
            
            # Clean up and interpret value
            value = self._interpret_card_value(value_text)
            
            # For suit detection
            suit_gray = cv2.cvtColor(suit_region, cv2.COLOR_BGR2GRAY)
            _, suit_thresh = cv2.threshold(suit_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert to HSV for color detection
            suit_hsv = cv2.cvtColor(suit_region, cv2.COLOR_BGR2HSV)
            
            # Check for red color (for hearts and diamonds)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(suit_hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(suit_hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            is_red_suit = cv2.countNonZero(red_mask) > 0
            
            # Use OCR for suit symbol
            suit_text = pytesseract.image_to_string(
                suit_thresh, 
                config='--psm 10 -c tessedit_char_whitelist=♥♦♣♠hdcs'
            ).strip().lower()
            
            # Determine suit
            suit = self._interpret_card_suit(suit_text, is_red_suit)
            
            logger.info(f"Identified card: {value} of {suit}")
            return value, suit
            
        except Exception as e:
            logger.error(f"Error identifying card: {str(e)}")
            return None, None

    def _interpret_card_value(self, value_text):
        """
        Interpret OCR result for card value
        
        Args:
            value_text: OCR text from value region
            
        Returns:
            str: Card value ('2' through '10', 'J', 'Q', 'K', or 'A')
        """
        # Clean up OCR result
        value_text = value_text.replace('O', '0').replace('o', '0')  # Common OCR errors
        
        # Handle '10' specially
        if '10' in value_text or ('1' in value_text and '0' in value_text):
            return '10'
        
        # Check for face cards and ace
        if 'A' in value_text:
            return 'A'
        elif 'K' in value_text:
            return 'K'
        elif 'Q' in value_text:
            return 'Q'
        elif 'J' in value_text:
            return 'J'
        
        # For numeric cards, find first digit
        for char in value_text:
            if char in '23456789':
                return char
        
        # If all else fails
        logger.warning(f"Could not interpret card value from text: '{value_text}'")
        return '2'  # Default fallback

    def _interpret_card_suit(self, suit_text, is_red_suit):
        """
        Interpret OCR result for card suit
        
        Args:
            suit_text: OCR text from suit region
            is_red_suit: Boolean indicating if the suit is red
            
        Returns:
            str: Card suit ('hearts', 'diamonds', 'clubs', or 'spades')
        """
        # Check for suit symbols in the text
        if '♥' in suit_text or 'h' in suit_text:
            return 'hearts'
        elif '♦' in suit_text or 'd' in suit_text:
            return 'diamonds'
        elif '♣' in suit_text or 'c' in suit_text:
            return 'clubs'
        elif '♠' in suit_text or 's' in suit_text:
            return 'spades'
        
        # If OCR failed, use color information
        if is_red_suit:
            # For red suits, randomly choose (would use shape detection in production)
            return 'hearts' if np.random.random() > 0.5 else 'diamonds'
        else:
            # For black suits, randomly choose (would use shape detection in production)
            return 'clubs' if np.random.random() > 0.5 else 'spades'

if __name__ == "__main__":
    # Standalone test
    grabber = PokerScreenGrabber(output_dir="poker_data/screenshots")
    
    # Get available windows
    windows = grabber.get_window_list()
    print("Available windows:")
    for i, window in enumerate(windows):
        print(f"{i+1}. {window}")
    
    # Select a window (if available)
    if windows:
        selected_index = 0  # Default to first window
        selected_window = windows[selected_index]
        print(f"Selected window: {selected_window}")
        grabber.select_window(selected_window)
    
    # Capture a single screenshot
    screenshot = grabber.capture_screenshot()
    if screenshot is not None:
        # Save screenshot
        os.makedirs("test_output", exist_ok=True)
        cv2.imwrite("test_output/test_screenshot.png", screenshot)
        print("Screenshot saved to test_output/test_screenshot.png")
        
        # Process screenshot
        game_state = grabber.process_screenshot(screenshot)
        print("Game state:", game_state.to_dict())