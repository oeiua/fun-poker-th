import cv2
import numpy as np
import pyautogui
import time
import os
import threading
import queue
import logging
import json
import datetime
import copy
import re
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PokerScreenGrabber")

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

class WindowInfo:
    def __init__(self, title="", handle=None, rect=None):
        self.title = title
        self.handle = handle
        self.rect = rect  # (left, top, right, bottom)
    
    def __str__(self):
        return f"{self.title} ({self.handle})"

class PokerTableDetector:
    """Detect poker table and player positions in screenshots"""
    
    def __init__(self):
        self.logger = logging.getLogger("PokerTableDetector")
        
        # Color ranges for detecting green poker table
        self.table_color_ranges = [
            # Light green
            [(35, 30, 40), (90, 255, 255)],
            # Dark green
            [(35, 30, 20), (90, 255, 120)]
        ]
        
        # Color ranges for detecting card positions (white areas)
        self.card_color_range = [(0, 0, 180), (180, 30, 255)]  # White in HSV
        
        # Cached results to avoid redundant processing
        self.cached_screenshot_hash = None
        self.cached_table_contour = None
        self.cached_player_positions = None
    
    def detect_table(self, img):
        """
        Detect poker table in the image
        
        Args:
            img: Source image
            
        Returns:
            Tuple: (table_contour, table_rect, center_point)
                where table_contour is the largest detected green contour,
                table_rect is (x, y, w, h) of bounding box,
                center_point is (cx, cy) of the table center
        """
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Create masks for each green color range
            masks = []
            for lower, upper in self.table_color_ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                masks.append(mask)
            
            # Combine masks
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            morphed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self.logger.warning("No table contours found")
                return None, None, None
            
            # Find the largest contour by area
            table_contour = max(contours, key=cv2.contourArea)
            
            # Get the bounding rectangle
            x, y, w, h = cv2.boundingRect(table_contour)
            
            # Calculate the center of the table
            moments = cv2.moments(table_contour)
            if moments["m00"] == 0:
                # Fallback to bounding rect center if moments fail
                cx = x + w // 2
                cy = y + h // 2
            else:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            
            return table_contour, (x, y, w, h), (cx, cy)
        
        except Exception as e:
            self.logger.error(f"Error detecting table: {str(e)}")
            return None, None, None
    
    def detect_player_positions(self, img, table_center=None):
        """
        Detect player positions around the poker table
        
        Args:
            img: Source image
            table_center: Optional center point of the table (cx, cy)
            
        Returns:
            List of player position points [(x1, y1), (x2, y2), ...]
        """
        try:
            if table_center is None:
                # Try to detect the table first
                _, _, table_center = self.detect_table(img)
                
                if table_center is None:
                    self.logger.warning("No table center found for player position detection")
                    return []
            
            cx, cy = table_center
            h, w = img.shape[:2]
            
            # Estimate table radius (considering it's an oval)
            table_radius_x = w // 3
            table_radius_y = h // 3
            
            # Define player positions around the table based on common layouts
            # For a 9-player table, positions are typically arranged in a circle
            angles = np.linspace(0, 2*np.pi, 9, endpoint=False)
            
            # Start at bottom center and go clockwise
            angles = (angles + np.pi/2) % (2*np.pi)
            
            # Calculate positions using elliptical coordinates
            player_positions = []
            for angle in angles:
                # Calculate position on table ellipse
                x = cx + int(table_radius_x * 0.8 * np.cos(angle))
                y = cy + int(table_radius_y * 0.8 * np.sin(angle))
                
                # Ensure within image bounds
                x = max(0, min(w-1, x))
                y = max(0, min(h-1, y))
                
                player_positions.append((x, y))
            
            # Try to refine positions by looking for actual player cards
            refined_positions = self._refine_positions_with_cards(img, player_positions)
            
            return refined_positions if refined_positions else player_positions
        
        except Exception as e:
            self.logger.error(f"Error detecting player positions: {str(e)}")
            return []
    
    def _refine_positions_with_cards(self, img, initial_positions, search_radius=50):
        """
        Refine player positions by looking for card-like regions
        
        Args:
            img: Source image
            initial_positions: Initial estimated player positions
            search_radius: Radius around each position to search for cards
            
        Returns:
            Refined list of player positions
        """
        try:
            # Convert to HSV for color detection
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Create mask for white (card color)
            lower, upper = self.card_color_range
            card_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find card contours
            card_contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not card_contours:
                return initial_positions
            
            # Filter contours to find card-like shapes
            card_contours = [c for c in card_contours if self._is_card_like(c)]
            
            if not card_contours:
                return initial_positions
            
            # Refine each position by finding the nearest card-like contour
            refined_positions = []
            
            for pos in initial_positions:
                px, py = pos
                
                # Find contours within search radius
                nearby_contours = []
                for contour in card_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    contour_center_x = x + w // 2
                    contour_center_y = y + h // 2
                    
                    distance = np.sqrt((px - contour_center_x)**2 + (py - contour_center_y)**2)
                    
                    if distance < search_radius:
                        nearby_contours.append((contour, distance))
                
                if nearby_contours:
                    # Use the nearest contour
                    nearest_contour, _ = min(nearby_contours, key=lambda x: x[1])
                    x, y, w, h = cv2.boundingRect(nearest_contour)
                    refined_pos = (x + w // 2, y + h // 2)
                    refined_positions.append(refined_pos)
                else:
                    # Keep original position if no nearby card contours
                    refined_positions.append(pos)
            
            return refined_positions
        
        except Exception as e:
            self.logger.error(f"Error refining positions: {str(e)}")
            return initial_positions
    
    def _is_card_like(self, contour, min_area=100, max_area=5000, aspect_ratio_range=(0.5, 0.9)):
        """Check if a contour has card-like properties"""
        # Get area and perimeter
        area = cv2.contourArea(contour)
        
        # Check area constraints
        if area < min_area or area > max_area:
            return False
        
        # Check aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
            return False
        
        return True

class AutoROICalibrator:
    """Automatically calibrate regions of interest for poker screenshots"""
    
    def __init__(self):
        self.logger = logging.getLogger("AutoROICalibrator")
        self.table_detector = PokerTableDetector()
        
        # Default card dimensions (will be scaled based on detected table)
        self.card_width = 45
        self.card_height = 65
        
        # Default chip stack region dimensions
        self.chip_region_width = 80
        self.chip_region_height = 20
        
        # Default bet region dimensions
        self.bet_region_width = 70
        self.bet_region_height = 20
        
        # Default pot region dimensions
        self.pot_region_width = 100
        self.pot_region_height = 20
    
    def calibrate_roi(self, img):
        """
        Automatically calibrate regions of interest for a poker screenshot
        
        Args:
            img: Source image
            
        Returns:
            dict: Calibrated ROI configuration
        """
        try:
            # Detect the poker table
            table_contour, table_rect, table_center = self.table_detector.detect_table(img)
            
            if table_center is None:
                self.logger.warning("Could not detect poker table, using default ROI")
                return self._get_default_roi()
            
            # Get image dimensions
            h, w = img.shape[:2]
            
            # Calculate scaled dimensions based on table size
            table_x, table_y, table_width, table_height = table_rect
            scale_factor = min(table_width / 800, table_height / 600)
            
            # Scale card dimensions
            card_width = int(self.card_width * scale_factor)
            card_height = int(self.card_height * scale_factor)
            
            # Scale chip region dimensions
            chip_width = int(self.chip_region_width * scale_factor)
            chip_height = int(self.chip_region_height * scale_factor)
            
            # Scale bet region dimensions
            bet_width = int(self.bet_region_width * scale_factor)
            bet_height = int(self.bet_region_height * scale_factor)
            
            # Scale pot region dimensions
            pot_width = int(self.pot_region_width * scale_factor)
            pot_height = int(self.pot_region_height * scale_factor)
            
            # Get table center
            cx, cy = table_center
            
            # Create ROI configuration
            roi = {}
            
            # Calibrate community cards (in the center of the table)
            community_spacing = int(card_width * 1.2)
            first_card_x = cx - (community_spacing * 2)
            roi['community_cards'] = [
                (first_card_x, cy - card_height // 2, card_width, card_height),
                (first_card_x + community_spacing, cy - card_height // 2, card_width, card_height),
                (first_card_x + community_spacing * 2, cy - card_height // 2, card_width, card_height),
                (first_card_x + community_spacing * 3, cy - card_height // 2, card_width, card_height),
                (first_card_x + community_spacing * 4, cy - card_height // 2, card_width, card_height)
            ]
            
            # Detect player positions
            player_positions = self.table_detector.detect_player_positions(img, table_center)
            
            if not player_positions:
                self.logger.warning("Could not detect player positions, using default patterns")
                # Generate positions in a circle around the table
                angles = np.linspace(0, 2*np.pi, 9, endpoint=False)
                angles = (angles + np.pi/2) % (2*np.pi)  # Start at bottom center
                
                table_radius_x = table_width // 2
                table_radius_y = table_height // 2
                
                player_positions = []
                for angle in angles:
                    x = cx + int(table_radius_x * 0.8 * np.cos(angle))
                    y = cy + int(table_radius_y * 0.8 * np.sin(angle))
                    player_positions.append((x, y))
            
            # Calibrate player cards
            roi['player_cards'] = {}
            for i, (px, py) in enumerate(player_positions, 1):
                # For player 1 (main player at bottom center)
                if i == 1:
                    card_spacing = int(card_width * 1.1)
                    roi['player_cards'][i] = [
                        (px - card_spacing // 2, py, card_width, card_height),
                        (px + card_spacing // 2, py, card_width, card_height)
                    ]
            
            # Calibrate player chips
            roi['player_chips'] = {}
            for i, (px, py) in enumerate(player_positions, 1):
                chip_offset_y = int(card_height * 0.8)  # Place chips below cards
                roi['player_chips'][i] = [(px - chip_width // 2, py + chip_offset_y, chip_width, chip_height)]
            
            # Special region for main player's chips (more precise)
            roi['main_player_chips'] = [(
                player_positions[0][0] - chip_width // 2,
                player_positions[0][1] + int(card_height * 1.0),
                chip_width,
                chip_height
            )]
            
            # Calibrate current bets
            roi['current_bets'] = {}
            for i, (px, py) in enumerate(player_positions, 1):
                bet_offset_y = int(card_height * 0.4)  # Place bet above cards
                roi['current_bets'][i] = [(px - bet_width // 2, py - bet_offset_y, bet_width, bet_height)]
            
            # Calibrate pot (in the center, above community cards)
            pot_offset_y = int(card_height * 0.8)
            roi['pot'] = [(cx - pot_width // 2, cy - card_height - pot_offset_y, pot_width, pot_height)]
            
            # Calibrate game stage regions (above community cards)
            game_stage_offset_y = int(card_height * 1.2)
            game_stage_width = int(80 * scale_factor)
            game_stage_height = int(25 * scale_factor)
            
            roi['game_stage'] = [
                (cx - game_stage_width - 10, cy - card_height - game_stage_offset_y, game_stage_width, game_stage_height),
                (cx + 10, cy - card_height - game_stage_offset_y, game_stage_width, game_stage_height)
            ]
            
            # Calibrate action buttons (bottom center, below main player)
            action_width = int(80 * scale_factor)
            action_height = int(20 * scale_factor)
            main_x, main_y = player_positions[0]  # Main player position
            
            action_spacing = int(action_height * 2.5)
            roi['actions'] = {
                'raise': [(main_x, main_y + card_height + chip_height + action_spacing, action_width, action_height)],
                'call': [(main_x, main_y + card_height + chip_height + action_spacing * 2, action_width, action_height)],
                'fold': [(main_x, main_y + card_height + chip_height + action_spacing * 3, action_width, action_height)]
            }
            
            return roi
        
        except Exception as e:
            self.logger.error(f"Error calibrating ROI: {str(e)}")
            return self._get_default_roi()
    
    def calibrate_and_save(self, img, output_file="roi_config.json"):
        """
        Calibrate and save ROI configuration to a file
        
        Args:
            img: Source image
            output_file: Path to save the configuration
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            roi = self.calibrate_roi(img)
            
            # Convert tuple keys to strings for JSON serialization
            json_roi = self._convert_to_json_serializable(roi)
            
            with open(output_file, 'w') as f:
                json.dump(json_roi, f, indent=2)
            
            self.logger.info(f"ROI configuration saved to {output_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error saving ROI configuration: {str(e)}")
            return False
    
    def _convert_to_json_serializable(self, roi):
        """Convert ROI dict to be JSON serializable"""
        # Create a deep copy of the ROI
        result = {}
        
        for key, value in roi.items():
            if isinstance(value, dict):
                # Handle nested dicts (like player_cards, player_chips)
                result[key] = {}
                for subkey, subvalue in value.items():
                    result[key][str(subkey)] = subvalue
            else:
                result[key] = value
        
        return result
    
    def load_and_apply_roi(self, img, roi_file="roi_config.json"):
        """
        Load ROI from file and adjust to current image dimensions
        
        Args:
            img: Source image
            roi_file: Path to ROI configuration file
            
        Returns:
            dict: Adjusted ROI configuration
        """
        try:
            if os.path.exists(roi_file):
                with open(roi_file, 'r') as f:
                    roi = json.load(f)
                
                # Detect the poker table
                table_contour, table_rect, table_center = self.table_detector.detect_table(img)
                
                if table_center is None:
                    self.logger.warning("Could not detect poker table, using loaded ROI without adjustment")
                    return roi
                
                # Get image dimensions
                h, w = img.shape[:2]
                
                # Get table dimensions
                table_x, table_y, table_width, table_height = table_rect
                
                # Adjust the loaded ROI to the current table dimensions
                adjusted_roi = self._adjust_roi_to_current_table(roi, table_center, table_width, table_height, w, h)
                
                return adjusted_roi
            else:
                self.logger.warning(f"ROI file {roi_file} not found, calibrating new ROI")
                return self.calibrate_roi(img)
        
        except Exception as e:
            self.logger.error(f"Error loading and applying ROI: {str(e)}")
            return self._get_default_roi()
    
    def _adjust_roi_to_current_table(self, roi, table_center, table_width, table_height, image_width, image_height):
        """Adjust loaded ROI to current table dimensions"""
        # Get table center
        cx, cy = table_center
        
        # Create a copy of the ROI
        adjusted_roi = {}
        
        # Default reference dimensions
        ref_width = 800
        ref_height = 600
        
        # Calculate scale factors
        scale_x = table_width / ref_width
        scale_y = table_height / ref_height
        
        # Adjust each region type
        for region_type, regions in roi.items():
            if isinstance(regions, list):
                # Simple list of regions (like community_cards)
                adjusted_roi[region_type] = []
                
                for region in regions:
                    x, y, w, h = region
                    
                    # Adjust coordinates relative to table center
                    adj_x = int(cx + (x - ref_width/2) * scale_x)
                    adj_y = int(cy + (y - ref_height/2) * scale_y)
                    adj_w = int(w * scale_x)
                    adj_h = int(h * scale_y)
                    
                    # Ensure within image bounds
                    adj_x = max(0, min(image_width-1, adj_x))
                    adj_y = max(0, min(image_height-1, adj_y))
                    adj_w = min(adj_w, image_width - adj_x)
                    adj_h = min(adj_h, image_height - adj_y)
                    
                    adjusted_roi[region_type].append((adj_x, adj_y, adj_w, adj_h))
            
            elif isinstance(regions, dict):
                # Nested dict (like player_cards, player_chips)
                adjusted_roi[region_type] = {}
                
                for player_id, player_regions in regions.items():
                    # Convert string keys back to integers if needed
                    player_id = int(player_id) if player_id.isdigit() else player_id
                    
                    adjusted_roi[region_type][player_id] = []
                    
                    for region in player_regions:
                        x, y, w, h = region
                        
                        # Adjust coordinates relative to table center
                        adj_x = int(cx + (x - ref_width/2) * scale_x)
                        adj_y = int(cy + (y - ref_height/2) * scale_y)
                        adj_w = int(w * scale_x)
                        adj_h = int(h * scale_y)
                        
                        # Ensure within image bounds
                        adj_x = max(0, min(image_width-1, adj_x))
                        adj_y = max(0, min(image_height-1, adj_y))
                        adj_w = min(adj_w, image_width - adj_x)
                        adj_h = min(adj_h, image_height - adj_y)
                        
                        adjusted_roi[region_type][player_id].append((adj_x, adj_y, adj_w, adj_h))
        
        return adjusted_roi
    
    def _get_default_roi(self):
        """Get default ROI configuration"""
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
                1: [(510, 330, 45, 65), (560, 330, 45, 65)]  # Main player's cards
            },
            
            # Chip counts
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
            
            # Main player chips
            'main_player_chips': [(280, 392, 100, 25)],
            
            # Current bets
            'current_bets': {
                1: [(280, 350, 70, 20)],   # Player 1 current bet
                2: [(120, 320, 70, 20)],   # Player 2 current bet
                3: [(120, 120, 70, 20)],   # Player 3 current bet
                4: [(280, 70, 70, 20)],    # Player 4 current bet
                5: [(400, 70, 70, 20)],    # Player 5 current bet
                6: [(520, 70, 70, 20)],    # Player 6 current bet
                7: [(680, 120, 70, 20)],   # Player 7 current bet
                8: [(680, 320, 70, 20)],   # Player 8 current bet
                9: [(520, 350, 70, 20)]    # Player 9 current bet
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
        
        # Create ROI calibrator
        self.roi_calibrator = AutoROICalibrator()
        
        # Initialize with default ROI
        self.roi = self._get_default_roi()
        
        # Show debugging overlay by default
        self.show_debug_overlay = True
        
        # Load ROI from file if available
        roi_file = "roi_config.json"
        if os.path.exists(roi_file):
            self.load_regions_from_file(roi_file)
        
        # Current game state and debug info
        self.current_state = None
        self.last_detection_info = {}
    
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
                        
                        # Take a screenshot and calibrate ROI
                        screenshot = self.capture_screenshot()
                        if screenshot is not None:
                            self.calibrate_roi_from_screenshot(screenshot)
                        
                        return True
                
                logger.warning(f"Window not found by title: {window_info}")
                return False
            else:
                # WindowInfo object provided
                self.selected_window = window_info.title
                self.window_handle = window_info.handle
                self.window_rect = window_info.rect
                logger.info(f"Selected window by object: {window_info.title}")
                
                # Take a screenshot and calibrate ROI
                screenshot = self.capture_screenshot()
                if screenshot is not None:
                    self.calibrate_roi_from_screenshot(screenshot)
                
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
                        if self.show_debug_overlay:
                            img = self.add_debugging_overlay(img)
                        return img
                    logger.warning("Windows-specific capture failed, falling back to region capture")
                
                if self.window_rect:
                    logger.info(f"Using region capture for rect: {self.window_rect}")
                    img = self.capture_window(rect=self.window_rect)
                    if img is not None:
                        # Add debugging overlay
                        if self.show_debug_overlay:
                            img = self.add_debugging_overlay(img)
                        return img
                    logger.warning("Region capture failed, falling back to mock screenshot")
                
                # If platform-specific capturing failed or isn't available, use mock screenshot
                logger.info("Using mock screenshot as fallback")
                img = self.create_mock_screenshot()
                # Add debugging overlay
                if self.show_debug_overlay:
                    img = self.add_debugging_overlay(img)
                return img
            else:
                logger.warning("No window selected for capture - using mock screenshot")
                img = self.create_mock_screenshot()
                # Add debugging overlay
                if self.show_debug_overlay:
                    img = self.add_debugging_overlay(img)
                return img
        
        except Exception as e:
            logger.error(f"Error in capture_screenshot: {str(e)}", exc_info=True)
            logger.info("Returning mock screenshot due to error")
            img = self.create_mock_screenshot()
            # Add debugging overlay - try/except to ensure we return something even if overlay fails
            try:
                if self.show_debug_overlay:
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
    
    def calibrate_roi_from_screenshot(self, img=None):
        """
        Calibrate ROI from current screenshot
        
        Args:
            img: Screenshot to use for calibration, or None to capture new screenshot
            
        Returns:
            bool: True if calibration was successful
        """
        try:
            if img is None:
                img = self.capture_screenshot()
                
            if img is not None:
                self.roi = self.roi_calibrator.calibrate_roi(img)
                self.save_regions_to_file("roi_config.json")
                logger.info("ROI calibrated successfully")
                return True
            else:
                logger.warning("Failed to capture screenshot for ROI calibration")
                return False
        except Exception as e:
            logger.error(f"Error calibrating ROI: {str(e)}", exc_info=True)
            return False
    
    def save_regions_to_file(self, filename="roi_config.json"):
        """Save the current ROI configuration to a file"""
        try:
            # Convert ROI to a JSON-serializable format
            json_roi = self.roi_calibrator._convert_to_json_serializable(self.roi)
            
            with open(filename, 'w') as f:
                json.dump(json_roi, f, indent=2)
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
                
                # Convert string player IDs to integers
                for key in ['player_cards', 'player_chips', 'current_bets']:
                    if key in loaded_roi:
                        loaded_roi[key] = {int(k) if k.isdigit() else k: v for k, v in loaded_roi[key].items()}
                
                self.roi = loaded_roi
                logger.info(f"ROI configuration loaded from {filename}")
                return True
            else:
                logger.warning(f"ROI configuration file not found: {filename}")
        except Exception as e:
            logger.error(f"Failed to load ROI configuration: {str(e)}")
        
        # If we get here, loading failed - use default ROI
        self.roi = self._get_default_roi()
        return False
    
    def _detect_card(self, img, card_region):
        """
        Detect a card in the given region
        
        Args:
            img: Source image
            card_region: Region tuple (x, y, w, h)
            
        Returns:
            tuple: (value, suit) or (None, None) if no card detected
        """
        # Simple card detection - for real implementation, use a more sophisticated method
        # This is a placeholder for the improved card detection that would be implemented
        try:
            x, y, w, h = card_region
            
            # Ensure region is within image bounds
            if x >= 0 and y >= 0 and x+w <= img.shape[1] and y+h <= img.shape[0]:
                card_img = img[y:y+h, x:x+w]
                
                # Check if a card is present
                if self._is_card_visible(card_img):
                    from improved_poker_cv_analyzer import ImprovedCardDetector
                    detector = ImprovedCardDetector()
                    return detector.detect_card(card_img)
            
            return None, None
        except Exception as e:
            logger.error(f"Error detecting card: {str(e)}")
            return None, None
    
    def _is_card_visible(self, card_img):
        """Check if a card is visible (not folded or hidden)"""
        if card_img is None or card_img.size == 0:
            return False
        
        # Convert to grayscale
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find white areas (cards are mostly white)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Count white pixels
        white_pixels = cv2.countNonZero(thresh)
        total_pixels = card_img.shape[0] * card_img.shape[1]
        
        # If more than 40% of pixels are white, it's likely a card
        return white_pixels > (total_pixels * 0.4)
    
    def _detect_chip_count(self, img, chip_region):
        """
        Detect chip count from the given region
        
        Args:
            img: Source image
            chip_region: Region tuple (x, y, w, h)
            
        Returns:
            int: Detected chip count or 0
        """
        try:
            x, y, w, h = chip_region
            
            # Ensure region is within image bounds
            if x >= 0 and y >= 0 and x+w <= img.shape[1] and y+h <= img.shape[0]:
                chip_img = img[y:y+h, x:x+w]
                
                from improved_poker_cv_analyzer import EnhancedTextRecognition
                text_recognizer = EnhancedTextRecognition()
                return text_recognizer.extract_chip_count(img, chip_region)
            
            return 0
        except Exception as e:
            logger.error(f"Error detecting chip count: {str(e)}")
            return 0
    
    def add_debugging_overlay(self, img):
        """
        Add visual debugging information to the screenshot showing analyzed regions
        
        Args:
            img: Screenshot image to annotate
        
        Returns:
            numpy.ndarray: Annotated screenshot with debugging information
        """
        # Check if debugging overlay is enabled
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
            if 'community_cards' in self.roi:
                for i, region in enumerate(self.roi['community_cards']):
                    x, y, w, h = region
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), colors['community_cards'], 2)
                    cv2.putText(debug_img, f"CC {i+1}", (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['community_cards'], 1)
            
            # Draw player card regions
            if 'player_cards' in self.roi:
                for player_id, regions in self.roi['player_cards'].items():
                    for i, region in enumerate(regions):
                        x, y, w, h = region
                        cv2.rectangle(debug_img, (x, y), (x+w, y+h), colors['player_cards'], 2)
                        cv2.putText(debug_img, f"P{player_id} C{i+1}", (x, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['player_cards'], 1)
            
            # Draw player chip regions
            if 'player_chips' in self.roi:
                for player_id, regions in self.roi['player_chips'].items():
                    for i, region in enumerate(regions):
                        x, y, w, h = region
                        cv2.rectangle(debug_img, (x, y), (x+w, y+h), colors['player_chips'], 2)
                        cv2.putText(debug_img, f"P{player_id} $", (x, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['player_chips'], 1)
            
            # Draw main player chips region
            if 'main_player_chips' in self.roi:
                for i, region in enumerate(self.roi['main_player_chips']):
                    x, y, w, h = region
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), colors['main_player_chips'], 2)
                    cv2.putText(debug_img, "Main $", (x, y-5), 
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
                    cv2.putText(debug_img, f"Stage {i+1}", (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['game_stage'], 1)
            
            # Draw pot region
            if 'pot' in self.roi:
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
                if region_type in self.roi:
                    cv2.putText(debug_img, region_type.replace('_', ' ').title(), (10, legend_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    legend_y += 20
            
            # Add detection info if available
            if self.last_detection_info:
                # Add header for detection info
                cv2.putText(debug_img, "Detection Info:", (10, legend_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                legend_y += 40
                
                # Display up to 3 detection results
                count = 0
                for key, value in self.last_detection_info.items():
                    if count < 3:
                        cv2.putText(debug_img, f"{key}: {value}", (10, legend_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        legend_y += 20
                        count += 1
            
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
            
            # Add player chips
            cv2.putText(img, "$4980", (280, 392), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Add pot
            cv2.putText(img, "$840", (280, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Add game stage
            cv2.putText(img, "FLOP", (265, 197), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
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

# Test function
def test_screen_grabber():
    """Test the screen grabber functionality"""
    grabber = PokerScreenGrabber(output_dir="poker_data/screenshots")
    
    # Get available windows
    windows = grabber.get_window_list()
    print("Available windows:")
    for i, window in enumerate(windows):
        print(f"{i+1}. {window}")
    
    # Select a window (if available)
    if windows:
        try:
            print("\nSelect a window number:")
            window_idx = int(input()) - 1
            
            if 0 <= window_idx < len(windows):
                selected_window = windows[window_idx]
                print(f"Selected window: {selected_window}")
                grabber.select_window(selected_window)
                
                # Capture a screenshot
                screenshot = grabber.capture_screenshot()
                if screenshot is not None:
                    # Save screenshot
                    os.makedirs("test_output", exist_ok=True)
                    cv2.imwrite("test_output/test_screenshot.png", screenshot)
                    print("Screenshot saved to test_output/test_screenshot.png")
                    
                    # Display screenshot
                    cv2.imshow("Screenshot", screenshot)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("Failed to capture screenshot")
            else:
                print("Invalid window selection")
        except ValueError:
            print("Invalid input. Using mock screenshot instead.")
            screenshot = grabber.create_mock_screenshot()
            
            # Save screenshot
            os.makedirs("test_output", exist_ok=True)
            cv2.imwrite("test_output/mock_screenshot.png", screenshot)
            print("Mock screenshot saved to test_output/mock_screenshot.png")
            
            # Display screenshot
            cv2.imshow("Mock Screenshot", screenshot)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("No windows available. Using mock screenshot.")
        screenshot = grabber.create_mock_screenshot()
        
        # Save screenshot
        os.makedirs("test_output", exist_ok=True)
        cv2.imwrite("test_output/mock_screenshot.png", screenshot)
        print("Mock screenshot saved to test_output/mock_screenshot.png")
        
        # Display screenshot
        cv2.imshow("Mock Screenshot", screenshot)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_screen_grabber()