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
from pathlib import Path

# Set up logging with a more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("PokerCV")

# Import platform-specific window handling libraries
try:
    import win32gui

    WINDOWS_PLATFORM = True

    # Test if PrintWindow is available
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
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGWindowListOptionOnScreenOnly,
        kCGNullWindowID,
    )

    MAC_PLATFORM = True
    logger.info("Using macOS-specific window capturing")
except ImportError:
    MAC_PLATFORM = False
    logger.info("macOS-specific libraries not available")


class WindowInfo:
    """Information about a window for capturing"""

    def __init__(self, title="", handle=None, rect=None):
        self.title = title
        self.handle = handle
        self.rect = rect  # (left, top, right, bottom)

    def __str__(self):
        return f"{self.title} ({self.handle})"

    @property
    def width(self):
        """Get window width"""
        if self.rect:
            return self.rect[2] - self.rect[0]
        return 0

    @property
    def height(self):
        """Get window height"""
        if self.rect:
            return self.rect[3] - self.rect[1]
        return 0


class PokerTableDetector:
    """Detect poker table and player positions in screenshots with improved accuracy"""

    def __init__(self):
        self.logger = logging.getLogger("PokerTableDetector")

        # Color ranges for detecting green poker table (HSV format)
        self.table_color_ranges = [
            # Light green
            [(35, 30, 40), (90, 255, 255)],
            # Dark green
            [(35, 30, 20), (90, 255, 120)],
        ]

        # Color ranges for detecting card positions (white areas)
        self.card_color_range = [(0, 0, 180), (180, 30, 255)]  # White in HSV

        # Results caching to improve performance
        self._cache = {
            "last_img_hash": None,
            "table_contour": None,
            "table_rect": None,
            "table_center": None,
            "player_positions": None,
        }

    def _compute_image_hash(self, img):
        """Compute a simple hash of the image for caching purposes"""
        if img is None:
            return None
        # Use the average of downsampled image as a simple hash
        small_img = cv2.resize(img, (32, 32))
        return hash(small_img.mean())

    def detect_table(self, img):
        """
        Detect poker table in the image with caching for improved performance

        Args:
            img: Source image

        Returns:
            Tuple: (table_contour, table_rect, center_point)
        """
        # Check cache first
        img_hash = self._compute_image_hash(img)
        if (
            img_hash
            and img_hash == self._cache["last_img_hash"]
            and self._cache["table_contour"] is not None
        ):
            return (
                self._cache["table_contour"],
                self._cache["table_rect"],
                self._cache["table_center"],
            )

        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Create masks for each green color range and combine them
            combined_mask = None
            for lower, upper in self.table_color_ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            morphed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(
                morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                self.logger.warning("No table contours found")
                return None, None, None

            # Find the largest contour by area (likely to be the poker table)
            table_contour = max(contours, key=cv2.contourArea)

            # Get the bounding rectangle
            x, y, w, h = cv2.boundingRect(table_contour)
            table_rect = (x, y, w, h)

            # Calculate the center of the table
            moments = cv2.moments(table_contour)
            if moments["m00"] == 0:
                # Fallback to bounding rect center if moments fail
                cx = x + w // 2
                cy = y + h // 2
            else:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])

            table_center = (cx, cy)

            # Update cache
            self._cache["last_img_hash"] = img_hash
            self._cache["table_contour"] = table_contour
            self._cache["table_rect"] = table_rect
            self._cache["table_center"] = table_center

            return table_contour, table_rect, table_center

        except Exception as e:
            self.logger.error(f"Error detecting table: {str(e)}", exc_info=True)
            return None, None, None

    def detect_player_positions(self, img, table_center=None):
        """
        Detect player positions around the poker table with caching for improved performance

        Args:
            img: Source image
            table_center: Optional center point of the table (cx, cy)

        Returns:
            List of player position points [(x1, y1), (x2, y2), ...]
        """
        # Check cache first
        img_hash = self._compute_image_hash(img)
        if (
            img_hash
            and img_hash == self._cache["last_img_hash"]
            and self._cache["player_positions"] is not None
        ):
            return self._cache["player_positions"]

        try:
            if table_center is None:
                # Try to detect the table first
                _, _, table_center = self.detect_table(img)

                if table_center is None:
                    self.logger.warning(
                        "No table center found for player position detection"
                    )
                    return []

            cx, cy = table_center
            h, w = img.shape[:2]

            # Estimate table radius (considering it's an oval)
            table_radius_x = w // 3
            table_radius_y = h // 3

            # Define player positions around the table based on common layouts
            # For a 9-player table, positions are typically arranged in a circle
            angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)

            # Start at bottom center and go clockwise
            angles = (angles + np.pi / 2) % (2 * np.pi)

            # Calculate positions using elliptical coordinates
            player_positions = []
            for angle in angles:
                # Calculate position on table ellipse
                x = cx + int(table_radius_x * 0.8 * np.cos(angle))
                y = cy + int(table_radius_y * 0.8 * np.sin(angle))

                # Ensure within image bounds
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))

                player_positions.append((x, y))

            # Try to refine positions by looking for actual player cards
            refined_positions = self._refine_positions_with_cards(img, player_positions)
            result = refined_positions if refined_positions else player_positions

            # Update cache
            self._cache["last_img_hash"] = img_hash
            self._cache["player_positions"] = result

            return result

        except Exception as e:
            self.logger.error(
                f"Error detecting player positions: {str(e)}", exc_info=True
            )
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
            contours, _ = cv2.findContours(
                card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return initial_positions

            # Filter contours to find card-like shapes
            card_contours = [c for c in contours if self._is_card_like(c)]

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

                    distance = np.sqrt(
                        (px - contour_center_x) ** 2 + (py - contour_center_y) ** 2
                    )

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

    def _is_card_like(
        self, contour, min_area=100, max_area=5000, aspect_ratio_range=(0.5, 0.9)
    ):
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
    """Automatically calibrate regions of interest for poker screenshots with improved accuracy"""

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

    def calibrate_roi(self, img, force_recalibrate=False):
        """
        Automatically calibrate regions of interest for a poker screenshot

        Args:
            img: Source image
            force_recalibrate: Force recalibration even if ROI exists

        Returns:
            dict: Calibrated ROI configuration
        """
        # Try loading existing ROI first, unless force_recalibrate is True
        roi_file = "roi_config.json"
        if os.path.exists(roi_file) and not force_recalibrate:
            try:
                with open(roi_file, "r") as f:
                    loaded_roi = json.load(f)

                self.logger.info(f"Loaded existing ROI configuration from {roi_file}")
                return self._convert_string_keys_to_int(loaded_roi)
            except Exception as e:
                self.logger.warning(f"Failed to load existing ROI: {str(e)}")
                # Continue with calibration

        try:
            # Detect the poker table
            table_contour, table_rect, table_center = self.table_detector.detect_table(
                img
            )

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
            roi["community_cards"] = [
                (first_card_x, cy - card_height // 2, card_width, card_height),
                (
                    first_card_x + community_spacing,
                    cy - card_height // 2,
                    card_width,
                    card_height,
                ),
                (
                    first_card_x + community_spacing * 2,
                    cy - card_height // 2,
                    card_width,
                    card_height,
                ),
                (
                    first_card_x + community_spacing * 3,
                    cy - card_height // 2,
                    card_width,
                    card_height,
                ),
                (
                    first_card_x + community_spacing * 4,
                    cy - card_height // 2,
                    card_width,
                    card_height,
                ),
            ]

            # Detect player positions
            player_positions = self.table_detector.detect_player_positions(
                img, table_center
            )

            if not player_positions:
                self.logger.warning(
                    "Could not detect player positions, using default patterns"
                )
                # Generate positions in a circle around the table
                angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
                angles = (angles + np.pi / 2) % (2 * np.pi)  # Start at bottom center

                table_radius_x = table_width // 2
                table_radius_y = table_height // 2

                player_positions = []
                for angle in angles:
                    x = cx + int(table_radius_x * 0.8 * np.cos(angle))
                    y = cy + int(table_radius_y * 0.8 * np.sin(angle))
                    player_positions.append((x, y))

            # Calibrate player cards
            roi["player_cards"] = {}
            for i, (px, py) in enumerate(player_positions, 1):
                # For player 1 (main player at bottom center)
                if i == 1:
                    card_spacing = int(card_width * 1.1)
                    roi["player_cards"][i] = [
                        (px - card_spacing // 2, py, card_width, card_height),
                        (px + card_spacing // 2, py, card_width, card_height),
                    ]

            # Calibrate player chips
            roi["player_chips"] = {}
            for i, (px, py) in enumerate(player_positions, 1):
                chip_offset_y = int(card_height * 0.8)  # Place chips below cards
                roi["player_chips"][i] = [
                    (px - chip_width // 2, py + chip_offset_y, chip_width, chip_height)
                ]

            # Special region for main player's chips (more precise)
            roi["main_player_chips"] = [
                (
                    player_positions[0][0] - chip_width // 2,
                    player_positions[0][1] + int(card_height * 1.0),
                    chip_width,
                    chip_height,
                )
            ]

            # Calibrate current bets
            roi["current_bets"] = {}
            for i, (px, py) in enumerate(player_positions, 1):
                bet_offset_y = int(card_height * 0.4)  # Place bet above cards
                roi["current_bets"][i] = [
                    (px - bet_width // 2, py - bet_offset_y, bet_width, bet_height)
                ]

            # Calibrate pot (in the center, above community cards)
            pot_offset_y = int(card_height * 0.8)
            roi["pot"] = [
                (
                    cx - pot_width // 2,
                    cy - card_height - pot_offset_y,
                    pot_width,
                    pot_height,
                )
            ]

            # Calibrate game stage regions (above community cards)
            game_stage_offset_y = int(card_height * 1.2)
            game_stage_width = int(80 * scale_factor)
            game_stage_height = int(25 * scale_factor)

            roi["game_stage"] = [
                (
                    cx - game_stage_width - 10,
                    cy - card_height - game_stage_offset_y,
                    game_stage_width,
                    game_stage_height,
                ),
                (
                    cx + 10,
                    cy - card_height - game_stage_offset_y,
                    game_stage_width,
                    game_stage_height,
                ),
            ]

            # Calibrate action buttons (bottom center, below main player)
            action_width = int(80 * scale_factor)
            action_height = int(20 * scale_factor)
            main_x, main_y = player_positions[0]  # Main player position

            action_spacing = int(action_height * 2.5)
            roi["actions"] = {
                "raise": [
                    (
                        main_x,
                        main_y + card_height + chip_height + action_spacing,
                        action_width,
                        action_height,
                    )
                ],
                "call": [
                    (
                        main_x,
                        main_y + card_height + chip_height + action_spacing * 2,
                        action_width,
                        action_height,
                    )
                ],
                "fold": [
                    (
                        main_x,
                        main_y + card_height + chip_height + action_spacing * 3,
                        action_width,
                        action_height,
                    )
                ],
            }

            # Save calibrated ROI for future use
            self.save_roi_to_file(roi, roi_file)

            return roi

        except Exception as e:
            self.logger.error(f"Error calibrating ROI: {str(e)}", exc_info=True)
            return self._get_default_roi()

    def save_roi_to_file(self, roi, output_file="roi_config.json"):
        """Save ROI configuration to a file"""
        try:
            # Convert tuple keys to strings for JSON serialization
            json_roi = self._convert_to_json_serializable(roi)

            with open(output_file, "w") as f:
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

    def _convert_string_keys_to_int(self, roi):
        """Convert string keys back to integers in the loaded ROI"""
        for key in ["player_cards", "player_chips", "current_bets"]:
            if key in roi:
                roi[key] = {
                    int(k) if k.isdigit() else k: v for k, v in roi[key].items()
                }
        return roi

    def _get_default_roi(self):
        """Get default ROI configuration"""
        return {
            # Community cards area
            "community_cards": [
                (390, 220, 45, 65),  # First card
                (450, 220, 45, 65),  # Second card
                (510, 220, 45, 65),  # Third card
                (570, 220, 45, 65),  # Fourth card
                (630, 220, 45, 65),  # Fifth card
            ],
            # Player cards area
            "player_cards": {
                1: [(510, 330, 45, 65), (560, 330, 45, 65)]  # Main player's cards
            },
            # Chip counts
            "player_chips": {
                1: [(280, 380, 80, 20)],  # Player 1 (bottom center)
                2: [(90, 345, 80, 20)],  # Player 2 (bottom left)
                3: [(80, 90, 80, 20)],  # Player 3 (middle left)
                4: [(280, 40, 80, 20)],  # Player 4 (top left)
                5: [(420, 40, 80, 20)],  # Player 5 (top center)
                6: [(550, 40, 80, 20)],  # Player 6 (top right)
                7: [(730, 90, 80, 20)],  # Player 7 (middle right)
                8: [(730, 345, 80, 20)],  # Player 8 (bottom right)
                9: [(550, 380, 80, 20)],  # Player 9 (bottom center right)
            },
            # Main player chips
            "main_player_chips": [(280, 392, 100, 25)],
            # Current bets
            "current_bets": {
                1: [(280, 350, 70, 20)],  # Player 1 current bet
                2: [(120, 320, 70, 20)],  # Player 2 current bet
                3: [(120, 120, 70, 20)],  # Player 3 current bet
                4: [(280, 70, 70, 20)],  # Player 4 current bet
                5: [(400, 70, 70, 20)],  # Player 5 current bet
                6: [(520, 70, 70, 20)],  # Player 6 current bet
                7: [(680, 120, 70, 20)],  # Player 7 current bet
                8: [(680, 320, 70, 20)],  # Player 8 current bet
                9: [(520, 350, 70, 20)],  # Player 9 current bet
            },
            # Game stage indicators
            "game_stage": [
                (265, 197, 80, 25),  # Game stage text (Preflop, Flop, Turn, River)
                (720, 197, 80, 25),  # Alternative location for game stage
            ],
            # Pot information
            "pot": [(280, 248, 100, 20)],  # Pot size area
            # Action buttons
            "actions": {
                "raise": [(510, 480, 80, 20)],  # Raise button/amount
                "call": [(510, 530, 80, 20)],  # Call button/amount
                "fold": [(510, 580, 80, 20)],  # Fold button
            },
        }


class PokerScreenGrabber:
    """Enhanced screen grabber for poker gameplay with optimized performance"""

    def __init__(self, capture_interval=2.0, output_dir="poker_data"):
        # Settings
        self.capture_interval = capture_interval
        self.output_dir = output_dir
        self.is_capturing = False
        self.capture_thread = None
        self.selected_window = None
        self.window_handle = None
        self.window_rect = None
        self.logger = logging.getLogger("PokerScreenGrabber")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create ROI calibrator
        self.roi_calibrator = AutoROICalibrator()

        # Initialize with default ROI
        self.roi = self.roi_calibrator._get_default_roi()

        # Show debugging overlay by default
        self.show_debug_overlay = True

        # Load ROI from file if available
        roi_file = "roi_config.json"
        if os.path.exists(roi_file):
            self.load_regions_from_file(roi_file)

        # Current game state and debug info
        self.current_state = None
        self.last_detection_info = {}

        # Screenshot cache to avoid redundant processing - updated to store both original and overlay versions
        self._screenshot_cache = {
            "last_time": 0,
            "screenshot_original": None,  # Original screenshot without overlay
            "screenshot_with_overlay": None,  # Screenshot with debug overlay
            "path": None,
        }

    def get_window_list(self):
        """Get a list of all visible windows with improved error handling"""
        windows = []

        if WINDOWS_PLATFORM:
            try:

                def enum_windows_callback(hwnd, results):
                    if win32gui.IsWindowVisible(hwnd):
                        try:
                            window_title = win32gui.GetWindowText(hwnd)
                            if window_title and window_title != "Program Manager":
                                rect = win32gui.GetWindowRect(hwnd)
                                windows.append(WindowInfo(window_title, hwnd, rect))
                        except Exception as e:
                            self.logger.debug(f"Error getting window info: {str(e)}")
                    return True

                win32gui.EnumWindows(enum_windows_callback, [])
            except Exception as e:
                self.logger.error(f"Error enumerating windows: {str(e)}")

        elif MAC_PLATFORM:
            try:
                window_list = CGWindowListCopyWindowInfo(
                    kCGWindowListOptionOnScreenOnly, kCGNullWindowID
                )
                for window in window_list:
                    window_title = window.get("kCGWindowOwnerName", "")
                    if window_title:
                        bounds = window.get("kCGWindowBounds", {})
                        if bounds:
                            rect = (
                                bounds["X"],
                                bounds["Y"],
                                bounds["X"] + bounds["Width"],
                                bounds["Y"] + bounds["Height"],
                            )
                            windows.append(
                                WindowInfo(
                                    window_title, window.get("kCGWindowNumber"), rect
                                )
                            )
            except Exception as e:
                self.logger.error(f"Error getting macOS windows: {str(e)}")

        else:
            # Fallback - just use mock windows for non-Windows/macOS platforms
            self.logger.info("Using mock windows list for unsupported platform")
            windows.append(WindowInfo("PokerTH", None, (0, 0, 1024, 768)))
            windows.append(WindowInfo("Other Window"))

        self.logger.info(f"Found {len(windows)} windows")
        return windows

    def select_window(self, window_info):
        """
        Select a window to capture with improved error handling

        Args:
            window_info: WindowInfo object or window title

        Returns:
            bool: True if the window was found and selected, False otherwise
        """
        try:
            self.logger.info(f"Selecting window: {window_info}")

            if isinstance(window_info, str):
                # Find window by title
                for window in self.get_window_list():
                    if window_info.lower() in window.title.lower():
                        self.selected_window = window.title
                        self.window_handle = window.handle
                        self.window_rect = window.rect
                        self.logger.info(f"Selected window by title: {window.title}")
                        return True

                self.logger.warning(f"Window not found by title: {window_info}")
                return False
            else:
                # WindowInfo object provided
                self.selected_window = window_info.title
                self.window_handle = window_info.handle
                self.window_rect = window_info.rect
                self.logger.info(f"Selected window by object: {window_info.title}")
                return True
        except Exception as e:
            self.logger.error(f"Error selecting window: {str(e)}", exc_info=True)
            return False

    def capture_window(self, hwnd=None, rect=None):
        """
        Capture a specific window with optimized implementation

        Args:
            hwnd: Window handle (Windows only)
            rect: Window rectangle (left, top, right, bottom)

        Returns:
            numpy.ndarray: Screenshot of the window or None on failure
        """
        try:
            # Use the most efficient capture method available
            if WINDOWS_PLATFORM and hwnd:
                try:
                    # Get window dimensions
                    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                    width = right - left
                    height = bottom - top

                    # Capture screen region using pyautogui (more reliable than PrintWindow)
                    screenshot = pyautogui.screenshot(region=(left, top, width, height))
                    img = np.array(screenshot)

                    # Convert from RGB to BGR for OpenCV
                    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    self.logger.warning(
                        f"Windows capture failed: {str(e)}, trying fallback"
                    )

            # Fallback: Use direct region capture
            if rect:
                try:
                    left, top, right, bottom = rect
                    width = right - left
                    height = bottom - top

                    screenshot = pyautogui.screenshot(region=(left, top, width, height))
                    img = np.array(screenshot)

                    # Convert from RGB to BGR for OpenCV
                    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    self.logger.warning(
                        f"Region capture failed: {str(e)}, trying fallback"
                    )

            # Final fallback: Use mock screenshot
            self.logger.warning("All capture methods failed, using mock screenshot")
            return self.create_mock_screenshot()

        except Exception as e:
            self.logger.error(f"Error capturing window: {str(e)}", exc_info=True)
            return self.create_mock_screenshot()

    def capture_screenshot(self, use_cache=True, max_cache_age=1.0, with_overlay=True):
        """
        Capture a screenshot of the selected window with caching for improved performance

        Args:
            use_cache: Whether to use the cached screenshot if available
            max_cache_age: Maximum age of the cached screenshot in seconds
            with_overlay: Whether to apply debugging overlay to the returned image

        Returns:
            numpy.ndarray: Screenshot image (with or without overlays)
        """
        try:
            current_time = time.time()

            # Check if we can use the cached screenshot
            if (
                use_cache
                and self._screenshot_cache["screenshot_original"] is not None
                and current_time - self._screenshot_cache["last_time"] < max_cache_age
            ):
                self.logger.debug("Using cached screenshot")
                # Return either original or overlay version
                if with_overlay and self.show_debug_overlay:
                    return self._screenshot_cache["screenshot_with_overlay"]
                else:
                    return self._screenshot_cache["screenshot_original"]

            self.logger.info("Capturing new screenshot")

            # Check if we have window information
            if self.selected_window:
                img = None

                # Try platform-specific window capture first
                if WINDOWS_PLATFORM and self.window_handle:
                    img = self.capture_window(hwnd=self.window_handle)

                # Fall back to region capture if needed
                if img is None and self.window_rect:
                    img = self.capture_window(rect=self.window_rect)

                # Fall back to mock screenshot as a last resort
                if img is None:
                    img = self.create_mock_screenshot()

                # Store the original image
                original_img = img.copy()

                # Create version with overlay if needed
                img_with_overlay = original_img.copy()
                if self.show_debug_overlay:
                    img_with_overlay = self.add_debugging_overlay(img_with_overlay)

                # Update cache with both versions
                self._screenshot_cache["screenshot_original"] = original_img
                self._screenshot_cache["screenshot_with_overlay"] = img_with_overlay
                self._screenshot_cache["last_time"] = current_time

                # Return the appropriate version
                if with_overlay and self.show_debug_overlay:
                    return img_with_overlay
                else:
                    return original_img
            else:
                self.logger.warning(
                    "No window selected for capture - using mock screenshot"
                )
                img = self.create_mock_screenshot()

                # Store the original image
                original_img = img.copy()

                # Create version with overlay if needed
                img_with_overlay = original_img.copy()
                if self.show_debug_overlay:
                    img_with_overlay = self.add_debugging_overlay(img_with_overlay)

                # Update cache with both versions
                self._screenshot_cache["screenshot_original"] = original_img
                self._screenshot_cache["screenshot_with_overlay"] = img_with_overlay
                self._screenshot_cache["last_time"] = current_time

                # Return the appropriate version
                if with_overlay and self.show_debug_overlay:
                    return img_with_overlay
                else:
                    return original_img

        except Exception as e:
            self.logger.error(f"Error capturing screenshot: {str(e)}", exc_info=True)
            img = self.create_mock_screenshot()

            # Store the original image
            original_img = img.copy()

            # Create version with overlay if needed
            img_with_overlay = original_img.copy()
            try:
                if self.show_debug_overlay:
                    img_with_overlay = self.add_debugging_overlay(img_with_overlay)
            except:
                pass

            # Update cache with both versions
            self._screenshot_cache["screenshot_original"] = original_img
            self._screenshot_cache["screenshot_with_overlay"] = img_with_overlay
            self._screenshot_cache["last_time"] = current_time

            # Return the appropriate version
            if with_overlay and self.show_debug_overlay:
                return img_with_overlay
            else:
                return original_img

    def save_screenshot(self, img, filepath):
        """
        Save a screenshot to a file with error handling.
        Always saves the original image without overlays.

        Args:
            img: Screenshot image as numpy array (can be with or without overlay)
            filepath: Path to save the screenshot
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Always save the original image without overlays
            if (
                "screenshot_original" in self._screenshot_cache
                and self._screenshot_cache["screenshot_original"] is not None
            ):
                # Save the cached original
                cv2.imwrite(filepath, self._screenshot_cache["screenshot_original"])
            else:
                # If no cached original, save the provided image
                # This is a fallback and might include overlays
                cv2.imwrite(filepath, img)

            self.logger.info(f"Screenshot saved to {filepath}")

            # Update cache path
            self._screenshot_cache["path"] = filepath

            return True
        except Exception as e:
            self.logger.error(f"Error saving screenshot: {str(e)}")
            return False

    def calibrate_roi_from_screenshot(self, img=None, force_calibrate=False):
        """Calibrate ROI from current screenshot with improved error handling"""
        try:
            if img is None:
                img = self.capture_screenshot(use_cache=False)

            if img is not None:
                self.roi = self.roi_calibrator.calibrate_roi(
                    img, force_recalibrate=force_calibrate
                )
                return True
            else:
                self.logger.warning("Failed to capture screenshot for ROI calibration")
                return False
        except Exception as e:
            self.logger.error(f"Error calibrating ROI: {str(e)}", exc_info=True)
            return False

    def load_regions_from_file(self, filename="roi_config.json"):
        """Load ROI configuration from a file with improved error handling"""
        try:
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    loaded_roi = json.load(f)

                # Convert string player IDs to integers
                for key in ["player_cards", "player_chips", "current_bets"]:
                    if key in loaded_roi:
                        loaded_roi[key] = {
                            int(k) if k.isdigit() else k: v
                            for k, v in loaded_roi[key].items()
                        }

                self.roi = loaded_roi
                self.logger.info(f"ROI configuration loaded from {filename}")
                return True
            else:
                self.logger.warning(f"ROI configuration file not found: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to load ROI configuration: {str(e)}")

        # If we get here, loading failed - use default ROI
        self.roi = self.roi_calibrator._get_default_roi()
        return False

    def save_regions_to_file(self, filename="roi_config.json"):
        """Save the current ROI configuration to a file"""
        return self.roi_calibrator.save_roi_to_file(self.roi, filename)

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
                "community_cards": (0, 255, 0),  # Green
                "player_cards": (255, 0, 0),  # Red
                "player_chips": (0, 0, 255),  # Blue
                "main_player_chips": (0, 150, 255),  # Light blue
                "pot": (255, 255, 0),  # Yellow
                "current_bets": (255, 0, 255),  # Magenta
                "game_stage": (0, 255, 255),  # Cyan
                "actions": (200, 200, 200),  # Gray
            }

            # Draw regions on the image
            for region_type, regions in self.roi.items():
                color = colors.get(region_type, (255, 255, 255))

                if isinstance(regions, list):
                    # Simple list of regions
                    for i, region in enumerate(regions):
                        x, y, w, h = region
                        if self._is_valid_region(region, debug_img.shape):
                            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(
                                debug_img,
                                f"{region_type.replace('_', ' ')} {i}",
                                (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                1,
                            )

                elif isinstance(regions, dict):
                    # Nested dict (player cards, chips, etc.)
                    for key, key_regions in regions.items():
                        for i, region in enumerate(key_regions):
                            x, y, w, h = region
                            if self._is_valid_region(region, debug_img.shape):
                                cv2.rectangle(
                                    debug_img, (x, y), (x + w, y + h), color, 2
                                )
                                cv2.putText(
                                    debug_img,
                                    f"{region_type.replace('_', ' ')} {key}",
                                    (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    1,
                                )

            # Add legend
            legend_y = 20
            for region_type, color in colors.items():
                if region_type in self.roi:
                    cv2.putText(
                        debug_img,
                        region_type.replace("_", " ").title(),
                        (10, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
                    legend_y += 20

            # Add image dimensions and timestamp
            h, w = debug_img.shape[:2]
            time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                debug_img,
                f"Image: {w}x{h} - {time_str}",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            return debug_img

        except Exception as e:
            self.logger.error(
                f"Error adding debugging overlay: {str(e)}", exc_info=True
            )
            return img  # Return original image if overlay fails

    def _is_valid_region(self, region, img_shape):
        """Check if a region is valid within the image bounds"""
        x, y, w, h = region
        h_img, w_img = img_shape[:2]
        return x >= 0 and y >= 0 and x + w <= w_img and y + h <= h_img

    def create_mock_screenshot(self):
        """
        Create a mock screenshot for testing or when window capture fails

        Returns:
            numpy.ndarray: Mock screenshot
        """
        try:
            # Create a green background similar to PokerTH
            img = np.ones((768, 1024, 3), dtype=np.uint8)
            img[:, :, 0] = 0  # B
            img[:, :, 1] = 100  # G
            img[:, :, 2] = 0  # R

            # Draw the poker table (oval in the center)
            cv2.ellipse(img, (512, 384), (400, 250), 0, 0, 360, (0, 50, 0), -1)

            # Add community cards (flop, turn, river)
            card_positions = [
                (390, 220),
                (450, 220),
                (510, 220),
                (570, 220),
                (630, 220),
            ]
            card_values = ["10", "A", "J", "3", "6"]
            card_suits = ["♦", "♦", "♥", "♦", "♣"]
            suit_colors = {
                "♥": (0, 0, 255),  # Red for hearts
                "♦": (0, 0, 255),  # Red for diamonds
                "♣": (0, 0, 0),  # Black for clubs
                "♠": (0, 0, 0),  # Black for spades
            }

            for i, pos in enumerate(card_positions):
                if i < 3:  # Just show 3 community cards (flop)
                    x, y = pos
                    # Draw card rectangle
                    cv2.rectangle(img, (x, y), (x + 45, y + 65), (255, 255, 255), -1)

                    # Add card value and suit
                    cv2.putText(
                        img,
                        card_values[i],
                        (x + 5, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        suit_colors[card_suits[i]],
                        2,
                    )

                    cv2.putText(
                        img,
                        card_suits[i],
                        (x + 5, y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        suit_colors[card_suits[i]],
                        2,
                    )

            # Add player cards (at the bottom)
            player_card_pos = [(510, 330), (560, 330)]
            player_values = ["A", "K"]
            player_suits = ["♥", "♥"]

            for i, pos in enumerate(player_card_pos):
                x, y = pos
                # Draw card rectangle
                cv2.rectangle(img, (x, y), (x + 45, y + 65), (255, 255, 255), -1)

                # Add card value and suit
                cv2.putText(
                    img,
                    player_values[i],
                    (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    suit_colors[player_suits[i]],
                    2,
                )

                cv2.putText(
                    img,
                    player_suits[i],
                    (x + 5, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    suit_colors[player_suits[i]],
                    2,
                )

            # Add player chips
            cv2.putText(
                img,
                "$4980",
                (280, 392),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            # Add pot
            cv2.putText(
                img, "$840", (280, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )

            # Add game stage
            cv2.putText(
                img,
                "FLOP",
                (265, 197),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Add a watermark
            cv2.putText(
                img,
                "MOCK SCREENSHOT - NO WINDOW SELECTED",
                (50, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            # Add timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                img,
                timestamp,
                (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            return img

        except Exception as e:
            self.logger.error(
                f"Error creating mock screenshot: {str(e)}", exc_info=True
            )
            # Return a very simple fallback image
            simple_img = np.ones((480, 640, 3), dtype=np.uint8) * 100
            cv2.putText(
                simple_img,
                "ERROR - FALLBACK IMAGE",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            return simple_img


# Test function
def test_screen_grabber():
    """Test the screen grabber functionality"""
    print("Testing PokerScreenGrabber...")

    # Create screen grabber
    grabber = PokerScreenGrabber(output_dir="poker_data/screenshots")

    # Get available windows
    windows = grabber.get_window_list()
    print(f"Found {len(windows)} windows:")
    for i, window in enumerate(windows):
        print(f"{i+1}. {window}")

    # Select a window or use mock
    if windows:
        try:
            print("\nSelect a window number (or 0 for mock screenshot):")
            window_idx = int(input())

            if window_idx == 0:
                print("Using mock screenshot")
                screenshot = grabber.create_mock_screenshot()
            elif 0 < window_idx <= len(windows):
                selected_window = windows[window_idx - 1]
                print(f"Selected window: {selected_window}")
                grabber.select_window(selected_window)

                # Capture screenshot
                screenshot = grabber.capture_screenshot()
            else:
                print("Invalid selection, using mock screenshot")
                screenshot = grabber.create_mock_screenshot()
        except ValueError:
            print("Invalid input, using mock screenshot")
            screenshot = grabber.create_mock_screenshot()
    else:
        print("No windows available, using mock screenshot")
        screenshot = grabber.create_mock_screenshot()

    # Save and display the screenshot
    if screenshot is not None:
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_dir}/test_screenshot_{timestamp}.png"
        cv2.imwrite(output_path, screenshot)
        print(f"Screenshot saved to {output_path}")

        # Display the image using OpenCV
        cv2.imshow("Screenshot", screenshot)
        print("Press any key to close the image")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to capture screenshot")


if __name__ == "__main__":
    test_screen_grabber()
