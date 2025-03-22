import cv2
import numpy as np
import os
import logging
import json
import time
from pathlib import Path
from functools import lru_cache
import numpy as np  # Make sure numpy is imported for np.bool_ type

# Import our improved components
from card_detector import ImprovedCardDetector, EnhancedTextRecognition

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PokerAnalyzer")

class OptimizedPokerAnalyzer:
    """Analyze poker screenshots with optimized performance and accuracy"""
    
    def __init__(self, template_dir="card_templates", model_path="card_model.h5"):
        self.card_detector = ImprovedCardDetector(template_dir=template_dir)
        self.card_detector.debug_mode = True

        self.text_recognizer = EnhancedTextRecognition()
        self.logger = logging.getLogger("PokerAnalyzer")
        
        # Analysis cache to avoid redundant processing
        self._analysis_cache = {}
        
        # Default ROI configuration
        self.roi = self._get_default_roi()
        
        # Load ROI from file if available
        roi_file = "roi_config.json"
        if os.path.exists(roi_file):
            self.load_roi_from_file(roi_file)
    
    def _get_default_roi(self):
        """Get default regions of interest for analysis"""
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
            "main_player_chips": [(280, 392, 100, 25)],
            
            # Current bets
            "current_bets": {
                1: [(280, 350, 70, 20)],  # Player 1 current bet
                2: [(120, 320, 70, 20)],  # Player 2 current bet
                3: [(120, 120, 70, 20)],  # Player 3 current bet
                4: [(280, 70, 70, 20)],   # Player 4 current bet
                5: [(400, 70, 70, 20)],   # Player 5 current bet
                6: [(520, 70, 70, 20)],   # Player 6 current bet
                7: [(680, 120, 70, 20)],  # Player 7 current bet
                8: [(680, 320, 70, 20)],  # Player 8 current bet
                9: [(520, 350, 70, 20)],  # Player 9 current bet
            },
            
            # Pot information
            'pot': [(280, 248, 100, 20)],  # Pot size area
            
            # Game stage indicators
            'game_stage': [
                (265, 197, 80, 25),  # Game stage text (Preflop, Flop, Turn, River)
                (720, 197, 80, 25)   # Alternative location for game stage
            ],
            
            # Action buttons
            "actions": {
                "raise": [(510, 480, 80, 20)],  # Raise button/amount
                "call": [(510, 530, 80, 20)],   # Call button/amount
                "fold": [(510, 580, 80, 20)],   # Fold button
            },
        }
    
    def load_roi_from_file(self, filename="roi_config.json"):
        """Load regions of interest from a file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    loaded_roi = json.load(f)
                
                # Convert string player IDs to integers
                for key in ['player_cards', 'player_chips', 'current_bets']:
                    if key in loaded_roi:
                        loaded_roi[key] = {int(k) if k.isdigit() else k: v for k, v in loaded_roi[key].items()}
                
                self.roi = loaded_roi
                self.logger.info(f"ROI configuration loaded from {filename}")
                return True
            else:
                self.logger.warning(f"ROI configuration file not found: {filename}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to load ROI configuration: {str(e)}")
            return False
    
    @lru_cache(maxsize=16)
    def _compute_image_hash(self, image_path):
        """Compute a hash of the image for caching"""
        if not os.path.exists(image_path):
            return None
            
        # Use file size and modified time for a quick hash
        file_stat = os.stat(image_path)
        return hash((file_stat.st_size, file_stat.st_mtime))
    
    def _ensure_json_serializable(self, obj):
        """
        Ensure all values in a nested structure are JSON serializable
        
        Args:
            obj: Any object (dict, list, etc.)
            
        Returns:
            JSON serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._ensure_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, (bool, np.bool_)):
            # Explicitly convert any boolean type to Python's built-in bool
            return bool(obj)
        elif isinstance(obj, (int, float, str, type(None))):
            return obj
        else:
            # Convert any other type to string for safety
            return str(obj)
            
    def analyze_image(self, image_path):
        """
        Analyze a poker table screenshot and extract game state with caching for improved performance
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Game state information
        """
        # Check if image exists
        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found: {image_path}")
            return self._create_default_game_state()
        
        # Check cache first to avoid redundant processing
        img_hash = self._compute_image_hash(image_path)
        if img_hash in self._analysis_cache:
            self.logger.info(f"Using cached analysis for {image_path}")
            return self._analysis_cache[img_hash]
        
        try:
            # Load the image
            start_time = time.time()
            original_img = cv2.imread(image_path)
            
            # Check if the image was loaded successfully
            if original_img is None or original_img.size == 0:
                self.logger.error(f"Failed to load image or empty image: {image_path}")
                return self._create_default_game_state()
            
            # Make a working copy of the image for visualization
            # This ensures we keep the original image clean for card extraction
            img = original_img.copy()
            
            # Create game state dictionary
            game_state = {
                'community_cards': [],
                'players': {},
                'pot': 0,
                'game_stage': 'preflop',
                'available_actions': {}  # New field for available actions
            }
            
            # Detect game stage first to know how many community cards to expect
            game_stage, visible_cards = self._detect_game_stage(original_img)
            game_state['game_stage'] = game_stage
            
            # Extract community cards - use the original image, not the working copy
            for i, card_region in enumerate(self.roi['community_cards']):
                if i < visible_cards:
                    self._extract_card(original_img, card_region, game_state['community_cards'])
            
            # Extract player information - use the original image, not the working copy
            for player_id, card_regions in self.roi.get('player_cards', {}).items():
                # Check if player is active
                if self._is_player_active(original_img, player_id):
                    player_cards = []
                    for card_region in card_regions:
                        self._extract_card(original_img, card_region, player_cards)
                    
                    # Get player's chip count - FIXED to prioritize main_player_chips for player 1
                    if player_id == 1 and 'main_player_chips' in self.roi and self.roi['main_player_chips']:
                        # Use the specialized main_player_chips ROI for player 1 (human player)
                        chip_region = self.roi['main_player_chips'][0]
                        self.logger.debug("Using main_player_chips ROI for player 1")
                    else:
                        # Use regular player_chips ROI for other players
                        chip_region = self.roi.get('player_chips', {}).get(player_id, [(0, 0, 0, 0)])[0]
                    
                    chips = self._extract_chip_count(original_img, chip_region)
                    
                    game_state['players'][player_id] = {
                        'chips': chips,
                        'cards': player_cards,
                        'position': player_id
                    }
            
            # Extract pot size
            if 'pot' in self.roi and self.roi['pot']:
                game_state['pot'] = self._extract_pot_size(original_img, self.roi['pot'][0])
            
            # NEW: Extract current bets for each player
            self.logger.debug("Extracting current bets for players")
            for player_id in list(game_state['players'].keys()):
                if 'current_bets' in self.roi and player_id in self.roi.get('current_bets', {}):
                    bet_region = self.roi['current_bets'][player_id][0]
                    current_bet = self._extract_current_bet(original_img, bet_region)
                    game_state['players'][player_id]['current_bet'] = current_bet
                    self.logger.debug(f"Player {player_id} current bet: {current_bet}")
            
            # NEW: Detect available actions
            self.logger.debug("Detecting available actions")
            game_state['available_actions'] = self._detect_available_actions(original_img)
            self.logger.debug(f"Available actions: {game_state['available_actions']}")
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            self.logger.info(f"Image analysis completed in {elapsed_time:.2f} seconds")
            
            # Ensure all values are JSON serializable
            json_safe_game_state = self._ensure_json_serializable(game_state)
            
            # Cache the result
            self._analysis_cache[img_hash] = json_safe_game_state
            
            return json_safe_game_state
            
        except Exception as e:
            self.logger.error(f"Error analyzing image {image_path}: {str(e)}", exc_info=True)
            # Return a default game state as fallback
            return self._create_default_game_state()

    def _extract_card(self, img, region, cards_list):
        """
        Extract a card from the image and add it to the cards list.
        Improved with better error handling and reliability.
        
        Args:
            img: Source image
            region: Region tuple (x, y, w, h)
            cards_list: List to append detected cards to
        """
        try:
            x, y, w, h = region
            
            # Ensure region is within image bounds
            if not self._is_valid_region(region, img.shape):
                return
                    
            # Make a clean copy of the original source image before extracting
            original_img = img.copy()
                    
            # Extract card image from the CLEAN copy (without any debug overlays)
            card_img = original_img[y:y+h, x:x+w]
            
            # Check if card is visible (not folded)
            if not self._is_card_visible(card_img):
                return
                
            # Detect the card using our neural network-enhanced detector
            # The detector will get a clean image without any overlays
            value, suit = self.card_detector.detect_card(card_img)
            
            # Only add valid cards
            if value not in ['?', ''] and suit not in ['?', '']:
                cards_list.append({
                    'value': value,
                    'suit': suit
                })
                
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Detected card: {value} of {suit}")
        except Exception as e:
            self.logger.error(f"Error extracting card: {str(e)}")

    def _is_valid_region(self, region, img_shape):
        """
        Check if a region is valid within the image bounds.
        Added padding to avoid boundary-related issues.
        
        Args:
            region: Region tuple (x, y, w, h)
            img_shape: Image shape (height, width, channels)
            
        Returns:
            bool: True if region is valid
        """
        x, y, w, h = region
        h_img, w_img = img_shape[:2]
        
        # Add padding check to avoid boundary issues
        padding = 2
        return (x >= padding and y >= padding and 
                x+w <= w_img-padding and y+h <= h_img-padding and
                w > 0 and h > 0)

    def _is_card_visible(self, card_img):
        """
        Check if a card is visible (not folded or hidden).
        Improved with multi-threshold analysis for better reliability.
        
        Args:
            card_img: Card image region
            
        Returns:
            bool: True if a card is likely present
        """
        if card_img is None or card_img.size == 0:
            return False
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            
            # Multi-threshold approach for better reliability
            thresholds = [150, 180, 200]
            white_percentages = []
            
            for threshold in thresholds:
                # Apply thresholding to find white/light areas
                _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                
                # Count white pixels
                white_pixels = cv2.countNonZero(thresh)
                total_pixels = card_img.shape[0] * card_img.shape[1]
                white_percent = (white_pixels / total_pixels) * 100
                white_percentages.append(white_percent)
            
            # Calculate average white percentage across thresholds
            avg_white_percent = sum(white_percentages) / len(white_percentages)
            
            # Check standard deviation to detect variance in white detection
            std_dev = np.std(white_percentages)
            
            # If very consistent across thresholds with high white percentage,
            # it's more likely to be a card
            is_consistent = std_dev < 10
            
            # Higher confidence if consistent readings
            min_white_threshold = 25 if is_consistent else 35
            
            return avg_white_percent > min_white_threshold
            
        except Exception as e:
            logger.error(f"Error checking card visibility: {str(e)}")
            return False

    def _is_player_active(self, img, player_id):
        """
        Check if a player is active in the game
        
        In a real implementation, this would look for visual cues indicating an active player.
        For this simplified version, we assume all players in the configuration are active.
        """
        return player_id in self.roi.get('player_cards', {})
    
    def _extract_chip_count(self, img, region):
        """Extract chip count using the enhanced text recognizer"""
        if not self._is_valid_region(region, img.shape):
            return 0
            
        return self.text_recognizer.extract_chip_count(img, region)
    
    def _extract_current_bet(self, img, region):
        """
        Extract current bet from the image using text recognition
        
        Args:
            img: Source image
            region: Region tuple (x, y, w, h) defining the location of the bet text
            
        Returns:
            int: Extracted bet amount, or 0 if extraction fails
        """
        if not self._is_valid_region(region, img.shape):
            return 0
                
        # Use the same text recognition approach as for chip counts
        return self.text_recognizer.extract_chip_count(img, region)
    
    def _extract_pot_size(self, img, region):
        """Extract pot size using the enhanced text recognizer"""
        if not self._is_valid_region(region, img.shape):
            return 0
            
        return self.text_recognizer.extract_chip_count(img, region)
    
    def _detect_available_actions(self, img):
        """
        Detect available actions based on action button regions in the ROI
        
        Args:
            img: Source image
            
        Returns:
            dict: Dictionary with actions as keys and boolean values indicating availability
        """
        # Initialize all actions as unavailable by default
        available_actions = {
            'raise': False,
            'call': False, 
            'fold': False
        }
        
        # Check if actions are defined in ROI
        if 'actions' not in self.roi:
            self.logger.warning("No action regions defined in ROI configuration")
            return available_actions
            
        # Check each action
        for action, regions in self.roi['actions'].items():
            if not regions:
                continue
                
            region = regions[0]  # Take the first region for each action
            
            if not self._is_valid_region(region, img.shape):
                self.logger.warning(f"Invalid action region for {action}: {region}")
                continue
                
            # Extract the action button region
            x, y, w, h = region
            button_img = img[y:y+h, x:x+w]
            
            # Simple brightness-based detection
            # A more sophisticated approach would analyze the button's appearance and text
            gray = cv2.cvtColor(button_img, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            
            # Consider the action available if the region is bright enough
            # This threshold might need adjustment based on the specific UI
            # Use Python's built-in bool for JSON serialization
            available_actions[action] = bool(avg_brightness > 50)
            
            self.logger.debug(f"Action {action}: brightness={avg_brightness}, available={available_actions[action]}")
        
        return available_actions
    
    def _detect_game_stage(self, img):
        """
        Detect the current game stage and how many community cards are visible
        
        Returns:
            tuple: (game_stage, num_visible_cards)
        """
        # For a real implementation, this would use OCR to detect the game stage text
        # For simplicity, we'll infer it by counting visible community cards
        
        # Check each community card position
        visible_cards = 0
        for region in self.roi['community_cards']:
            if not self._is_valid_region(region, img.shape):
                continue
                
            card_img = img[region[1]:region[1]+region[3], region[0]:region[0]+region[2]]
            if self._is_card_visible(card_img):
                visible_cards += 1
        
        # Determine game stage based on number of visible cards
        if visible_cards == 0:
            return 'preflop', 0
        elif visible_cards == 3:
            return 'flop', 3
        elif visible_cards == 4:
            return 'turn', 4
        elif visible_cards == 5:
            return 'river', 5
        else:
            return 'unknown', visible_cards
    
    def _create_default_game_state(self):
        """Create a default game state when analysis fails"""
        self.logger.info("Creating default game state")
        
        game_state = {
            'game_stage': 'flop',
            'community_cards': [
                {'value': '9', 'suit': 'diamonds'},
                {'value': '6', 'suit': 'clubs'},
                {'value': '2', 'suit': 'spades'}
            ],
            'players': {
                1: {
                    'chips': 4980,
                    'cards': [
                        {'value': 'A', 'suit': 'hearts'},
                        {'value': 'K', 'suit': 'hearts'}
                    ],
                    'position': 1,
                    'current_bet': 20  # Added default current bet
                }
            },
            'pot': 200,
            'available_actions': {  # Added available actions
                'raise': True,
                'call': True,
                'fold': True
            }
        }
        
        return game_state
    
    def clear_cache(self):
        """Clear the analysis cache"""
        self._analysis_cache.clear()
        self.card_detector.clear_cache()
        self.text_recognizer.clear_cache()
        self.logger.info("All caches cleared")
    
    def analyze_batch(self, image_paths):
        """
        Analyze multiple images in batch for improved performance
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            list: List of game state dictionaries
        """
        results = []
        
        for path in image_paths:
            results.append(self.analyze_image(path))
            
        return results
    
    def analyze_sequence(self, directory, pattern="screenshot_*.png"):
        """
        Analyze a sequence of screenshots in a directory
        
        Args:
            directory: Directory containing screenshots
            pattern: Filename pattern to match
            
        Returns:
            list: List of (timestamp, game_state) tuples
        """
        try:
            # Find all matching files
            path = Path(directory)
            files = sorted(path.glob(pattern))
            
            if not files:
                self.logger.warning(f"No files matching {pattern} found in {directory}")
                return []
            
            self.logger.info(f"Found {len(files)} files to analyze")
            
            # Analyze each file
            results = []
            for file_path in files:
                # Extract timestamp from filename
                try:
                    # Try to parse timestamp from filename (assuming format like screenshot_20220101_120000.png)
                    timestamp_str = file_path.stem.split('_', 1)[1]
                    timestamp = time.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                except (ValueError, IndexError):
                    # If parsing fails, use file modification time
                    timestamp = time.localtime(os.path.getmtime(file_path))
                
                # Analyze the image
                game_state = self.analyze_image(str(file_path))
                
                # Add the result
                results.append((timestamp, game_state))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing sequence: {str(e)}", exc_info=True)
            return []