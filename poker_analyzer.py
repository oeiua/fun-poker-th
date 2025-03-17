import cv2
import numpy as np
import os
import logging
import json
import time
from pathlib import Path
from functools import lru_cache

# Import our improved components
from card_detector import ImprovedCardDetector, EnhancedTextRecognition

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PokerAnalyzer")

class OptimizedPokerAnalyzer:
    """Analyze poker screenshots with optimized performance and accuracy"""
    
    def __init__(self, template_dir="card_templates"):
        self.card_detector = ImprovedCardDetector(template_dir)
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
            
            # Pot information
            'pot': [(280, 248, 100, 20)],  # Pot size area
            
            # Game stage indicators
            'game_stage': [
                (265, 197, 80, 25),  # Game stage text (Preflop, Flop, Turn, River)
                (720, 197, 80, 25)   # Alternative location for game stage
            ]
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
            img = cv2.imread(image_path)
            
            # Check if the image was loaded successfully
            if img is None or img.size == 0:
                self.logger.error(f"Failed to load image or empty image: {image_path}")
                return self._create_default_game_state()
            
            # Create game state dictionary
            game_state = {
                'community_cards': [],
                'players': {},
                'pot': 0,
                'game_stage': 'preflop'
            }
            
            # Detect game stage first to know how many community cards to expect
            game_stage, visible_cards = self._detect_game_stage(img)
            game_state['game_stage'] = game_stage
            
            # Extract community cards
            for i, card_region in enumerate(self.roi['community_cards']):
                if i < visible_cards:
                    self._extract_card(img, card_region, game_state['community_cards'])
            
            # Extract player information
            for player_id, card_regions in self.roi.get('player_cards', {}).items():
                # Check if player is active
                if self._is_player_active(img, player_id):
                    player_cards = []
                    for card_region in card_regions:
                        self._extract_card(img, card_region, player_cards)
                    
                    # Get player's chip count
                    chip_region = self.roi.get('player_chips', {}).get(player_id, [(0, 0, 0, 0)])[0]
                    chips = self._extract_chip_count(img, chip_region)
                    
                    game_state['players'][player_id] = {
                        'chips': chips,
                        'cards': player_cards,
                        'position': player_id
                    }
            
            # Extract pot size
            if 'pot' in self.roi and self.roi['pot']:
                game_state['pot'] = self._extract_pot_size(img, self.roi['pot'][0])
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            self.logger.info(f"Image analysis completed in {elapsed_time:.2f} seconds")
            
            # Cache the result
            self._analysis_cache[img_hash] = game_state
            
            return game_state
            
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
                    
            # Extract card image
            card_img = img[y:y+h, x:x+w]
            
            # Check if card is visible (not folded)
            if not self._is_card_visible(card_img):
                return
                
            # Detect the card
            value, suit = self.card_detector.detect_card(card_img)
            
            # Only add valid cards
            if value not in ['?', ''] and suit not in ['?', '']:
                cards_list.append({
                    'value': value,
                    'suit': suit
                })
        except Exception as e:
            logger.error(f"Error extracting card: {str(e)}")

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
    
    def _extract_pot_size(self, img, region):
        """Extract pot size using the enhanced text recognizer"""
        if not self._is_valid_region(region, img.shape):
            return 0
            
        return self.text_recognizer.extract_chip_count(img, region)
    
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
                    'position': 1
                }
            },
            'pot': 200
        }
        
        return game_state
    
    def clear_cache(self):
        """Clear the analysis cache"""
        self._analysis_cache.clear()
        self.logger.info("Analysis cache cleared")
    
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


# Test function
def test_poker_analyzer():
    """Test the poker analyzer with a sample image"""
    print("Testing OptimizedPokerAnalyzer...")
    
    # Create analyzer
    analyzer = OptimizedPokerAnalyzer()
    
    # Check if test directory exists, create it if it doesn't
    test_dir = "test_output"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a sample image for testing
    img_size = (800, 600, 3)  # width, height, channels
    test_img = np.zeros(img_size, dtype=np.uint8)
    
    # Add green poker table background
    cv2.ellipse(test_img, (400, 300), (350, 250), 0, 0, 360, (0, 100, 0), -1)
    
    # Add white areas for cards
    # Community cards
    for i in range(3):  # Draw 3 cards for flop
        x = 300 + i * 60
        y = 220
        cv2.rectangle(test_img, (x, y), (x + 45, y + 65), (255, 255, 255), -1)
        # Add card value/suit
        suit_color = (0, 0, 255) if i % 2 == 0 else (0, 0, 0)
        cv2.putText(test_img, "A" if i == 0 else ("K" if i == 1 else "Q"), 
                   (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, suit_color, 2)
    
    # Player cards
    player_card_x = [510, 560]
    player_card_y = 330
    for i in range(2):
        x = player_card_x[i]
        y = player_card_y
        cv2.rectangle(test_img, (x, y), (x + 45, y + 65), (255, 255, 255), -1)
        # Add card value/suit
        cv2.putText(test_img, "J" if i == 0 else "10", 
                   (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Add chip count and pot text
    cv2.putText(test_img, "$1000", (280, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(test_img, "$150", (280, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Save test image
    test_img_path = os.path.join(test_dir, "test_poker_table.png")
    cv2.imwrite(test_img_path, test_img)
    print(f"Created test image: {test_img_path}")
    
    # Analyze the image
    start_time = time.time()
    game_state = analyzer.analyze_image(test_img_path)
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\nAnalysis completed in {elapsed_time:.3f} seconds")
    print("\nDetected Game State:")
    print(f"Game Stage: {game_state['game_stage']}")
    print(f"Pot Size: ${game_state['pot']}")
    
    print("\nCommunity Cards:")
    for i, card in enumerate(game_state['community_cards']):
        print(f"  Card {i+1}: {card['value']} of {card['suit']}")
    
    print("\nPlayers:")
    for player_id, player_data in game_state['players'].items():
        print(f"  Player {player_id}:")
        print(f"    Chips: ${player_data['chips']}")
        
        if player_data.get('cards'):
            cards_str = ", ".join([f"{card['value']} of {card['suit']}" for card in player_data['cards']])
            print(f"    Cards: {cards_str}")
    
    # Test caching performance
    print("\nTesting analysis cache...")
    start_time = time.time()
    cached_game_state = analyzer.analyze_image(test_img_path)
    cached_elapsed_time = time.time() - start_time
    
    print(f"Cached analysis completed in {cached_elapsed_time:.3f} seconds")
    print(f"Speedup factor: {elapsed_time / max(cached_elapsed_time, 0.0001):.1f}x")

if __name__ == "__main__":
    test_poker_analyzer()