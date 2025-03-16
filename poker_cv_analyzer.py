import cv2
import numpy as np
import pytesseract
import logging
import json
import os

# Set up logging
logger = logging.getLogger("CardDetector")

class PokerCardDetector:
    def __init__(self):
        self.values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.suits = ['hearts', 'diamonds', 'clubs', 'spades']
        
        # Symbol to suit mapping
        self.symbol_to_suit = {
            '♥': 'hearts',
            '♦': 'diamonds',
            '♣': 'clubs',
            '♠': 'spades',
            'h': 'hearts',    # Fallback for OCR errors
            'd': 'diamonds',  # Fallback for OCR errors
            'c': 'clubs',     # Fallback for OCR errors
            's': 'spades'     # Fallback for OCR errors
        }
        
        # Color ranges for suits (HSV)
        self.suit_colors = {
            'red': [(0, 100, 100), (10, 255, 255)],   # Red range 1
            'red2': [(170, 100, 100), (180, 255, 255)], # Red range 2 (wraps around hue)
            'black': [(0, 0, 0), (180, 255, 50)]      # Black
        }
        
        # Mapping of color to suits
        self.color_to_suits = {
            'red': ['hearts', 'diamonds'],
            'black': ['clubs', 'spades']
        }
        
        logger.info("PokerCardDetector initialized")

    def detect_card(self, card_img):
        """
        Detect card value and suit from an image
        
        Args:
            card_img: Image of a playing card (cropped to show just the card)
            
        Returns:
            tuple: (value, suit) e.g. ('A', 'hearts')
        """
        try:
            if card_img is None or card_img.size == 0:
                logger.warning("Empty card image provided")
                return self.values[0], self.suits[0]
            
            # Get image dimensions
            h, w = card_img.shape[:2]
            
            # Split card into top and bottom regions for value and suit
            value_region = card_img[0:int(h*0.4), 0:w]  # Top 40% for value
            suit_region = card_img[int(h*0.4):h, 0:w]   # Bottom 60% for suit
            
            # Detect card value (top part)
            value = self._detect_card_value(value_region)
            
            # Detect card suit (bottom part) using both OCR and color
            suit = self._detect_card_suit(suit_region)
            
            logger.info(f"Detected card: {value} of {suit}")
            return value, suit
            
        except Exception as e:
            logger.error(f"Error detecting card: {str(e)}")
            # Return a default value if detection fails
            return self.values[0], self.suits[0]
    
    def _detect_card_value(self, value_region):
        """
        Detect the card value from the top part of the card
        
        Args:
            value_region: Image of the top part of the card
            
        Returns:
            str: Card value ('2' through '10', 'J', 'Q', 'K', or 'A')
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(value_region, cv2.COLOR_BGR2GRAY)
            
            # Threshold to isolate the value
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Use OCR to extract text
            text = pytesseract.image_to_string(
                thresh, 
                config='--psm 10 -c tessedit_char_whitelist=0123456789AJQK'
            ).strip()
            
            # Clean up OCR result
            text = text.upper().replace('O', '0')  # Common OCR error
            
            # Handle '10' specially
            if '10' in text or ('1' in text and '0' in text):
                return '10'
            
            # Check for face cards and ace
            if 'A' in text:
                return 'A'
            elif 'K' in text:
                return 'K'
            elif 'Q' in text:
                return 'Q'
            elif 'J' in text:
                return 'J'
            
            # For numeric cards, find first digit
            for char in text:
                if char in '23456789':
                    return char
            
            # If OCR failed, try template matching as fallback
            # This would be implemented for production but is simplified here
            
            # Return the most likely card value based on visual features
            # For now, use a fallback
            logger.warning(f"Could not determine card value, OCR text: '{text}'")
            return '2'  # Default fallback
            
        except Exception as e:
            logger.error(f"Error detecting card value: {str(e)}")
            return '2'  # Default fallback
    
    def _detect_card_suit(self, suit_region):
        """
        Detect the card suit from the bottom part of the card
        
        Args:
            suit_region: Image of the bottom part of the card
            
        Returns:
            str: Card suit ('hearts', 'diamonds', 'clubs', or 'spades')
        """
        try:
            # First attempt: OCR to detect suit symbol
            gray = cv2.cvtColor(suit_region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Use OCR to extract text that might contain suit symbols
            text = pytesseract.image_to_string(
                thresh, 
                config='--psm 10 -c tessedit_char_whitelist=♥♦♣♠hdcs'
            ).strip().lower()
            
            # Check for suit symbols in the text
            for symbol, suit in self.symbol_to_suit.items():
                if symbol in text:
                    return suit
            
            # Second attempt: color detection
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(suit_region, cv2.COLOR_BGR2HSV)
            
            # Check for red suits (hearts, diamonds)
            red_mask1 = cv2.inRange(hsv, np.array(self.suit_colors['red'][0]), np.array(self.suit_colors['red'][1]))
            red_mask2 = cv2.inRange(hsv, np.array(self.suit_colors['red2'][0]), np.array(self.suit_colors['red2'][1]))
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            red_pixels = cv2.countNonZero(red_mask)
            
            # Check for black suits (clubs, spades)
            black_mask = cv2.inRange(hsv, np.array(self.suit_colors['black'][0]), np.array(self.suit_colors['black'][1]))
            black_pixels = cv2.countNonZero(black_mask)
            
            # Determine color based on pixel count
            if red_pixels > black_pixels * 2:  # If there are significantly more red pixels
                # Differentiate hearts and diamonds based on shape analysis
                # This would use contour analysis in production
                # For simplicity, randomly choose between hearts and diamonds
                return 'hearts' if np.random.random() > 0.5 else 'diamonds'
            else:
                # Differentiate clubs and spades based on shape analysis
                # This would use contour analysis in production
                # For simplicity, randomly choose between clubs and spades
                return 'clubs' if np.random.random() > 0.5 else 'spades'
            
        except Exception as e:
            logger.error(f"Error detecting card suit: {str(e)}")
            return 'spades'  # Default fallback
    
    def enhance_card_image(self, card_img):
        """
        Enhance the card image for better detection
        
        Args:
            card_img: Original card image
            
        Returns:
            numpy.ndarray: Enhanced card image
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up the image
            kernel = np.ones((2, 2), np.uint8)
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return morph
            
        except Exception as e:
            logger.error(f"Error enhancing card image: {str(e)}")
            return card_img  # Return original image if enhancement fails


class PokerImageAnalyzer:
    def __init__(self):
        self.card_detector = PokerCardDetector()
        
        # Define regions of interest for PokerTH 1.1.2
        # These would need to be calibrated for the actual game window
        self.roi = {
            'community_cards': [
                (320, 320, 60, 80),  # First card
                (390, 320, 60, 80),  # Second card
                (460, 320, 60, 80),  # Third card
                (530, 320, 60, 80),  # Fourth card
                (600, 320, 60, 80)   # Fifth card
            ],
            'player_cards': {
                1: [(380, 600, 60, 80), (450, 600, 60, 80)],  # Player 1's cards
                # Add more player card regions as needed
            },
            'player_chips': {
                1: (390, 650, 100, 30),  # Player 1's chip count region
                # Add more player chip regions as needed
            },
            'pot': (320, 250, 100, 30)  # Pot size region
        }
    
    def analyze_image(self, image_path):
        """Analyze a poker table screenshot and extract game state"""
        # Check if the image path exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
            
        # Load the image
        img = cv2.imread(image_path)
        
        # Check if the image was loaded successfully
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
            
        # Check if the image is empty
        if img.size == 0:
            logger.error(f"Image is empty: {image_path}")
            return None
        
        try:
            # Create game state dictionary
            game_state = {
                'community_cards': [],
                'players': {},
                'pot': 0
            }
            
            # Extract community cards
            visible_cards = self._detect_visible_cards(img)
            for i, card_region in enumerate(self.roi['community_cards']):
                if i < visible_cards:
                    x, y, w, h = card_region
                    # Ensure region is within image bounds
                    if x >= 0 and y >= 0 and x+w <= img.shape[1] and y+h <= img.shape[0]:
                        card_img = img[y:y+h, x:x+w]
                        value, suit = self.card_detector.detect_card(card_img)
                        game_state['community_cards'].append({
                            'value': value,
                            'suit': suit
                        })
            
            # Extract player information
            for player_id, card_regions in self.roi['player_cards'].items():
                # Check if player is active
                if self._is_player_active(img, player_id):
                    player_cards = []
                    for card_region in card_regions:
                        x, y, w, h = card_region
                        # Ensure region is within image bounds
                        if x >= 0 and y >= 0 and x+w <= img.shape[1] and y+h <= img.shape[0]:
                            card_img = img[y:y+h, x:x+w]
                            # Check if card is visible (not folded)
                            if self._is_card_visible(card_img):
                                value, suit = self.card_detector.detect_card(card_img)
                                player_cards.append({
                                    'value': value,
                                    'suit': suit
                                })
                    
                    # Get player's chip count
                    chip_region = self.roi['player_chips'][player_id]
                    chips = self._extract_chip_count(img, chip_region)
                    
                    game_state['players'][player_id] = {
                        'chips': chips,
                        'cards': player_cards,
                        'position': player_id
                    }
            
            # Extract pot size
            game_state['pot'] = self._extract_pot_size(img, self.roi['pot'])
            
            logger.info(f"Successfully analyzed image: {image_path}")
            return game_state
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            # Return a default game state as fallback
            return self._create_default_game_state()
    
    def _create_default_game_state(self):
        """Create a default game state when analysis fails"""
        logger.info("Creating default game state")
        
        game_state = {
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
    
    def _detect_visible_cards(self, img):
        """Detect how many community cards are visible"""
        # For demo, return 3 for flop
        # In a real implementation, this would check the regions for card presence
        return 3
    
    def _is_player_active(self, img, player_id):
        """Check if a player is active in the game"""
        # For demo, always return True
        # Real implementation would check for visual cues
        return True
    
    def _is_card_visible(self, card_img):
        """Check if a card is visible (not folded or hidden)"""
        # For demo, always return True
        # Real implementation would check for the card back pattern
        if card_img is None or card_img.size == 0:
            return False
        return True
    
    def _extract_chip_count(self, img, region):
        """Extract chip count using OCR"""
        try:
            x, y, w, h = region
            # Ensure region is within image bounds
            if x >= 0 and y >= 0 and x+w <= img.shape[1] and y+h <= img.shape[0]:
                chip_img = img[y:y+h, x:x+w]
                
                # Preprocess for OCR
                gray = cv2.cvtColor(chip_img, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                
                # OCR
                text = pytesseract.image_to_string(binary, config='--psm 7 -c tessedit_char_whitelist=0123456789$')
                
                # Extract the number
                import re
                match = re.search(r'\$?(\d+)', text)
                if match:
                    return int(match.group(1))
        except Exception as e:
            logger.error(f"Error extracting chip count: {str(e)}")
        
        # For demo, return a default value
        return 1000
    
    def _extract_pot_size(self, img, region):
        """Extract pot size using OCR"""
        try:
            return self._extract_chip_count(img, region)
        except Exception as e:
            logger.error(f"Error extracting pot size: {str(e)}")
            return 200
    
    def calibrate_roi(self, reference_image):
        """Calibrate regions of interest using a reference image"""
        # This would use computer vision techniques to automatically detect
        # the regions of interest in a reference image
        # For simplicity, we'll skip implementation details
        pass


if __name__ == "__main__":
    # Example usage
    analyzer = PokerImageAnalyzer()
    
    # Test with a sample image
    test_image = "screenshot_example.png"
    if os.path.exists(test_image):
        game_state = analyzer.analyze_image(test_image)
        if game_state:
            print(json.dumps(game_state, indent=2))
    else:
        print(f"Test image {test_image} not found")