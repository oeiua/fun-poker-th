import cv2
import numpy as np
import pytesseract
import logging
import json
import os

from improved_poker_cv_analyzer import ImprovedCardDetector

# Set up logging
logger = logging.getLogger("CardDetector")

class PokerImageAnalyzer:
    def __init__(self):
        self.card_detector = ImprovedCardDetector()
        
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