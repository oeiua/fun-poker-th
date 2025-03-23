#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Card Detector Module

This module provides functionality to detect and recognize playing cards
in screenshots of poker games. It can detect both community cards and
player cards.
"""

import logging
import numpy as np
import cv2
import os
import time
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

logger = logging.getLogger("PokerVision.CardDetector")

# Define card suits and values
class CardSuit(Enum):
    """Enumeration of card suits."""
    CLUBS = auto()
    DIAMONDS = auto()
    HEARTS = auto()
    SPADES = auto()
    UNKNOWN = auto()
    
    def __str__(self) -> str:
        """String representation of card suit."""
        if self == CardSuit.CLUBS:
            return "♣"
        elif self == CardSuit.DIAMONDS:
            return "♦"
        elif self == CardSuit.HEARTS:
            return "♥"
        elif self == CardSuit.SPADES:
            return "♠"
        else:
            return "?"
    
    @property
    def symbol(self) -> str:
        """Get the symbol for this suit."""
        return str(self)
    
    @property
    def color(self) -> Tuple[int, int, int]:
        """Get the color for this suit."""
        if self == CardSuit.DIAMONDS or self == CardSuit.HEARTS:
            return (0, 0, 255)  # Red
        else:
            return (0, 0, 0)    # Black


class CardValue(Enum):
    """Enumeration of card values."""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    UNKNOWN = 0
    
    def __str__(self) -> str:
        """String representation of card value."""
        if self == CardValue.JACK:
            return "J"
        elif self == CardValue.QUEEN:
            return "Q"
        elif self == CardValue.KING:
            return "K"
        elif self == CardValue.ACE:
            return "A"
        elif self == CardValue.UNKNOWN:
            return "?"
        else:
            return str(self.value)
    
    @property
    def symbol(self) -> str:
        """Get the symbol for this value."""
        return str(self)
    
    @classmethod
    def from_string(cls, value_str: str) -> 'CardValue':
        """
        Create a CardValue from a string.
        
        Args:
            value_str: String representation of the card value
            
        Returns:
            CardValue enum value
        """
        value_str = value_str.upper().strip()
        
        if value_str == 'J' or value_str == 'JACK':
            return cls.JACK
        elif value_str == 'Q' or value_str == 'QUEEN':
            return cls.QUEEN
        elif value_str == 'K' or value_str == 'KING':
            return cls.KING
        elif value_str == 'A' or value_str == 'ACE':
            return cls.ACE
        elif value_str.isdigit():
            value = int(value_str)
            if 2 <= value <= 10:
                return cls(value)
        
        return cls.UNKNOWN


@dataclass
class Card:
    """Class representing a playing card."""
    value: CardValue
    suit: CardSuit
    confidence: float = 1.0
    
    def __str__(self) -> str:
        """String representation of the card."""
        return f"{self.value.symbol}{self.suit.symbol}"
    
    def __eq__(self, other: object) -> bool:
        """Check if two cards are equal."""
        if not isinstance(other, Card):
            return False
        return self.value == other.value and self.suit == other.suit


class CardDetector:
    """Class for detecting and recognizing playing cards in images."""
    
    def __init__(self, model_path: Optional[str] = None, template_path: Optional[str] = None, debug_mode: bool = False):
        """
        Initialize the CardDetector.
        
        Args:
            model_path: Path to the card recognition model directory
            template_path: Path to the card template directory
            debug_mode: Whether to save debug images
        """
        logger.info("Initializing CardDetector")
        
        # Set default paths if not provided
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "..", "models", "card_recognition_model")
        
        if template_path is None:
            template_path = os.path.join(os.path.dirname(__file__), "..", "models", "card_templates")
        
        self.model_path = model_path
        self.template_path = template_path
        self.debug_mode = debug_mode
        
        # Debug directory
        self.debug_dir = os.path.join(os.path.dirname(__file__), "..", "debug", "card_detector")
        if self.debug_mode:
            os.makedirs(self.debug_dir, exist_ok=True)
            os.makedirs(os.path.join(self.debug_dir, "roi"), exist_ok=True)
            os.makedirs(os.path.join(self.debug_dir, "contours"), exist_ok=True)
            os.makedirs(os.path.join(self.debug_dir, "preprocessing"), exist_ok=True)
            os.makedirs(os.path.join(self.debug_dir, "recognition"), exist_ok=True)
            logger.info(f"Debug images will be saved to {self.debug_dir}")
        
        # Load templates
        self.card_templates = self._load_card_templates()
        
        # Initialize feature detector and matcher
        self.feature_detector = cv2.SIFT_create()
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        # Load suit and value templates
        self.suit_templates = self._load_suit_templates()
        self.value_templates = self._load_value_templates()
        
        # Initialize card parameters
        self.card_width = 60   # Expected width of a card in pixels
        self.card_height = 80  # Expected height of a card in pixels
        self.min_card_area = 2000  # Minimum area to consider as a card
        
        # Color ranges for suits (HSV format)
        self.red_range = [(0, 100, 100), (10, 255, 255), (170, 100, 100), (180, 255, 255)]
        self.black_range = [(0, 0, 0), (180, 100, 70)]
        
        # Debug counter for unique filenames
        self.debug_counter = 0
        
    def _load_card_templates(self) -> Dict[str, np.ndarray]:
        """
        Load card templates from the template directory.
        
        Returns:
            Dictionary of card templates (key = card name, value = image)
        """
        templates = {}
        template_dir = Path(self.template_path)
        
        if not template_dir.exists():
            logger.warning(f"Template directory not found: {template_dir}")
            return templates
        
        for template_file in template_dir.glob("*.png"):
            card_name = template_file.stem
            template = cv2.imread(str(template_file))
            if template is not None:
                templates[card_name] = template
                logger.debug(f"Loaded template: {card_name}")
        
        logger.info(f"Loaded {len(templates)} card templates")
        return templates
    
    def _load_suit_templates(self) -> Dict[CardSuit, np.ndarray]:
        """
        Load suit templates.
        
        Returns:
            Dictionary of suit templates (key = suit enum, value = image)
        """
        templates = {}
        suit_dir = Path(self.template_path) / "suits"
        
        if not suit_dir.exists():
            logger.warning(f"Suit template directory not found: {suit_dir}")
            return templates
        
        for suit in CardSuit:
            if suit == CardSuit.UNKNOWN:
                continue
                
            filename = f"{suit.name.lower()}.png"
            filepath = suit_dir / filename
            
            if filepath.exists():
                template = cv2.imread(str(filepath))
                if template is not None:
                    templates[suit] = template
                    logger.debug(f"Loaded suit template: {suit.name}")
        
        logger.info(f"Loaded {len(templates)} suit templates")
        return templates
    
    def _load_value_templates(self) -> Dict[CardValue, np.ndarray]:
        """
        Load value templates.
        
        Returns:
            Dictionary of value templates (key = value enum, value = image)
        """
        templates = {}
        value_dir = Path(self.template_path) / "values"
        
        if not value_dir.exists():
            logger.warning(f"Value template directory not found: {value_dir}")
            return templates
        
        for value in CardValue:
            if value == CardValue.UNKNOWN:
                continue
                
            filename = f"{value.name.lower()}.png"
            filepath = value_dir / filename
            
            if filepath.exists():
                template = cv2.imread(str(filepath))
                if template is not None:
                    templates[value] = template
                    logger.debug(f"Loaded value template: {value.name}")
        
        logger.info(f"Loaded {len(templates)} value templates")
        return templates
    
    def detect_community_cards(self, image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> List[Card]:
        """
        Detect community cards in the image.
        
        Args:
            image: Input image
            roi: Region of interest (x, y, width, height), if None then use the entire image
            
        Returns:
            List of detected community cards
        """
        logger.debug("Detecting community cards")
        
        # Extract ROI if provided
        if roi is not None:
            x, y, w, h = roi
            roi_image = image[y:y+h, x:x+w]
            if self.debug_mode:
                self._save_debug_image(roi_image, "roi", "community_cards")
        else:
            roi_image = image.copy()
        
        # Apply preprocessing
        preprocessed = self._preprocess_image(roi_image)
        
        # Detect cards
        card_regions = self._detect_card_regions(preprocessed)
        
        # Recognize cards
        cards = []
        for i, card_region in enumerate(card_regions):
            x, y, w, h = card_region
            roi_card = roi_image[y:y+h, x:x+w]
            
            if self.debug_mode:
                self._save_roi_image(roi_card, "community_card", i)
                
            card = self._recognize_card(roi_image, card_region)
            if card is not None:
                cards.append(card)
                if self.debug_mode and card is not None:
                    logger.debug(f"Detected community card {i}: {card}")
        
        logger.info(f"Detected {len(cards)} community cards")
        return cards
    
    def detect_player_cards(self, image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> List[Card]:
        """
        Detect player cards in the image.
        
        Args:
            image: Input image
            roi: Region of interest (x, y, width, height), if None then use the entire image
            
        Returns:
            List of detected player cards
        """
        logger.debug("Detecting player cards")
        
        # Extract ROI if provided
        if roi is not None:
            x, y, w, h = roi
            roi_image = image[y:y+h, x:x+w]
            if self.debug_mode:
                self._save_debug_image(roi_image, "roi", "player_cards")
        else:
            roi_image = image.copy()
        
        # Apply preprocessing
        preprocessed = self._preprocess_image(roi_image)
        
        # Detect cards
        card_regions = self._detect_card_regions(preprocessed)
        
        # Recognize cards
        cards = []
        for i, card_region in enumerate(card_regions):
            x, y, w, h = card_region
            roi_card = roi_image[y:y+h, x:x+w]
            
            if self.debug_mode:
                self._save_roi_image(roi_card, "player_card", i)
                
            card = self._recognize_card(roi_image, card_region)
            if card is not None:
                cards.append(card)
                if self.debug_mode and card is not None:
                    logger.debug(f"Detected player card {i}: {card}")
        
        logger.info(f"Detected {len(cards)} player cards")
        return cards
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for card detection.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Save original image for debugging
        if self.debug_mode:
            self._save_debug_image(image, "preprocessing", "original")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.debug_mode:
            self._save_debug_image(gray, "preprocessing", "grayscale")
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.debug_mode:
            self._save_debug_image(blurred, "preprocessing", "blurred")
        
        # Apply adaptive thresholding
        threshold = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        if self.debug_mode:
            self._save_debug_image(threshold, "preprocessing", "threshold")
        
        # Apply morphological operations to remove noise
        kernel = np.ones((3, 3), np.uint8)
        morphed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        if self.debug_mode:
            self._save_debug_image(morphed, "preprocessing", "morphed")
        
        return morphed
    
    def _detect_card_regions(self, preprocessed: np.ndarray) -> List[np.ndarray]:
        """
        Detect card regions in the preprocessed image.
        
        Args:
            preprocessed: Preprocessed image
            
        Returns:
            List of card region images
        """
        # Find contours
        contours, hierarchy = cv2.findContours(
            preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Save all contours for debugging
        if self.debug_mode:
            self._save_contours_image(
                cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR) if len(preprocessed.shape) == 2 else preprocessed,
                contours,
                "all_contours"
            )
        
        # Filter contours by area and shape
        card_regions = []
        filtered_contours = []
        
        for contour in contours:
            # Get contour area
            area = cv2.contourArea(contour)
            
            # Skip small contours
            if area < self.min_card_area:
                continue
            
            # Approximate contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Check if the contour has 4 vertices (card shape)
            if len(approx) == 4:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(approx)
                
                # Check if the aspect ratio is close to expected card aspect ratio
                aspect_ratio = h / w
                if 1.2 <= aspect_ratio <= 1.5:
                    # Extract card region
                    card_region = preprocessed[y:y+h, x:x+w]
                    
                    # Only add if the region is not empty
                    if card_region.size > 0:
                        card_regions.append((x, y, w, h))
                        filtered_contours.append(contour)
        
        # Save filtered contours for debugging
        if self.debug_mode:
            self._save_contours_image(
                cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR) if len(preprocessed.shape) == 2 else preprocessed,
                filtered_contours,
                "filtered_card_contours"
            )
            
            # Save each card region
            for i, (x, y, w, h) in enumerate(card_regions):
                roi = preprocessed[y:y+h, x:x+w]
                self._save_roi_image(roi, "card_region", i)
        
        logger.debug(f"Detected {len(card_regions)} potential card regions")
        return card_regions
    
    def _recognize_card(self, image: np.ndarray, card_region: Tuple[int, int, int, int]) -> Optional[Card]:
        """
        Recognize a card in the given region.
        
        Args:
            image: Input image
            card_region: Region of the card (x, y, width, height)
            
        Returns:
            Recognized card or None if recognition failed
        """
        x, y, w, h = card_region
        
        # Extract card region
        card_img = image[y:y+h, x:x+w]
        
        # Save the original card region for debugging
        if self.debug_mode:
            self._save_debug_image(card_img, "recognition", f"card_original_{x}_{y}")
        
        # Resize to standard size
        card_img = cv2.resize(card_img, (self.card_width, self.card_height))
        
        if self.debug_mode:
            self._save_debug_image(card_img, "recognition", f"card_resized_{x}_{y}")
        
        # Split into value and suit regions
        value_region = card_img[0:int(h/5), 0:int(w/2)]
        suit_region = card_img[int(h/5):int(h/3), 0:int(w/2)]
        
        if self.debug_mode:
            self._save_debug_image(value_region, "recognition", f"value_region_{x}_{y}")
            self._save_debug_image(suit_region, "recognition", f"suit_region_{x}_{y}")
        
        # Recognize value and suit
        value = self._recognize_value(value_region)
        suit = self._recognize_suit(suit_region)
        
        # Create card object
        if value != CardValue.UNKNOWN and suit != CardSuit.UNKNOWN:
            card = Card(value, suit)
            if self.debug_mode:
                logger.debug(f"Recognized card at {card_region}: {card}")
            return card
        
        # Try template matching if feature-based matching failed
        for card_name, template in self.card_templates.items():
            # Resize template to match card size
            template = cv2.resize(template, (self.card_width, self.card_height))
            
            # Apply template matching
            result = cv2.matchTemplate(card_img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            # If match confidence is high enough
            if max_val > 0.8:
                # Parse card name (e.g., "ace_of_hearts")
                parts = card_name.split("_of_")
                if len(parts) == 2:
                    value_str, suit_str = parts
                    
                    # Convert to enum values
                    value = CardValue.from_string(value_str)
                    
                    if suit_str.lower() == "clubs":
                        suit = CardSuit.CLUBS
                    elif suit_str.lower() == "diamonds":
                        suit = CardSuit.DIAMONDS
                    elif suit_str.lower() == "hearts":
                        suit = CardSuit.HEARTS
                    elif suit_str.lower() == "spades":
                        suit = CardSuit.SPADES
                    else:
                        suit = CardSuit.UNKNOWN
                    
                    # Create card object
                    if value != CardValue.UNKNOWN and suit != CardSuit.UNKNOWN:
                        card = Card(value, suit, max_val)
                        if self.debug_mode:
                            logger.debug(f"Recognized card at {card_region} using template matching: {card} (confidence: {max_val:.2f})")
                            self._save_debug_image(card_img, "recognition", f"card_matched_{card}_{x}_{y}")
                        return card
        
        # If all recognition methods failed
        logger.warning(f"Failed to recognize card at {card_region}")
        return None
    
    def _recognize_value(self, value_region: np.ndarray) -> CardValue:
        """
        Recognize the card value.
        
        Args:
            value_region: Region containing the card value
            
        Returns:
            Recognized card value
        """
        # Convert to grayscale
        gray = cv2.cvtColor(value_region, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Try template matching for each value
        best_match = None
        best_score = 0
        
        for value, template in self.value_templates.items():
            # Convert template to grayscale
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # Resize template to match value region
            template_resized = cv2.resize(template_gray, (value_region.shape[1], value_region.shape[0]))
            
            # Apply threshold
            _, template_thresh = cv2.threshold(template_resized, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Apply template matching
            result = cv2.matchTemplate(thresh, template_thresh, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            # If match confidence is high enough
            if max_val > best_score:
                best_score = max_val
                best_match = value
        
        # Use OCR as fallback
        if best_match is None or best_score < 0.7:
            # TODO: Use EasyOCR for value recognition
            pass
        
        return best_match if best_match is not None else CardValue.UNKNOWN
    
    def _recognize_suit(self, suit_region: np.ndarray) -> CardSuit:
        """
        Recognize the card suit.
        
        Args:
            suit_region: Region containing the card suit
            
        Returns:
            Recognized card suit
        """
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(suit_region, cv2.COLOR_BGR2HSV)
        
        # Create masks for red and black colors
        red_mask1 = cv2.inRange(hsv, np.array(self.red_range[0]), np.array(self.red_range[1]))
        red_mask2 = cv2.inRange(hsv, np.array(self.red_range[2]), np.array(self.red_range[3]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        black_mask = cv2.inRange(hsv, np.array(self.black_range[0]), np.array(self.black_range[1]))
        
        # Calculate the percentage of red and black pixels
        red_pixels = cv2.countNonZero(red_mask)
        black_pixels = cv2.countNonZero(black_mask)
        total_pixels = suit_region.shape[0] * suit_region.shape[1]
        
        red_percentage = red_pixels / total_pixels
        black_percentage = black_pixels / total_pixels
        
        # Determine the color of the suit
        is_red = red_percentage > 0.1 and red_percentage > black_percentage
        
        # Try template matching for each suit
        best_match = None
        best_score = 0
        
        for suit, template in self.suit_templates.items():
            # Skip suits of wrong color
            if (is_red and suit not in [CardSuit.HEARTS, CardSuit.DIAMONDS]) or \
               (not is_red and suit not in [CardSuit.CLUBS, CardSuit.SPADES]):
                continue
            
            # Convert template to grayscale
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # Resize template to match suit region
            template_resized = cv2.resize(template_gray, (suit_region.shape[1], suit_region.shape[0]))
            
            # Apply threshold
            _, template_thresh = cv2.threshold(template_resized, 127, 255, cv2.THRESH_BINARY)
            
            # Convert suit region to grayscale
            gray = cv2.cvtColor(suit_region, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Apply template matching
            result = cv2.matchTemplate(thresh, template_thresh, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            # If match confidence is high enough
            if max_val > best_score:
                best_score = max_val
                best_match = suit
        
        # Determine suit based on color and shape if template matching failed
        if best_match is None or best_score < 0.7:
            # Create a simplified shape detector
            contours, _ = cv2.findContours(
                black_mask if not is_red else red_mask,
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Find the largest contour
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                # Approximate the contour
                peri = cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, 0.04 * peri, True)
                
                # Check shape
                if is_red:
                    # Heart or Diamond
                    if len(approx) > 6:  # More complex shape = heart
                        best_match = CardSuit.HEARTS
                    else:  # Simpler shape = diamond
                        best_match = CardSuit.DIAMONDS
                else:
                    # Club or Spade
                    if len(approx) > 6:  # More complex shape = club
                        best_match = CardSuit.CLUBS
                    else:  # Simpler shape = spade
                        best_match = CardSuit.SPADES
        
        return best_match if best_match is not None else CardSuit.UNKNOWN
    
    def draw_detected_cards(self, image: np.ndarray, cards: List[Card], 
                           positions: List[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Draw the detected cards on the image.
        
        Args:
            image: Input image
            cards: List of detected cards
            positions: List of card positions (x, y, width, height), if None use default positions
            
        Returns:
            Image with drawn cards
        """
        result = image.copy()
        
        # Use default positions if not provided
        if positions is None:
            # Calculate default positions
            image_height, image_width = image.shape[:2]
            card_spacing = 10
            x_start = 10
            y_start = 10
            
            positions = []
            for i in range(len(cards)):
                x = x_start + i * (self.card_width + card_spacing)
                if x + self.card_width > image_width:
                    x_start = 10
                    y_start += self.card_height + card_spacing
                    x = x_start
                
                positions.append((x, y_start, self.card_width, self.card_height))
        
        # Draw each card
        for i, (card, pos) in enumerate(zip(cards, positions)):
            x, y, w, h = pos
            
            # Draw card rectangle
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
            # Draw card value and suit
            text = str(card)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2
            
            # Draw with the appropriate color
            color = card.suit.color
            cv2.putText(result, text, (text_x, text_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return result

    def _save_debug_image(self, image: np.ndarray, subfolder: str, name: str) -> str:
        """
        Save a debug image.
        
        Args:
            image: Image to save
            subfolder: Subfolder within the debug directory
            name: Base name for the image
            
        Returns:
            Path to saved image
        """
        if not self.debug_mode:
            return ""
            
        # Create a timestamp and counter-based filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.debug_counter += 1
        filename = f"{timestamp}_{self.debug_counter:04d}_{name}.png"
        
        # Save the image
        path = os.path.join(self.debug_dir, subfolder, filename)
        try:
            cv2.imwrite(path, image)
            logger.debug(f"Saved debug image: {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to save debug image: {e}")
            return ""
        
    def _save_roi_image(self, image: np.ndarray, region_type: str, index: int) -> str:
        """
        Save a region of interest image.
        
        Args:
            image: ROI image
            region_type: Type of ROI (e.g., "community_card", "player_card")
            index: Index of the ROI
            
        Returns:
            Path to saved image
        """
        return self._save_debug_image(image, "roi", f"{region_type}_{index}")
        
    def _save_contours_image(self, image: np.ndarray, contours: List[np.ndarray], name: str) -> str:
        """
        Save an image with drawn contours.
        
        Args:
            image: Original image
            contours: List of contours
            name: Base name for the image
            
        Returns:
            Path to saved image
        """
        if not self.debug_mode:
            return ""
            
        # Create a copy of the image
        contour_image = image.copy()
        
        # Draw contours
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        
        # Draw bounding rectangles
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(contour_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(contour_image, f"{i}: {w}x{h}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return self._save_debug_image(contour_image, "contours", name)

# Test function
def test_card_detector():
    """Test the card detector functionality."""
    import time
    import os
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Initialize card detector
    detector = CardDetector()
    
    # Load test image
    test_image_path = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "test_cards.png")
    if os.path.exists(test_image_path):
        # Load image
        image = cv2.imread(test_image_path)
        
        # Detect community cards
        start_time = time.time()
        community_cards = detector.detect_community_cards(image)
        end_time = time.time()
        
        print(f"Detected {len(community_cards)} community cards in {end_time - start_time:.3f} seconds:")
        for card in community_cards:
            print(f"  {card}")
        
        # Draw detected cards
        result = detector.draw_detected_cards(image, community_cards)
        
        # Save result
        cv2.imwrite("detected_cards.png", result)
        print(f"Saved result to detected_cards.png")
    else:
        print(f"Test image not found: {test_image_path}")


if __name__ == "__main__":
    # Run test
    test_card_detector()