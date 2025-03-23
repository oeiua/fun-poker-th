#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Utilities

This module provides OCR (Optical Character Recognition) helper functions
for extracting text from images.
"""

import logging
import os
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path

logger = logging.getLogger("PokerVision.OCRUtils")

class OCRHelper:
    """Helper class for OCR operations."""
    
    def __init__(self, languages: List[str] = None):
        """
        Initialize the OCRHelper.
        
        Args:
            languages: List of languages to use (default: ['en'])
        """
        logger.info("Initializing OCRHelper")
        
        if languages is None:
            languages = ['en']
        
        self.languages = languages
        self.reader = None
        
        # Initialize EasyOCR reader
        try:
            import easyocr
            self.reader = easyocr.Reader(self.languages)
            logger.info("EasyOCR initialized successfully")
        except ImportError:
            logger.warning("EasyOCR not available, OCR functionality will not work")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}", exc_info=True)
    
    def read_text(self, image: np.ndarray, 
                 min_confidence: float = 0.3,
                 allowlist: str = None) -> str:
        """
        Extract text from an image.
        
        Args:
            image: Input image
            min_confidence: Minimum confidence for OCR results
            allowlist: Optional allowlist of characters to recognize
            
        Returns:
            Extracted text
        """
        if self.reader is None:
            logger.warning("OCR reader not available")
            return ""
        
        try:
            # Extract text using EasyOCR
            results = self.reader.readtext(
                image,
                detail=1,
                paragraph=False,
                allowlist=allowlist
            )
            
            # Filter results by confidence
            filtered_results = [result for result in results if result[2] >= min_confidence]
            
            # Sort results by position (top to bottom, left to right)
            sorted_results = sorted(filtered_results, key=lambda x: (x[0][0][1], x[0][0][0]))
            
            # Extract text from results
            text = " ".join([result[1] for result in sorted_results])
            
            return text
        
        except Exception as e:
            logger.error(f"Error during OCR: {e}", exc_info=True)
            return ""
    
    def read_numbers(self, image: np.ndarray, 
                    min_confidence: float = 0.3,
                    include_decimals: bool = True) -> str:
        """
        Extract numbers from an image.
        
        Args:
            image: Input image
            min_confidence: Minimum confidence for OCR results
            include_decimals: Whether to include decimal points
            
        Returns:
            Extracted numbers as string
        """
        allowlist = "0123456789"
        if include_decimals:
            allowlist += "."
        
        return self.read_text(image, min_confidence, allowlist)
    
    def read_currency(self, image: np.ndarray, 
                     min_confidence: float = 0.3) -> float:
        """
        Extract currency amount from an image.
        
        Args:
            image: Input image
            min_confidence: Minimum confidence for OCR results
            
        Returns:
            Extracted currency amount as float
        """
        # Include currency symbols in allowlist
        allowlist = "0123456789 $€£¥"
        
        text = self.read_text(image, min_confidence, allowlist)
        
        # Parse currency amount
        if text:
            # Remove currency symbols
            text = text.replace("$", "").replace("€", "").replace("£", "").replace("¥", "")
            
            # Remove commas
            text = text.replace(",", "")
            
            # Try to convert to float
            try:
                return float(text)
            except ValueError:
                logger.warning(f"Failed to parse currency value: {text}")
        
        return 0.0
    
    def enhance_text_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance an image for better OCR results.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations to enhance text
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # Invert back to black text on white background for OCR
        inverted = cv2.bitwise_not(eroded)
        
        return inverted
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for OCR.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Resize image for better OCR (if too small)
        height, width = image.shape[:2]
        min_width = 200
        
        if width < min_width:
            # Calculate new height while maintaining aspect ratio
            new_height = int(height * (min_width / width))
            image = cv2.resize(image, (min_width, new_height))
        
        # Apply text enhancement
        enhanced = self.enhance_text_image(image)
        
        return enhanced
    
    def extract_text_regions(self, image: np.ndarray,
                            min_area: int = 100,
                            max_area: int = 10000) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """
        Extract regions that likely contain text.
        
        Args:
            image: Input image
            min_area: Minimum area of text regions
            max_area: Maximum area of text regions
            
        Returns:
            List of tuples (region_rect, region_image)
        """
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Apply dilation to connect edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by area
        text_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area <= area <= max_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract region
                region = image[y:y+h, x:x+w]
                
                # Add to list
                text_regions.append(((x, y, w, h), region))
        
        return text_regions
    
    def detect_numbers(self, image: np.ndarray, 
                      min_confidence: float = 0.3) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """
        Detect numbers in an image.
        
        Args:
            image: Input image
            min_confidence: Minimum confidence for OCR results
            
        Returns:
            List of tuples (number, bounding_box, confidence)
        """
        if self.reader is None:
            logger.warning("OCR reader not available")
            return []
        
        try:
            # Extract text using EasyOCR with allowlist for numbers
            results = self.reader.readtext(
                image,
                detail=1,
                paragraph=False,
                allowlist="0123456789."
            )
            
            # Filter results by confidence
            filtered_results = [result for result in results if result[2] >= min_confidence]
            
            # Convert to desired format
            detected_numbers = []
            for result in filtered_results:
                bbox = result[0]
                text = result[1]
                confidence = result[2]
                
                # Convert bbox to (x, y, w, h) format
                x_min = min(point[0] for point in bbox)
                y_min = min(point[1] for point in bbox)
                x_max = max(point[0] for point in bbox)
                y_max = max(point[1] for point in bbox)
                
                rect = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
                
                detected_numbers.append((text, rect, confidence))
            
            return detected_numbers
        
        except Exception as e:
            logger.error(f"Error during number detection: {e}", exc_info=True)
            return []


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image for better OCR.
    
    Args:
        image: Input image
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    return thresh


def apply_threshold(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    Apply binary threshold to an image.
    
    Args:
        image: Input image
        threshold: Threshold value (0-255)
        
    Returns:
        Thresholded image
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    return thresh


# Test function
def test_ocr_helper():
    """Test the OCR helper functionality."""
    import time
    import os
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Initialize OCR helper
    ocr_helper = OCRHelper()
    
    # Test with a simple image
    test_image = np.ones((100, 300), dtype=np.uint8) * 255
    cv2.putText(test_image, "Pot: $123 45", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Save test image
    cv2.imwrite("test_ocr.png", test_image)
    
    # Extract text
    start_time = time.time()
    text = ocr_helper.read_text(test_image)
    end_time = time.time()
    
    print(f"Extracted text: '{text}' in {end_time - start_time:.3f} seconds")
    
    # Extract currency
    start_time = time.time()
    amount = ocr_helper.read_currency(test_image)
    end_time = time.time()
    
    print(f"Extracted currency: ${amount:.2f} in {end_time - start_time:.3f} seconds")
    
    # Enhance text image
    enhanced = ocr_helper.enhance_text_image(test_image)
    cv2.imwrite("enhanced_ocr.png", enhanced)
    
    print("OCR helper test completed")


if __name__ == "__main__":
    # Run test
    test_ocr_helper()