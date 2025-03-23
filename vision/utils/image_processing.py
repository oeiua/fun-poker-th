#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Processing Utilities

This module provides common image processing functions used by various
components of the poker vision assistant.
"""

import logging
import cv2
import numpy as np
from typing import List, Tuple, Optional, Union, Any

logger = logging.getLogger("PokerVision.Utils.ImageProcessing")

def preprocess_image(image: np.ndarray, 
                    grayscale: bool = True, 
                    blur: bool = True,
                    threshold: bool = True,
                    adaptive: bool = True,
                    morphology: bool = False) -> np.ndarray:
    """
    Preprocess an image for better feature detection and OCR.
    
    Args:
        image: Input image (BGR format)
        grayscale: Whether to convert to grayscale
        blur: Whether to apply Gaussian blur
        threshold: Whether to apply thresholding
        adaptive: Whether to use adaptive thresholding (ignored if threshold is False)
        morphology: Whether to apply morphological operations
        
    Returns:
        Preprocessed image
    """
    # Make a copy of the input image
    result = image.copy()
    
    # Convert to grayscale
    if grayscale and len(result.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    if blur:
        result = cv2.GaussianBlur(result, (5, 5), 0)
    
    # Apply thresholding
    if threshold:
        if adaptive:
            # Adaptive thresholding
            result = cv2.adaptiveThreshold(
                result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            # Global thresholding
            _, result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations
    if morphology:
        kernel = np.ones((3, 3), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    return result

def apply_threshold(image: np.ndarray, threshold_value: int = 127, 
                   inverse: bool = False) -> np.ndarray:
    """
    Apply simple binary thresholding to an image.
    
    Args:
        image: Input image
        threshold_value: Threshold value (0-255)
        inverse: Whether to apply inverse thresholding
        
    Returns:
        Thresholded image
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply threshold
    thresh_type = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(gray, threshold_value, 255, thresh_type)
    
    return thresh

def resize_image(image: np.ndarray, width: Optional[int] = None, 
               height: Optional[int] = None,
               max_size: Optional[int] = None,
               keep_aspect: bool = True) -> np.ndarray:
    """
    Resize an image to the specified width and/or height.
    
    Args:
        image: Input image
        width: Target width, or None to calculate from height
        height: Target height, or None to calculate from width
        max_size: Maximum dimension (overrides width and height)
        keep_aspect: Whether to keep the aspect ratio
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    # Apply max_size constraint if provided
    if max_size is not None:
        if w > h and w > max_size:
            # Width is the limiting factor
            width = max_size
            height = None
        elif h > w and h > max_size:
            # Height is the limiting factor
            height = max_size
            width = None
        elif w == h and w > max_size:
            # Both dimensions equal and too large
            width = max_size
            height = max_size
    
    # Calculate new dimensions
    if width is None and height is None:
        return image.copy()  # No resizing needed
    
    if keep_aspect:
        if width is None:
            # Calculate width from height
            width = int(w * (height / h))
        elif height is None:
            # Calculate height from width
            height = int(h * (width / w))
    else:
        # Use default dimensions if not specified
        if width is None:
            width = w
        if height is None:
            height = h
    
    # Resize the image
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
    return resized

def rotate_image(image: np.ndarray, angle: float, 
               center: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Rotate an image around its center.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees (positive for counterclockwise)
        center: Rotation center, or None for image center
        
    Returns:
        Rotated image
    """
    h, w = image.shape[:2]
    
    # Default to image center
    if center is None:
        center = (w // 2, h // 2)
    
    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    
    return rotated

def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Crop an image to the specified region.
    
    Args:
        image: Input image
        x: X-coordinate of the top-left corner
        y: Y-coordinate of the top-left corner
        width: Width of the crop region
        height: Height of the crop region
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    
    # Ensure coordinates are within image boundaries
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    width = max(1, min(width, w - x))
    height = max(1, min(height, h - y))
    
    # Crop the image
    cropped = image[y:y+height, x:x+width]
    
    return cropped

def detect_edges(image: np.ndarray, low_threshold: int = 50, 
               high_threshold: int = 150,
               blur_size: int = 5) -> np.ndarray:
    """
    Detect edges in an image using the Canny edge detector.
    
    Args:
        image: Input image
        low_threshold: Low threshold for the hysteresis procedure
        high_threshold: High threshold for the hysteresis procedure
        blur_size: Size of the Gaussian blur kernel
        
    Returns:
        Edge mask image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    return edges

def find_contours(image: np.ndarray, external_only: bool = True, 
                min_area: int = 100, max_area: Optional[int] = None) -> List[np.ndarray]:
    """
    Find contours in a binary image.
    
    Args:
        image: Binary input image
        external_only: Whether to return only external contours
        min_area: Minimum contour area
        max_area: Maximum contour area, or None for no limit
        
    Returns:
        List of contours
    """
    # Ensure the image is binary
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contour_mode = cv2.RETR_EXTERNAL if external_only else cv2.RETR_LIST
    contours, _ = cv2.findContours(image, contour_mode, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area and (max_area is None or area <= max_area):
            filtered_contours.append(contour)
    
    return filtered_contours

def draw_contours(image: np.ndarray, contours: List[np.ndarray], 
                color: Tuple[int, int, int] = (0, 255, 0),
                thickness: int = 2) -> np.ndarray:
    """
    Draw contours on an image.
    
    Args:
        image: Input image
        contours: List of contours to draw
        color: Color of the contours (BGR format)
        thickness: Thickness of the contour lines
        
    Returns:
        Image with drawn contours
    """
    # Create a copy of the image
    result = image.copy()
    
    # Draw the contours
    cv2.drawContours(result, contours, -1, color, thickness)
    
    return result

def adjust_brightness_contrast(image: np.ndarray, 
                             alpha: float = 1.0, 
                             beta: int = 0) -> np.ndarray:
    """
    Adjust the brightness and contrast of an image.
    
    Args:
        image: Input image
        alpha: Contrast control (1.0 for unchanged)
        beta: Brightness control (0 for unchanged)
        
    Returns:
        Adjusted image
    """
    # Apply contrast and brightness adjustment
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    return adjusted

def enhance_text(image: np.ndarray) -> np.ndarray:
    """
    Enhance an image for better text recognition.
    
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
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply morphological operations to enhance text
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    return eroded

def detect_circles(image: np.ndarray, 
                 min_radius: int = 10, 
                 max_radius: int = 50,
                 param1: int = 50,
                 param2: int = 30) -> List[Tuple[int, int, int]]:
    """
    Detect circles in an image using the Hough Circle Transform.
    
    Args:
        image: Input image
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
        param1: First method-specific parameter (Canny edge detector threshold)
        param2: Second method-specific parameter (accumulator threshold)
        
    Returns:
        List of detected circles as (x, y, radius)
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect circles
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius
    )
    
    # Process results
    result = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for x, y, r in circles:
            result.append((x, y, r))
    
    return result

def template_matching(image: np.ndarray, template: np.ndarray, 
                    threshold: float = 0.8,
                    method: int = cv2.TM_CCOEFF_NORMED) -> List[Tuple[int, int, float]]:
    """
    Find occurrences of a template in an image.
    
    Args:
        image: Input image
        template: Template to search for
        threshold: Minimum match confidence
        method: Template matching method
        
    Returns:
        List of matches as (x, y, confidence)
    """
    # Ensure image and template have the same number of channels
    if len(image.shape) != len(template.shape):
        if len(image.shape) == 3 and len(template.shape) == 2:
            # Convert image to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2 and len(template.shape) == 3:
            # Convert template to grayscale
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Get template dimensions
    h, w = template.shape[:2]
    
    # Apply template matching
    result = cv2.matchTemplate(image, template, method)
    
    # Find locations where the match exceeds the threshold
    locations = np.where(result >= threshold)
    
    # Process results
    matches = []
    for pt in zip(*locations[::-1]):
        confidence = result[pt[1], pt[0]]
        matches.append((pt[0], pt[1], confidence))
    
    # Apply non-maximum suppression to remove overlapping matches
    matches = non_max_suppression(matches, h, w)
    
    return matches

def non_max_suppression(matches: List[Tuple[int, int, float]], 
                       template_height: int, 
                       template_width: int,
                       overlap_threshold: float = 0.3) -> List[Tuple[int, int, float]]:
    """
    Apply non-maximum suppression to remove overlapping matches.
    
    Args:
        matches: List of matches as (x, y, confidence)
        template_height: Height of the template
        template_width: Width of the template
        overlap_threshold: Maximum allowed overlap ratio
        
    Returns:
        Filtered list of matches
    """
    if not matches:
        return []
    
    # Sort matches by confidence (highest first)
    matches.sort(key=lambda x: x[2], reverse=True)
    
    # Initialize list of kept matches
    keep = []
    
    # Iterate through matches
    for match in matches:
        x1, y1, conf = match
        x2, y2 = x1 + template_width, y1 + template_height
        
        # Check for overlap with kept matches
        overlap = False
        for kept_match in keep:
            kx1, ky1, _ = kept_match
            kx2, ky2 = kx1 + template_width, ky1 + template_height
            
            # Calculate overlap area
            overlap_width = max(0, min(x2, kx2) - max(x1, kx1))
            overlap_height = max(0, min(y2, ky2) - max(y1, ky1))
            overlap_area = overlap_width * overlap_height
            
            # Calculate areas
            area1 = template_width * template_height
            area2 = template_width * template_height
            
            # Calculate overlap ratio
            overlap_ratio = overlap_area / min(area1, area2)
            
            if overlap_ratio > overlap_threshold:
                overlap = True
                break
        
        # If no significant overlap, keep the match
        if not overlap:
            keep.append(match)
    
    return keep

def color_filter(image: np.ndarray, 
               lower_hsv: Tuple[int, int, int], 
               upper_hsv: Tuple[int, int, int]) -> np.ndarray:
    """
    Filter an image by HSV color range.
    
    Args:
        image: Input image (BGR format)
        lower_hsv: Lower bound of HSV range
        upper_hsv: Upper bound of HSV range
        
    Returns:
        Binary mask of the filtered colors
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    
    return mask

def create_color_mask(image: np.ndarray, 
                    colors: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]) -> np.ndarray:
    """
    Create a binary mask for multiple color ranges.
    
    Args:
        image: Input image (BGR format)
        colors: List of (lower_hsv, upper_hsv) tuples
        
    Returns:
        Binary mask of the filtered colors
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create empty mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Apply each color range
    for lower_hsv, upper_hsv in colors:
        color_mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
        mask = cv2.bitwise_or(mask, color_mask)
    
    return mask

def debug_show(image: np.ndarray, title: str = "Debug", wait: bool = True):
    """
    Show an image for debugging purposes.
    
    Args:
        image: Image to display
        title: Window title
        wait: Whether to wait for a key press
    """
    cv2.imshow(title, image)
    if wait:
        cv2.waitKey(0)
        cv2.destroyWindow(title)

def save_debug_image(image: np.ndarray, filename: str) -> bool:
    """
    Save an image for debugging purposes.
    
    Args:
        image: Image to save
        filename: Output filename
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cv2.imwrite(filename, image)
        logger.debug(f"Saved debug image to {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to save debug image: {e}")
        return False


# Test function
def test_image_processing():
    """Test the image processing functions."""
    import os
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create a test image
    test_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    # Draw some elements
    cv2.rectangle(test_image, (50, 50), (150, 100), (0, 0, 255), 2)
    cv2.rectangle(test_image, (200, 50), (300, 100), (0, 255, 0), 2)
    cv2.circle(test_image, (100, 200), 30, (255, 0, 0), 2)
    cv2.putText(test_image, "Test Image", (120, 180), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Save original image
    save_debug_image(test_image, "test_original.png")
    
    # Test preprocess_image
    preprocessed = preprocess_image(test_image)
    save_debug_image(preprocessed, "test_preprocessed.png")
    
    # Test apply_threshold
    thresholded = apply_threshold(test_image, 200)
    save_debug_image(thresholded, "test_thresholded.png")
    
    # Test resize_image
    resized = resize_image(test_image, width=200)
    save_debug_image(resized, "test_resized.png")
    
    # Test rotate_image
    rotated = rotate_image(test_image, 45)
    save_debug_image(rotated, "test_rotated.png")
    
    # Test crop_image
    cropped = crop_image(test_image, 100, 50, 200, 150)
    save_debug_image(cropped, "test_cropped.png")
    
    # Test detect_edges
    edges = detect_edges(test_image)
    save_debug_image(edges, "test_edges.png")
    
    # Test find_contours and draw_contours
    contours = find_contours(thresholded)
    contour_image = draw_contours(test_image.copy(), contours)
    save_debug_image(contour_image, "test_contours.png")
    
    # Test adjust_brightness_contrast
    adjusted = adjust_brightness_contrast(test_image, alpha=1.5, beta=30)
    save_debug_image(adjusted, "test_adjusted.png")
    
    # Test enhance_text
    enhanced = enhance_text(test_image)
    save_debug_image(enhanced, "test_enhanced.png")
    
    # Test detect_circles
    circle_image = test_image.copy()
    circles = detect_circles(test_image, min_radius=20, max_radius=40)
    for x, y, r in circles:
        cv2.circle(circle_image, (x, y), r, (0, 255, 255), 2)
    save_debug_image(circle_image, "test_circles.png")
    
    # Test color_filter
    mask_red = color_filter(test_image, (0, 100, 100), (10, 255, 255))
    save_debug_image(mask_red, "test_mask_red.png")
    
    logger.info("Image processing tests completed")


if __name__ == "__main__":
    # Run test
    test_image_processing()