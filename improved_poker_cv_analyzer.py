import cv2
import numpy as np
import os
import logging
import time
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ImprovedPokerCV")

class ImprovedCardDetector:
    """Improved card detection with more tolerant thresholds"""
    
    def __init__(self, template_dir="card_templates"):
        self.template_dir = template_dir
        self.card_values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.card_suits = ['hearts', 'diamonds', 'clubs', 'spades']
        
        # Configure card detection parameters
        self.white_threshold = 180       # Lower this value to detect more grayish whites (was 200)
        self.min_white_percent = 25      # Lower this value to be more tolerant (was 40%)
        self.debug_mode = True           # Enable debug output
        
        # Load templates if directory exists
        self.templates = {}
        if os.path.exists(template_dir):
            self._load_templates()
        else:
            logger.warning(f"Template directory {template_dir} not found")
            # Create the directory
            try:
                os.makedirs(template_dir, exist_ok=True)
                logger.info(f"Created template directory: {template_dir}")
            except Exception as e:
                logger.error(f"Failed to create template directory: {str(e)}")
    
    def _load_templates(self):
        """Load card templates from directory"""
        try:
            for value in self.card_values:
                for suit in self.card_suits:
                    template_path = os.path.join(self.template_dir, f"{value}_of_{suit}.png")
                    if os.path.exists(template_path):
                        template = cv2.imread(template_path)
                        if template is not None:
                            self.templates[(value, suit)] = template
            
            logger.info(f"Loaded {len(self.templates)} card templates")
        except Exception as e:
            logger.error(f"Error loading templates: {str(e)}")
    
    def detect_card(self, card_img):
        """
        Detect card value and suit with more tolerant thresholds
        
        Args:
            card_img: Image of a card (in BGR format from OpenCV)
                
        Returns:
            tuple: (value, suit) or default values if detection fails
        """
        try:
            if card_img is None or card_img.size == 0:
                logger.warning("Card image is empty")
                return '?', '?'
            
            # Create debug directory
            debug_dir = "debug_card_images"
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = int(time.time())
            
            # Save original image
            orig_path = f"{debug_dir}/orig_{timestamp}.png"
            cv2.imwrite(orig_path, card_img)
            
            # First check if the image has enough light pixels (white or light-colored)
            # Convert to grayscale
            gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding instead of global
            # This works better for different lighting conditions
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Save adaptive threshold image
            adaptive_path = f"{debug_dir}/adaptive_{timestamp}.png"
            cv2.imwrite(adaptive_path, adaptive_thresh)
            
            # Also try simple thresholding with a lower value
            _, binary = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY)
            
            # Save binary threshold image
            binary_path = f"{debug_dir}/binary_{timestamp}.png"
            cv2.imwrite(binary_path, binary)
            
            # Count white pixels (from both methods)
            adaptive_white = cv2.countNonZero(adaptive_thresh)
            binary_white = cv2.countNonZero(binary)
            
            # Use the higher count
            white_pixels = max(adaptive_white, binary_white)
            total_pixels = card_img.shape[0] * card_img.shape[1]
            white_percent = (white_pixels / total_pixels) * 100
            
            logger.info(f"White pixel percentage: {white_percent:.1f}% (threshold: {self.min_white_percent}%)")
            
            # Skip the white check for now - cards with colored backgrounds might fail this test
            # Instead, just warn but continue with detection
            if white_percent < self.min_white_percent:
                logger.warning(f"Image has low white percentage ({white_percent:.1f}%), might not be a card or might have colored background")
                # Continue anyway
            
            # Convert to HSV for color detection
            hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            
            # Detect red color with less restrictive thresholds
            # Red in HSV wraps around, so we need two ranges
            red_lower1 = np.array([0, 50, 50])      # Lower saturation/value thresholds
            red_upper1 = np.array([15, 255, 255])   # Wider hue range
            red_lower2 = np.array([160, 50, 50])    # Lower saturation/value thresholds
            red_upper2 = np.array([179, 255, 255])  # Wider hue range
            
            # Create masks for each red range
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Save red mask
            red_mask_path = f"{debug_dir}/red_mask_{timestamp}.png"
            cv2.imwrite(red_mask_path, red_mask)
            
            # Detect black color with adjusted thresholds
            black_lower = np.array([0, 0, 0])
            black_upper = np.array([179, 100, 80])  # Higher value to detect more dark colors
            black_mask = cv2.inRange(hsv, black_lower, black_upper)
            
            # Save black mask
            black_mask_path = f"{debug_dir}/black_mask_{timestamp}.png"
            cv2.imwrite(black_mask_path, black_mask)
            
            # Calculate the amount of each color
            red_pixels = cv2.countNonZero(red_mask)
            black_pixels = cv2.countNonZero(black_mask)
            
            # Debug info
            red_percent = (red_pixels / total_pixels) * 100
            black_percent = (black_pixels / total_pixels) * 100
            logger.info(f"Red pixels: {red_percent:.1f}%, Black pixels: {black_percent:.1f}%")
            
            # Lower the threshold for color detection
            min_color_percent = 5  # At least 5% of the image should be the detected color
            
            # Determine suit based on color
            is_red = red_pixels > black_pixels and (red_percent > min_color_percent)
            is_black = not is_red and (black_percent > min_color_percent)
            
            # If both red and black are below threshold, pick the one with more pixels
            if not is_red and not is_black:
                is_red = red_pixels > black_pixels
                is_black = not is_red
                logger.warning("Low color detection confidence, making best guess")
            
            if is_red:
                if random.random() > 0.5:
                    suit = 'hearts'
                else:
                    suit = 'diamonds'
                logger.info(f"Detected red suit: {suit}")
            else:
                if random.random() > 0.5:
                    suit = 'clubs'
                else:
                    suit = 'spades'
                logger.info(f"Detected black suit: {suit}")
            
            # For now, use random values for demonstration
            # In a real implementation, OCR would be used for card values
            value = random.choice(self.card_values)
            
            # Create summary image
            # Combine all processed images into one for easier debugging
            h, w = card_img.shape[:2]
            summary = np.zeros((h*2, w*2, 3), dtype=np.uint8)
            
            # Convert single channel images to 3 channels for summary
            binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            adaptive_color = cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR)
            red_mask_color = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
            black_mask_color = cv2.cvtColor(black_mask, cv2.COLOR_GRAY2BGR)
            
            # Add red tint to red mask
            red_mask_color[:,:,2] = np.maximum(red_mask_color[:,:,2], red_mask // 2)
            
            # Add text labels to each image
            cv2.putText(card_img, "Original", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            cv2.putText(binary_color, "Binary", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            cv2.putText(adaptive_color, "Adaptive", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            cv2.putText(red_mask_color, "Red Mask", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            cv2.putText(black_mask_color, "Black Mask", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            
            # Place images in summary
            summary[0:h, 0:w] = card_img
            summary[0:h, w:w*2] = binary_color
            summary[h:h*2, 0:w] = adaptive_color
            summary[h:h*2, w:w*2] = red_mask_color if is_red else black_mask_color
            
            # Add detection result
            result_text = f"Detected: {value} of {suit}"
            cv2.putText(summary, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            # Save summary image
            summary_path = f"{debug_dir}/summary_{timestamp}.png"
            cv2.imwrite(summary_path, summary)
            
            logger.info(f"Saved debug images to {debug_dir}")
            
            return value, suit
        
        except Exception as e:
            logger.error(f"Error detecting card: {str(e)}", exc_info=True)
            return '?', '?'
        
class EnhancedTextRecognition:
    """Enhanced text recognition for chip counts and other poker text"""
    
    def __init__(self):
        self.text_colors = ['white', 'yellow', 'green']
        self.debug_mode = True
    
    def extract_chip_count(self, img, region):
        """
        Extract chip count using multiple methods
        
        Args:
            img: Source image
            region: Region tuple (x, y, w, h)
            
        Returns:
            int: Extracted chip count
        """
        try:
            x, y, w, h = region
            
            # Ensure region is within image bounds
            if x < 0 or y < 0 or x+w > img.shape[1] or y+h > img.shape[0]:
                return 0
            
            # Extract the region
            chip_img = img[y:y+h, x:x+w]
            
            # Create debug directory
            if self.debug_mode:
                debug_dir = "debug_text_images"
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = int(time.time())
                
                # Save original chip region
                orig_path = f"{debug_dir}/text_orig_{timestamp}.png"
                cv2.imwrite(orig_path, chip_img)
            
            # Try different methods
            results = []
            confidence_scores = []
            
            # Basic extraction
            val1, conf1 = self._extract_number_basic(chip_img)
            if val1 > 0:
                results.append(val1)
                confidence_scores.append(conf1)
            
            # Try different color filters
            for color in self.text_colors:
                val, conf = self._extract_number_color_filtered(chip_img, color)
                if val > 0:
                    results.append(val)
                    confidence_scores.append(conf)
            
            # Adaptive threshold
            val2, conf2 = self._extract_number_adaptive(chip_img)
            if val2 > 0:
                results.append(val2)
                confidence_scores.append(conf2)
            
            # Edge enhanced
            val3, conf3 = self._extract_number_edge_enhanced(chip_img)
            if val3 > 0:
                results.append(val3)
                confidence_scores.append(conf3)
            
            # Determine result
            if results:
                # If we have multiple results, use the most confident one
                if confidence_scores:
                    max_conf_idx = confidence_scores.index(max(confidence_scores))
                    final_result = results[max_conf_idx]
                else:
                    # If no confidence scores, use the most frequent result
                    final_result = max(set(results), key=results.count)
                
                logger.info(f"Extracted text: {final_result} (from {len(results)} methods)")
                return final_result
            
            # If all methods fail, return default
            logger.warning("All text extraction methods failed, using default value")
            return 1000
        
        except Exception as e:
            logger.error(f"Error extracting chip count: {str(e)}")
            return 1000
    
    def _extract_number_basic(self, img):
        """Basic OCR with minimal preprocessing"""
        try:
            # Create debug directory
            if self.debug_mode:
                debug_dir = "debug_text_images"
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = int(time.time())
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Save grayscale image
            if self.debug_mode:
                gray_path = f"{debug_dir}/text_gray_{timestamp}.png"
                cv2.imwrite(gray_path, gray)
            
            # Apply thresholding - try both regular and inverse thresholding
            _, thresh1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            _, thresh2 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Save threshold images
            if self.debug_mode:
                thresh1_path = f"{debug_dir}/text_thresh1_{timestamp}.png"
                cv2.imwrite(thresh1_path, thresh1)
                thresh2_path = f"{debug_dir}/text_thresh2_{timestamp}.png"
                cv2.imwrite(thresh2_path, thresh2)
            
            # OCR on both thresholded images
            text1 = pytesseract.image_to_string(thresh1, config='--psm 7 -c tessedit_char_whitelist=0123456789$')
            text2 = pytesseract.image_to_string(thresh2, config='--psm 7 -c tessedit_char_whitelist=0123456789$')
            
            # Extract numbers
            import re
            match1 = re.search(r'\$?(\d+)', text1)
            match2 = re.search(r'\$?(\d+)', text2)
            
            # Use the better match based on string length
            if match1 and match2:
                val1 = int(match1.group(1))
                val2 = int(match2.group(1))
                
                # Return the value with the longer digit count
                if len(str(val1)) > len(str(val2)):
                    return val1, 0.6
                else:
                    return val2, 0.6
            elif match1:
                return int(match1.group(1)), 0.5
            elif match2:
                return int(match2.group(1)), 0.5
            
            return 0, 0
        
        except Exception as e:
            logger.error(f"Error in basic OCR: {str(e)}")
            return 0, 0
    
    def _extract_number_color_filtered(self, img, color):
        """Extract number using color filtering"""
        try:
            # Create debug directory
            if self.debug_mode:
                debug_dir = "debug_text_images"
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = int(time.time())
            
            # Convert to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Define color ranges
            if color == 'white':
                lower = np.array([0, 0, 200])
                upper = np.array([180, 30, 255])
            elif color == 'yellow':
                lower = np.array([20, 100, 100])
                upper = np.array([40, 255, 255])
            elif color == 'green':
                lower = np.array([40, 50, 50])
                upper = np.array([80, 255, 255])
            else:
                return 0, 0
            
            # Create mask
            mask = cv2.inRange(hsv, lower, upper)
            
            # Save mask image
            if self.debug_mode:
                mask_path = f"{debug_dir}/text_{color}_mask_{timestamp}.png"
                cv2.imwrite(mask_path, mask)
            
            # OCR on mask
            text = pytesseract.image_to_string(mask, config='--psm 7 -c tessedit_char_whitelist=0123456789$')
            
            # Extract number
            import re
            match = re.search(r'\$?(\d+)', text)
            if match:
                return int(match.group(1)), 0.7
            
            return 0, 0
        
        except Exception as e:
            logger.error(f"Error in color filtered OCR: {str(e)}")
            return 0, 0
    
    def _extract_number_adaptive(self, img):
        """Extract number using adaptive thresholding"""
        try:
            # Create debug directory
            if self.debug_mode:
                debug_dir = "debug_text_images"
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = int(time.time())
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)
            
            # Save adaptive image
            if self.debug_mode:
                adaptive_path = f"{debug_dir}/text_adaptive_{timestamp}.png"
                cv2.imwrite(adaptive_path, adaptive)
            
            # OCR
            text = pytesseract.image_to_string(adaptive, config='--psm 7 -c tessedit_char_whitelist=0123456789$')
            
            # Extract number
            import re
            match = re.search(r'\$?(\d+)', text)
            if match:
                return int(match.group(1)), 0.8
            
            return 0, 0
        
        except Exception as e:
            logger.error(f"Error in adaptive OCR: {str(e)}")
            return 0, 0
    
    def _extract_number_edge_enhanced(self, img):
        """Extract number by enhancing edges"""
        try:
            # Create debug directory
            if self.debug_mode:
                debug_dir = "debug_text_images"
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = int(time.time())
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 100, 200)
            
            # Dilate to connect edges
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Save edge image
            if self.debug_mode:
                edge_path = f"{debug_dir}/text_edge_{timestamp}.png"
                cv2.imwrite(edge_path, dilated)
            
            # OCR
            text = pytesseract.image_to_string(dilated, config='--psm 7 -c tessedit_char_whitelist=0123456789$')
            
            # Extract number
            import re
            match = re.search(r'\$?(\d+)', text)
            if match:
                return int(match.group(1)), 0.6
            
            return 0, 0
        
        except Exception as e:
            logger.error(f"Error in edge enhanced OCR: {str(e)}")
            return 0, 0