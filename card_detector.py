import cv2
import numpy as np
import os
import logging
import json
from collections import Counter
import tensorflow as tf
import time

# Set up logging - reduced verbosity
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("CardDetector")

class ImprovedCardDetector:
    """
    Neural network-based card detector with OpenCV fallback for poker games
    """
    
    def __init__(self, model_path="card_model.h5", template_dir="card_templates", debug_mode=True, save_debug_images=True):
        self.template_dir = template_dir
        self.card_values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.card_suits = ['hearts', 'diamonds', 'clubs', 'spades']
        
        # Configure detection parameters
        self.white_threshold = 170
        self.min_white_percent = 20
        self.debug_mode = debug_mode
        self.save_debug_images = save_debug_images
        
        # Create debug directory if needed
        if self.save_debug_images:
            self.debug_dir = os.path.join(template_dir, "debug_images")
            os.makedirs(self.debug_dir, exist_ok=True)
        
        # Ensure template directory exists
        os.makedirs(template_dir, exist_ok=True)
        
        # Load corrections
        self.corrections = self._load_corrections()
        
        # Cache for detected cards to improve performance
        self._detection_cache = {}
        
        # Target size for the neural network model
        self.img_height, self.img_width = 20, 50
        
        # Define class mapping for model predictions
        self.class_mapping = self._create_class_mapping()
        
        # Load the neural network model
        self.model = None
        self.model_path = model_path
        self._load_model()
        
        # Counter for processed images
        self.processed_count = 0
    
    def _load_model(self):
        """Load the pre-trained TensorFlow model if available"""
        try:
            if os.path.exists(self.model_path):
                start_time = time.time()
                self.model = tf.keras.models.load_model(self.model_path)
                load_time = time.time() - start_time
                logger.info(f"Card detection model loaded from {self.model_path} in {load_time:.2f} seconds")
                
                # Warm up the model with a dummy prediction
                dummy_input = np.zeros((1, self.img_height, self.img_width, 3), dtype=np.float32)
                self.model.predict(dummy_input)
                
                return True
            else:
                logger.warning(f"Card detection model not found at {self.model_path}, using fallback detection methods")
                return False
        except Exception as e:
            logger.error(f"Error loading card detection model: {str(e)}")
            logger.warning("Using fallback detection methods")
            return False
    
    def _load_corrections(self):
        """Load corrections from file"""
        corrections = {}
        correction_file = os.path.join(self.template_dir, "direct_corrections.json")
        
        if os.path.exists(correction_file):
            try:
                with open(correction_file, 'r') as f:
                    corrections = json.load(f)
                if self.debug_mode:
                    logger.info(f"Loaded {len(corrections)} corrections from {correction_file}")
                return corrections
            except Exception as e:
                logger.error(f"Error loading corrections: {str(e)}")
        
        # Create default corrections if file doesn't exist
        default_corrections = {
            # Format: "red_percent|aspect_ratio|complexity": [value, suit]
            "3.5|0.58|22.5": ["3", "clubs"],  # Example: correcting 3 of clubs
            "2.1|0.85|18.2": ["A", "spades"]   # Example: correcting A of spades
        }
        
        # Save default corrections
        try:
            os.makedirs(os.path.dirname(correction_file), exist_ok=True)
            with open(correction_file, 'w') as f:
                json.dump(default_corrections, f, indent=4)
            if self.debug_mode:
                logger.info(f"Created default corrections file with {len(default_corrections)} entries")
        except Exception as e:
            logger.error(f"Error creating corrections file: {str(e)}")
        
        return default_corrections
    
    def save_corrections(self):
        """Save current corrections to file"""
        correction_file = os.path.join(self.template_dir, "direct_corrections.json")
        try:
            with open(correction_file, 'w') as f:
                json.dump(self.corrections, f, indent=4)
            if self.debug_mode:
                logger.info(f"Saved {len(self.corrections)} corrections to {correction_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving corrections: {str(e)}")
            return False
    
    def add_correction(self, feature_key, correct_value, correct_suit):
        """
        Add a correction using the feature key
        
        Args:
            feature_key: Feature signature (red_percent|aspect_ratio|complexity)
            correct_value: Correct card value
            correct_suit: Correct card suit
        """
        # Store the correction
        self.corrections[feature_key] = [correct_value, correct_suit]
        
        # Save to file
        self.save_corrections()
        if self.debug_mode:
            logger.info(f"Added correction: {feature_key} -> {correct_value} of {correct_suit}")
    
    def _compute_card_hash(self, card_img):
        """Compute a hash for a card image to use with caching"""
        if card_img is None or card_img.size == 0:
            return None
        # Simple hashing approach using downsized image mean values
        small_img = cv2.resize(card_img, (16, 16))
        return hash(small_img.tobytes()[:100])
    
    def _preprocess_image(self, card_img):
        """Preprocess the card image for the neural network"""
        try:
            if card_img is None or card_img.size == 0:
                return None
            
            # In this version, we don't resize the image - we pass it as is
            # Just ensure it's in the right color format (RGB) as neural networks typically expect RGB
            if len(card_img.shape) == 3 and card_img.shape[2] == 3:
                # Convert from BGR (OpenCV) to RGB (what most models expect)
                rgb_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)
            else:
                # If grayscale, convert to RGB
                rgb_img = cv2.cvtColor(card_img, cv2.COLOR_GRAY2RGB)
            
            # Normalize pixel values to [0, 1]
            normalized_img = rgb_img.astype(np.float32) / 255.0
            
            # Add batch dimension
            return np.expand_dims(normalized_img, axis=0)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None
            
    def _save_debug_image(self, card_img):
        """Save original image for debugging"""
        try:
            # Create unique filename
            self.processed_count += 1
            filename = f"card_{self.processed_count:04d}"
            
            # Save original image
            original_path = os.path.join(self.debug_dir, f"{filename}_original.png")
            cv2.imwrite(original_path, card_img)
            
            # Convert to RGB for visualization
            if len(card_img.shape) == 3 and card_img.shape[2] == 3:
                rgb_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = cv2.cvtColor(card_img, cv2.COLOR_GRAY2RGB)
            
            # Save visualization of the RGB version
            viz_path = os.path.join(self.debug_dir, f"{filename}_rgb.png")
            rgb_for_save = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
            cv2.imwrite(viz_path, rgb_for_save)
            
            logger.info(f"Saved debug image for card {self.processed_count} to {self.debug_dir}")
        except Exception as e:
            logger.error(f"Error saving debug images: {str(e)}")
    
    def _create_class_mapping(self):
        """Create a mapping between class indices and card names"""
        # Create a list of all possible card combinations
        all_cards = []
        for value in self.card_values:
            for suit in self.card_suits:
                card_name = f"{value.lower()}_of_{suit}"
                all_cards.append(card_name)
        
        # Add special cases if needed
        for extra in ["joker_red", "joker_black", "card_back"]:
            all_cards.append(extra)
        
        # Create mapping dictionary
        class_mapping = {i: card_name for i, card_name in enumerate(all_cards)}
        
        if self.debug_mode:
            logger.info(f"Created class mapping with {len(class_mapping)} classes")
        
        return class_mapping
            
    def _parse_prediction(self, prediction):
        """Parse the model prediction to get value and suit"""
        try:
            # Get the class with highest probability
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_idx])
            
            # If confidence is too low, return None
            if confidence < 0.5:  # Confidence threshold
                logger.warning(f"Low confidence prediction: {confidence:.2f}")
                return None, None
            
            # If we have access to class names from the model, use them
            if hasattr(self.model, 'class_names'):
                class_name = self.model.class_names[predicted_class_idx]
            # Otherwise use our predefined mapping
            elif predicted_class_idx in self.class_mapping:
                class_name = self.class_mapping[predicted_class_idx]
            else:
                logger.warning(f"Unknown class index: {predicted_class_idx}")
                return None, None
            
            # Parse the card value and suit from the class name
            if '_of_' in class_name:
                value, suit = class_name.split('_of_')
                # Convert value to standard format
                value = value.upper() if value in ['j', 'q', 'k', 'a'] else value
                
                # Log the successful prediction
                if self.debug_mode:
                    logger.warning(f"Predicted {value} of {suit} with confidence {confidence:.2f}")
                
                return value, suit
            else:
                logger.warning(f"Unexpected prediction format: {class_name}")
                return None, None
        except Exception as e:
            logger.error(f"Error parsing prediction: {str(e)}")
            return None, None

    def _save_normalized_debug_image(self, normalized_img, img_batch):
        """Save normalized image used for neural network input for debugging"""
        try:
            # Create unique filename based on processed count
            filename = f"card_{self.processed_count:04d}"
            
            # Convert normalized image back to uint8 for saving (scale back to 0-255)
            normalized_for_save = (normalized_img * 255).astype(np.uint8)
            
            # Save the normalized RGB image
            normalized_path = os.path.join(self.debug_dir, f"{filename}_normalized.png")
            
            # OpenCV expects BGR for saving, so convert from RGB
            normalized_for_save_bgr = cv2.cvtColor(normalized_for_save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(normalized_path, normalized_for_save_bgr)
            
            # Also save the exact batch input (first item from batch) as a numpy array
            # This is the exact data fed to the neural network
            batch_path = os.path.join(self.debug_dir, f"{filename}_batch_input.npy")
            np.save(batch_path, img_batch[0])
            
            logger.info(f"Saved normalized debug image for card {self.processed_count} to {normalized_path}")
        except Exception as e:
            logger.error(f"Error saving normalized debug image: {str(e)}")

    def detect_card(self, card_img):
        """
        Detect card value and suit using neural network with OpenCV fallback
        
        Args:
            card_img: Image of a card (in BGR format from OpenCV)
                    
        Returns:
            tuple: (value, suit) or default values if detection fails
        """
        try:
            if card_img is None or card_img.size == 0:
                return '?', '?'
            
            # Save original image for debugging without modifications
            if self.save_debug_images:
                self._save_debug_image(card_img.copy())
            
            # Check cache first for improved performance
            card_hash = self._compute_card_hash(card_img)
            if card_hash in self._detection_cache:
                return self._detection_cache[card_hash]
            
            # Try neural network prediction first if model is available
            if self.model is not None:
                try:
                    # Make a clean copy of the image
                    img = card_img.copy()
                    
                    # Resize the image to target size for the model
                    resized_img = cv2.resize(img, ( self.img_height, self.img_width))
                    
                    # Convert from BGR (OpenCV) to RGB (what most models expect)
                    if len(resized_img.shape) == 3 and resized_img.shape[2] == 3:
                        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                    else:
                        # If grayscale, convert to RGB
                        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
                    
                    # Normalize pixel values to [0,1]
                    normalized_img = rgb_img.astype(np.float32) / 255.0
                    
                    # Add batch dimension
                    img_batch = np.expand_dims(normalized_img, axis=0)
                    
                    # Save normalized image for debugging
                    if self.save_debug_images:
                        self._save_normalized_debug_image(normalized_img, img_batch)
                    
                    # Make prediction
                    prediction = self.model.predict(img_batch, verbose=1)
                    
                    # Get the class with highest probability
                    predicted_class_idx = np.argmax(prediction[0])
                    confidence = float(prediction[0][predicted_class_idx])
                    
                    # If confidence is too low, fall back to traditional methods
                    if confidence < 0.5:  # Confidence threshold
                        logger.warning(f"Low confidence prediction: {confidence:.2f}")
                    else:
                        # If we have access to class names from the model, use them
                        if hasattr(self.model, 'class_names'):
                            class_name = self.model.class_names[predicted_class_idx]
                        # Otherwise use our predefined mapping
                        elif predicted_class_idx in self.class_mapping:
                            class_name = self.class_mapping[predicted_class_idx]
                        else:
                            logger.warning(f"Unknown class index: {predicted_class_idx}")
                            # Fall back to traditional methods
                            pass
                        
                        # Parse the card value and suit from the class name
                        if '_of_' in class_name:
                            value, suit = class_name.split('_of_')
                            # Convert value to standard format
                            value = value.upper() if value in ['j', 'q', 'k', 'a'] else value
                            
                            # Log the successful prediction
                            if self.debug_mode:
                                logger.info(f"Predicted {value} of {suit} with confidence {confidence:.2f}")
                            
                            # Cache the result
                            self._detection_cache[card_hash] = (value, suit)
                            return value, suit
                except Exception as e:
                    logger.warning(f"Neural network prediction failed: {str(e)}")
                    # Fall back to traditional detection methods
            
            # FALLBACK: Use traditional OpenCV-based detection
            
            # STEP 1: Determine if card is red or black
            red_percent = self._calculate_red_percentage(card_img)
            is_red = red_percent > 5
            
            # STEP 2: Extract value and suit regions
            height, width = card_img.shape[:2]
            
            # Value is typically in the top-left corner
            value_region = card_img[0:int(height*0.6), 0:int(width)]
            
            # Suit is typically in the center
            suit_region = card_img[int(height*0.6):int(height), 0:int(width)]
            
            # STEP 3: Analyze shapes to determine value and suit
            value_features = self._calculate_shape_features(value_region, not is_red)
            suit_features = self._calculate_shape_features(suit_region, not is_red)
            
            # STEP 4: Create a feature signature that uniquely identifies this card
            feature_key = f"{red_percent:.1f}|{value_features['aspect_ratio']:.2f}|{value_features['complexity']:.1f}"
            
            # STEP 5: Check if we have a direct correction for this feature signature
            if feature_key in self.corrections:
                # Use the correction
                value, suit = self.corrections[feature_key]
                
                # Cache the result
                self._detection_cache[card_hash] = (value, suit)
                return value, suit
            
            # STEP 6: Perform direct classification using shape features
            value = self._classify_value(value_features, is_red)
            suit = self._classify_suit(suit_features, is_red)
            
            # STEP 7: Apply specific rules for problematic cases like 3 of clubs vs Ace of spades
            if value == 'A' and suit == 'spades' and not is_red:
                # The classic 3 of clubs vs Ace of spades problem
                if 0.5 <= value_features['aspect_ratio'] <= 0.65 and value_features['complexity'] >= 20:
                    value = '3'
                    suit = 'clubs'
            
            # Cache the result
            self._detection_cache[card_hash] = (value, suit)
            
            return value, suit
            
        except Exception as e:
            logger.error(f"Error detecting card: {str(e)}")
            return '?', '?'
     
    def _calculate_red_percentage(self, img):
        """Calculate percentage of red pixels in the image"""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Red detection (two ranges)
            red_lower1 = np.array([0, 50, 50])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([160, 50, 50])
            red_upper2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Calculate percentage of red pixels
            red_pixels = cv2.countNonZero(red_mask)
            total_pixels = img.shape[0] * img.shape[1]
            red_percent = red_pixels / max(1, total_pixels) * 100
            
            return red_percent
            
        except Exception as e:
            logger.error(f"Error calculating red percentage: {str(e)}")
            return 0  # Default to 0% red
    
    def _calculate_shape_features(self, region_img, invert_threshold=False):
        """
        Calculate shape features for value or suit region
        
        Args:
            region_img: Image of the region (value or suit)
            invert_threshold: Whether to invert thresholding (for black regions)
            
        Returns:
            dict: Shape features for classification
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            thresh_mode = cv2.THRESH_BINARY_INV if invert_threshold else cv2.THRESH_BINARY
            _, thresh = cv2.threshold(gray, 150, 255, thresh_mode)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Default features if no contours found
            default_features = {
                'aspect_ratio': 1.0,
                'complexity': 10.0,
                'area_ratio': 0.5,
                'contour_count': 0
            }
            
            if not contours:
                return default_features
            
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate basic measurements
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / max(h, 1)  # Avoid division by zero
            
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Calculate complexity (perimeter^2 / area)
            complexity = (perimeter**2) / max(area, 1)
            
            # Calculate area ratio (contour area / bounding rect area)
            rect_area = w * h
            area_ratio = area / max(rect_area, 1)
            
            return {
                'aspect_ratio': aspect_ratio,
                'complexity': complexity,
                'area_ratio': area_ratio,
                'contour_count': len(contours)
            }
            
        except Exception as e:
            logger.error(f"Error calculating shape features: {str(e)}")
            return {
                'aspect_ratio': 1.0,
                'complexity': 10.0,
                'area_ratio': 0.5,
                'contour_count': 0
            }
    
    def _classify_value(self, features, is_red):
        """
        Classify card value based on shape features
        
        Args:
            features: Shape features
            is_red: Whether the card is red
            
        Returns:
            str: Card value
        """
        # Simple rule-based classification based on shape metrics
        aspect_ratio = features['aspect_ratio']
        complexity = features['complexity']
        
        # Start with highly distinctive shapes
        if aspect_ratio < 0.5:
            return '1'  # Very narrow, likely '1' from '10'
        elif aspect_ratio > 1.2:
            return '2'  # Very wide shape
        
        # Next, check for specific value shapes
        if 0.5 <= aspect_ratio <= 0.65:
            if complexity >= 20:
                return '3'  # '3' has a complex, curvy shape
            else:
                return 'J'  # 'J' is narrow but simpler
        elif 0.65 <= aspect_ratio <= 0.75:
            if complexity >= 25:
                return 'Q'  # 'Q' has high complexity
            else:
                return '5'  # '5' has medium complexity
        elif 0.75 <= aspect_ratio <= 0.85:
            if complexity >= 20:
                return 'K'  # 'K' has high complexity
            else:
                return '7'  # '7' has medium-high aspect ratio
        elif 0.85 <= aspect_ratio <= 0.95:
            if complexity <= 15:
                return '4'  # '4' has a simple shape
            else:
                return '8'  # '8' has a more complex shape
        elif 0.95 <= aspect_ratio <= 1.05:
            if complexity <= 18:
                return '9'  # '9' has medium complexity
            else:
                return '6'  # '6' has higher complexity
        elif aspect_ratio > 1.05:
            if complexity < 15:
                return 'A'  # 'A' typically has a wide, simple shape
            else:
                return '10'  # '10' has a wide, complex shape
        
        # Default fallback
        return 'A'
    
    def _classify_suit(self, features, is_red):
        """
        Classify card suit based on shape features
        
        Args:
            features: Shape features
            is_red: Whether the card is red
            
        Returns:
            str: Card suit
        """
        # Filter possible suits by color first
        if is_red:
            # Red suits: hearts and diamonds
            if features['aspect_ratio'] > 1.0 or features['complexity'] < 20:
                return 'diamonds'  # Diamonds typically have a simpler shape and higher aspect ratio
            else:
                return 'hearts'    # Hearts typically have a more complex shape
        else:
            # Black suits: clubs and spades
            if features['complexity'] > 20:
                return 'clubs'     # Clubs typically have a more complex shape
            else:
                return 'spades'    # Spades typically have a simpler shape
    
    def clear_cache(self):
        """Clear the detection cache"""
        self._detection_cache.clear()


class EnhancedTextRecognition:
    """Enhanced text recognition for chip counts and other poker text with improved performance"""
    
    def __init__(self, debug_mode=False):
        self.text_colors = ['white', 'yellow', 'green']
        self.debug_mode = debug_mode
        
        # Results cache to avoid redundant processing
        self._cache = {}
    
    def _compute_region_hash(self, img, region):
        """Compute a hash for the region to use in caching"""
        if img is None or region is None:
            return None
            
        x, y, w, h = region
        
        # Ensure region is within image bounds
        if x < 0 or y < 0 or x+w > img.shape[1] or y+h > img.shape[0]:
            return None
            
        # Extract the region
        chip_img = img[y:y+h, x:x+w]
        
        # Use a simple hash of the image bytes
        if hasattr(chip_img, 'tobytes'):
            return hash(chip_img.tobytes()[:1000])  # Use first 1000 bytes for speed
        return hash(str(chip_img.mean()))  # Fallback hash
    
    def extract_chip_count(self, img, region):
        """
        Extract chip count using multiple methods with improved performance
        
        Args:
            img: Source image
            region: Region tuple (x, y, w, h)
            
        Returns:
            int: Extracted chip count
        """
        try:
            # Check cache first to avoid redundant processing
            region_hash = self._compute_region_hash(img, region)
            if region_hash in self._cache:
                return self._cache[region_hash]
            
            x, y, w, h = region
            
            # Ensure region is within image bounds
            if x < 0 or y < 0 or x+w > img.shape[1] or y+h > img.shape[0]:
                return 0
            
            # Extract the region
            chip_img = img[y:y+h, x:x+w]
            
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
                    result_counts = Counter(results)
                    final_result = result_counts.most_common(1)[0][0]
                
                # Cache the result
                self._cache[region_hash] = final_result
                
                return final_result
            
            # If all methods fail, return default
            default_value = 1000
            self._cache[region_hash] = default_value
            return default_value
        
        except Exception as e:
            logger.error(f"Error extracting chip count: {str(e)}")
            return 1000
    
    def _extract_number_basic(self, img):
        """Basic OCR with minimal preprocessing"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding - try both regular and inverse thresholding
            _, thresh1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            _, thresh2 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # OCR on both thresholded images
            # In a real implementation, this would use pytesseract
            # For simplicity, let's simulate OCR with a random number
            text1 = "$" + str(np.random.randint(1000, 9999))
            text2 = "$" + str(np.random.randint(1000, 9999))
            
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
            
            # OCR on mask
            # In a real implementation, this would use pytesseract
            # For simplicity, let's simulate OCR with a random number
            text = "$" + str(np.random.randint(1000, 9999))
            
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
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)
            
            # OCR
            # In a real implementation, this would use pytesseract
            # For simplicity, let's simulate OCR with a random number
            text = "$" + str(np.random.randint(1000, 9999))
            
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
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 100, 200)
            
            # Dilate to connect edges
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # OCR
            # In a real implementation, this would use pytesseract
            # For simplicity, let's simulate OCR with a random number
            text = "$" + str(np.random.randint(1000, 9999))
            
            # Extract number
            import re
            match = re.search(r'\$?(\d+)', text)
            if match:
                return int(match.group(1)), 0.6
            
            return 0, 0
        
        except Exception as e:
            logger.error(f"Error in edge enhanced OCR: {str(e)}")
            return 0, 0
            
    def clear_cache(self):
        """Clear the recognition cache"""
        self._cache.clear()