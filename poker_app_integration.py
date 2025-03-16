import sys
import os
import time
import json
import argparse
import threading
import queue
import logging
from datetime import datetime

# Filename: poker_app_integration.py

# Import the modules from our other files
# Assuming they are in the same directory
from poker_screen_grabber import PokerScreenGrabber, WindowInfo
from poker_cv_analyzer import PokerImageAnalyzer
from poker_neural_engine_torch import PokerNeuralNetwork  # Updated import for PyTorch version

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("poker_assistant.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PokerAssistant")

class PokerAssistant:
    """
    Main application that integrates screen grabbing, image analysis, and neural network
    decision making for poker gameplay assistance.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize the poker assistant
        
        Args:
            config_file: Path to configuration file (optional)
        """
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.screen_grabber = PokerScreenGrabber(
            capture_interval=self.config.get('capture_interval', 2.0),
            output_dir=self.config.get('screenshot_dir', 'poker_data/screenshots')
        )
        
        self.image_analyzer = PokerImageAnalyzer()
        
        self.neural_network = PokerNeuralNetwork(
            model_path=self.config.get('model_path', None)
        )
        
        # Queues for communication between threads
        self.screenshot_queue = queue.Queue(maxsize=10)
        self.game_state_queue = queue.Queue(maxsize=10)
        
        # Control flags
        self.running = False
        self.threads = []

        # Latest data
        self.latest_game_state = None
        self.latest_decision = None
        self.latest_screenshot = None
        
        logger.info("Poker Assistant initialized")
    
    def _load_config(self, config_file):
        """Load configuration from a JSON file"""
        default_config = {
            'capture_interval': 2.0,
            'screenshot_dir': 'poker_data/screenshots',
            'game_state_dir': 'poker_data/game_states',
            'model_path': None,
            'auto_play': False,
            'confidence_threshold': 0.7,
            'auto_detect_poker': True,
            'use_mock_data': True
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {str(e)}")
        
        return default_config
    
    def start(self):
        """Start the poker assistant"""
        if self.running:
            logger.warning("Poker Assistant is already running")
            return False
        
        # Check if we have a window selected
        if not hasattr(self.screen_grabber, 'selected_window') or not self.screen_grabber.selected_window:
            if self.config.get('auto_detect_poker', True):
                # Try to automatically find and select a PokerTH window
                window_list = self.screen_grabber.get_window_list()
                poker_windows = [w for w in window_list if "poker" in w.title.lower()]
                
                if poker_windows:
                    # Select the first poker window found
                    success = self.screen_grabber.select_window(poker_windows[0])
                    logger.info(f"Auto-selected window: {poker_windows[0].title}, result: {success}")
                else:
                    if not self.config.get('use_mock_data', True):
                        logger.error("No PokerTH window found and mock data disabled")
                        return False
                    logger.warning("No PokerTH window found, using mock screenshots")
        else:
            logger.info(f"Using previously selected window: {self.screen_grabber.selected_window}")
        
        self.running = True
        
        # Start threads
        self.threads = []
        
        # Screenshot capture thread
        capture_thread = threading.Thread(
            target=self._capture_loop,
            name="ScreenshotCapture"
        )
        capture_thread.daemon = True
        self.threads.append(capture_thread)
        
        # Image analysis thread
        analysis_thread = threading.Thread(
            target=self._analysis_loop,
            name="ImageAnalysis"
        )
        analysis_thread.daemon = True
        self.threads.append(analysis_thread)
        
        # Decision making thread
        decision_thread = threading.Thread(
            target=self._decision_loop,
            name="DecisionMaking"
        )
        decision_thread.daemon = True
        self.threads.append(decision_thread)
        
        # Start all threads
        for thread in self.threads:
            thread.start()
        
        logger.info("Poker Assistant started")
        return True

    def stop(self):
        """Stop the poker assistant"""
        if not self.running:
            logger.warning("Poker Assistant is not running")
            return
        
        self.running = False
        
        # Wait for threads to terminate
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        self.threads = []
        logger.info("Poker Assistant stopped")
    
    def _capture_loop(self):
        """Continuously capture screenshots of the poker game"""
        logger.info("Screenshot capture thread started")
        
        while self.running:
            try:
                # Capture screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Debug info before capture
                if hasattr(self.screen_grabber, 'selected_window'):
                    logger.info(f"About to capture screenshot. Selected window: {self.screen_grabber.selected_window}")
                else:
                    logger.info("About to capture screenshot. No window selected.")
                
                screenshot = self.screen_grabber.capture_screenshot()
                
                if screenshot is not None:
                    logger.info(f"Screenshot captured successfully, shape: {screenshot.shape}")
                    
                    # Store the latest screenshot
                    self.latest_screenshot = screenshot
                    
                    # Save screenshot
                    screenshot_path = os.path.join(
                        self.config['screenshot_dir'],
                        f"screenshot_{timestamp}.png"
                    )
                    os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
                    self.screen_grabber.save_screenshot(screenshot, screenshot_path)
                    
                    # Put in queue for analysis
                    if not self.screenshot_queue.full():
                        self.screenshot_queue.put((screenshot, screenshot_path, timestamp))
                        logger.info(f"Screenshot added to analysis queue: {screenshot_path}")
                    else:
                        logger.warning("Screenshot queue is full, skipping analysis")
                else:
                    logger.warning("Screenshot capture returned None")
                
                # Sleep until next capture
                capture_interval = self.config.get('capture_interval', 2.0)
                logger.info(f"Sleeping for {capture_interval} seconds before next capture")
                time.sleep(capture_interval)
            
            except Exception as e:
                logger.error(f"Error in screenshot capture: {str(e)}", exc_info=True)
                time.sleep(1.0)  # Sleep before retrying
        
        logger.info("Screenshot capture thread stopped")
    
    def _analysis_loop(self):
        """Analyze screenshots to extract game state"""
        logger.info("Image analysis thread started")
        
        while self.running:
            try:
                # Get screenshot from queue
                try:
                    screenshot, screenshot_path, timestamp = self.screenshot_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Analyze the screenshot
                game_state = self.image_analyzer.analyze_image(screenshot_path)
                
                if game_state:
                    # Add timestamp
                    game_state['timestamp'] = timestamp
                    
                    # Save game state
                    game_state_path = os.path.join(
                        self.config['game_state_dir'],
                        f"game_state_{timestamp}.json"
                    )
                    os.makedirs(os.path.dirname(game_state_path), exist_ok=True)
                    
                    with open(game_state_path, 'w') as f:
                        json.dump(game_state, f, indent=2)
                    
                    # Put in queue for decision making
                    if not self.game_state_queue.full():
                        self.game_state_queue.put((game_state, game_state_path))
                        self.latest_game_state = game_state
                    else:
                        logger.warning("Game state queue is full, skipping decision")
                
                # Mark task as done
                self.screenshot_queue.task_done()
            
            except Exception as e:
                logger.error(f"Error in image analysis: {str(e)}")
                time.sleep(1.0)  # Sleep before retrying
        
        logger.info("Image analysis thread stopped")
    
    def _decision_loop(self):
        """Make decisions based on the analyzed game state"""
        logger.info("Decision making thread started")
        
        while self.running:
            try:
                # Get game state from queue
                try:
                    game_state, game_state_path = self.game_state_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Determine the player position
                # This needs to be adapted based on how the game identifies the player
                player_position = 1  # Assuming player is always at position 1
                
                # Make a decision using the PyTorch model
                decision = self.neural_network.predict_action(game_state, player_position)
                self.latest_decision = decision
                
                # Log the decision
                logger.info(f"Decision: {decision['action']} with confidence {decision['confidence']:.2f}")
                logger.info(f"Bet size: {decision['bet_size_percentage'] * 100:.1f}% of pot")
                
                # Save the decision with the game state
                with open(game_state_path, 'r') as f:
                    gs_data = json.load(f)
                
                gs_data['decision'] = decision
                
                with open(game_state_path, 'w') as f:
                    json.dump(gs_data, f, indent=2)
                
                # Take automatic action if enabled and confidence is high enough
                if (self.config['auto_play'] and 
                    decision['confidence'] >= self.config['confidence_threshold']):
                    self._take_action(decision)
                
                # Mark task as done
                self.game_state_queue.task_done()
            
            except Exception as e:
                logger.error(f"Error in decision making: {str(e)}")
                time.sleep(1.0)  # Sleep before retrying
        
        logger.info("Decision making thread stopped")
    
    def _take_action(self, decision):
        """
        Take an action in the game based on the decision
        
        This would integrate with the game's UI through mouse/keyboard automation
        For safety, this is left as a stub that just logs the intended action
        """
        action = decision['action']
        bet_size = decision['bet_size_percentage']
        
        logger.info(f"Auto-play: Would {action} with bet size {bet_size * 100:.1f}% of pot")
        
        # In a real implementation, this would use pyautogui or similar
        # to interact with the game UI
    
    def get_latest_state(self):
        """Get the latest game state and decision"""
        return {
            'game_state': self.latest_game_state,
            'decision': self.latest_decision,
            'screenshot': self.latest_screenshot,
            'timestamp': datetime.now().isoformat()
        }
    
    def train_from_collected_data(self, data_dir=None):
        """
        Train the neural network using collected game states and expert actions
        
        Args:
            data_dir: Directory containing labeled training data
                    If None, uses default from config
        """
        if data_dir is None:
            data_dir = self.config.get('training_data_dir', 'poker_data/training')
        
        logger.info(f"Loading training data from {data_dir}")
        
        # Load training data
        from poker_neural_engine_torch import PokerDataCollector  # Updated import
        data_collector = PokerDataCollector(output_dir=data_dir)
        game_states, expert_actions = data_collector.load_training_data()
        
        if not game_states:
            logger.warning("No training data found")
            return
        
        logger.info(f"Training PyTorch model with {len(game_states)} samples")
        
        # Train the model
        history = self.neural_network.train(
            game_states,
            expert_actions,
            epochs=50,
            batch_size=32
        )
        
        # Save the trained model
        model_save_path = self.config.get('model_save_path', 'poker_model.pt')  # Changed extension to .pt for PyTorch
        self.neural_network.save_model(model_save_path)
        logger.info(f"PyTorch model saved to {model_save_path}")
        
        return history