import sys
import os
import time
import json
import threading
import queue
import logging
from datetime import datetime
from pathlib import Path
import argparse

# Import our improved modules
from screen_grabber import PokerScreenGrabber, WindowInfo
from poker_analyzer import OptimizedPokerAnalyzer
from neural_engine import OptimizedPokerNeuralNetwork
from poker_integration import RLCardAdapter

# Set up logging with a more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler("poker_assistant.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PokerAssistant")

class PokerAssistant:
    """
    Main application that integrates screen grabbing, image analysis, and neural network
    decision making for poker gameplay assistance with improved performance and reliability.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize the poker assistant
        
        Args:
            config_file: Path to configuration file (optional)
        """
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Configure logger
        self._configure_logger()
        
        # Initialize components with improved error handling
        try:
            # Screen grabber for capturing the poker game
            self.screen_grabber = PokerScreenGrabber(
                capture_interval=self.config.get('capture_interval', 2.0),
                output_dir=self.config.get('screenshot_dir', 'poker_data/screenshots')
            )
            logger.info("Screen grabber initialized")
            
            # Poker analyzer for extracting game state from screenshots
            template_dir = self.config.get('template_dir', 'card_templates')
            self.image_analyzer = OptimizedPokerAnalyzer(template_dir)
            logger.info("Poker analyzer initialized")
            
            # Neural network for decision making
            self.neural_network = OptimizedPokerNeuralNetwork(
                model_path=self.config.get('model_path', None),
                device=self.config.get('device', None)
            )
            logger.info("Neural network initialized")
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Poker Assistant: {str(e)}")
        
        # Queues for communication between threads
        self.screenshot_queue = queue.Queue(maxsize=10)
        self.game_state_queue = queue.Queue(maxsize=10)
        
        # Control flags
        self.running = False
        self.threads = []
        self.exit_flag = threading.Event()

        # Latest data
        self.latest_game_state = None
        self.latest_decision = None
        self.latest_screenshot = None
        self.latest_screenshot_path = None
        
        # Performance stats
        self.stats = {
            'screenshots_captured': 0,
            'analyses_completed': 0,
            'decisions_made': 0,
            'errors': 0,
            'avg_analysis_time': 0,
            'avg_decision_time': 0
        }
        
        logger.info("Poker Assistant initialization complete")
    
    def _configure_logger(self):
        """Configure logger based on settings"""
        log_level = self.config.get('log_level', 'INFO').upper()
        
        # Set log level
        if log_level == 'DEBUG':
            logging.getLogger().setLevel(logging.DEBUG)
        elif log_level == 'WARNING':
            logging.getLogger().setLevel(logging.WARNING)
        elif log_level == 'ERROR':
            logging.getLogger().setLevel(logging.ERROR)
        else:
            logging.getLogger().setLevel(logging.INFO)
        
        # Configure file logging if specified
        log_file = self.config.get('log_file')
        if log_file:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Add file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            ))
            logging.getLogger().addHandler(file_handler)
    
    def _load_config(self, config_file):
        """Load configuration from a JSON file with improved error handling"""
        default_config = {
            'capture_interval': 2.0,
            'screenshot_dir': 'poker_data/screenshots',
            'game_state_dir': 'poker_data/game_states',
            'model_path': None,
            'model_save_path': 'models/poker_model.pt',
            'auto_play': False,
            'confidence_threshold': 0.7,
            'auto_detect_poker': True,
            'use_mock_data': True,
            'device': None,  # Auto-select device
            'log_level': 'INFO',
            'template_dir': 'card_templates',
            'max_cache_size': 1000,
            'analysis_batch_size': 1  # How many screenshots to analyze in batch
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                
                # Update default config with user settings
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_file}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in configuration file: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {str(e)}")
        else:
            if config_file:
                logger.warning(f"Configuration file not found: {config_file}")
            logger.info("Using default configuration")
        
        # Create required directories
        required_dirs = [
            default_config['screenshot_dir'],
            default_config['game_state_dir'],
            os.path.dirname(default_config['model_save_path'])
        ]
        
        for directory in required_dirs:
            if directory and not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                    logger.info(f"Created directory: {directory}")
                except Exception as e:
                    logger.warning(f"Failed to create directory {directory}: {str(e)}")
        
        return default_config
    
    def start(self):
        """Start the poker assistant with improved thread management"""
        if self.running:
            logger.warning("Poker Assistant is already running")
            return False
        
        # Reset exit flag
        self.exit_flag.clear()
        
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
                        logger.error("No poker window found and mock data disabled")
                        return False
                    logger.warning("No poker window found, using mock screenshots")
            else:
                if not self.config.get('use_mock_data', True):
                    logger.error("No window selected and auto-detect disabled and mock data disabled")
                    return False
                logger.warning("No window selected, using mock screenshots")
        else:
            logger.info(f"Using previously selected window: {self.screen_grabber.selected_window}")
        
        # Mark as running
        self.running = True
        
        # Start threads
        self.threads = []
        
        # Screenshot capture thread
        capture_thread = threading.Thread(
            target=self._capture_loop,
            name="ScreenshotCapture",
            daemon=True
        )
        self.threads.append(capture_thread)
        
        # Image analysis thread
        analysis_thread = threading.Thread(
            target=self._analysis_loop,
            name="ImageAnalysis",
            daemon=True
        )
        self.threads.append(analysis_thread)
        
        # Decision making thread
        decision_thread = threading.Thread(
            target=self._decision_loop,
            name="DecisionMaking",
            daemon=True
        )
        self.threads.append(decision_thread)
        
        # Start all threads with proper error handling
        try:
            for thread in self.threads:
                thread.start()
                logger.info(f"Started thread: {thread.name}")
            
            logger.info("Poker Assistant started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start threads: {str(e)}", exc_info=True)
            self.running = False
            return False

    def stop(self):
        """Stop the poker assistant with proper cleanup"""
        if not self.running:
            logger.warning("Poker Assistant is not running")
            return
        
        logger.info("Stopping Poker Assistant...")
        
        # Set the exit flag to signal threads to stop
        self.exit_flag.set()
        
        # Clear the running flag
        self.running = False
        
        # Wait for threads to terminate
        timeout = 5.0  # seconds
        for thread in self.threads:
            logger.info(f"Waiting for thread to terminate: {thread.name}")
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(f"Thread {thread.name} did not terminate within {timeout} seconds")
        
        # Clear thread list
        self.threads = []
        
        # Clear any queued items
        self._clear_queues()
        
        # Save analysis cache to improve startup performance next time
        self._save_cache()
        
        logger.info("Poker Assistant stopped")
    
    def _clear_queues(self):
        """Clear all queues"""
        try:
            # Clear screenshot queue
            while not self.screenshot_queue.empty():
                try:
                    self.screenshot_queue.get_nowait()
                    self.screenshot_queue.task_done()
                except queue.Empty:
                    break
            
            # Clear game state queue
            while not self.game_state_queue.empty():
                try:
                    self.game_state_queue.get_nowait()
                    self.game_state_queue.task_done()
                except queue.Empty:
                    break
                    
            logger.debug("All queues cleared")
        except Exception as e:
            logger.error(f"Error clearing queues: {str(e)}")
    
    def _save_cache(self):
        """Save analysis cache for future use"""
        try:
            # Currently not implemented - would save cached results to disk
            # for improved performance on restart
            pass
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    def _capture_loop(self):
        """Continuously capture screenshots of the poker game with improved error handling"""
        logger.info("Screenshot capture thread started")
        
        # Initialize variables
        capture_count = 0
        error_count = 0
        last_log_time = time.time()
        
        while self.running and not self.exit_flag.is_set():
            try:
                # Capture screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Debug info before capture
                logger.debug("About to capture screenshot")
                
                # Capture the screenshot with a suitable timeout to prevent hangs
                screenshot = None
                
                # Use a timeout approach to prevent thread from hanging
                capture_start_time = time.time()
                capture_timeout = 5.0  # seconds
                
                while (time.time() - capture_start_time) < capture_timeout:
                    try:
                        # Get screenshot with overlay for display purposes
                        screenshot = self.screen_grabber.capture_screenshot(with_overlay=True)
                        break
                    except Exception as e:
                        logger.error(f"Error in screenshot capture attempt: {str(e)}")
                        time.sleep(0.1)  # Short sleep before retry
                
                if screenshot is None:
                    logger.warning("Screenshot capture timed out, using mock screenshot")
                    screenshot = self.screen_grabber.create_mock_screenshot()
                
                # Process the screenshot
                if screenshot is not None:
                    logger.debug(f"Screenshot captured successfully, shape: {screenshot.shape}")
                    
                    # Store the latest screenshot (with overlay for display)
                    self.latest_screenshot = screenshot
                    
                    # Save screenshot - Note: save_screenshot will save the original version without overlay
                    screenshot_path = os.path.join(
                        self.config['screenshot_dir'],
                        f"screenshot_{timestamp}.png"
                    )
                    os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
                    self.screen_grabber.save_screenshot(screenshot, screenshot_path)
                    self.latest_screenshot_path = screenshot_path
                    
                    # Put in queue for analysis
                    if not self.screenshot_queue.full():
                        self.screenshot_queue.put((screenshot, screenshot_path, timestamp))
                        capture_count += 1
                        logger.debug(f"Screenshot added to analysis queue: {screenshot_path}")
                    else:
                        logger.warning("Screenshot queue is full, skipping analysis")
                else:
                    logger.warning("Screenshot capture returned None")
                    error_count += 1
                
                # Periodically log capture stats
                current_time = time.time()
                if current_time - last_log_time > 60:  # Log stats every minute
                    logger.info(f"Capture stats: {capture_count} captures, {error_count} errors in the last minute")
                    capture_count = 0
                    error_count = 0
                    last_log_time = current_time
                
                # Sleep until next capture
                capture_interval = self.config.get('capture_interval', 2.0)
                logger.debug(f"Sleeping for {capture_interval} seconds before next capture")
                
                # Use a more responsive sleep approach that can be interrupted
                sleep_start = time.time()
                while (time.time() - sleep_start < capture_interval and 
                    self.running and not self.exit_flag.is_set()):
                    time.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error in screenshot capture: {str(e)}", exc_info=True)
                self.stats['errors'] += 1
                
                # Sleep before retrying to avoid tight error loops
                time.sleep(1.0)
        
        logger.info("Screenshot capture thread stopped")

    def _analysis_loop(self):
        """Analyze screenshots to extract game state with improved error handling and batch processing"""
        logger.info("Image analysis thread started")
        
        # Initialize variables
        total_analysis_time = 0
        analysis_count = 0
        
        # Determine batch size
        batch_size = max(1, self.config.get('analysis_batch_size', 1))
        
        while self.running and not self.exit_flag.is_set():
            try:
                # Collect a batch of screenshots
                screenshot_batch = []
                for _ in range(batch_size):
                    try:
                        screenshot, screenshot_path, timestamp = self.screenshot_queue.get(timeout=1.0)
                        screenshot_batch.append((screenshot, screenshot_path, timestamp))
                    except queue.Empty:
                        break
                
                # Skip if no screenshots to process
                if not screenshot_batch:
                    continue
                
                # Process each screenshot in the batch
                for screenshot, screenshot_path, timestamp in screenshot_batch:
                    analysis_start_time = time.time()
                    
                    # Analyze the screenshot
                    game_state = self.image_analyzer.analyze_image(screenshot_path)
                    
                    # Calculate analysis time
                    analysis_time = time.time() - analysis_start_time
                    
                    # Update performance stats
                    total_analysis_time += analysis_time
                    analysis_count += 1
                    self.stats['analyses_completed'] += 1
                    self.stats['avg_analysis_time'] = total_analysis_time / analysis_count
                    
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
                            logger.debug(f"Game state added to decision queue: {game_state_path}")
                        else:
                            logger.warning("Game state queue is full, skipping decision")
                    
                    # Mark task as done
                    self.screenshot_queue.task_done()
                    
                # Log performance periodically
                if analysis_count % 10 == 0:
                    logger.info(f"Analysis performance: Avg time {self.stats['avg_analysis_time']:.2f}s over {analysis_count} analyses")
            
            except Exception as e:
                logger.error(f"Error in image analysis: {str(e)}", exc_info=True)
                self.stats['errors'] += 1
                
                # Ensure we mark any tasks as done to prevent queue blocking
                try:
                    for _ in range(len(screenshot_batch)):
                        self.screenshot_queue.task_done()
                except:
                    pass
                
                # Sleep before retrying to avoid tight error loops
                time.sleep(1.0)
        
        logger.info("Image analysis thread stopped")
    
    def _decision_loop(self):
        """Make decisions based on the analyzed game state with improved error handling"""
        logger.info("Decision making thread started")
        
        # Initialize variables
        total_decision_time = 0
        decision_count = 0
        
        while self.running and not self.exit_flag.is_set():
            try:
                # Get game state from queue
                try:
                    game_state, game_state_path = self.game_state_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Start timing decision process
                decision_start_time = time.time()
                
                # Determine the player position (main player is position 1)
                player_position = 1  # Main player
                
                
                adapter = RLCardAdapter(model_path='models/poker_model.pt')
 
                # Use it to predict actions during gameplay
                decision = adapter.predict_action(game_state, player_position)
                
                # Make a decision using the neural network
                # decision = self.neural_network.predict_action(game_state, player_position)
                self.latest_decision = decision
                
                # Calculate decision time
                decision_time = time.time() - decision_start_time
                
                # Update performance stats
                total_decision_time += decision_time
                decision_count += 1
                self.stats['decisions_made'] += 1
                self.stats['avg_decision_time'] = total_decision_time / decision_count
                
                # Log the decision
                logger.info(f"Decision: {decision['action']} with confidence {decision['confidence']:.2f}")
                logger.info(f"Bet size: {decision['bet_size_percentage'] * 100:.1f}% of pot")
                
                # Save the decision with the game state
                try:
                    with open(game_state_path, 'r') as f:
                        gs_data = json.load(f)
                    
                    gs_data['decision'] = decision
                    
                    with open(game_state_path, 'w') as f:
                        json.dump(gs_data, f, indent=2)
                except Exception as e:
                    logger.error(f"Error saving decision to game state file: {str(e)}")
                
                # Take automatic action if enabled and confidence is high enough
                if (self.config['auto_play'] and 
                    decision['confidence'] >= self.config['confidence_threshold']):
                    self._take_action(decision)
                
                # Mark task as done
                self.game_state_queue.task_done()
                
                # Log performance periodically
                if decision_count % 10 == 0:
                    logger.info(f"Decision performance: Avg time {self.stats['avg_decision_time']:.2f}s over {decision_count} decisions")
            
            except Exception as e:
                logger.error(f"Error in decision making: {str(e)}", exc_info=True)
                self.stats['errors'] += 1
                
                # Ensure we mark any game state tasks as done
                try:
                    self.game_state_queue.task_done()
                except:
                    pass
                
                # Sleep before retrying to avoid tight error loops
                time.sleep(1.0)
        
        logger.info("Decision making thread stopped")
    
    def _take_action(self, decision):
        """
        Take an action in the game based on the decision by sending keyboard commands
        
        Args:
            decision: Decision dictionary with 'action' and 'bet_size_percentage' keys
        
        Returns:
            bool: True if action was taken successfully, False otherwise
        """
        try:
            action = decision['action']
            bet_size = decision['bet_size_percentage']
            confidence = decision.get('confidence', 0)
            
            logger.info(f"Auto-play: Taking action {action} with bet size {bet_size * 100:.1f}% of pot (confidence: {confidence:.2f})")
            
            # Check if we have a window selected
            if not hasattr(self, 'screen_grabber') or not self.screen_grabber.selected_window:
                logger.warning("Auto-play: No window selected for sending commands")
                return False
            
            # Map decision to function key
            key_mapping = {
                'fold': 'F1',
                'check/call': 'F2',
                'bet/raise': 'F3',
                'all-in': 'F4'  # Special case
            }
            
            # Special case for all-in decisions
            if action == 'bet/raise' and bet_size >= 0.95:  # If bet size is ≥95% of pot, consider it all-in
                action_key = key_mapping['all-in']
                logger.info(f"Auto-play: Large bet detected ({bet_size * 100:.1f}% of pot), treating as all-in")
            else:
                # Get corresponding key for the action
                action_key = key_mapping.get(action)
                
                if not action_key:
                    logger.error(f"Auto-play: Unknown action '{action}'")
                    return False
            
            # Ensure the window is in focus before sending keys
            try:
                # Check if we're on Windows and have win32gui available
                import platform
                import importlib.util
                
                is_windows = platform.system() == 'Windows'
                has_win32gui = importlib.util.find_spec('win32gui') is not None
                
                if is_windows and has_win32gui and hasattr(self.screen_grabber, 'window_handle'):
                    import win32gui
                    win32gui.SetForegroundWindow(self.screen_grabber.window_handle)
                    time.sleep(0.3)  # Short delay to ensure window is focused
                    logger.info("Auto-play: Window focused successfully")
            except Exception as e:
                logger.warning(f"Auto-play: Failed to focus window: {str(e)}")
            
            # Press the corresponding function key
            import pyautogui
            pyautogui.press(action_key)
            logger.info(f"Auto-play: Pressed {action_key} for action '{action}'")
            
            # Add a small delay after action
            time.sleep(0.5)
            
            return True
            
        except Exception as e:
            logger.error(f"Auto-play: Error taking action: {str(e)}", exc_info=True)
            return False
    
    def get_latest_state(self):
        """Get the latest game state and decision with timing information"""
        return {
            'game_state': self.latest_game_state,
            'decision': self.latest_decision,
            'screenshot': self.latest_screenshot,
            'screenshot_path': self.latest_screenshot_path,
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats
        }
    
    def train_model(self, data_dir=None, epochs=50, batch_size=32, learning_rate=0.001):
        """
        Train the neural network using collected game states and expert actions
        
        Args:
            data_dir: Directory containing labeled training data
                    If None, uses default from config
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            
        Returns:
            dict: Training history
        """
        if data_dir is None:
            data_dir = self.config.get('training_data_dir', 'poker_data/training')
        
        logger.info(f"Loading training data from {data_dir}")
        
        # Load training data
        from neural_engine import PokerDataCollector
        data_collector = PokerDataCollector(output_dir=data_dir)
        game_states, expert_actions = data_collector.load_training_data()
        
        if not game_states:
            logger.warning("No training data found")
            return {'error': 'No training data found'}
        
        logger.info(f"Training model with {len(game_states)} samples")
        
        # Train the model
        history = self.neural_network.train(
            game_states,
            expert_actions,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Save the trained model
        model_save_path = self.config.get('model_save_path', 'models/poker_model.pt')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        success = self.neural_network.save_model(model_save_path)
        if success:
            logger.info(f"Model saved to {model_save_path}")
        else:
            logger.error(f"Failed to save model to {model_save_path}")
        
        return history
    
    def analyze_screenshot(self, screenshot_path):
        """
        Analyze a specific screenshot (useful for testing and debugging)
        
        Args:
            screenshot_path: Path to the screenshot to analyze
            
        Returns:
            dict: Analysis result with game state and decision
        """
        try:
            if not os.path.exists(screenshot_path):
                logger.error(f"Screenshot not found: {screenshot_path}")
                return {'error': 'Screenshot not found'}
            
            # Analyze the screenshot
            logger.info(f"Analyzing screenshot: {screenshot_path}")
            game_state = self.image_analyzer.analyze_image(screenshot_path)
            
            if not game_state:
                logger.error(f"Failed to analyze screenshot: {screenshot_path}")
                return {'error': 'Analysis failed'}
            
            # Make a decision
            # decision = self.neural_network.predict_action(game_state, 1)  # Assume player position 1
            
            adapter = RLCardAdapter(model_path='models/poker_model.pt')

            # Use it to predict actions during gameplay
            decision = adapter.predict_action(game_state, 1)
            
            return {
                'game_state': game_state,
                'decision': decision,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing screenshot: {str(e)}", exc_info=True)
            return {'error': str(e)}
    
    def get_stats(self):
        """Get current statistics"""
        return self.stats
    
    def select_window(self, window_title):
        """
        Select a window for capture based on title
        
        Args:
            window_title: Title or partial title of the window to select
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get available windows
            windows = self.screen_grabber.get_window_list()
            
            # Find matching windows
            matching_windows = [w for w in windows if window_title.lower() in w.title.lower()]
            
            if not matching_windows:
                logger.error(f"No windows found matching '{window_title}'")
                return False
            
            # Select the first matching window
            window = matching_windows[0]
            success = self.screen_grabber.select_window(window)
            
            if success:
                logger.info(f"Selected window: {window.title}")
                return True
            else:
                logger.error(f"Failed to select window: {window.title}")
                return False
                
        except Exception as e:
            logger.error(f"Error selecting window: {str(e)}", exc_info=True)
            return False
    
    def get_available_windows(self):
        """Get a list of available windows"""
        try:
            windows = self.screen_grabber.get_window_list()
            return [{'title': w.title, 'size': (w.width, w.height)} for w in windows]
        except Exception as e:
            logger.error(f"Error getting window list: {str(e)}")
            return []


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Poker Assistant")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--window", help="Select window by title (partial match)")
    parser.add_argument("--list-windows", action="store_true", help="List available windows")
    parser.add_argument("--analyze", help="Analyze a specific screenshot")
    parser.add_argument("--train", action="store_true", help="Train the neural network with collected data")
    parser.add_argument("--train-epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--train-batch-size", type=int, default=32, help="Training batch size")
    
    args = parser.parse_args()
    
    try:
        # Create the assistant
        assistant = PokerAssistant(config_file=args.config)
        
        # List windows if requested
        if args.list_windows:
            print("Available windows:")
            for i, window in enumerate(assistant.get_available_windows(), 1):
                print(f"{i}. {window['title']} - {window['size']}")
            return
        
        # Select window if specified
        if args.window:
            if assistant.select_window(args.window):
                print(f"Selected window matching '{args.window}'")
            else:
                print(f"No window found matching '{args.window}'")
                return
        
        # Analyze screenshot if specified
        if args.analyze:
            result = assistant.analyze_screenshot(args.analyze)
            if 'error' in result:
                print(f"Analysis error: {result['error']}")
            else:
                print("\nAnalysis Result:")
                print(f"Game Stage: {result['game_state'].get('game_stage', 'unknown')}")
                print(f"Pot Size: ${result['game_state'].get('pot', 0)}")
                
                print("\nCommunity Cards:")
                for card in result['game_state'].get('community_cards', []):
                    print(f"  {card['value']} of {card['suit']}")
                
                print("\nPlayers:")
                for player_id, player in result['game_state'].get('players', {}).items():
                    print(f"  Player {player_id}:")
                    print(f"    Chips: ${player.get('chips', 0)}")
                    if 'cards' in player:
                        cards_str = ", ".join([f"{c['value']} of {c['suit']}" for c in player['cards']])
                        print(f"    Cards: {cards_str}")
                
                print("\nDecision:")
                print(f"  Action: {result['decision']['action']}")
                print(f"  Confidence: {result['decision']['confidence']:.2f}")
                print(f"  Bet Size: {result['decision']['bet_size_percentage']*100:.1f}% of pot")
            return
        
        # Train model if requested
        if args.train:
            print("Training neural network...")
            history = assistant.train_model(
                epochs=args.train_epochs,
                batch_size=args.train_batch_size
            )
            
            if 'error' in history:
                print(f"Training error: {history['error']}")
            else:
                print("\nTraining completed:")
                print(f"Final accuracy: {history['accuracy'][-1]:.2f}")
                print(f"Final loss: {history['total_loss'][-1]:.4f}")
            return
        
        # Otherwise, start the assistant
        print("Starting Poker Assistant...")
        if assistant.start():
            print("Poker Assistant started. Press Ctrl+C to stop.")
            
            try:
                while True:
                    time.sleep(1)
                    
                    # Periodically print status
                    if assistant.stats['analyses_completed'] % 10 == 0 and assistant.stats['analyses_completed'] > 0:
                        state = assistant.get_latest_state()
                        if state['decision']:
                            print(f"\nLatest decision: {state['decision']['action']} with confidence {state['decision']['confidence']:.2f}")
                            print(f"Stats: {state['stats']['screenshots_captured']} screenshots, {state['stats']['analyses_completed']} analyses")
            
            except KeyboardInterrupt:
                print("\nStopping Poker Assistant...")
                assistant.stop()
                print("Poker Assistant stopped.")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()