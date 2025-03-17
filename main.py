#!/usr/bin/env python3
# main.py - Unified entry point for the Poker Computer Vision project

import os
import sys
import argparse
import json
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
import time
import glob
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler("poker_app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PokerApp")

# Try to import our modules and fail gracefully if there are problems
IMPORT_ERROR = None
try:
    # Import optimized components
    from screen_grabber import PokerScreenGrabber, WindowInfo
    from poker_analyzer import OptimizedPokerAnalyzer
    from neural_engine import OptimizedPokerNeuralNetwork
    from poker_assistant import PokerAssistant
except ImportError as e:
    IMPORT_ERROR = str(e)
    logger.error(f"Import error: {IMPORT_ERROR}")
    # We'll handle this gracefully in the UI

class PokerApplication:
    """Main application for Poker Computer Vision with improved UI and error handling"""
    
    def __init__(self, root, config_file=None):
        """Initialize the application with improved error recovery"""
        self.root = root
        self.root.title("Poker CV Assistant")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        # Check for import errors and warn the user if necessary
        if IMPORT_ERROR:
            messagebox.showerror(
                "Import Error",
                f"There was a problem importing required modules: {IMPORT_ERROR}\n\n"
                "The application may not function correctly. "
                "Please check your installation and dependencies."
            )
        
        # Load config
        self.config = self._load_config(config_file)
        
        # Set up UI variables
        self.setup_ui_variables()
        
        # Window selection attributes
        self.window_list = []
        self.selected_window_info = None
        
        # Initialize components
        try:
            # Screen grabber for window detection
            self.window_detector = PokerScreenGrabber(
                capture_interval=float(self.config.get('capture_interval', 2.0)),
                output_dir=self.config.get('screenshot_dir', 'poker_data/screenshots')
            )
            
            # Initialize Poker Assistant
            self.poker_assistant = None  # Will be initialized when started
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}", exc_info=True)
            messagebox.showerror(
                "Initialization Error",
                f"Failed to initialize application components: {str(e)}\n\n"
                "The application may have limited functionality."
            )
        
        # Create UI
        self._create_menu()
        self._create_main_frame()
        
        # Initialize status display
        self.status_update_id = None
        self._schedule_status_update()
        
        # Populate window list
        self._refresh_windows()
        
        logger.info("Application initialized")
    
    def setup_ui_variables(self):
        """Set up UI variables with default values"""
        self.status_var = tk.StringVar(value="Ready")
        self.selected_window_var = tk.StringVar(value="None")
        self.window_handle_var = tk.StringVar(value="None")
        self.window_rect_var = tk.StringVar(value="None")
        
        self.action_var = tk.StringVar(value="N/A")
        self.confidence_var = tk.StringVar(value="N/A")
        self.bet_size_var = tk.StringVar(value="N/A")
        self.hand_strength_var = tk.StringVar(value="N/A")
        self.equity_var = tk.StringVar(value="N/A")
        self.model_info_var = tk.StringVar(value="Default model")
        
        # Configuration variables
        self.interval_var = tk.StringVar(value=str(self.config.get('capture_interval', 2.0)))
        self.auto_play_var = tk.BooleanVar(value=self.config.get('auto_play', False))
    
    def _load_config(self, config_file=None):
        """Load configuration with improved error handling and defaults"""
        default_config = {
            'capture_interval': 2.0,
            'screenshot_dir': 'poker_data/screenshots',
            'game_state_dir': 'poker_data/game_states',
            'training_data_dir': 'poker_data/training',
            'model_path': None,
            'model_save_path': 'models/poker_model.pt',
            'auto_play': False,
            'confidence_threshold': 0.7,
            'auto_detect_poker': True,
            'use_mock_data': True,
            'show_debug_overlay': False,
            'device': None,  # Auto-select device (CPU/GPU)
            'ui_refresh_rate': 1.0  # UI update interval in seconds
        }
        
        # Create directories
        for dir_path in [
            default_config['screenshot_dir'], 
            default_config['game_state_dir'], 
            default_config['training_data_dir'],
            os.path.dirname(default_config['model_save_path'])
        ]:
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create directory {dir_path}: {str(e)}")
        
        # If config file is specified, try to load it
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    
                # Update default config with loaded values
                default_config.update(loaded_config)
                logger.info(f"Loaded configuration from {config_file}")
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in config file: {config_file}")
                messagebox.showwarning(
                    "Configuration Error",
                    f"The configuration file {config_file} contains invalid JSON. "
                    "Using default configuration."
                )
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
                messagebox.showwarning(
                    "Configuration Error",
                    f"Failed to load configuration from {config_file}: {str(e)}\n"
                    "Using default configuration."
                )
        elif config_file:
            logger.warning(f"Configuration file not found: {config_file}")
            messagebox.showinfo(
                "Configuration",
                f"Configuration file {config_file} was not found. "
                "Using default configuration."
            )
        
        return default_config
    
    def _create_menu(self):
        """Create the application menu"""
        menu_bar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Load Configuration", command=self._load_config_file)
        file_menu.add_command(label="Save Configuration", command=self._save_config_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_exit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menu_bar, tearoff=0)
        tools_menu.add_command(label="Settings", command=self._show_settings)
        tools_menu.add_command(label="Train Model", command=self._train_model)
        tools_menu.add_command(label="Calibrate Regions", command=self._calibrate_regions)
        tools_menu.add_separator()
        tools_menu.add_command(label="Analyze Single Screenshot", command=self._analyze_screenshot)
        menu_bar.add_cascade(label="Tools", menu=tools_menu)
        
        # View menu
        view_menu = tk.Menu(menu_bar, tearoff=0)
        
        # Debug overlay toggle
        self.show_overlay_var = tk.BooleanVar(value=self.config.get('show_debug_overlay', True))
        view_menu.add_checkbutton(
            label="Show Debug Overlay", 
            variable=self.show_overlay_var,
            command=self._toggle_debug_overlay
        )
        
        # Device info
        device_info = "Using GPU" if torch.cuda.is_available() else "Using CPU"
        view_menu.add_separator()
        view_menu.add_command(label=device_info, state=tk.DISABLED)
        
        menu_bar.add_cascade(label="View", menu=view_menu)
        
        # Help menu
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Help", command=self._show_help)
        help_menu.add_command(label="About", command=self._show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menu_bar)

    def _create_main_frame(self):
        """Create the main application layout with improved UI design"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        # Window selection
        ttk.Label(control_frame, text="Select Window:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.window_selector = ttk.Combobox(control_frame, width=40)
        self.window_selector.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="Refresh", command=self._refresh_windows).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Select", command=self._select_window).grid(row=0, column=3, padx=5)
        
        # Window preview button
        ttk.Button(control_frame, text="Preview Window", command=self._preview_selected_window).grid(row=0, column=4, padx=5)
        
        # Capture controls
        ttk.Label(control_frame, text="Capture Interval (s):").grid(row=1, column=0, sticky=tk.W, pady=5)
        interval_entry = ttk.Entry(control_frame, width=10, textvariable=self.interval_var)
        interval_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.start_btn = ttk.Button(control_frame, text="Start Assistant", command=self._start_assistant)
        self.start_btn.grid(row=1, column=2, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Assistant", command=self._stop_assistant, state=tk.DISABLED)
        self.stop_btn.grid(row=1, column=3, padx=5)
        
        # Screenshot refresh button
        self.refresh_screenshot_btn = ttk.Button(control_frame, text="Refresh Screenshot", command=self._update_latest_screenshot_display)
        self.refresh_screenshot_btn.grid(row=1, column=4, padx=5)
        
        # Auto-play checkbox
        auto_play_cb = ttk.Checkbutton(
            control_frame, 
            text="Enable Auto-Play",
            variable=self.auto_play_var,
            command=self._toggle_auto_play
        )
        auto_play_cb.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Device info label
        device_str = "Using GPU" if torch.cuda.is_available() else "Using CPU"
        ttk.Label(
            control_frame,
            text=device_str,
            font=("TkDefaultFont", 10, "italic")
        ).grid(row=2, column=2, columnspan=2, sticky=tk.E, pady=5)
        
        # Status bar at the bottom
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Content panes - main area with preview and analysis
        content_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        content_paned.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Left pane - Game preview
        preview_frame = ttk.LabelFrame(content_paned, text="Game Preview", padding="10")
        content_paned.add(preview_frame, weight=2)
        
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # Right pane - Analysis and information
        analysis_frame = ttk.LabelFrame(content_paned, text="Analysis", padding="10")
        content_paned.add(analysis_frame, weight=1)
        
        # Tabbed interface for analysis
        tab_control = ttk.Notebook(analysis_frame)
        
        # Game state tab
        game_state_tab = ttk.Frame(tab_control)
        tab_control.add(game_state_tab, text="Game State")
        
        # Game state display (scrollable)
        game_state_scroll = ttk.Scrollbar(game_state_tab)
        game_state_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.game_state_text = tk.Text(game_state_tab, width=40, height=20, wrap=tk.WORD,
                                       yscrollcommand=game_state_scroll.set)
        self.game_state_text.pack(fill=tk.BOTH, expand=True)
        game_state_scroll.config(command=self.game_state_text.yview)
        
        # Decision tab
        decision_tab = ttk.Frame(tab_control)
        tab_control.add(decision_tab, text="Decision")
        
        # Decision display
        ttk.Label(decision_tab, text="Recommended Action:").pack(anchor=tk.W, pady=(10, 5))
        ttk.Label(decision_tab, textvariable=self.action_var, font=("TkDefaultFont", 12, "bold")).pack(anchor=tk.W, padx=10)
        
        ttk.Label(decision_tab, text="Confidence:").pack(anchor=tk.W, pady=(10, 5))
        ttk.Label(decision_tab, textvariable=self.confidence_var, font=("TkDefaultFont", 12, "bold")).pack(anchor=tk.W, padx=10)
        
        ttk.Label(decision_tab, text="Bet Size:").pack(anchor=tk.W, pady=(10, 5))
        ttk.Label(decision_tab, textvariable=self.bet_size_var, font=("TkDefaultFont", 12, "bold")).pack(anchor=tk.W, padx=10)
        
        # Hand strength section
        ttk.Separator(decision_tab, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(decision_tab, text="Hand Strength:").pack(anchor=tk.W, pady=(5, 5))
        ttk.Label(decision_tab, textvariable=self.hand_strength_var, font=("TkDefaultFont", 12, "bold")).pack(anchor=tk.W, padx=10)
        
        ttk.Label(decision_tab, text="Equity:").pack(anchor=tk.W, pady=(10, 5))
        ttk.Label(decision_tab, textvariable=self.equity_var, font=("TkDefaultFont", 12, "bold")).pack(anchor=tk.W, padx=10)
        
        # Model information
        ttk.Separator(decision_tab, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(decision_tab, text="Model:").pack(anchor=tk.W, pady=(5, 5))
        ttk.Label(decision_tab, textvariable=self.model_info_var, font=("TkDefaultFont", 10)).pack(anchor=tk.W, padx=10)
        
        # Performance stats
        ttk.Label(decision_tab, text="Performance Stats:").pack(anchor=tk.W, pady=(10, 5))
        self.stats_text = tk.Text(decision_tab, width=40, height=6, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.X, expand=False, padx=10)
        self.stats_text.insert(tk.END, "No statistics available")
        self.stats_text.config(state=tk.DISABLED)
        
        # Window info tab
        window_tab = ttk.Frame(tab_control)
        tab_control.add(window_tab, text="Window Info")
        
        # Window info display
        ttk.Label(window_tab, text="Selected Window:").pack(anchor=tk.W, pady=(10, 5))
        ttk.Label(window_tab, textvariable=self.selected_window_var, font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, padx=10)
        
        ttk.Label(window_tab, text="Window Handle:").pack(anchor=tk.W, pady=(10, 5))
        ttk.Label(window_tab, textvariable=self.window_handle_var).pack(anchor=tk.W, padx=10)
        
        ttk.Label(window_tab, text="Window Rect:").pack(anchor=tk.W, pady=(10, 5))
        ttk.Label(window_tab, textvariable=self.window_rect_var).pack(anchor=tk.W, padx=10)
        
        # Available Windows list
        ttk.Separator(window_tab, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(window_tab, text="Available Windows:").pack(anchor=tk.W, pady=(5, 5))
        
        # Scrollable list of windows
        window_list_frame = ttk.Frame(window_tab)
        window_list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        window_list_scroll = ttk.Scrollbar(window_list_frame)
        window_list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.window_list_box = tk.Listbox(window_list_frame, height=8, yscrollcommand=window_list_scroll.set)
        self.window_list_box.pack(fill=tk.BOTH, expand=True)
        window_list_scroll.config(command=self.window_list_box.yview)
        
        # Bind select event
        self.window_list_box.bind('<<ListboxSelect>>', self._on_window_list_select)
        
        # Add tabs to notebook
        tab_control.pack(fill=tk.BOTH, expand=True)
    
    def _refresh_windows(self):
        """Refresh the list of available windows with error handling"""
        try:
            # Get windows from the window detector
            if hasattr(self, 'window_detector'):
                self.window_list = self.window_detector.get_window_list()
            else:
                self.window_list = []
                self.status_var.set("Window detector not initialized")
                return
            
            # Update dropdown
            window_titles = [w.title for w in self.window_list]
            self.window_selector['values'] = window_titles
            
            if window_titles:
                # Look for poker windows
                poker_indices = [i for i, title in enumerate(window_titles) if "poker" in title.lower()]
                if poker_indices:
                    # Select the first poker window found
                    self.window_selector.current(poker_indices[0])
                else:
                    # Default to first window
                    self.window_selector.current(0)
            
            # Update window list box
            self.window_list_box.delete(0, tk.END)
            for i, window in enumerate(self.window_list):
                self.window_list_box.insert(tk.END, f"{i+1}. {window.title}")
                
                # Highlight poker windows
                if "poker" in window.title.lower():
                    self.window_list_box.itemconfig(i, {'bg': '#e0f0e0'})  # Light green highlight
            
            self.status_var.set(f"Found {len(self.window_list)} windows")
        except Exception as e:
            logger.error(f"Error refreshing windows: {str(e)}", exc_info=True)
            self.status_var.set(f"Error refreshing windows: {str(e)}")
            messagebox.showerror("Error", f"Failed to refresh window list: {str(e)}")
    
    def _on_window_list_select(self, event):
        """Handle window selection from the listbox"""
        try:
            selected_idx = self.window_list_box.curselection()[0]
            if 0 <= selected_idx < len(self.window_list):
                selected_window = self.window_list[selected_idx]
                
                # Update dropdown to match selection
                window_titles = [w.title for w in self.window_list]
                idx = window_titles.index(selected_window.title)
                self.window_selector.current(idx)
                
                # Select this window
                self._select_window()
        except (IndexError, ValueError) as e:
            logger.error(f"Error in window list selection: {str(e)}")
    
    def _select_window(self):
        """Select a window for capturing with improved error handling"""
        try:
            selected_title = self.window_selector.get()
            
            if not selected_title:
                self.status_var.set("No window selected")
                return
            
            # Find the window info by title
            matching_windows = [w for w in self.window_list if w.title == selected_title]
            
            if matching_windows:
                # Select the first matching window
                self.selected_window_info = matching_windows[0]
                
                # Update UI
                self.selected_window_var.set(self.selected_window_info.title)
                self.window_handle_var.set(str(self.selected_window_info.handle))
                
                if self.selected_window_info.rect:
                    rect_str = f"({self.selected_window_info.rect[0]}, {self.selected_window_info.rect[1]}, " \
                            f"{self.selected_window_info.rect[2]}, {self.selected_window_info.rect[3]})"
                    self.window_rect_var.set(rect_str)
                else:
                    self.window_rect_var.set("Unknown")
                
                # Also update the window detector's selection
                if hasattr(self, 'window_detector'):
                    success = self.window_detector.select_window(self.selected_window_info)
                    
                    if success:
                        self.status_var.set(f"Selected window: {self.selected_window_info.title}")
                        
                        # Preview the window
                        self._preview_selected_window()
                    else:
                        self.status_var.set(f"Error selecting window: {self.selected_window_info.title}")
                else:
                    self.status_var.set(f"Window detector not initialized, can't select {self.selected_window_info.title}")
            else:
                self.status_var.set(f"Window not found: {selected_title}")
        except Exception as e:
            logger.error(f"Error selecting window: {str(e)}", exc_info=True)
            self.status_var.set(f"Error selecting window: {str(e)}")
            messagebox.showerror("Error", f"Failed to select window: {str(e)}")
    
    def _preview_selected_window(self):
        """Preview the selected window with error handling"""
        try:
            if not self.selected_window_info:
                self.status_var.set("No window selected for preview")
                return
            
            # Use the window detector to capture a screenshot
            if hasattr(self, 'window_detector'):
                self.window_detector.select_window(self.selected_window_info)
                img = self.window_detector.capture_screenshot()
                
                if img is not None:
                    # Update preview
                    self._update_preview(img)
                    self.status_var.set(f"Previewing window: {self.selected_window_info.title}")
                else:
                    self.status_var.set("Failed to capture window preview")
            else:
                self.status_var.set("Window detector not initialized")
        except Exception as e:
            logger.error(f"Error previewing window: {str(e)}", exc_info=True)
            self.status_var.set(f"Error previewing window: {str(e)}")
    
    def _update_preview(self, img):
        """Update the preview image"""
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            
            # Resize to fit the preview area while maintaining aspect ratio
            preview_width = self.preview_label.winfo_width() or 640
            preview_height = self.preview_label.winfo_height() or 480
            
            img.thumbnail((preview_width, preview_height), Image.LANCZOS)
            
            img = ImageTk.PhotoImage(image=img)
            self.preview_label.configure(image=img)
            self.preview_label.image = img  # Keep a reference
        except Exception as e:
            logger.error(f"Error updating preview: {str(e)}", exc_info=True)
            self.status_var.set(f"Error updating preview: {str(e)}")
    
    def _update_latest_screenshot_display(self):
        """Update the preview with the latest screenshot from the poker assistant"""
        try:
            # Check if poker assistant is running
            if not hasattr(self, 'poker_assistant') or not self.poker_assistant:
                self.status_var.set("Poker assistant not running")
                return
            
            # Get latest state
            latest_state = self.poker_assistant.get_latest_state()
            
            # Check if there's a latest screenshot
            if latest_state['screenshot'] is not None:
                # Update preview
                self._update_preview(latest_state['screenshot'])
                self.status_var.set("Displaying latest screenshot")
                return
            
            # If no screenshot in state, try to find the latest from disk
            screenshot_dir = self.config.get('screenshot_dir', 'poker_data/screenshots')
            
            # Find the most recent screenshot
            list_of_files = glob.glob(os.path.join(screenshot_dir, 'screenshot_*.png'))
            if not list_of_files:
                self.status_var.set("No screenshots found in directory")
                return
                
            # Get the most recent file
            latest_screenshot = max(list_of_files, key=os.path.getctime)
            
            # Load and display the image
            img = cv2.imread(latest_screenshot)
            if img is not None:
                # Update preview
                self._update_preview(img)
                
                # Update status
                filename = os.path.basename(latest_screenshot)
                self.status_var.set(f"Displaying latest screenshot: {filename}")
            else:
                self.status_var.set(f"Failed to load latest screenshot")
        except Exception as e:
            logger.error(f"Error updating screenshot display: {str(e)}", exc_info=True)
            self.status_var.set(f"Error updating screenshot: {str(e)}")
    
    def _start_assistant(self):
        """Start the poker assistant with improved error handling"""
        try:
            # Check if assistant is already running
            if hasattr(self, 'poker_assistant') and self.poker_assistant and hasattr(self.poker_assistant, 'running') and self.poker_assistant.running:
                messagebox.showinfo("Info", "Poker Assistant is already running")
                return
            
            # Check if a window is selected
            if not self.selected_window_info and not self.config.get('auto_detect_poker', True):
                result = messagebox.askokcancel(
                    "No Window Selected", 
                    "No window is currently selected. The application will use mock screenshots instead of live capture.\n\n"
                    "Do you want to continue?",
                    icon=messagebox.WARNING
                )
                if not result:
                    return
            
            # Update config with current values
            self.config['capture_interval'] = float(self.interval_var.get())
            self.config['auto_play'] = self.auto_play_var.get()
            self.config['show_debug_overlay'] = self.show_overlay_var.get()
            
            # Initialize and start the poker assistant
            if not hasattr(self, 'poker_assistant') or not self.poker_assistant:
                # Create a temporary config file
                temp_config_path = "temp_config.json"
                with open(temp_config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
                
                # Create the poker assistant with our config
                self.poker_assistant = PokerAssistant(config_file=temp_config_path)
                
                # Clean up temp file
                try:
                    os.remove(temp_config_path)
                except:
                    pass
            else:
                # Update existing assistant's config
                self.poker_assistant.config.update(self.config)
            
            # Pass window info if available
            if self.selected_window_info:
                self.poker_assistant.select_window(self.selected_window_info.title)
            
            # Start the assistant
            success = self.poker_assistant.start()
            
            if not success:
                messagebox.showerror("Error", "Failed to start Poker Assistant")
                return
            
            # Update UI
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_var.set("Poker Assistant started")
            
            # Update model info display
            model_path = self.config.get('model_path', 'Default model')
            self.model_info_var.set(f"Using: {os.path.basename(model_path) if model_path else 'Default model'}")
        
        except Exception as e:
            logger.error(f"Error starting assistant: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Failed to start Poker Assistant: {str(e)}")
            self.status_var.set(f"Error starting assistant: {str(e)}")
    
    def _stop_assistant(self):
        """Stop the poker assistant with improved error handling"""
        try:
            if hasattr(self, 'poker_assistant') and self.poker_assistant:
                self.poker_assistant.stop()
                
                # Update UI
                self.start_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
                self.status_var.set("Poker Assistant stopped")
            else:
                self.status_var.set("Poker Assistant not running")
        except Exception as e:
            logger.error(f"Error stopping assistant: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Failed to stop Poker Assistant: {str(e)}")
            self.status_var.set(f"Error stopping assistant: {str(e)}")
    
    def _toggle_auto_play(self):
        """Toggle auto-play mode"""
        try:
            if hasattr(self, 'poker_assistant') and self.poker_assistant:
                self.poker_assistant.config['auto_play'] = self.auto_play_var.get()
            
            self.config['auto_play'] = self.auto_play_var.get()
            
            if self.auto_play_var.get():
                self.status_var.set("Auto-play enabled")
            else:
                self.status_var.set("Auto-play disabled")
        except Exception as e:
            logger.error(f"Error toggling auto-play: {str(e)}")
            self.status_var.set(f"Error toggling auto-play: {str(e)}")
    
    def _toggle_debug_overlay(self):
        """Toggle the debug overlay display"""
        try:
            overlay_enabled = self.show_overlay_var.get()
            
            # Update window detector
            if hasattr(self, 'window_detector'):
                self.window_detector.show_debug_overlay = overlay_enabled
            
            # Update config
            self.config['show_debug_overlay'] = overlay_enabled
            
            # Update poker assistant if running
            if hasattr(self, 'poker_assistant') and self.poker_assistant:
                if hasattr(self.poker_assistant, 'screen_grabber'):
                    self.poker_assistant.screen_grabber.show_debug_overlay = overlay_enabled
            
            self.status_var.set(f"Debug overlay {'enabled' if overlay_enabled else 'disabled'}")
            
            # Refresh the preview
            self._preview_selected_window()
        except Exception as e:
            logger.error(f"Error toggling debug overlay: {str(e)}")
            self.status_var.set(f"Error toggling debug overlay: {str(e)}")
    
    def _update_ui_status(self):
        """Update UI with the latest game state and decision"""
        try:
            if hasattr(self, 'poker_assistant') and self.poker_assistant and hasattr(self.poker_assistant, 'running') and self.poker_assistant.running:
                latest = self.poker_assistant.get_latest_state()
                
                if latest['game_state']:
                    # Update game state display
                    self.game_state_text.config(state=tk.NORMAL)
                    self.game_state_text.delete(1.0, tk.END)
                    self.game_state_text.insert(tk.END, json.dumps(latest['game_state'], indent=2))
                    self.game_state_text.config(state=tk.DISABLED)
                
                if latest['decision']:
                    # Update decision display
                    action = latest['decision']['action']
                    confidence = latest['decision']['confidence']
                    bet_size = latest['decision']['bet_size_percentage']
                    
                    self.action_var.set(action.upper())
                    self.confidence_var.set(f"{confidence:.2f}")
                    self.bet_size_var.set(f"{bet_size * 100:.1f}% of pot")
                    
                    # Hand strength would be calculated from the game state
                    # This is a simplification
                    if 'game_state' in latest and latest['game_state']:
                        if 'players' in latest['game_state'] and 1 in latest['game_state']['players']:
                            player = latest['game_state']['players'][1]
                            if 'cards' in player and len(player['cards']) == 2:
                                # Simple display for now
                                cards_str = ", ".join([f"{c['value']} of {c['suit']}" for c in player['cards']])
                                self.hand_strength_var.set(cards_str)
                                
                                # Equity is usually calculated through simulations
                                self.equity_var.set("~30%")  # Placeholder
                
                # Update stats
                if 'stats' in latest:
                    stats_text = (
                        f"Screenshots: {latest['stats'].get('screenshots_captured', 0)}\n"
                        f"Analyses: {latest['stats'].get('analyses_completed', 0)}\n"
                        f"Decisions: {latest['stats'].get('decisions_made', 0)}\n"
                        f"Analysis time: {latest['stats'].get('avg_analysis_time', 0):.2f}s\n"
                        f"Decision time: {latest['stats'].get('avg_decision_time', 0):.2f}s\n"
                        f"Errors: {latest['stats'].get('errors', 0)}"
                    )
                    
                    self.stats_text.config(state=tk.NORMAL)
                    self.stats_text.delete(1.0, tk.END)
                    self.stats_text.insert(tk.END, stats_text)
                    self.stats_text.config(state=tk.DISABLED)
        except Exception as e:
            logger.error(f"Error updating UI status: {str(e)}")
        
        # Schedule the next update
        refresh_rate = float(self.config.get('ui_refresh_rate', 1.0))
        self.status_update_id = self.root.after(int(refresh_rate * 1000), self._update_ui_status)
    
    def _schedule_status_update(self):
        """Schedule periodic UI status updates"""
        self._update_ui_status()
    
    def _load_config_file(self):
        """Load configuration from a file"""
        try:
            filename = filedialog.askopenfilename(
                title="Load Configuration",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if not filename:
                return
                
            # Load the configuration
            with open(filename, 'r') as f:
                loaded_config = json.load(f)
            
            # Update our config
            self.config.update(loaded_config)
            
            # Update UI elements with loaded config
            self.interval_var.set(str(self.config.get('capture_interval', 2.0)))
            self.auto_play_var.set(self.config.get('auto_play', False))
            self.show_overlay_var.set(self.config.get('show_debug_overlay', True))
            
            # Update window detector if needed
            if hasattr(self, 'window_detector'):
                self.window_detector.show_debug_overlay = self.config.get('show_debug_overlay', True)
            
            self.status_var.set(f"Configuration loaded from {filename}")
            
            # Update model info if available
            if 'model_path' in self.config and self.config['model_path']:
                self.model_info_var.set(f"Using: {os.path.basename(self.config['model_path'])}")
            
            messagebox.showinfo("Configuration", f"Configuration loaded from {filename}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
            self.status_var.set(f"Error loading configuration: {str(e)}")
    
    def _save_config_file(self):
        """Save configuration to a file"""
        try:
            # Update config with current UI values
            self.config['capture_interval'] = float(self.interval_var.get())
            self.config['auto_play'] = self.auto_play_var.get()
            self.config['show_debug_overlay'] = self.show_overlay_var.get()
            
            # Get save file path
            filename = filedialog.asksaveasfilename(
                title="Save Configuration",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if not filename:
                return
                
            # Save the config
            with open(filename, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            self.status_var.set(f"Configuration saved to {filename}")
            messagebox.showinfo("Configuration", f"Configuration saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
            self.status_var.set(f"Error saving configuration: {str(e)}")
    
    def _show_settings(self):
        """Show settings dialog"""
        try:
            settings_window = tk.Toplevel(self.root)
            settings_window.title("Settings")
            settings_window.geometry("800x600")
            settings_window.transient(self.root)
            settings_window.grab_set()
            
            # Create settings form
            frame = ttk.Frame(settings_window, padding="10")
            frame.pack(fill=tk.BOTH, expand=True)
            
            # Notebook for tabbed settings
            settings_notebook = ttk.Notebook(frame)
            settings_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # General settings tab
            general_tab = ttk.Frame(settings_notebook)
            settings_notebook.add(general_tab, text="General")
            
            # Directories section
            ttk.Label(general_tab, text="Directories", font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
            
            # Screenshot directory
            ttk.Label(general_tab, text="Screenshot Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
            screenshot_dir_var = tk.StringVar(value=self.config.get('screenshot_dir', 'poker_data/screenshots'))
            ttk.Entry(general_tab, textvariable=screenshot_dir_var, width=40).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
            ttk.Button(general_tab, text="...", width=3, command=lambda: self._browse_directory(screenshot_dir_var)).grid(row=1, column=2, padx=5)
            
            ttk.Label(general_tab, text="Game State Directory:").grid(row=2, column=0, sticky=tk.W, pady=5)
            game_state_dir_var = tk.StringVar(value=self.config.get('game_state_dir', 'poker_data/game_states'))
            ttk.Entry(general_tab, textvariable=game_state_dir_var, width=40).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
            ttk.Button(general_tab, text="...", width=3, command=lambda: self._browse_directory(game_state_dir_var)).grid(row=2, column=2, padx=5)
            
            # Training data directory
            ttk.Label(general_tab, text="Training Data Directory:").grid(row=3, column=0, sticky=tk.W, pady=5)
            training_dir_var = tk.StringVar(value=self.config.get('training_data_dir', 'poker_data/training'))
            ttk.Entry(general_tab, textvariable=training_dir_var, width=40).grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
            ttk.Button(general_tab, text="...", width=3, command=lambda: self._browse_directory(training_dir_var)).grid(row=3, column=2, padx=5)
            
            # Model settings section
            ttk.Label(general_tab, text="Model Settings", font=("TkDefaultFont", 12, "bold")).grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(20, 10))
            
            # Model path
            ttk.Label(general_tab, text="Model Path:").grid(row=5, column=0, sticky=tk.W, pady=5)
            model_path_var = tk.StringVar(value=self.config.get('model_path', ''))
            ttk.Entry(general_tab, textvariable=model_path_var, width=40).grid(row=5, column=1, padx=5, pady=5, sticky=tk.W)
            ttk.Button(general_tab, text="...", width=3, command=lambda: self._browse_file(model_path_var, [("Model files", "*.pt")])).grid(row=5, column=2, padx=5)
            
            # Model save path
            ttk.Label(general_tab, text="Model Save Path:").grid(row=6, column=0, sticky=tk.W, pady=5)
            model_save_var = tk.StringVar(value=self.config.get('model_save_path', 'models/poker_model.pt'))
            ttk.Entry(general_tab, textvariable=model_save_var, width=40).grid(row=6, column=1, padx=5, pady=5, sticky=tk.W)
            ttk.Button(general_tab, text="...", width=3, command=lambda: self._browse_save_file(model_save_var, [("Model files", "*.pt")])).grid(row=6, column=2, padx=5)
            
            # Device selection
            ttk.Label(general_tab, text="Device:").grid(row=7, column=0, sticky=tk.W, pady=5)
            device_var = tk.StringVar(value=self.config.get('device', 'auto'))
            device_options = ['auto', 'cpu', 'cuda']
            if not torch.cuda.is_available():
                # If CUDA not available, show only CPU option
                device_options = ['cpu']
                device_var.set('cpu')
            
            device_combo = ttk.Combobox(general_tab, textvariable=device_var, values=device_options, state="readonly", width=10)
            device_combo.grid(row=7, column=1, sticky=tk.W, padx=5, pady=5)
            
            # Advanced tab
            advanced_tab = ttk.Frame(settings_notebook)
            settings_notebook.add(advanced_tab, text="Advanced")
            
            # Auto-play settings
            ttk.Label(advanced_tab, text="Auto-play Settings", font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
            
            # Enable auto-play
            auto_play_var = tk.BooleanVar(value=self.config.get('auto_play', False))
            ttk.Checkbutton(advanced_tab, text="Enable Auto-play", variable=auto_play_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)
            
            # Confidence threshold
            ttk.Label(advanced_tab, text="Confidence Threshold:").grid(row=2, column=0, sticky=tk.W, pady=5)
            confidence_var = tk.StringVar(value=str(self.config.get('confidence_threshold', 0.7)))
            ttk.Spinbox(advanced_tab, from_=0.0, to=1.0, increment=0.05, textvariable=confidence_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
            
            # Window capture settings
            ttk.Label(advanced_tab, text="Window Capture Settings", font=("TkDefaultFont", 12, "bold")).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(20, 10))
            
            # Auto-detect poker
            auto_detect_var = tk.BooleanVar(value=self.config.get('auto_detect_poker', True))
            ttk.Checkbutton(advanced_tab, text="Auto-detect poker window", variable=auto_detect_var).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)
            
            # Use mock data
            mock_data_var = tk.BooleanVar(value=self.config.get('use_mock_data', True))
            ttk.Checkbutton(advanced_tab, text="Use mock data if no window available", variable=mock_data_var).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=5)
            
            # Show debug overlay
            show_overlay_var = tk.BooleanVar(value=self.config.get('show_debug_overlay', True))
            ttk.Checkbutton(advanced_tab, text="Show debug overlay on screenshots", variable=show_overlay_var).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=5)
            
            # Capture interval
            ttk.Label(advanced_tab, text="Capture Interval (seconds):").grid(row=7, column=0, sticky=tk.W, pady=5)
            interval_var = tk.StringVar(value=str(self.config.get('capture_interval', 2.0)))
            ttk.Spinbox(advanced_tab, from_=0.5, to=10.0, increment=0.5, textvariable=interval_var, width=10).grid(row=7, column=1, sticky=tk.W, padx=5, pady=5)
            
            # UI refresh rate
            ttk.Label(advanced_tab, text="UI Refresh Rate (seconds):").grid(row=8, column=0, sticky=tk.W, pady=5)
            refresh_var = tk.StringVar(value=str(self.config.get('ui_refresh_rate', 1.0)))
            ttk.Spinbox(advanced_tab, from_=0.1, to=5.0, increment=0.1, textvariable=refresh_var, width=10).grid(row=8, column=1, sticky=tk.W, padx=5, pady=5)
            
            # Debugging
            ttk.Label(advanced_tab, text="Debugging", font=("TkDefaultFont", 12, "bold")).grid(row=9, column=0, columnspan=3, sticky=tk.W, pady=(20, 10))
            
            # Log level
            ttk.Label(advanced_tab, text="Log Level:").grid(row=10, column=0, sticky=tk.W, pady=5)
            log_level_var = tk.StringVar(value=self.config.get('log_level', 'INFO'))
            log_level_combo = ttk.Combobox(advanced_tab, textvariable=log_level_var, values=['DEBUG', 'INFO', 'WARNING', 'ERROR'], state="readonly", width=10)
            log_level_combo.grid(row=10, column=1, sticky=tk.W, padx=5, pady=5)
            
            # Buttons
            button_frame = ttk.Frame(frame)
            button_frame.pack(pady=20)
            
            ttk.Button(button_frame, text="Save", command=lambda: self._save_settings(
                screenshot_dir_var.get(),
                game_state_dir_var.get(),
                training_dir_var.get(),
                model_path_var.get(),
                model_save_var.get(),
                device_var.get(),
                auto_play_var.get(),
                float(confidence_var.get()),
                auto_detect_var.get(),
                mock_data_var.get(),
                show_overlay_var.get(),
                float(interval_var.get()),
                float(refresh_var.get()),
                log_level_var.get(),
                settings_window
            )).pack(side=tk.LEFT, padx=10)
            
            ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.LEFT, padx=10)
        except Exception as e:
            logger.error(f"Error showing settings dialog: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Failed to show settings dialog: {str(e)}")
    
    def _browse_directory(self, string_var):
        """Browse for a directory and update the string variable"""
        directory = filedialog.askdirectory(initialdir=string_var.get())
        if directory:
            string_var.set(directory)
    
    def _browse_file(self, string_var, filetypes):
        """Browse for a file and update the string variable"""
        file_path = filedialog.askopenfilename(
            initialdir=os.path.dirname(string_var.get()) if string_var.get() else None,
            filetypes=filetypes
        )
        if file_path:
            string_var.set(file_path)
    
    def _browse_save_file(self, string_var, filetypes):
        """Browse for a save file location and update the string variable"""
        file_path = filedialog.asksaveasfilename(
            initialdir=os.path.dirname(string_var.get()) if string_var.get() else None,
            filetypes=filetypes,
            defaultextension=filetypes[0][1].split(".")[-1]
        )
        if file_path:
            string_var.set(file_path)
    
    def _save_settings(self, screenshot_dir, game_state_dir, training_dir, 
                      model_path, model_save_path, device, auto_play,
                      confidence, auto_detect_poker, use_mock_data,
                      show_debug_overlay, capture_interval, ui_refresh_rate,
                      log_level, window):
        """Save settings from the dialog"""
        try:
            # Update configuration
            self.config['screenshot_dir'] = screenshot_dir
            self.config['game_state_dir'] = game_state_dir
            self.config['training_data_dir'] = training_dir
            self.config['model_path'] = model_path
            self.config['model_save_path'] = model_save_path
            self.config['device'] = device
            self.config['auto_play'] = auto_play
            self.config['confidence_threshold'] = confidence
            self.config['auto_detect_poker'] = auto_detect_poker
            self.config['use_mock_data'] = use_mock_data
            self.config['show_debug_overlay'] = show_debug_overlay
            self.config['capture_interval'] = capture_interval
            self.config['ui_refresh_rate'] = ui_refresh_rate
            self.config['log_level'] = log_level
            
            # Create directories if they don't exist
            for dir_path in [screenshot_dir, game_state_dir, training_dir]:
                os.makedirs(dir_path, exist_ok=True)
            
            # Update the assistant's configuration if it exists
            if hasattr(self, 'poker_assistant') and self.poker_assistant:
                self.poker_assistant.config.update(self.config)
            
            # Update UI elements
            self.auto_play_var.set(auto_play)
            self.interval_var.set(str(capture_interval))
            self.show_overlay_var.set(show_debug_overlay)
            
            # Update window detector debug overlay if it exists
            if hasattr(self, 'window_detector'):
                self.window_detector.show_debug_overlay = show_debug_overlay
            
            # Update model info display
            self.model_info_var.set(f"Using: {os.path.basename(model_path) if model_path else 'Default model'}")
            
            self.status_var.set("Settings saved")
            window.destroy()
            
            # Offer to save to file
            result = messagebox.askyesno(
                "Save Settings", 
                "Would you like to save these settings to a configuration file for future use?"
            )
            
            if result:
                self._save_config_file()
        
        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
    
    def _train_model(self):
        """Train the neural network model"""
        try:
            # Check if poker assistant is available
            if not hasattr(self, 'poker_assistant') or not self.poker_assistant:
                # Create a new assistant for training
                self.status_var.set("Initializing Poker Assistant for training...")
                
                # Create a temporary config file
                temp_config_path = "temp_config.json"
                with open(temp_config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
                
                # Create the poker assistant with our config
                self.poker_assistant = PokerAssistant(config_file=temp_config_path)
                
                # Clean up temp file
                try:
                    os.remove(temp_config_path)
                except:
                    pass
            
            # Show training options dialog
            train_window = tk.Toplevel(self.root)
            train_window.title("Train Model")
            train_window.geometry("400x300")
            train_window.transient(self.root)
            train_window.grab_set()
            
            # Create form
            frame = ttk.Frame(train_window, padding="10")
            frame.pack(fill=tk.BOTH, expand=True)
            
            # Training parameters
            ttk.Label(frame, text="Training Parameters", font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
            
            # Epochs
            ttk.Label(frame, text="Epochs:").grid(row=1, column=0, sticky=tk.W, pady=5)
            epochs_var = tk.StringVar(value="50")
            ttk.Spinbox(frame, from_=1, to=1000, textvariable=epochs_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
            
            # Batch size
            ttk.Label(frame, text="Batch Size:").grid(row=2, column=0, sticky=tk.W, pady=5)
            batch_size_var = tk.StringVar(value="32")
            ttk.Spinbox(frame, from_=1, to=256, textvariable=batch_size_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
            
            # Learning rate
            ttk.Label(frame, text="Learning Rate:").grid(row=3, column=0, sticky=tk.W, pady=5)
            learning_rate_var = tk.StringVar(value="0.001")
            ttk.Entry(frame, textvariable=learning_rate_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
            
            # Data directory
            ttk.Label(frame, text="Training Data:").grid(row=4, column=0, sticky=tk.W, pady=5)
            data_dir_var = tk.StringVar(value=self.config.get('training_data_dir', 'poker_data/training'))
            ttk.Entry(frame, textvariable=data_dir_var, width=30).grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)
            ttk.Button(frame, text="...", width=3, command=lambda: self._browse_directory(data_dir_var)).grid(row=4, column=2, padx=5)
            
            # Buttons
            button_frame = ttk.Frame(frame)
            button_frame.grid(row=5, column=0, columnspan=3, pady=20)
            
            ttk.Button(button_frame, text="Start Training", command=lambda: self._start_training(
                data_dir_var.get(),
                int(epochs_var.get()),
                int(batch_size_var.get()),
                float(learning_rate_var.get()),
                train_window
            )).pack(side=tk.LEFT, padx=10)
            
            ttk.Button(button_frame, text="Cancel", command=train_window.destroy).pack(side=tk.LEFT, padx=10)
        except Exception as e:
            logger.error(f"Error preparing model training: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Failed to prepare model training: {str(e)}")
            self.status_var.set(f"Error preparing model training: {str(e)}")
    
    def _start_training(self, data_dir, epochs, batch_size, learning_rate, window):
        """Start the training process in a separate thread"""
        try:
            # Close the training dialog
            window.destroy()
            
            # Create a progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Training Model")
            progress_window.geometry("400x200")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            ttk.Label(
                progress_window, 
                text="Training Neural Network...",
                font=("TkDefaultFont", 12, "bold")
            ).pack(pady=10)
            
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(
                progress_window, 
                variable=progress_var,
                maximum=100
            )
            progress_bar.pack(fill=tk.X, padx=20, pady=10)
            
            status_var = tk.StringVar(value="Starting training...")
            status_label = ttk.Label(progress_window, textvariable=status_var)
            status_label.pack(pady=10)
            
            # Device info
            device_info = f"Using {'GPU' if torch.cuda.is_available() else 'CPU'}"
            ttk.Label(progress_window, text=device_info).pack(pady=5)
            
            # Use a thread to avoid blocking the UI
            def training_thread():
                try:
                    # Start training
                    status_var.set("Checking training data...")
                    progress_var.set(5)
                    
                    # Train the model
                    history = self.poker_assistant.train_model(
                        data_dir=data_dir,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate
                    )
                    
                    # Check for errors
                    if 'error' in history:
                        status_var.set(f"Error: {history['error']}")
                        messagebox.showerror("Training Error", history['error'])
                        return
                    
                    # Training successful
                    status_var.set("Training complete!")
                    progress_var.set(100)
                    
                    # Update model info
                    model_path = self.config.get('model_save_path', 'models/poker_model.pt')
                    self.model_info_var.set(f"Using: {os.path.basename(model_path)}")
                    
                    messagebox.showinfo(
                        "Training Complete", 
                        f"Model training completed successfully!\n\n"
                        f"Final accuracy: {history.get('accuracy', [0])[-1]:.2f}\n"
                        f"Model saved to: {model_path}"
                    )
                    
                    # Close progress window
                    progress_window.destroy()
                
                except Exception as e:
                    logger.error(f"Error during training: {str(e)}", exc_info=True)
                    status_var.set(f"Error: {str(e)}")
                    messagebox.showerror("Training Error", f"An error occurred during training: {str(e)}")
            
            # Start the training thread
            threading.Thread(target=training_thread, daemon=True).start()
            
            # Update progress periodically (this is a separate update thread)
            def update_progress():
                if progress_window.winfo_exists():
                    # Get the latest state if possible
                    if hasattr(self, 'poker_assistant') and self.poker_assistant:
                        latest = self.poker_assistant.get_latest_state()
                        if 'training_progress' in latest:
                            progress = latest['training_progress']
                            progress_var.set(progress)
                            status_var.set(f"Training: {progress:.1f}% complete")
                    
                    # Schedule the next update
                    progress_window.after(500, update_progress)
            
            # Start progress updates
            update_progress()
            
        except Exception as e:
            logger.error(f"Error starting training: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Failed to start training: {str(e)}")
            self.status_var.set(f"Error starting training: {str(e)}")
    
    def _calibrate_regions(self):
        """Show dialog to calibrate ROI regions"""
        try:
            # Check if we have a window selected
            if not self.selected_window_info:
                messagebox.showwarning("Warning", "No window selected. Please select a window first.")
                return
            
            # Show confirmation dialog
            result = messagebox.askyesno(
                "Calibrate Regions",
                "This will recalibrate regions of interest based on the current window, " +
                "overwriting any customizations. Continue?",
                icon=messagebox.WARNING
            )
            
            if not result:
                return
            
            # Take a screenshot
            if hasattr(self, 'window_detector'):
                screenshot = self.window_detector.capture_screenshot(use_cache=False)
                
                if screenshot is not None:
                    # Force recalibration
                    success = self.window_detector.calibrate_roi_from_screenshot(screenshot, force_calibrate=True)
                    
                    if success:
                        messagebox.showinfo("Success", "Regions successfully recalibrated.")
                        
                        # Sync with poker assistant if it exists
                        if hasattr(self, 'poker_assistant') and self.poker_assistant:
                            if hasattr(self.poker_assistant, 'screen_grabber') and hasattr(self.poker_assistant.screen_grabber, 'roi'):
                                self.poker_assistant.screen_grabber.roi = self.window_detector.roi
                                logger.info("ROI configuration synced with poker assistant")
                    else:
                        messagebox.showerror("Error", "Failed to recalibrate regions.")
                else:
                    messagebox.showerror("Error", "Failed to capture screenshot for calibration.")
            else:
                messagebox.showerror("Error", "Window detector not initialized")
        except Exception as e:
            logger.error(f"Error calibrating regions: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Failed to calibrate regions: {str(e)}")
            self.status_var.set(f"Error calibrating regions: {str(e)}")
    
    def _analyze_screenshot(self):
        """Analyze a single screenshot file"""
        try:
            # Get screenshot file
            screenshot_path = filedialog.askopenfilename(
                title="Select Screenshot to Analyze",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")]
            )
            
            if not screenshot_path:
                return
            
            # Check if poker assistant is initialized
            if not hasattr(self, 'poker_assistant') or not self.poker_assistant:
                self.status_var.set("Initializing Poker Assistant for analysis...")
                
                # Create a temporary config file
                temp_config_path = "temp_config.json"
                with open(temp_config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
                
                # Create the poker assistant with our config
                self.poker_assistant = PokerAssistant(config_file=temp_config_path)
                
                # Clean up temp file
                try:
                    os.remove(temp_config_path)
                except:
                    pass
            
            # Show progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Analyzing Screenshot")
            progress_window.geometry("400x150")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            ttk.Label(
                progress_window, 
                text="Analyzing screenshot...",
                font=("TkDefaultFont", 12, "bold")
            ).pack(pady=10)
            
            status_var = tk.StringVar(value="Starting analysis...")
            status_label = ttk.Label(progress_window, textvariable=status_var)
            status_label.pack(pady=10)
            
            # Use a thread to avoid blocking the UI
            def analysis_thread():
                try:
                    # Start analysis
                    status_var.set("Analyzing screenshot...")
                    
                    # Analyze the screenshot
                    result = self.poker_assistant.analyze_screenshot(screenshot_path)
                    
                    # Close progress window
                    progress_window.destroy()
                    
                    # Check for errors
                    if 'error' in result:
                        messagebox.showerror("Analysis Error", result['error'])
                        return
                    
                    # Display results in a new window
                    result_window = tk.Toplevel(self.root)
                    result_window.title("Analysis Results")
                    result_window.geometry("800x600")
                    result_window.transient(self.root)
                    
                    # Create paned window for results
                    paned = ttk.PanedWindow(result_window, orient=tk.HORIZONTAL)
                    paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    
                    # Left side - Screenshot preview
                    preview_frame = ttk.LabelFrame(paned, text="Screenshot")
                    paned.add(preview_frame, weight=1)
                    
                    # Load and display the image
                    img = cv2.imread(screenshot_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(img)
                        
                        # Resize to fit
                        max_width = 400
                        max_height = 500
                        img.thumbnail((max_width, max_height), Image.LANCZOS)
                        
                        img_tk = ImageTk.PhotoImage(image=img)
                        label = ttk.Label(preview_frame, image=img_tk)
                        label.image = img_tk  # Keep a reference
                        label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    
                    # Right side - Analysis results
                    results_frame = ttk.LabelFrame(paned, text="Analysis Results")
                    paned.add(results_frame, weight=1)
                    
                    # Create a notebook for tabbed results
                    results_notebook = ttk.Notebook(results_frame)
                    results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    
                    # Game state tab
                    game_state_tab = ttk.Frame(results_notebook)
                    results_notebook.add(game_state_tab, text="Game State")
                    
                    # Game state display
                    game_state_text = tk.Text(game_state_tab, wrap=tk.WORD, width=40, height=20)
                    game_state_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    game_state_text.insert(tk.END, json.dumps(result['game_state'], indent=2))
                    game_state_text.config(state=tk.DISABLED)
                    
                    # Decision tab
                    decision_tab = ttk.Frame(results_notebook)
                    results_notebook.add(decision_tab, text="Decision")
                    
                    # Decision display
                    ttk.Label(decision_tab, text="Recommended Action:").pack(anchor=tk.W, pady=(10, 5))
                    ttk.Label(
                        decision_tab, 
                        text=result['decision']['action'].upper(),
                        font=("TkDefaultFont", 12, "bold")
                    ).pack(anchor=tk.W, padx=10)
                    
                    ttk.Label(decision_tab, text="Confidence:").pack(anchor=tk.W, pady=(10, 5))
                    ttk.Label(
                        decision_tab, 
                        text=f"{result['decision']['confidence']:.2f}",
                        font=("TkDefaultFont", 12, "bold")
                    ).pack(anchor=tk.W, padx=10)
                    
                    ttk.Label(decision_tab, text="Bet Size:").pack(anchor=tk.W, pady=(10, 5))
                    ttk.Label(
                        decision_tab, 
                        text=f"{result['decision']['bet_size_percentage'] * 100:.1f}% of pot",
                        font=("TkDefaultFont", 12, "bold")
                    ).pack(anchor=tk.W, padx=10)
                    
                    # Close button
                    ttk.Button(
                        result_window, 
                        text="Close",
                        command=result_window.destroy
                    ).pack(pady=10)
                    
                except Exception as e:
                    # Close progress window if still open
                    if progress_window.winfo_exists():
                        progress_window.destroy()
                    
                    logger.error(f"Error analyzing screenshot: {str(e)}", exc_info=True)
                    messagebox.showerror("Error", f"Failed to analyze screenshot: {str(e)}")
            
            # Start the analysis thread
            threading.Thread(target=analysis_thread, daemon=True).start()
        except Exception as e:
            logger.error(f"Error preparing screenshot analysis: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Failed to prepare screenshot analysis: {str(e)}")
            self.status_var.set(f"Error preparing screenshot analysis: {str(e)}")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
        Poker Computer Vision Assistant
        
        This application allows you to analyze poker games using computer vision and machine learning.
        
        Getting Started:
        1. Select a poker game window from the dropdown
        2. Click "Select" to confirm the window
        3. Click "Preview Window" to check that detection works
        4. Set the capture interval (in seconds)
        5. Click "Start Assistant" to begin analysis
        
        The application will:
        - Take periodic screenshots of the selected poker game
        - Analyze the screenshots to identify cards, chips, etc.
        - Use a neural network to make optimal poker decisions
        - Display the recommended actions
        
        If auto-play is enabled, the assistant can automatically take actions in the game.
        
        For more information, see the documentation.
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        help_window.geometry("600x500")
        help_window.transient(self.root)
        
        # Create scrollable text area
        text_frame = ttk.Frame(help_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text_widget.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)
        
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)
        
        # Close button
        ttk.Button(help_window, text="Close", command=help_window.destroy).pack(pady=10)
    
    def _show_about(self):
        """Show about information"""
        about_text = """
        Poker Computer Vision Assistant
        Version 2.0.0
        
        A Python application that uses computer vision and neural networks
        to analyze poker games and provide decision assistance.
        
        Key technologies:
        • OpenCV for image processing and card detection
        • PyTorch for neural network decision making
        • Tkinter for the user interface
        
        © 2025
        """
        
        messagebox.showinfo("About", about_text)
    
    def _on_exit(self):
        """Handle application exit with proper cleanup"""
        try:
            # Check if the assistant is running
            if hasattr(self, 'poker_assistant') and self.poker_assistant and hasattr(self.poker_assistant, 'running') and self.poker_assistant.running:
                result = messagebox.askyesno(
                    "Exit",
                    "Poker Assistant is still running. Stop it and exit?"
                )
                
                if result:
                    self._stop_assistant()
                else:
                    return  # Don't exit
            
            # Cancel any scheduled updates
            if self.status_update_id:
                self.root.after_cancel(self.status_update_id)
            
            # Destroy the root window
            self.root.destroy()
        except Exception as e:
            logger.error(f"Error during exit: {str(e)}", exc_info=True)
            # Force exit
            self.root.destroy()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Poker Computer Vision Assistant")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--window", help="Title of the window to capture (partial match)")
    
    args = parser.parse_args()
    
    # Create and run the application
    root = tk.Tk()
    app = PokerApplication(root, config_file=args.config)
    
    # Try to select window if specified
    if args.window and hasattr(app, 'window_detector'):
        # Get window list
        windows = app.window_detector.get_window_list()
        
        # Find matching windows
        matching_windows = [w for w in windows if args.window.lower() in w.title.lower()]
        
        if matching_windows:
            # Select the first matching window
            app.selected_window_info = matching_windows[0]
            app.window_detector.select_window(app.selected_window_info)
            
            # Update UI
            app.selected_window_var.set(app.selected_window_info.title)
            app.window_handle_var.set(str(app.selected_window_info.handle))
            
            if app.selected_window_info.rect:
                rect_str = f"({app.selected_window_info.rect[0]}, {app.selected_window_info.rect[1]}, " \
                        f"{app.selected_window_info.rect[2]}, {app.selected_window_info.rect[3]})"
                app.window_rect_var.set(rect_str)
            
            app.status_var.set(f"Selected window: {app.selected_window_info.title}")
            
            # Preview the window
            app._preview_selected_window()
    
    # Start the application main loop
    app.root.protocol("WM_DELETE_WINDOW", app._on_exit)
    app.root.mainloop()

if __name__ == "__main__":
    main()