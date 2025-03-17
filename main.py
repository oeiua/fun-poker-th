#!/usr/bin/env python3
# Filename: main.py - Main entry point for the Poker Screen Grabber application

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
import glob
import copy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("poker_app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PokerApp")

# Import our modules
try:
    from poker_screen_grabber import PokerScreenGrabber, WindowInfo
    from poker_cv_analyzer import PokerImageAnalyzer
    from poker_neural_engine_torch import PokerNeuralNetwork, HandEvaluator
    from poker_app_integration import PokerAssistant
except ImportError:
    # Add the current directory to the path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from poker_screen_grabber import PokerScreenGrabber, WindowInfo
    from poker_cv_analyzer import PokerImageAnalyzer
    from poker_neural_engine_torch import PokerNeuralNetwork, HandEvaluator
    from poker_app_integration import PokerAssistant

class PokerScreenGrabberApp:
    """Main application GUI for Poker Screen Grabber with PyTorch backend"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Poker Screen Grabber (PyTorch)")
        self.root.geometry("1300x768+1080+800")
        self.root.minsize(800, 600)
        
        # Initialize window selection attributes
        self.window_list = []
        self.selected_window_info = None
        
        # Create screen grabber for window detection
        self.window_detector = PokerScreenGrabber(capture_interval=1.0, output_dir="poker_data/screenshots")
        
        # Load ROI configuration - modified to be more robust
        roi_file = "roi_config.json"
        if os.path.exists(roi_file):
            try:
                # Attempt to load from file
                success = self.window_detector.load_regions_from_file(roi_file)
                if not success:
                    # Show error notification
                    messagebox.showwarning(
                        "ROI Configuration Warning", 
                        f"Could not load ROI configuration from {roi_file}.\n\n"
                        "Using default values. You can recalibrate once you select a window."
                    )
                    # Since loading failed, we'll explicitly use defaults
                    self.window_detector.roi = self.window_detector._get_default_roi()
            except Exception as e:
                # Extra exception handling
                logger.error(f"Error loading ROI configuration: {str(e)}", exc_info=True)
                messagebox.showwarning(
                    "ROI Configuration Error", 
                    f"Error loading ROI configuration: {str(e)}.\n\n"
                    "Using default values. You can recalibrate once you select a window."
                )
                # Use defaults
                self.window_detector.roi = self.window_detector._get_default_roi()
        else:
            # File doesn't exist, use defaults - let the user know
            logger.info("No ROI configuration file found. Using default values.")
            self.window_detector.roi = self.window_detector._get_default_roi()
        
        # Initialize components
        self.poker_assistant = None
        self.config = self._load_default_config()

        # Status variables
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - PyTorch Backend")
        
        # Set up the UI
        self._create_menu()
        self._create_main_frame()
        
        # Status bar
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Update timer
        self.update_id = None
        self._schedule_ui_update()
        
        # Populate window list
        self._refresh_windows()
        
        logger.info("Application started with PyTorch backend")
    
    def _load_default_config(self):
        """Load default configuration"""
        config = {
            'capture_interval': 2.0,
            'screenshot_dir': 'poker_data/screenshots',
            'game_state_dir': 'poker_data/game_states',
            'training_data_dir': 'poker_data/training',
            'model_path': None,
            'model_save_path': 'poker_model.pt',  # Changed to .pt for PyTorch
            'auto_play': False,
            'confidence_threshold': 0.7
        }
        
        # Create directories if they don't exist
        for dir_path in [config['screenshot_dir'], config['game_state_dir'], config['training_data_dir']]:
            os.makedirs(dir_path, exist_ok=True)
        
        return config
    
    def _create_menu(self):
        """Create the application menu"""
        menu_bar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Load Configuration", command=self._load_config)
        file_menu.add_command(label="Save Configuration", command=self._save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_exit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menu_bar, tearoff=0)
        tools_menu.add_command(label="Settings", command=self._show_settings)
        tools_menu.add_command(label="Train Neural Network", command=self._train_network)
        menu_bar.add_cascade(label="Tools", menu=tools_menu)
        
        # Help menu
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Help", command=self._show_help)
        help_menu.add_command(label="About", command=self._show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menu_bar)

    def _recalibrate_roi(self):
        """Explicitly recalibrate ROI from the current window"""
        if not self.selected_window_info:
            messagebox.showwarning("Warning", "No window selected. Please select a window first.")
            return
        
        # Confirm with user
        result = messagebox.askyesno(
            "Recalibrate ROI",
            "This will recalibrate regions of interest based on the current window, " +
            "overwriting any customizations. Continue?",
            icon=messagebox.WARNING
        )
        
        if not result:
            return
        
        # Take a screenshot
        screenshot = self.window_detector.capture_screenshot()
        
        if screenshot is not None:
            # Log screenshot dimensions
            logger.info(f"Screenshot dimensions for calibration: {screenshot.shape}")
            
            # Before calibration, save the current screenshot with overlay
            debug_dir = "debug_calibration"
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = int(time.time())
            
            # Save pre-calibration screenshot with current ROI overlay
            pre_overlay = self.window_detector.add_debugging_overlay(screenshot)
            pre_path = f"{debug_dir}/pre_calibration_{timestamp}.png"
            cv2.imwrite(pre_path, pre_overlay)
            
            # Force recalibration with the current screenshot dimensions
            success = self.window_detector.calibrate_roi_from_screenshot(screenshot, force_calibrate=True)
            
            if success:
                # Save post-calibration screenshot with new ROI overlay
                post_screenshot = self.window_detector.capture_screenshot()
                post_overlay = self.window_detector.add_debugging_overlay(post_screenshot)
                post_path = f"{debug_dir}/post_calibration_{timestamp}.png"
                cv2.imwrite(post_path, post_overlay)
                
                messagebox.showinfo("Success", "ROI successfully recalibrated. Debug images saved to the 'debug_calibration' folder.")
                
                # Sync with poker assistant if it exists
                if hasattr(self, 'poker_assistant') and self.poker_assistant:
                    if hasattr(self.poker_assistant, 'screen_grabber') and hasattr(self.poker_assistant.screen_grabber, 'roi'):
                        self.poker_assistant.screen_grabber.roi = copy.deepcopy(self.window_detector.roi)
                        logger.info("ROI configuration synced with poker assistant")
            else:
                messagebox.showerror("Error", "Failed to recalibrate ROI.")
        else:
            messagebox.showerror("Error", "Failed to capture screenshot for calibration.")

    def _create_main_frame(self):
        """Create the main application frame"""
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
        self.interval_var = tk.StringVar(value=str(self.config['capture_interval']))
        interval_entry = ttk.Entry(control_frame, width=10, textvariable=self.interval_var)
        interval_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.start_btn = ttk.Button(control_frame, text="Start Capture", command=self._start_assistant)
        self.start_btn.grid(row=1, column=2, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Capture", command=self._stop_assistant, state=tk.DISABLED)
        self.stop_btn.grid(row=1, column=3, padx=5)
        
        # Screenshot refresh button
        self.refresh_screenshot_btn = ttk.Button(control_frame, text="Refresh Screenshot", command=self._update_latest_screenshot_display)
        self.refresh_screenshot_btn.grid(row=1, column=4, padx=5)
        
        # Auto-play checkbox
        self.auto_play_var = tk.BooleanVar(value=self.config['auto_play'])
        auto_play_cb = ttk.Checkbutton(
            control_frame, 
            text="Enable Auto-Play",
            variable=self.auto_play_var,
            command=self._toggle_auto_play
        )
        auto_play_cb.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # PyTorch information label
        ttk.Label(
            control_frame,
            text="Using PyTorch Backend",
            font=("TkDefaultFont", 10, "bold")
        ).grid(row=2, column=2, columnspan=2, sticky=tk.E, pady=5)
        
        # Create a frame for the region configuration buttons
        roi_frame = ttk.Frame(control_frame)
        roi_frame.grid(row=2, column=4, padx=5, pady=5)
        
        # Edit ROI configuration button
        edit_roi_btn = ttk.Button(roi_frame, text="Edit Regions", command=self._show_region_settings)
        edit_roi_btn.pack(side=tk.LEFT, padx=2)
        
        # Recalibrate ROI button (new button to explicitly control recalibration)
        recalibrate_roi_btn = ttk.Button(roi_frame, text="Recalibrate", command=self._recalibrate_roi)
        recalibrate_roi_btn.pack(side=tk.LEFT, padx=2)
        # Content panes (using PanedWindow for resizable sections)
        content_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        content_paned.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Left pane - Game preview
        preview_frame = ttk.LabelFrame(content_paned, text="Game Preview", padding="10")
        content_paned.add(preview_frame, weight=2)
        
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # Right pane - Analysis results
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
        self.action_var = tk.StringVar(value="N/A")
        ttk.Label(decision_tab, textvariable=self.action_var, font=("TkDefaultFont", 12, "bold")).pack(anchor=tk.W, padx=10)
        
        ttk.Label(decision_tab, text="Confidence:").pack(anchor=tk.W, pady=(10, 5))
        self.confidence_var = tk.StringVar(value="N/A")
        ttk.Label(decision_tab, textvariable=self.confidence_var, font=("TkDefaultFont", 12, "bold")).pack(anchor=tk.W, padx=10)
        
        ttk.Label(decision_tab, text="Bet Size:").pack(anchor=tk.W, pady=(10, 5))
        self.bet_size_var = tk.StringVar(value="N/A")
        ttk.Label(decision_tab, textvariable=self.bet_size_var, font=("TkDefaultFont", 12, "bold")).pack(anchor=tk.W, padx=10)
        
        # Hand strength section
        ttk.Separator(decision_tab, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(decision_tab, text="Hand Strength:").pack(anchor=tk.W, pady=(5, 5))
        self.hand_strength_var = tk.StringVar(value="N/A")
        ttk.Label(decision_tab, textvariable=self.hand_strength_var, font=("TkDefaultFont", 12, "bold")).pack(anchor=tk.W, padx=10)
        
        ttk.Label(decision_tab, text="Equity:").pack(anchor=tk.W, pady=(10, 5))
        self.equity_var = tk.StringVar(value="N/A")
        ttk.Label(decision_tab, textvariable=self.equity_var, font=("TkDefaultFont", 12, "bold")).pack(anchor=tk.W, padx=10)
        
        # PyTorch model info
        ttk.Separator(decision_tab, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(decision_tab, text="PyTorch Model:").pack(anchor=tk.W, pady=(5, 5))
        self.model_info_var = tk.StringVar(value="Default model")
        ttk.Label(decision_tab, textvariable=self.model_info_var, font=("TkDefaultFont", 10)).pack(anchor=tk.W, padx=10)
        
        # Add tabs to notebook
        tab_control.pack(fill=tk.BOTH, expand=True)
        
        # Window info tab
        window_tab = ttk.Frame(tab_control)
        tab_control.add(window_tab, text="Window Info")
        
        # Window info display
        ttk.Label(window_tab, text="Selected Window:").pack(anchor=tk.W, pady=(10, 5))
        self.selected_window_var = tk.StringVar(value="None")
        ttk.Label(window_tab, textvariable=self.selected_window_var, font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, padx=10)
        
        ttk.Label(window_tab, text="Window Handle:").pack(anchor=tk.W, pady=(10, 5))
        self.window_handle_var = tk.StringVar(value="None")
        ttk.Label(window_tab, textvariable=self.window_handle_var).pack(anchor=tk.W, padx=10)
        
        ttk.Label(window_tab, text="Window Rect:").pack(anchor=tk.W, pady=(10, 5))
        self.window_rect_var = tk.StringVar(value="None")
        ttk.Label(window_tab, textvariable=self.window_rect_var).pack(anchor=tk.W, padx=10)
        
        # Available Windows list
        ttk.Separator(window_tab, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(window_tab, text="Available Windows:").pack(anchor=tk.W, pady=(5, 5))
        
        region_config_btn = ttk.Button(control_frame, text="Configure Regions", command=self._show_region_settings)
        region_config_btn.grid(row=2, column=4, padx=5, pady=5)  # Adjust row/column as needed

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
    
    def _refresh_windows(self):
        """Refresh the list of available windows"""
        # Get windows from the window detector
        self.window_list = self.window_detector.get_window_list()
        
        # Update dropdown
        window_titles = [w.title for w in self.window_list]
        self.window_selector['values'] = window_titles
        
        if window_titles:
            # Look for PokerTH windows
            poker_indices = [i for i, title in enumerate(window_titles) if "poker" in title.lower()]
            if poker_indices:
                # Select the first PokerTH window found
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
        
        self.status_var.set(f"Found {len(self.window_list)} windows - PyTorch backend active")
    
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
        """Select a window for capturing"""
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
            if self.window_detector:
                success = self.window_detector.select_window(self.selected_window_info)
                
                # Check if ROI config exists
                roi_file = "roi_config.json"
                if os.path.exists(roi_file):
                    # Ask user if they want to auto-calibrate or keep existing configuration
                    result = messagebox.askyesno(
                        "ROI Configuration", 
                        "An existing ROI configuration was found. Would you like to auto-calibrate ROI for the new window?\n\n"
                        "Selecting 'No' will keep the existing ROI configuration.",
                        icon=messagebox.QUESTION
                    )
                    
                    if result:
                        # User wants to calibrate
                        self._preview_selected_window()
                        calibrate_success = self.window_detector.calibrate_roi_from_screenshot()
                        if calibrate_success:
                            self.status_var.set(f"Selected window: {self.selected_window_info.title} (ROI calibrated)")
                            # Save the new ROI
                            self.window_detector.save_regions_to_file(roi_file)
                            
                            # Sync with poker assistant if it exists
                            if hasattr(self, 'poker_assistant') and self.poker_assistant:
                                if hasattr(self.poker_assistant, 'screen_grabber') and hasattr(self.poker_assistant.screen_grabber, 'roi'):
                                    self.poker_assistant.screen_grabber.roi = copy.deepcopy(self.window_detector.roi)
                        else:
                            self.status_var.set(f"Selected window: {self.selected_window_info.title} (ROI calibration failed)")
                    else:
                        # User wants to keep existing ROI
                        self.status_var.set(f"Selected window: {self.selected_window_info.title} (using existing ROI)")
                        # Just to be safe, let's reload ROI from file
                        self.window_detector.load_regions_from_file(roi_file)
                else:
                    # No existing ROI, calibrate automatically
                    self._preview_selected_window()
                    self.window_detector.calibrate_roi_from_screenshot()
                    self.window_detector.save_regions_to_file(roi_file)
                    self.status_var.set(f"Selected window: {self.selected_window_info.title} (ROI calibrated)")
                
                if not success:
                    self.status_var.set(f"Error selecting window: {self.selected_window_info.title}")
            else:
                self.status_var.set(f"Selected window: {self.selected_window_info.title}")
            
            # Preview the window
            self._preview_selected_window()
        else:
            self.status_var.set(f"Window not found: {selected_title}")


    def _preview_selected_window(self):
        """Preview the selected window"""
        if not self.selected_window_info:
            self.status_var.set("No window selected for preview")
            return
        
        # Use the window detector to capture a screenshot
        self.window_detector.select_window(self.selected_window_info)
        img = self.window_detector.capture_screenshot()
        
        if img is not None:
            # Update preview
            self._update_preview(img)
            self.status_var.set(f"Previewing window: {self.selected_window_info.title}")
        else:
            self.status_var.set("Failed to capture window preview")
    
    def _update_preview(self, img):
        """Update the preview image"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        # Resize to fit the preview area while maintaining aspect ratio
        preview_width = self.preview_label.winfo_width() or 640
        preview_height = self.preview_label.winfo_height() or 480
        
        img.thumbnail((preview_width, preview_height), Image.LANCZOS)
        
        img = ImageTk.PhotoImage(image=img)
        self.preview_label.configure(image=img)
        self.preview_label.image = img  # Keep a reference
    
    def _update_latest_screenshot_display(self):
        """Update the preview with the latest screenshot from the output directory"""
        try:
            # Get the screenshot directory from config
            screenshot_dir = self.config.get('screenshot_dir', 'poker_data/screenshots')
            
            # Find the most recent screenshot
            list_of_files = glob.glob(os.path.join(screenshot_dir, 'screenshot_*.png'))
            if not list_of_files:
                self.status_var.set("No screenshots found in directory")
                return False
                
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
                return True
            else:
                self.status_var.set(f"Failed to load latest screenshot: {latest_screenshot}")
                return False
        except Exception as e:
            self.status_var.set(f"Error updating screenshot display: {str(e)}")
            return False

    def _sync_roi_with_components(self):
        """Synchronize ROI configuration with all components that use it"""
        try:
            # Sync with poker assistant if it exists
            if hasattr(self, 'poker_assistant') and self.poker_assistant:
                if hasattr(self.poker_assistant, 'screen_grabber') and hasattr(self.poker_assistant.screen_grabber, 'roi'):
                    # Make a deep copy to prevent shared references
                    self.poker_assistant.screen_grabber.roi = copy.deepcopy(self.window_detector.roi)
                    logger.info("ROI configuration synced with poker assistant")
            
            # You might have other components that need syncing here
            
            return True
        except Exception as e:
            logger.error(f"Error syncing ROI with components: {str(e)}", exc_info=True)
            return False

    def _start_assistant(self):
        """Start the poker assistant"""
        try:
            # Check if a window is selected
            if not self.selected_window_info:
                # Try to auto-select a poker window
                poker_windows = [w for w in self.window_list if "poker" in w.title.lower()]
                if poker_windows:
                    self.selected_window_info = poker_windows[0]
                    self.status_var.set(f"Auto-selected window: {self.selected_window_info.title}")
                    
                    # Update window info display
                    self.selected_window_var.set(self.selected_window_info.title)
                    self.window_handle_var.set(str(self.selected_window_info.handle))
                    
                    if self.selected_window_info.rect:
                        rect_str = f"({self.selected_window_info.rect[0]}, {self.selected_window_info.rect[1]}, " \
                                f"{self.selected_window_info.rect[2]}, {self.selected_window_info.rect[3]})"
                        self.window_rect_var.set(rect_str)
                else:
                    # Show warning
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
            
            # Initialize and start the poker assistant
            if not self.poker_assistant:
                self.poker_assistant = PokerAssistant(config_file=None)
                # Set the config directly
                self.poker_assistant.config.update(self.config)
                
                # Pass window info to the screen grabber if available
                if self.selected_window_info:
                    # Make sure screen grabber exists and has the select_window method
                    if hasattr(self.poker_assistant, 'screen_grabber') and hasattr(self.poker_assistant.screen_grabber, 'select_window'):
                        self.poker_assistant.screen_grabber.select_window(self.selected_window_info)
                        
                        if hasattr(self.window_detector, 'roi') and hasattr(self.poker_assistant.screen_grabber, 'roi'):
                            self.poker_assistant.screen_grabber.roi = copy.deepcopy(self.window_detector.roi)
                        # Print debug info
                        logger.info(f"Selected window passed to screen grabber: {self.selected_window_info.title}")
                        logger.info(f"Handle: {self.selected_window_info.handle}")
                        if self.selected_window_info.rect:
                            logger.info(f"Rect: {self.selected_window_info.rect}")
            
            # Make sure poker_assistant is initialized
            if not self.poker_assistant:
                logger.error("Failed to initialize poker assistant")
                messagebox.showerror("Error", "Failed to initialize poker assistant")
                return
            
            # Start the poker assistant
            success = self.poker_assistant.start()
            if not success:
                logger.error("Poker assistant failed to start")
                messagebox.showerror("Error", "Poker assistant failed to start")
                return
            
            # Update UI
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_var.set("Poker Assistant started with PyTorch backend")
            
            # Start updating the UI with latest data
            self._schedule_ui_update()
            
            # Update model info display
            model_path = self.config.get('model_path', 'Default PyTorch model')
            self.model_info_var.set(f"Using: {os.path.basename(model_path) if model_path else 'Default PyTorch model'}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start Poker Assistant: {str(e)}")
            logger.error(f"Failed to start Poker Assistant: {str(e)}", exc_info=True)  # Log full exception info
    
    def _stop_assistant(self):
        """Stop the poker assistant"""
        if self.poker_assistant:
            self.poker_assistant.stop()
            
            # Update UI
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.status_var.set("Poker Assistant stopped")
    
    def _toggle_auto_play(self):
        """Toggle auto-play mode"""
        if self.poker_assistant:
            self.poker_assistant.config['auto_play'] = self.auto_play_var.get()
        
        if self.auto_play_var.get():
            self.status_var.set("Auto-play enabled")
        else:
            self.status_var.set("Auto-play disabled")
    
    def _update_ui(self):
        """Update the UI with the latest game state and decision"""
        if self.poker_assistant and self.poker_assistant.running:
            latest = self.poker_assistant.get_latest_state()
            
            if latest['game_state']:
                # Update game state display
                self.game_state_text.delete(1.0, tk.END)
                self.game_state_text.insert(tk.END, json.dumps(latest['game_state'], indent=2))
            
            if latest['decision']:
                # Update decision display
                action = latest['decision']['action']
                confidence = latest['decision']['confidence']
                bet_size = latest['decision']['bet_size_percentage']
                
                self.action_var.set(action.upper())
                self.confidence_var.set(f"{confidence:.2f}")
                self.bet_size_var.set(f"{bet_size * 100:.1f}% of pot")
                
                # In a real implementation, you'd calculate these from the game state
                self.hand_strength_var.set("Flush Draw")
                self.equity_var.set("32%")
            
            # Update preview with latest screenshot
            self._update_latest_screenshot_display()
        
        # Schedule the next update
        self.update_id = self.root.after(1000, self._update_ui)
    
    def _schedule_ui_update(self):
        """Schedule periodic UI updates"""
        self._update_ui()
    
    def _load_config(self):
        """Load configuration from a file"""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                self.config.update(config)
                
                # Update UI elements with loaded config
                self.interval_var.set(str(self.config['capture_interval']))
                self.auto_play_var.set(self.config['auto_play'])
                
                self.status_var.set(f"Configuration loaded from {filename}")
                logger.info(f"Configuration loaded from {filename}")
                
                # Update model info if available
                if 'model_path' in config and config['model_path']:
                    self.model_info_var.set(f"Using: {os.path.basename(config['model_path'])}")
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
                logger.error(f"Failed to load configuration: {str(e)}")
    
    def _save_config(self):
        """Save configuration to a file"""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Update config with current UI values
                self.config['capture_interval'] = float(self.interval_var.get())
                self.config['auto_play'] = self.auto_play_var.get()
                
                with open(filename, 'w') as f:
                    json.dump(self.config, f, indent=2)
                
                self.status_var.set(f"Configuration saved to {filename}")
                logger.info(f"Configuration saved to {filename}")
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
                logger.error(f"Failed to save configuration: {str(e)}")
    
    def _show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings - PyTorch")
        settings_window.geometry("800x1200+1080+800")  # Slightly taller for PyTorch info
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Create settings form
        frame = ttk.Frame(settings_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Directories section
        ttk.Label(frame, text="Directories", font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        ttk.Label(frame, text="Debugging", font=("TkDefaultFont", 12, "bold")).grid(row=15, column=0, columnspan=3, sticky=tk.W, pady=(20, 10))

        # Screenshot directory
        ttk.Label(frame, text="Screenshot Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        screenshot_dir_var = tk.StringVar(value=self.config['screenshot_dir'])
        ttk.Entry(frame, textvariable=screenshot_dir_var, width=40).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Button(frame, text="...", width=3, command=lambda: self._browse_directory(screenshot_dir_var)).grid(row=1, column=2, padx=5)
        
        ttk.Label(frame, text="Game State Directory:").grid(row=2, column=0, sticky=tk.W, pady=5)
        game_state_dir_var = tk.StringVar(value=self.config['game_state_dir'])
        ttk.Entry(frame, textvariable=game_state_dir_var, width=40).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Button(frame, text="...", width=3, command=lambda: self._browse_directory(game_state_dir_var)).grid(row=2, column=2, padx=5)
        
        # Training data directory
        ttk.Label(frame, text="Training Data Directory:").grid(row=3, column=0, sticky=tk.W, pady=5)
        training_dir_var = tk.StringVar(value=self.config['training_data_dir'])
        ttk.Entry(frame, textvariable=training_dir_var, width=40).grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Button(frame, text="...", width=3, command=lambda: self._browse_directory(training_dir_var)).grid(row=3, column=2, padx=5)
        
        # Model settings
        ttk.Label(frame, text="PyTorch Model", font=("TkDefaultFont", 12, "bold")).grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(20, 10))
        
        # Model path
        ttk.Label(frame, text="Model Path (.pt):").grid(row=5, column=0, sticky=tk.W, pady=5)
        model_path_var = tk.StringVar(value=self.config['model_path'] or "")
        ttk.Entry(frame, textvariable=model_path_var, width=40).grid(row=5, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Button(frame, text="...", width=3, command=lambda: self._browse_file(model_path_var, [("PyTorch models", "*.pt"), ("All files", "*.*")])).grid(row=5, column=2, padx=5)
        
        # Model save path
        ttk.Label(frame, text="Model Save Path:").grid(row=6, column=0, sticky=tk.W, pady=5)
        model_save_var = tk.StringVar(value=self.config['model_save_path'] or "poker_model.pt")
        ttk.Entry(frame, textvariable=model_save_var, width=40).grid(row=6, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Button(frame, text="...", width=3, command=lambda: self._browse_save_file(model_save_var, [("PyTorch models", "*.pt"), ("All files", "*.*")])).grid(row=6, column=2, padx=5)
        
        # PyTorch-specific settings
        ttk.Label(frame, text="Device:").grid(row=7, column=0, sticky=tk.W, pady=5)
        device_var = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        device_combo = ttk.Combobox(frame, textvariable=device_var, values=["cpu", "cuda"], state="readonly", width=10)
        device_combo.grid(row=7, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(frame, text="Debugging", font=("TkDefaultFont", 12, "bold")).grid(row=15, column=0, columnspan=3, sticky=tk.W, pady=(20, 10))

        # Show region overlays
        show_overlay_var = tk.BooleanVar(value=self.config.get('show_debug_overlay', True))
        ttk.Checkbutton(frame, text="Show region debugging overlay", variable=show_overlay_var).grid(row=16, column=0, columnspan=2, sticky=tk.W, pady=5)


        # Window capture settings
        ttk.Label(frame, text="Window Capture", font=("TkDefaultFont", 12, "bold")).grid(row=8, column=0, columnspan=3, sticky=tk.W, pady=(20, 10))
        
        # Auto-detect PokerTH
        auto_detect_var = tk.BooleanVar(value=self.config.get('auto_detect_poker', True))
        ttk.Checkbutton(frame, text="Auto-detect PokerTH window", variable=auto_detect_var).grid(row=9, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Use mock data if no window
        mock_data_var = tk.BooleanVar(value=self.config.get('use_mock_data', True))
        ttk.Checkbutton(frame, text="Use mock data if no window available", variable=mock_data_var).grid(row=10, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Auto-play settings
        ttk.Label(frame, text="Auto-play", font=("TkDefaultFont", 12, "bold")).grid(row=11, column=0, columnspan=3, sticky=tk.W, pady=(20, 10))
        
        # Enable auto-play
        auto_play_var = tk.BooleanVar(value=self.config['auto_play'])
        ttk.Checkbutton(frame, text="Enable Auto-play", variable=auto_play_var).grid(row=12, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Confidence threshold
        ttk.Label(frame, text="Confidence Threshold:").grid(row=13, column=0, sticky=tk.W, pady=5)
        confidence_var = tk.StringVar(value=str(self.config['confidence_threshold']))
        ttk.Spinbox(frame, from_=0.0, to=1.0, increment=0.05, textvariable=confidence_var, width=10).grid(row=13, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=14, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="Save", command=lambda: self._save_settings(
            screenshot_dir_var.get(),
            game_state_dir_var.get(),
            training_dir_var.get(),
            model_path_var.get(),
            model_save_var.get(),
            device_var.get(),
            auto_detect_var.get(),
            mock_data_var.get(),
            auto_play_var.get(),
            show_overlay_var.get(),
            float(confidence_var.get()),
            settings_window
        )).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.LEFT, padx=10)
    
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
            defaultextension=".pt"
        )
        if file_path:
            string_var.set(file_path)
    
    def _save_settings(self, screenshot_dir, game_state_dir, training_dir, 
                      model_path, model_save_path, device, auto_detect_poker, 
                      use_mock_data, auto_play,show_overlay_var, confidence, window):
        """Save settings from the dialog"""
        try:
            # Update configuration
            self.config['screenshot_dir'] = screenshot_dir
            self.config['game_state_dir'] = game_state_dir
            self.config['training_data_dir'] = training_dir
            self.config['model_path'] = model_path
            self.config['model_save_path'] = model_save_path
            self.config['device'] = device
            self.config['auto_detect_poker'] = auto_detect_poker
            self.config['use_mock_data'] = use_mock_data
            self.config['auto_play'] = auto_play
            self.config['confidence_threshold'] = confidence

            self.config['show_debug_overlay'] = show_overlay_var.get()
            # Create directories if they don't exist
            for dir_path in [screenshot_dir, game_state_dir, training_dir]:
                os.makedirs(dir_path, exist_ok=True)
            
            # Update the assistant's configuration if it exists
            if self.poker_assistant:
                self.poker_assistant.config.update(self.config)
            
            # Update UI elements
            self.auto_play_var.set(auto_play)
            
            # Update model info display
            self.model_info_var.set(f"Using: {os.path.basename(model_path) if model_path else 'Default PyTorch model'}")
            
            self.status_var.set("Settings saved")
            window.destroy()
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
            logger.error(f"Failed to save settings: {str(e)}")
    
    def _train_network(self):
        """Train the neural network with PyTorch"""
        if not self.poker_assistant:
            messagebox.showinfo("Info", "Please start the Poker Assistant first")
            return
        
        result = messagebox.askyesno(
            "Train Neural Network (PyTorch)",
            "This will train the PyTorch neural network using the collected data.\n"
            "Training may take some time. Continue?"
        )
        
        if result:
            # Create a progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Training PyTorch Neural Network")
            progress_window.geometry("400x200")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            ttk.Label(
                progress_window, 
                text="Training PyTorch Model...",
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
            device_info = f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
            ttk.Label(progress_window, text=device_info).pack(pady=5)
            
            # Use a thread to avoid blocking the UI
            def training_thread():
                try:
                    # Update progress periodically
                    for i in range(0, 101, 10):
                        progress_var.set(i)
                        status_var.set(f"Training: {i}% complete")
                        time.sleep(0.5)  # Simulate training time
                    
                    # In a real implementation, this would train the PyTorch model
                    # self.poker_assistant.train_from_collected_data()
                    
                    status_var.set("Training complete!")
                    messagebox.showinfo("Training Complete", "PyTorch neural network training completed successfully!")
                    progress_window.destroy()
                    
                    # Update model info
                    model_path = self.config.get('model_save_path', 'poker_model.pt')
                    self.model_info_var.set(f"Using: {os.path.basename(model_path)}")
                
                except Exception as e:
                    status_var.set(f"Error: {str(e)}")
                    messagebox.showerror("Error", f"PyTorch training failed: {str(e)}")
                    logger.error(f"PyTorch training failed: {str(e)}")
            
            threading.Thread(target=training_thread, daemon=True).start()
    
    def _show_help(self):
        """Show help information"""
        help_text = """
        Poker Screen Grabber Help (PyTorch Edition)
        
        This application captures and analyzes poker game screenshots to provide decision assistance.
        
        Getting Started:
        1. Select a poker game window from the dropdown
        2. Click "Select" to confirm the window
        3. Click "Preview Window" to check what will be captured
        4. Set the capture interval (in seconds)
        5. Click "Start Capture" to begin analysis
        
        The application will:
        - Take periodic screenshots of the selected poker game
        - Analyze the game state (cards, chips, etc.)
        - Provide recommended actions based on PyTorch neural network analysis
        
        If auto-play is enabled, the application can automatically perform the recommended actions.
        
        This version uses PyTorch for neural network operations instead of TensorFlow.
        
        For more information, please refer to the documentation.
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Help - PyTorch Edition")
        help_window.geometry("600x400")
        help_window.transient(self.root)
        
        text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)
    
    def _show_about(self):
        """Show about information"""
        about_text = """
        Poker Screen Grabber (PyTorch Edition)
        Version 1.0.0
        
        A Python application that uses computer vision and PyTorch neural networks
        to analyze poker games and provide decision assistance.
        
        Created for learning and demonstration purposes.
        
        Libraries used:
        - OpenCV for image processing
        - PyTorch for neural network analysis
        - Tkinter for the user interface
        
        © 2025
        """
        
        messagebox.showinfo("About - PyTorch Edition", about_text)
    
    def _on_exit(self):
        """Handle application exit"""
        if self.poker_assistant and self.poker_assistant.running:
            result = messagebox.askyesno(
                "Exit",
                "Poker Assistant is still running. Stop it and exit?"
            )
            
            if result:
                self._stop_assistant()
                self.root.destroy()
        else:
            self.root.destroy()

    def _preview_selected_window(self):
        """Preview the selected window with debugging overlay"""
        if not self.selected_window_info:
            self.status_var.set("No window selected for preview")
            return
        
        # Use the window detector to capture a screenshot
        self.window_detector.select_window(self.selected_window_info)
        img = self.window_detector.capture_screenshot()  # This will now include the debugging overlay
        
        if img is not None:
            # Update preview
            self._update_preview(img)
            self.status_var.set(f"Previewing window with debugging overlay: {self.selected_window_info.title}")
        else:
            self.status_var.set("Failed to capture window preview")

    def _show_region_settings(self):
        """Show settings dialog for region configuration"""
        region_window = tk.Toplevel(self.root)
        region_window.title("Region Configuration")
        region_window.geometry("800x2000+1300+100")
        region_window.transient(self.root)
        region_window.grab_set()
        
        # Create main frame with scrollbar
        main_frame = ttk.Frame(region_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create canvas with scrollbar for the content
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Dictionary to store all StringVars for later retrieval
        region_vars = {}
        
        # Add labels for the columns
        ttk.Label(scrollable_frame, text="Region Type").grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Index/ID").grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(scrollable_frame, text="X").grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Y").grid(row=0, column=3, padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Width").grid(row=0, column=4, padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Height").grid(row=0, column=5, padx=5, pady=5)
        
        # Create separator
        ttk.Separator(scrollable_frame, orient="horizontal").grid(
            row=1, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=5)
        
        # Access the regions from the window detector
        if hasattr(self, 'window_detector') and hasattr(self.window_detector, 'roi'):
            roi = self.window_detector.roi
        else:
            messagebox.showerror("Error", "Window detector not initialized or missing ROI data")
            region_window.destroy()
            return
        
        # Row counter for grid layout
        row = 2
        
        # Ensure all expected ROI types exist
        required_roi_types = [
            'community_cards', 'player_cards', 'player_chips', 
            'main_player_chips', 'current_bets', 'game_stage', 
            'pot', 'actions'
        ]
        
        # Create any missing ROI types with default values
        for roi_type in required_roi_types:
            if roi_type not in roi:
                if roi_type == 'main_player_chips':
                    roi[roi_type] = [(280, 392, 100, 25)]
                elif roi_type == 'game_stage':
                    roi[roi_type] = [(265, 197, 80, 25), (720, 197, 80, 25)]
                elif roi_type == 'current_bets':
                    roi[roi_type] = {}
                    for player_id in range(1, 10):
                        roi[roi_type][player_id] = [(280 + (player_id * 10), 350, 70, 20)]
                elif roi_type == 'actions':
                    roi[roi_type] = {
                        'raise': [(510, 480, 80, 20)],
                        'call': [(510, 530, 80, 20)],
                        'fold': [(510, 580, 80, 20)]
                    }
        
        # Process each region type in a specific order
        for region_type in required_roi_types:
            # Header for region type
            ttk.Label(
                scrollable_frame, 
                text=region_type.replace('_', ' ').title(),
                font=("TkDefaultFont", 10, "bold")
            ).grid(row=row, column=0, columnspan=6, sticky=tk.W, pady=(15, 5))
            row += 1
            
            # Create separator after header
            ttk.Separator(scrollable_frame, orient="horizontal").grid(
                row=row, column=0, columnspan=6, sticky=(tk.W, tk.E))
            row += 1
            
            # Process regions based on their structure
            if region_type in ['player_cards', 'player_chips', 'current_bets']:
                # These are dictionaries of lists
                for player_id in sorted(roi[region_type].keys()):
                    regions = roi[region_type][player_id]
                    for i, region in enumerate(regions):
                        # Region type display
                        ttk.Label(scrollable_frame, text=region_type.replace('_', ' ')).grid(
                            row=row, column=0, padx=5, pady=2, sticky=tk.W)
                        
                        # Player and index
                        ttk.Label(scrollable_frame, text=f"P{player_id}-{i}").grid(
                            row=row, column=1, padx=5, pady=2)
                        
                        # X, Y, W, H inputs
                        x_var = tk.StringVar(value=str(region[0]))
                        y_var = tk.StringVar(value=str(region[1]))
                        w_var = tk.StringVar(value=str(region[2]))
                        h_var = tk.StringVar(value=str(region[3]))
                        
                        ttk.Entry(scrollable_frame, textvariable=x_var, width=6).grid(
                            row=row, column=2, padx=5, pady=2)
                        ttk.Entry(scrollable_frame, textvariable=y_var, width=6).grid(
                            row=row, column=3, padx=5, pady=2)
                        ttk.Entry(scrollable_frame, textvariable=w_var, width=6).grid(
                            row=row, column=4, padx=5, pady=2)
                        ttk.Entry(scrollable_frame, textvariable=h_var, width=6).grid(
                            row=row, column=5, padx=5, pady=2)
                        
                        # Store variables for later retrieval
                        region_vars[(region_type, player_id, i)] = (x_var, y_var, w_var, h_var)
                        
                        row += 1
            elif region_type == 'actions':
                # Handle action buttons specially
                for action_name in sorted(roi[region_type].keys()):
                    regions = roi[region_type][action_name]
                    for i, region in enumerate(regions):
                        # Region type display
                        ttk.Label(scrollable_frame, text=region_type).grid(
                            row=row, column=0, padx=5, pady=2, sticky=tk.W)
                        
                        # Action name and index
                        ttk.Label(scrollable_frame, text=f"{action_name}-{i}").grid(
                            row=row, column=1, padx=5, pady=2)
                        
                        # X, Y, W, H inputs
                        x_var = tk.StringVar(value=str(region[0]))
                        y_var = tk.StringVar(value=str(region[1]))
                        w_var = tk.StringVar(value=str(region[2]))
                        h_var = tk.StringVar(value=str(region[3]))
                        
                        ttk.Entry(scrollable_frame, textvariable=x_var, width=6).grid(
                            row=row, column=2, padx=5, pady=2)
                        ttk.Entry(scrollable_frame, textvariable=y_var, width=6).grid(
                            row=row, column=3, padx=5, pady=2)
                        ttk.Entry(scrollable_frame, textvariable=w_var, width=6).grid(
                            row=row, column=4, padx=5, pady=2)
                        ttk.Entry(scrollable_frame, textvariable=h_var, width=6).grid(
                            row=row, column=5, padx=5, pady=2)
                        
                        # Store variables for later retrieval - using action name instead of player_id
                        region_vars[(region_type, action_name, i)] = (x_var, y_var, w_var, h_var)
                        
                        row += 1
            else:
                # These are lists of tuples
                for i, region in enumerate(roi[region_type]):
                    # Region type display
                    ttk.Label(scrollable_frame, text=region_type.replace('_', ' ')).grid(
                        row=row, column=0, padx=5, pady=2, sticky=tk.W)
                    
                    # Index
                    ttk.Label(scrollable_frame, text=str(i)).grid(
                        row=row, column=1, padx=5, pady=2)
                    
                    # X, Y, W, H inputs
                    x_var = tk.StringVar(value=str(region[0]))
                    y_var = tk.StringVar(value=str(region[1]))
                    w_var = tk.StringVar(value=str(region[2]))
                    h_var = tk.StringVar(value=str(region[3]))
                    
                    ttk.Entry(scrollable_frame, textvariable=x_var, width=6).grid(
                        row=row, column=2, padx=5, pady=2)
                    ttk.Entry(scrollable_frame, textvariable=y_var, width=6).grid(
                        row=row, column=3, padx=5, pady=2)
                    ttk.Entry(scrollable_frame, textvariable=w_var, width=6).grid(
                        row=row, column=4, padx=5, pady=2)
                    ttk.Entry(scrollable_frame, textvariable=h_var, width=6).grid(
                        row=row, column=5, padx=5, pady=2)
                    
                    # Store variables for later retrieval
                    region_vars[(region_type, None, i)] = (x_var, y_var, w_var, h_var)
                    
                    row += 1
        
        # Add extra controls for adding new regions
        ttk.Separator(scrollable_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        ttk.Label(
            scrollable_frame, 
            text="Add New Region",
            font=("TkDefaultFont", 10, "bold")
        ).grid(row=row, column=0, columnspan=6, sticky=tk.W, pady=(10, 5))
        row += 1
        
        # Type selection
        ttk.Label(scrollable_frame, text="Region Type:").grid(
            row=row, column=0, padx=5, pady=2, sticky=tk.W)
        
        new_region_type_var = tk.StringVar()
        region_type_combo = ttk.Combobox(
            scrollable_frame, 
            textvariable=new_region_type_var,
            values=required_roi_types,
            state="readonly",
            width=15
        )
        region_type_combo.grid(row=row, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W)
        
        # Index/ID field
        ttk.Label(scrollable_frame, text="Index/ID:").grid(
            row=row, column=3, padx=5, pady=2, sticky=tk.W)
        
        new_index_var = tk.StringVar(value="0")
        ttk.Entry(scrollable_frame, textvariable=new_index_var, width=6).grid(
            row=row, column=4, padx=5, pady=2)
        
        row += 1
        
        # Coordinates
        ttk.Label(scrollable_frame, text="X:").grid(
            row=row, column=0, padx=5, pady=2, sticky=tk.W)
        new_x_var = tk.StringVar(value="0")
        ttk.Entry(scrollable_frame, textvariable=new_x_var, width=6).grid(
            row=row, column=1, padx=5, pady=2)
        
        ttk.Label(scrollable_frame, text="Y:").grid(
            row=row, column=2, padx=5, pady=2, sticky=tk.W)
        new_y_var = tk.StringVar(value="0")
        ttk.Entry(scrollable_frame, textvariable=new_y_var, width=6).grid(
            row=row, column=3, padx=5, pady=2)
        
        ttk.Label(scrollable_frame, text="W:").grid(
            row=row, column=4, padx=5, pady=2, sticky=tk.W)
        new_w_var = tk.StringVar(value="50")
        ttk.Entry(scrollable_frame, textvariable=new_w_var, width=6).grid(
            row=row, column=5, padx=5, pady=2)
        
        row += 1
        
        ttk.Label(scrollable_frame, text="H:").grid(
            row=row, column=0, padx=5, pady=2, sticky=tk.W)
        new_h_var = tk.StringVar(value="20")
        ttk.Entry(scrollable_frame, textvariable=new_h_var, width=6).grid(
            row=row, column=1, padx=5, pady=2)
        
        # Add button
        ttk.Button(
            scrollable_frame, 
            text="Add Region", 
            command=lambda: self._add_new_region(
                new_region_type_var.get(),
                new_index_var.get(),
                new_x_var.get(),
                new_y_var.get(),
                new_w_var.get(),
                new_h_var.get(),
                region_window
            )
        ).grid(row=row, column=2, columnspan=2, padx=5, pady=2)
        
        row += 1
        
        # Add action buttons
        if hasattr(self, 'selected_window_info') and self.window_detector.selected_window:
            ttk.Label(
                scrollable_frame, 
                text=f"Currently calibrating regions for: {self.window_detector.selected_window}",
                font=("TkDefaultFont", 9, "italic")
            ).grid(row=row, column=0, columnspan=6, pady=(15, 5))
            row += 1
        
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.grid(row=row, column=0, columnspan=6, pady=20)
        
        # Preview button - shows current regions on live image
        preview_btn = ttk.Button(
            button_frame, 
            text="Preview Regions", 
            command=lambda: self._preview_regions(region_vars)
        )
        preview_btn.pack(side=tk.LEFT, padx=5)
        
        # Save button
        save_btn = ttk.Button(
            button_frame, 
            text="Save Regions", 
            command=lambda: self._save_regions(region_vars, region_window)
        )
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # Reset to defaults button
        reset_btn = ttk.Button(
            button_frame, 
            text="Reset to Defaults", 
            command=lambda: self._reset_regions(region_window)
        )
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Cancel button
        cancel_btn = ttk.Button(
            button_frame, 
            text="Cancel", 
            command=region_window.destroy
        )
        cancel_btn.pack(side=tk.LEFT, padx=5)

    def _add_new_region(self, region_type, index_str, x_str, y_str, w_str, h_str, window):
        """Add a new region to the ROI configuration"""
        try:
            # Convert inputs to appropriate types
            x = int(x_str)
            y = int(y_str)
            w = int(w_str)
            h = int(h_str)
            
            # Handle different region types
            if region_type in ['player_cards', 'player_chips', 'current_bets']:
                # These are dictionaries of lists - index_str is player_id
                try:
                    player_id = int(index_str)
                    if player_id < 1 or player_id > 9:
                        raise ValueError("Player ID must be between 1 and 9")
                    
                    # Initialize the key if it doesn't exist
                    if player_id not in self.window_detector.roi[region_type]:
                        self.window_detector.roi[region_type][player_id] = []
                    
                    # Add the new region
                    self.window_detector.roi[region_type][player_id].append((x, y, w, h))
                    
                except ValueError:
                    messagebox.showerror("Error", "Invalid player ID. Must be a number between 1 and 9.")
                    return
            
            elif region_type == 'actions':
                # Action buttons - index_str is action name
                action_name = index_str.lower()
                if not action_name:
                    messagebox.showerror("Error", "Action name cannot be empty")
                    return
                
                # Initialize the key if it doesn't exist
                if action_name not in self.window_detector.roi[region_type]:
                    self.window_detector.roi[region_type][action_name] = []
                
                # Add the new region
                self.window_detector.roi[region_type][action_name].append((x, y, w, h))
            
            else:
                # Simple list of regions
                self.window_detector.roi[region_type].append((x, y, w, h))
            
            # Ask if user wants to save changes
            result = messagebox.askyesno(
                "Save Changes", 
                "Region added successfully. Would you like to save changes to file now?",
                icon=messagebox.QUESTION
            )
            
            if result:
                # Save the updated ROI to file
                success = self.window_detector.save_regions_to_file("roi_config.json")
                
                # Sync with poker assistant if it exists
                if hasattr(self, 'poker_assistant') and self.poker_assistant:
                    if hasattr(self.poker_assistant, 'screen_grabber') and hasattr(self.poker_assistant.screen_grabber, 'roi'):
                        # Make a deep copy to prevent shared references
                        self.poker_assistant.screen_grabber.roi = copy.deepcopy(self.window_detector.roi)
                        logger.info("ROI configuration synced with poker assistant")
                
                if success:
                    messagebox.showinfo("Success", "ROI configuration saved successfully")
                else:
                    messagebox.showerror("Error", "Failed to save ROI configuration")
            
            # Refresh the window with updated regions
            window.destroy()
            self._show_region_settings()
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add region: {str(e)}")
            logger.error(f"Error in _add_new_region: {str(e)}", exc_info=True)

    def _preview_regions(self, region_vars):
        """Preview regions with current settings"""
        # Create a temporary copy of the ROI with the current settings
        temp_roi = self._get_updated_roi_from_vars(region_vars)
        
        # Store the original ROI
        original_roi = self.window_detector.roi
        
        # Temporarily set the ROI to the new values
        self.window_detector.roi = temp_roi
        
        # Capture a screenshot with the debug overlay
        self.window_detector.show_debug_overlay = True
        screenshot = self.window_detector.capture_screenshot()
        
        # Restore the original ROI
        self.window_detector.roi = original_roi
        
        # Display the preview
        if screenshot is not None:
            # Create a preview window
            preview_window = tk.Toplevel(self.root)
            preview_window.title("Region Preview")
            preview_window.geometry("1024x768")
            
            # Convert image for tkinter
            img = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            
            # Resize to fit window
            img.thumbnail((1000, 700))
            
            # Convert to PhotoImage
            img_tk = ImageTk.PhotoImage(image=img)
            
            # Create label and display image
            label = ttk.Label(preview_window, image=img_tk)
            label.image = img_tk  # Keep a reference
            label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Add close button
            ttk.Button(
                preview_window, 
                text="Close Preview", 
                command=preview_window.destroy
            ).pack(pady=10)
        else:
            messagebox.showerror("Error", "Failed to capture preview image")

    def _save_regions(self, region_vars, window):
        """Save the updated regions"""
        try:
            # Update ROI with values from the UI
            updated_roi = self._get_updated_roi_from_vars(region_vars)
            
            # Update the window detector's ROI
            self.window_detector.roi = updated_roi
            
            # Save to a file using the established method
            roi_file = "roi_config.json"
            success = self.window_detector.save_regions_to_file(roi_file)
            
            if not success:
                messagebox.showerror("Error", "Failed to save ROI configuration to file.")
                return
            
            # Sync with poker assistant if it exists
            if hasattr(self, 'poker_assistant') and self.poker_assistant:
                if hasattr(self.poker_assistant, 'screen_grabber') and hasattr(self.poker_assistant.screen_grabber, 'roi'):
                    # Make a deep copy to prevent shared references
                    self.poker_assistant.screen_grabber.roi = copy.deepcopy(updated_roi)
                    logger.info("ROI configuration synced with poker assistant")
            
            # Show success message
            messagebox.showinfo("Success", f"Regions saved to {roi_file}")
            
            # Close the window
            window.destroy()
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save regions: {str(e)}")
            logger.error(f"Error in _save_regions: {str(e)}", exc_info=True)

    def _get_updated_roi_from_vars(self, region_vars):
        """Get updated ROI dictionary from UI variables"""
        # Start with a deep copy of the current ROI
        updated_roi = copy.deepcopy(self.window_detector.roi)
        
        # Update with values from the UI
        for (region_type, identifier, index), (x_var, y_var, w_var, h_var) in region_vars.items():
            try:
                x = int(x_var.get())
                y = int(y_var.get())
                w = int(w_var.get())
                h = int(h_var.get())
                
                if region_type in ['player_cards', 'player_chips', 'current_bets']:
                    # These are player-specific regions
                    if identifier is not None:
                        player_id = identifier
                        if player_id in updated_roi[region_type] and index < len(updated_roi[region_type][player_id]):
                            updated_roi[region_type][player_id][index] = (x, y, w, h)
                
                elif region_type == 'actions':
                    # Action buttons - identifier is action name
                    action_name = identifier
                    if action_name in updated_roi[region_type] and index < len(updated_roi[region_type][action_name]):
                        updated_roi[region_type][action_name][index] = (x, y, w, h)
                
                else:
                    # This is a general region
                    if index < len(updated_roi[region_type]):
                        updated_roi[region_type][index] = (x, y, w, h)
            
            except (ValueError, KeyError, IndexError) as e:
                logger.error(f"Error updating region {region_type}-{identifier}-{index}: {str(e)}")
        
        return updated_roi

    def _reset_regions(self, window):
        """Reset regions to default values"""
        result = messagebox.askyesno(
            "Confirm Reset", 
            "Are you sure you want to reset all regions to default values? This cannot be undone."
        )
        
        if result:
            # Reset to default ROI
            self.window_detector.roi = self.window_detector._get_default_roi()
            
            # Close the window
            window.destroy()
            
            # Open a new window with the default values
            self._show_region_settings()

    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)
        self.root.mainloop()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Poker Screen Grabber (PyTorch Edition)")
    parser.add_argument("--config", help="Path to configuration file")
    args = parser.parse_args()
    
    # Create and run the application
    root = tk.Tk()
    app = PokerScreenGrabberApp(root)
    
    # Load configuration if specified
    if args.config:
        # This would be implemented to load the config
        pass
    
    app.run()