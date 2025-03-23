#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI Editor Dialog

This module implements a dialog for editing ROI (Region Of Interest) positions
in the poker vision assistant.
"""

import logging
import os
import sys
from typing import List, Dict, Tuple, Optional, Union, Any

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtWidgets import (QDialog, QWidget, QLabel, QPushButton, QComboBox,
                           QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
                           QScrollArea, QSizePolicy, QMessageBox, QFrame,
                           QListWidget, QListWidgetItem, QSplitter)

from vision.game_state_extractor import GameStateExtractor

logger = logging.getLogger("PokerVision.UI.ROIEditor")

class ROIEditorDialog(QDialog):
    """Dialog for editing ROI positions."""
    
    def __init__(self, parent, screenshot, game_state_extractor):
        """
        Initialize the ROI editor dialog.
        
        Args:
            parent: Parent widget
            screenshot: Screenshot image
            game_state_extractor: Game state extractor with ROI config
        """
        super().__init__(parent)
        
        self.screenshot = screenshot
        self.game_state_extractor = game_state_extractor
        
        # Make a copy of the ROIs to avoid modifying the original
        self.rois = {}
        for roi_name, roi_value in game_state_extractor.rois.items():
            if isinstance(roi_value, list):
                # For lists, make a deep copy
                if roi_name == "player_positions":
                    # Special handling for player positions
                    player_positions = []
                    for pos in roi_value:
                        if isinstance(pos, (list, tuple)) and len(pos) == 4:
                            player_positions.append(list(pos))
                        else:
                            # Add a default position if invalid
                            player_positions.append([100, 100, 100, 100])
                    self.rois[roi_name] = player_positions
                else:
                    # Regular ROI
                    if len(roi_value) == 4:
                        self.rois[roi_name] = list(roi_value)
                    else:
                        # Add a default ROI if invalid
                        self.rois[roi_name] = [100, 100, 100, 100]
            else:
                # For non-lists, just copy
                self.rois[roi_name] = roi_value
        
        # ROI drawing state
        self.current_roi_name = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        
        # Set up the UI
        self.setup_ui()
        
        # Initialize ROI list
        self.refresh_roi_list()
        
        # Set dialog properties
        self.setWindowTitle("ROI Editor")
        self.resize(1000, 700)
        
        logger.info("ROI editor dialog initialized")
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main layout
        self.main_layout = QVBoxLayout(self)
        
        # Splitter for left panel and image view
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Control panel
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        
        # ROI list group
        self.roi_list_group = QGroupBox("ROI List")
        self.roi_list_layout = QVBoxLayout()
        self.roi_list_group.setLayout(self.roi_list_layout)
        
        # ROI list widget
        self.roi_list = QListWidget()
        self.roi_list.itemClicked.connect(self.roi_selected)
        self.roi_list_layout.addWidget(self.roi_list)
        
        # ROI controls
        self.roi_controls_layout = QHBoxLayout()
        
        # Add ROI button
        self.add_roi_button = QPushButton("Add")
        self.add_roi_button.clicked.connect(self.add_roi)
        self.roi_controls_layout.addWidget(self.add_roi_button)
        
        # Remove ROI button
        self.remove_roi_button = QPushButton("Remove")
        self.remove_roi_button.clicked.connect(self.remove_roi)
        self.roi_controls_layout.addWidget(self.remove_roi_button)
        
        # Add ROI controls to layout
        self.roi_list_layout.addLayout(self.roi_controls_layout)
        
        # Add ROI list group to control panel
        self.control_layout.addWidget(self.roi_list_group)
        
        # ROI details group
        self.roi_details_group = QGroupBox("ROI Details")
        self.roi_details_layout = QVBoxLayout()
        self.roi_details_group.setLayout(self.roi_details_layout)
        
        # ROI name
        self.roi_name_layout = QHBoxLayout()
        self.roi_name_label = QLabel("Name:")
        self.roi_name_value = QLabel("")
        self.roi_name_layout.addWidget(self.roi_name_label)
        self.roi_name_layout.addWidget(self.roi_name_value)
        self.roi_details_layout.addLayout(self.roi_name_layout)
        
        # ROI coordinates
        self.roi_coords_layout = QHBoxLayout()
        self.roi_coords_label = QLabel("Coordinates:")
        self.roi_coords_value = QLabel("")
        self.roi_coords_layout.addWidget(self.roi_coords_label)
        self.roi_coords_layout.addWidget(self.roi_coords_value)
        self.roi_details_layout.addLayout(self.roi_coords_layout)
        
        # ROI size
        self.roi_size_layout = QHBoxLayout()
        self.roi_size_label = QLabel("Size:")
        self.roi_size_value = QLabel("")
        self.roi_size_layout.addWidget(self.roi_size_label)
        self.roi_size_layout.addWidget(self.roi_size_value)
        self.roi_details_layout.addLayout(self.roi_size_layout)
        
        # Add ROI details group to control panel
        self.control_layout.addWidget(self.roi_details_group)
        
        # Drawing instructions
        self.instructions_group = QGroupBox("Instructions")
        self.instructions_layout = QVBoxLayout()
        self.instructions_group.setLayout(self.instructions_layout)
        
        self.instructions_label = QLabel(
            "1. Select an ROI from the list or add a new one\n"
            "2. Click and drag on the image to draw the ROI\n"
            "3. Release to set the ROI position\n"
            "4. Click Save when finished"
        )
        self.instructions_layout.addWidget(self.instructions_label)
        
        # Add instructions group to control panel
        self.control_layout.addWidget(self.instructions_group)
        
        # Add a stretch to push everything up
        self.control_layout.addStretch()
        
        # Add control panel to splitter
        self.splitter.addWidget(self.control_panel)
        
        # Image view panel
        self.image_view_panel = QWidget()
        self.image_view_layout = QVBoxLayout(self.image_view_panel)
        
        # Image view scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        # Custom label for image display and ROI drawing
        self.image_label = ROIDrawingLabel(self)
        self.image_label.roi_drawn.connect(self.roi_drawn)
        
        # Add image label to scroll area
        self.scroll_area.setWidget(self.image_label)
        
        # Add scroll area to image view layout
        self.image_view_layout.addWidget(self.scroll_area)
        
        # Add image view panel to splitter
        self.splitter.addWidget(self.image_view_panel)
        
        # Add splitter to main layout
        self.main_layout.addWidget(self.splitter)
        
        # Set splitter sizes
        self.splitter.setSizes([300, 700])
        
        # Buttons layout
        self.buttons_layout = QHBoxLayout()
        
        # Save button
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_rois)
        self.buttons_layout.addWidget(self.save_button)
        
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.buttons_layout.addWidget(self.cancel_button)
        
        # Add buttons layout to main layout
        self.main_layout.addLayout(self.buttons_layout)
        
        # Update the display
        self.update_screenshot_display()
    
    def refresh_roi_list(self):
        """Refresh the ROI list."""
        # Clear the list
        self.roi_list.clear()
        
        # Add ROIs to the list
        for roi_name in sorted(self.rois.keys()):
            item = QListWidgetItem(roi_name)
            self.roi_list.addItem(item)
    
    def update_screenshot_display(self):
        """Update the screenshot display."""
        if self.screenshot is not None:
            # Create a copy of the screenshot
            display_img = self.screenshot.copy()
            
            # Draw all ROIs
            for roi_name, roi in self.rois.items():
                if isinstance(roi, list):
                    if len(roi) == 4 and all(isinstance(val, (int, float)) for val in roi):  # Standard ROI format: [x, y, width, height]
                        x, y, w, h = roi
                        color = (0, 255, 0)  # Default color: green
                        
                        # Highlight current ROI
                        if roi_name == self.current_roi_name:
                            color = (0, 0, 255)  # Blue for current ROI
                        
                        # Draw rectangle
                        cv2.rectangle(display_img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                        
                        # Draw label
                        cv2.putText(display_img, roi_name, (int(x), int(y) - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    elif roi_name == "player_positions" and len(roi) > 0:
                        # Special case for player positions
                        for i, player_roi in enumerate(roi):
                            if isinstance(player_roi, (list, tuple)) and len(player_roi) == 4:
                                x, y, w, h = player_roi
                                color = (0, 255, 0)  # Green
                                
                                # Draw rectangle
                                cv2.rectangle(display_img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                                
                                # Draw label
                                cv2.putText(display_img, f"Player {i}", (int(x), int(y) - 5), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Convert to QImage and then QPixmap
            height, width, channel = display_img.shape
            bytes_per_line = 3 * width
            q_img = QtGui.QImage(display_img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            q_img = q_img.rgbSwapped()  # Convert BGR to RGB
            
            pixmap = QtGui.QPixmap.fromImage(q_img)
            
            # Set the pixmap
            self.image_label.setPixmap(pixmap)
            self.image_label.setMinimumSize(pixmap.width(), pixmap.height())
            
        else:
            # No screenshot available
            self.image_label.setText("No screenshot available")
            self.image_label.setPixmap(QtGui.QPixmap())
    
    def roi_selected(self, item):
        """
        Handle ROI selection.
        
        Args:
            item: Selected list item
        """
        roi_name = item.text()
        self.current_roi_name = roi_name
        
        # Update ROI details
        self.update_roi_details()
        
        # Update the display
        self.update_screenshot_display()
        
        # Enable drawing
        self.image_label.set_drawing_enabled(True)
    
    def update_roi_details(self):
        """Update the ROI details display."""
        if self.current_roi_name and self.current_roi_name in self.rois:
            # Set ROI name
            self.roi_name_value.setText(self.current_roi_name)
            
            # Get ROI coordinates
            roi = self.rois[self.current_roi_name]
            
            if isinstance(roi, list):
                if len(roi) == 4:  # Standard ROI format: [x, y, width, height]
                    x, y, w, h = roi
                    
                    # Set coordinates
                    self.roi_coords_value.setText(f"({x}, {y})")
                    
                    # Set size
                    self.roi_size_value.setText(f"{w} x {h}")
                
                elif self.current_roi_name == "player_positions" and len(roi) > 0:
                    # Special case for player positions
                    self.roi_coords_value.setText("Multiple")
                    self.roi_size_value.setText("Multiple")
            
        else:
            # No ROI selected
            self.roi_name_value.setText("")
            self.roi_coords_value.setText("")
            self.roi_size_value.setText("")
    
    def add_roi(self):
        """Add a new ROI."""
        # Prompt for ROI name
        roi_name, ok = QtWidgets.QInputDialog.getText(self, "Add ROI", "ROI Name:")
        
        if ok and roi_name:
            # Check if ROI already exists
            if roi_name in self.rois:
                QMessageBox.warning(self, "Add ROI", f"ROI '{roi_name}' already exists.")
                return
            
            # Add ROI with default coordinates
            self.rois[roi_name] = [10, 10, 100, 100]
            
            # Refresh ROI list
            self.refresh_roi_list()
            
            # Select the new ROI
            items = self.roi_list.findItems(roi_name, Qt.MatchExactly)
            if items:
                self.roi_list.setCurrentItem(items[0])
                self.roi_selected(items[0])
    
    def remove_roi(self):
        """Remove the selected ROI."""
        if not self.current_roi_name:
            QMessageBox.warning(self, "Remove ROI", "No ROI selected.")
            return
        
        # Confirm removal
        reply = QMessageBox.question(
            self, "Remove ROI", 
            f"Are you sure you want to remove ROI '{self.current_roi_name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Remove ROI
            if self.current_roi_name in self.rois:
                del self.rois[self.current_roi_name]
            
            # Clear current ROI
            self.current_roi_name = None
            
            # Refresh ROI list
            self.refresh_roi_list()
            
            # Update ROI details
            self.update_roi_details()
            
            # Update the display
            self.update_screenshot_display()
    
    def roi_drawn(self, rect):
        """
        Handle ROI drawing.
        
        Args:
            rect: Drawn rectangle
        """
        if self.current_roi_name:
            # Check if this is a player position ROI
            if self.current_roi_name == "player_positions":
                # Get the current player positions
                player_positions = self.rois.get("player_positions", [])
                
                # Show a dialog to select the player
                player_index, ok = QtWidgets.QInputDialog.getInt(
                    self, "Select Player", "Player Index (0-8):", 0, 0, 8, 1
                )
                
                if ok:
                    # Ensure the list is long enough
                    while len(player_positions) <= player_index:
                        player_positions.append([100, 100, 100, 100])
                    
                    # Update the player position
                    player_positions[player_index] = [rect.x(), rect.y(), rect.width(), rect.height()]
                    self.rois["player_positions"] = player_positions
            else:
                # Update regular ROI coordinates
                self.rois[self.current_roi_name] = [rect.x(), rect.y(), rect.width(), rect.height()]
            
            # Update ROI details
            self.update_roi_details()
            
            # Update the display
            self.update_screenshot_display()
    
    def save_rois(self):
        """Save the ROIs and close the dialog."""
        try:
            # Validate ROIs before saving
            all_valid = True
            error_message = ""
            
            for roi_name, roi_value in self.rois.items():
                if roi_name == "player_positions":
                    # Check player positions
                    if not isinstance(roi_value, list):
                        all_valid = False
                        error_message = f"Player positions must be a list, got {type(roi_value)}"
                        break
                    
                    for i, pos in enumerate(roi_value):
                        if not isinstance(pos, (list, tuple)) or len(pos) != 4:
                            all_valid = False
                            error_message = f"Invalid player position at index {i}: {pos}"
                            break
                        
                        for j, val in enumerate(pos):
                            if not isinstance(val, (int, float)):
                                all_valid = False
                                error_message = f"Invalid value in player position {i}: {val} (index {j})"
                                break
                elif isinstance(roi_value, (list, tuple)):
                    # Check standard ROIs
                    if len(roi_value) != 4:
                        all_valid = False
                        error_message = f"ROI {roi_name} must have 4 values, got {len(roi_value)}"
                        break
                    
                    for i, val in enumerate(roi_value):
                        if not isinstance(val, (int, float)):
                            all_valid = False
                            error_message = f"Invalid value in ROI {roi_name}: {val} (index {i})"
                            break
            
            if not all_valid:
                # Show error message
                QtWidgets.QMessageBox.critical(self, "Error", f"Invalid ROI configuration: {error_message}")
                return
            
            # Update game state extractor ROIs
            for roi_name, roi_value in self.rois.items():
                self.game_state_extractor.update_roi_config(roi_name, roi_value)
            
            # Accept the dialog
            self.accept()
            
        except Exception as e:
            logger.error(f"Error saving ROIs: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save ROIs: {str(e)}")


class ROIDrawingLabel(QLabel):
    """
    Custom label for drawing ROIs.
    
    Signals:
        roi_drawn(QRect): Signal emitted when an ROI is drawn
    """
    
    roi_drawn = QtCore.pyqtSignal(QRect)
    
    def __init__(self, parent=None):
        """
        Initialize the ROI drawing label.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.drawing_enabled = False
        self.drawing = False
        self.start_point = None
        self.end_point = None
        
        # Enable mouse tracking
        self.setMouseTracking(True)
    
    def set_drawing_enabled(self, enabled):
        """
        Set whether drawing is enabled.
        
        Args:
            enabled: Whether drawing is enabled
        """
        self.drawing_enabled = enabled
    
    def mousePressEvent(self, event):
        """
        Handle mouse press events.
        
        Args:
            event: Mouse event
        """
        if not self.drawing_enabled:
            return
        
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.update()
    
    def mouseMoveEvent(self, event):
        """
        Handle mouse move events.
        
        Args:
            event: Mouse event
        """
        if not self.drawing:
            return
        
        self.end_point = event.pos()
        self.update()
    
    def mouseReleaseEvent(self, event):
        """
        Handle mouse release events.
        
        Args:
            event: Mouse event
        """
        if not self.drawing:
            return
        
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.end_point = event.pos()
            
            # Create the rectangle
            rect = QRect(self.start_point, self.end_point).normalized()
            
            # Emit the signal
            self.roi_drawn.emit(rect)
    
    def paintEvent(self, event):
        """
        Handle paint events.
        
        Args:
            event: Paint event
        """
        # Call the parent's paint event
        super().paintEvent(event)
        
        # If we're drawing, draw the rectangle
        if self.drawing and self.start_point and self.end_point:
            painter = QtGui.QPainter(self)
            
            # Set pen
            pen = QtGui.QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)
            
            # Draw the rectangle
            rect = QRect(self.start_point, self.end_point).normalized()
            painter.drawRect(rect)


def main():
    """Run the ROI editor dialog standalone (for testing)."""
    import sys
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create application
    app = QtWidgets.QApplication(sys.argv)
    
    # Create a mock game state extractor
    class MockExtractor:
        def __init__(self):
            self.rois = {
                "community_cards": [300, 150, 400, 100],
                "player_cards": [450, 400, 100, 80],
                "pot": [400, 100, 100, 30],
                "player_positions": [
                    [100, 400, 150, 100],
                    [50, 300, 150, 100],
                    [300, 100, 150, 100],
                    [500, 100, 150, 100]
                ]
            }
        
        def update_roi_config(self, roi_name, roi_value):
            self.rois[roi_name] = roi_value
            print(f"Updated ROI: {roi_name} = {roi_value}")
    
    # Create a test image
    test_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw some elements
    cv2.rectangle(test_image, (300, 150), (700, 250), (0, 0, 0), 2)
    cv2.putText(test_image, "Community Cards", (310, 130), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.rectangle(test_image, (450, 400), (550, 480), (0, 0, 0), 2)
    cv2.putText(test_image, "Player Cards", (460, 380), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.rectangle(test_image, (400, 100), (500, 130), (0, 0, 0), 2)
    cv2.putText(test_image, "Pot", (410, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.rectangle(test_image, (100, 400), (250, 500), (0, 0, 0), 2)
    cv2.putText(test_image, "Player 1", (110, 380), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.rectangle(test_image, (50, 300), (200, 400), (0, 0, 0), 2)
    cv2.putText(test_image, "Player 2", (60, 280), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.rectangle(test_image, (300, 100), (450, 200), (0, 0, 0), 2)
    cv2.putText(test_image, "Player 3", (310, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.rectangle(test_image, (500, 100), (650, 200), (0, 0, 0), 2)
    cv2.putText(test_image, "Player 4", (510, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Create and show the dialog
    extractor = MockExtractor()
    dialog = ROIEditorDialog(None, test_image, extractor)
    dialog.exec_()
    
    # Print the results
    print("Final ROIs:")
    for roi_name, roi in extractor.rois.items():
        print(f"  {roi_name}: {roi}")


if __name__ == "__main__":
    main()