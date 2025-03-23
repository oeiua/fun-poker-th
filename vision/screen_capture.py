#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Screen Capture Module

This module provides functionality to capture screenshots of poker game windows.
It supports capturing specific windows by title or the entire screen.
"""

import logging
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union

logger = logging.getLogger("PokerVision.ScreenCapture")

# Import platform-specific modules
try:
    import win32gui
    import win32ui
    import win32con
    import win32api
    from ctypes import windll
    IS_WINDOWS = True
except ImportError:
    IS_WINDOWS = False
    try:
        from PyQt5 import QtWidgets, QtCore, QtGui
        HAS_QT = True
    except ImportError:
        HAS_QT = False
        try:
            import pyscreenshot as ImageGrab
            HAS_PYSCREENSHOT = True
        except ImportError:
            HAS_PYSCREENSHOT = False
            logger.warning("No screen capture method available. Install either PyQt5 or pyscreenshot.")


class WindowInfo:
    """Class to store information about a window."""
    
    def __init__(self, handle: int = 0, title: str = "", rect: Tuple[int, int, int, int] = (0, 0, 0, 0)):
        """
        Initialize WindowInfo object.
        
        Args:
            handle: Window handle (Windows-specific)
            title: Window title
            rect: Window rectangle (left, top, right, bottom)
        """
        self.handle = handle
        self.title = title
        self.rect = rect
        
    @property
    def width(self) -> int:
        """Get window width."""
        return self.rect[2] - self.rect[0]
    
    @property
    def height(self) -> int:
        """Get window height."""
        return self.rect[3] - self.rect[1]
    
    def __str__(self) -> str:
        """String representation of WindowInfo."""
        return f"Window '{self.title}' ({self.width}x{self.height})"


class ScreenCaptureManager:
    """Manager class for screen capture operations."""
    
    def __init__(self):
        """Initialize the ScreenCaptureManager."""
        logger.info("Initializing ScreenCaptureManager")
        self.windows_cache = []
        self.last_capture = None
    
    def get_window_list(self) -> List[WindowInfo]:
        """
        Get a list of all visible windows.
        
        Returns:
            List of WindowInfo objects representing visible windows
        """
        logger.debug("Getting window list")
        windows = []
        
        if IS_WINDOWS:
            def enum_windows_callback(hwnd, result):
                if win32gui.IsWindowVisible(hwnd):
                    window_title = win32gui.GetWindowText(hwnd)
                    if window_title and len(window_title) > 0:
                        rect = win32gui.GetWindowRect(hwnd)
                        if rect[2] - rect[0] > 0 and rect[3] - rect[1] > 0:
                            windows.append(WindowInfo(hwnd, window_title, rect))
                return True
            
            win32gui.EnumWindows(enum_windows_callback, None)
        elif HAS_QT:
            # Use Qt to list windows (limited functionality)
            desktop = QtWidgets.QApplication.desktop()
            for i in range(desktop.screenCount()):
                screen_geometry = desktop.screenGeometry(i)
                windows.append(WindowInfo(
                    0, 
                    f"Screen {i}", 
                    (screen_geometry.left(), screen_geometry.top(), 
                     screen_geometry.right(), screen_geometry.bottom())
                ))
        
        self.windows_cache = windows
        return windows
    
    def find_window_by_title(self, title: str) -> Optional[WindowInfo]:
        """
        Find a window by its title (can be a partial match).
        
        Args:
            title: Window title to search for
            
        Returns:
            WindowInfo if found, None otherwise
        """
        if not self.windows_cache:
            self.get_window_list()
            
        for window in self.windows_cache:
            if title.lower() in window.title.lower():
                return window
        
        return None
    
    def capture_window(self, window_info: Union[WindowInfo, str, None] = None) -> np.ndarray:
        """
        Capture a screenshot of the specified window.
        
        Args:
            window_info: WindowInfo object, window title, or None for full screen
            
        Returns:
            Screenshot as a numpy array (BGR format)
        """
        # If window_info is a string, try to find the window by title
        if isinstance(window_info, str):
            window_info = self.find_window_by_title(window_info)
            if not window_info:
                logger.warning(f"Window with title '{window_info}' not found")
                return self.capture_screen()
        
        # If no window specified, capture the entire screen
        if window_info is None:
            return self.capture_screen()
        
        logger.debug(f"Capturing window: {window_info}")
        
        if IS_WINDOWS:
            try:
                # Get window handle and dimensions
                hwnd = window_info.handle
                left, top, right, bottom = window_info.rect
                width = right - left
                height = bottom - top
                
                # Create device context
                hwnd_dc = win32gui.GetWindowDC(hwnd)
                mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
                save_dc = mfc_dc.CreateCompatibleDC()
                
                # Create bitmap
                save_bitmap = win32ui.CreateBitmap()
                save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
                save_dc.SelectObject(save_bitmap)
                
                # Copy screen to bitmap
                result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)
                
                # If PrintWindow failed, try alternative method
                if not result:
                    logger.warning("PrintWindow failed, trying BitBlt")
                    save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)
                
                # Convert bitmap to numpy array
                signed_ints_array = save_bitmap.GetBitmapBits(True)
                img = np.frombuffer(signed_ints_array, dtype='uint8')
                img.shape = (height, width, 4)
                
                # Clean up resources
                win32gui.DeleteObject(save_bitmap.GetHandle())
                save_dc.DeleteDC()
                mfc_dc.DeleteDC()
                win32gui.ReleaseDC(hwnd, hwnd_dc)
                
                # Convert from BGRA to BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
            except Exception as e:
                logger.error(f"Error capturing window: {e}", exc_info=True)
                return self.capture_screen()
                
        elif HAS_QT:
            # Use Qt to capture screen
            screen = QtWidgets.QApplication.primaryScreen()
            screenshot = screen.grabWindow(0,
                                         window_info.rect[0],
                                         window_info.rect[1],
                                         window_info.width,
                                         window_info.height)
            img = self._qpixmap_to_numpy(screenshot)
            
        else:
            img = self.capture_screen(window_info.rect)
        
        self.last_capture = img
        return img
    
    def capture_screen(self, region: Tuple[int, int, int, int] = None) -> np.ndarray:
        """
        Capture the entire screen or a region.
        
        Args:
            region: Optional region to capture (left, top, right, bottom)
            
        Returns:
            Screenshot as a numpy array (BGR format)
        """
        logger.debug("Capturing full screen")
        
        if IS_WINDOWS:
            # Get screen dimensions
            screen_width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
            screen_height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
            screen_left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
            screen_top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
            
            if region:
                left, top, right, bottom = region
            else:
                left, top, right, bottom = screen_left, screen_top, screen_width + screen_left, screen_height + screen_top
            
            width = right - left
            height = bottom - top
            
            # Create device context
            hdc = win32gui.GetDC(0)
            mfc_dc = win32ui.CreateDCFromHandle(hdc)
            save_dc = mfc_dc.CreateCompatibleDC()
            
            # Create bitmap
            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(save_bitmap)
            
            # Copy screen to bitmap
            save_dc.BitBlt((0, 0), (width, height), mfc_dc, (left, top), win32con.SRCCOPY)
            
            # Convert bitmap to numpy array
            signed_ints_array = save_bitmap.GetBitmapBits(True)
            img = np.frombuffer(signed_ints_array, dtype='uint8')
            img.shape = (height, width, 4)
            
            # Clean up resources
            win32gui.DeleteObject(save_bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(0, hdc)
            
            # Convert from BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
        elif HAS_QT:
            # Use Qt to capture screen
            screen = QtWidgets.QApplication.primaryScreen()
            if region:
                left, top, right, bottom = region
                width = right - left
                height = bottom - top
                screenshot = screen.grabWindow(0, left, top, width, height)
            else:
                screenshot = screen.grabWindow(0)
            img = self._qpixmap_to_numpy(screenshot)
            
        elif HAS_PYSCREENSHOT:
            # Use pyscreenshot
            if region:
                screenshot = ImageGrab.grab(bbox=region)
            else:
                screenshot = ImageGrab.grab()
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        else:
            logger.error("No screen capture method available")
            img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        self.last_capture = img
        return img
    
    def _qpixmap_to_numpy(self, pixmap: 'QtGui.QPixmap') -> np.ndarray:
        """
        Convert a QPixmap to a numpy array.
        
        Args:
            pixmap: QPixmap to convert
            
        Returns:
            Numpy array (BGR format)
        """
        # Convert QPixmap to QImage
        qimage = pixmap.toImage()
        
        # Convert QImage to numpy array
        width = qimage.width()
        height = qimage.height()
        
        # Get the pointer to image data
        ptr = qimage.constBits()
        ptr.setsize(qimage.byteCount())
        
        # Create numpy array from data
        arr = np.array(ptr).reshape(height, width, 4)  # RGBA format
        
        # Convert from RGBA to BGR
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    
    def get_last_capture(self) -> Optional[np.ndarray]:
        """
        Get the last captured screenshot.
        
        Returns:
            Last captured screenshot or None
        """
        return self.last_capture
    
    def save_screenshot(self, filename: str, img: Optional[np.ndarray] = None) -> bool:
        """
        Save a screenshot to a file.
        
        Args:
            filename: Filename to save to
            img: Optional image to save, uses last_capture if None
            
        Returns:
            True if successful, False otherwise
        """
        if img is None:
            img = self.last_capture
            
        if img is None:
            logger.error("No screenshot to save")
            return False
        
        try:
            cv2.imwrite(filename, img)
            logger.info(f"Screenshot saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving screenshot: {e}")
            return False


# Test function
def test_screen_capture():
    """Test the screen capture functionality."""
    import time
    import os
    
    capture_manager = ScreenCaptureManager()
    
    # List available windows
    windows = capture_manager.get_window_list()
    print(f"Found {len(windows)} windows:")
    for i, window in enumerate(windows):
        print(f"{i}: {window}")
    
    # Capture full screen
    print("Capturing full screen...")
    full_screen = capture_manager.capture_screen()
    capture_manager.save_screenshot("full_screen.png")
    
    # Try to find and capture a poker window
    poker_window = capture_manager.find_window_by_title("Poker")
    if poker_window:
        print(f"Found poker window: {poker_window}")
        poker_screenshot = capture_manager.capture_window(poker_window)
        capture_manager.save_screenshot("poker_window.png")
    else:
        print("No poker window found")
    
    print("Screen capture test completed")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Run test
    test_screen_capture()