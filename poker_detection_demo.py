import cv2
import numpy as np
import os
import argparse
import json

from improved_poker_cv_analyzer import ImprovedCardDetector, EnhancedTextRecognition, PokerImageAnalyzer
from improved_poker_screen_grabber import PokerScreenGrabber
from auto_roi_calibrator import AutoROICalibrator

def demo_full_analysis(image_path, template_dir="card_templates"):
    """
    Demonstrate full poker image analysis with improved card and text detection
    
    Args:
        image_path: Path to the poker screenshot
        template_dir: Path to the card templates directory
    """
    print(f"\n=== Running full analysis on {image_path} ===\n")
    
    # Load the image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Create the image analyzer with improved card detection
    analyzer = PokerImageAnalyzer(template_dir)
    
    # Analyze the image
    game_state = analyzer.analyze_image(image_path)
    
    if game_state:
        # Print the game state
        print("=== Detected Game State ===")
        print(f"Game Stage: {game_state.get('game_stage', 'unknown')}")
        print(f"Pot Size: ${game_state.get('pot', 0)}")
        
        print("\nCommunity Cards:")
        for i, card in enumerate(game_state.get('community_cards', []), 1):
            print(f"  Card {i}: {card['value']} of {card['suit']}")
        
        print("\nPlayers:")
        for player_id, player_data in game_state.get('players', {}).items():
            print(f"  Player {player_id}:")
            print(f"    Chips: ${player_data.get('chips', 0)}")
            
            if player_data.get('cards'):
                print(f"    Cards: ", end="")
                for card in player_data['cards']:
                    print(f"{card['value']} of {card['suit']}", end=", ")
                print()
            
        # Save the game state to a file
        output_file = os.path.splitext(image_path)[0] + "_game_state.json"
        with open(output_file, 'w') as f:
            json.dump(game_state, f, indent=2)
        
        print(f"\nGame state saved to {output_file}")
    else:
        print("Analysis failed - no game state detected")

def demo_card_detection(image_path, template_dir="card_templates"):
    """
    Demonstrate card detection capabilities
    
    Args:
        image_path: Path to the image containing cards
        template_dir: Path to the card templates directory
    """
    print(f"\n=== Testing card detection on {image_path} ===\n")
    
    # Load the image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Create card detector with template matching
    card_detector = ImprovedCardDetector(template_dir)
    
    # Allow user to select card regions to detect
    print("Select a card region (click and drag) and press ENTER to detect, or ESC to finish")
    
    while True:
        # Show the image
        display_img = img.copy()
        cv2.imshow("Select Card", display_img)
        
        # Get region selection
        r = cv2.selectROI("Select Card", display_img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Card")
        
        if r == (0, 0, 0, 0):
            break
        
        # Extract the selected region
        x, y, w, h = r
        card_img = img[y:y+h, x:x+w]
        
        # Detect the card
        value, suit = card_detector.detect_card(card_img)
        
        # Show results
        if value and suit:
            print(f"Detected Card: {value} of {suit}")
            
            # Draw detection on image
            result_img = img.copy()
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                result_img, 
                f"{value} of {suit}", 
                (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (0, 255, 0), 
                2
            )
            
            # Show detected card
            cv2.imshow("Detected Card", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No card detected in the selected region")
    
    cv2.destroyAllWindows()

def demo_text_recognition(image_path):
    """
    Demonstrate text recognition capabilities
    
    Args:
        image_path: Path to the image containing text
    """
    print(f"\n=== Testing text recognition on {image_path} ===\n")
    
    # Load the image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Create text recognizer
    text_recognizer = EnhancedTextRecognition()
    
    # Allow user to select regions to recognize
    print("Select a text region (click and drag) and press ENTER to recognize, or ESC to finish")
    
    while True:
        # Show the image
        display_img = img.copy()
        cv2.imshow("Select Text Region", display_img)
        
        # Get region selection
        r = cv2.selectROI("Select Text Region", display_img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Text Region")
        
        if r == (0, 0, 0, 0):
            break
        
        # Extract the number using different methods
        x, y, w, h = r
        
        # Try all different color filters
        results = {}
        results['basic'] = text_recognizer._extract_number_basic(img[y:y+h, x:x+w])
        
        for color in text_recognizer.text_colors:
            results[color] = text_recognizer._extract_number_color_filtered(img[y:y+h, x:x+w], color)
        
        results['adaptive'] = text_recognizer._extract_number_adaptive(img[y:y+h, x:x+w])
        results['edge'] = text_recognizer._extract_number_edge_enhanced(img[y:y+h, x:x+w])
        
        # Final combined result
        final_result = text_recognizer.extract_chip_count(img, (x, y, w, h))
        
        # Show results
        print("\nDetection Results:")
        for method, value in results.items():
            print(f"  {method.ljust(10)}: {value}")
        
        print(f"\nFinal result: {final_result}")
        
        # Draw detected text on image
        result_img = img.copy()
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            result_img, 
            f"Detected: {final_result}", 
            (x, y-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (0, 255, 0), 
            2
        )
        
        # Show result
        cv2.imshow("Detected Text", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    cv2.destroyAllWindows()

def demo_roi_calibration(image_path):
    """
    Demonstrate automatic ROI calibration
    
    Args:
        image_path: Path to the poker screenshot
    """
    print(f"\n=== Testing ROI calibration on {image_path} ===\n")
    
    # Load the image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Create ROI calibrator
    calibrator = AutoROICalibrator()
    
    # Calibrate ROI
    roi = calibrator.calibrate_roi(img)
    
    # Show detected regions
    result_img = img.copy()
    
    # Define colors for different region types
    colors = {
        'community_cards': (0, 255, 0),      # Green
        'player_cards': (255, 0, 0),         # Red
        'player_chips': (0, 0, 255),         # Blue
        'main_player_chips': (0, 150, 255),  # Light blue
        'pot': (255, 255, 0),                # Yellow
        'current_bets': (255, 0, 255),       # Magenta
        'game_stage': (0, 255, 255),         # Cyan
        'actions': (200, 200, 200)           # Gray
    }
    
    # Draw regions on image
    for region_type, regions in roi.items():
        color = colors.get(region_type, (255, 255, 255))
        
        # Handle different region types
        if isinstance(regions, list):
            for i, region in enumerate(regions):
                x, y, w, h = region
                cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(result_img, f"{region_type} {i}", (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        elif isinstance(regions, dict):
            for key, key_regions in regions.items():
                for i, region in enumerate(key_regions):
                    x, y, w, h = region
                    cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(result_img, f"{region_type} {key}-{i}", (x, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Save calibrated ROI to file
    output_file = os.path.splitext(image_path)[0] + "_roi.json"
    json_roi = calibrator._convert_to_json_serializable(roi)
    with open(output_file, 'w') as f:
        json.dump(json_roi, f, indent=2)
    
    print(f"Calibrated ROI saved to {output_file}")
    
    # Show result
    cv2.imshow("Calibrated ROI", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo_screen_grabber():
    """Demonstrate the screen grabber functionality"""
    print("\n=== Testing Screen Grabber ===\n")
    
    # Create screen grabber
    grabber = PokerScreenGrabber(output_dir="poker_screenshots")
    
    # Get available windows
    windows = grabber.get_window_list()
    
    print("Available windows:")
    for i, window in enumerate(windows):
        print(f"{i+1}. {window}")
    
    # Select a window
    if windows:
        try:
            print("\nSelect a window number (or 0 for mock screenshot):")
            window_idx = int(input())
            
            if window_idx == 0:
                screenshot = grabber.create_mock_screenshot()
                window_name = "Mock Screenshot"
            elif 1 <= window_idx <= len(windows):
                selected_window = windows[window_idx - 1]
                print(f"Selected window: {selected_window}")
                grabber.select_window(selected_window)
                
                # Capture screenshot
                screenshot = grabber.capture_screenshot()
                window_name = selected_window.title if hasattr(selected_window, 'title') else "Screenshot"
            else:
                print("Invalid window selection, using mock screenshot")
                screenshot = grabber.create_mock_screenshot()
                window_name = "Mock Screenshot"
            
            # Save screenshot
            os.makedirs("demo_output", exist_ok=True)
            output_path = f"demo_output/{window_name.replace(' ', '_')}.png"
            cv2.imwrite(output_path, screenshot)
            print(f"Screenshot saved to {output_path}")
            
            # Show screenshot
            cv2.imshow(window_name, screenshot)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except ValueError:
            print("Invalid input. Using mock screenshot.")
            screenshot = grabber.create_mock_screenshot()
            
            # Save and show screenshot
            os.makedirs("demo_output", exist_ok=True)
            cv2.imwrite("demo_output/mock_screenshot.png", screenshot)
            print("Mock screenshot saved to demo_output/mock_screenshot.png")
            
            cv2.imshow("Mock Screenshot", screenshot)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("No windows available. Using mock screenshot.")
        screenshot = grabber.create_mock_screenshot()
        
        # Save and show screenshot
        os.makedirs("demo_output", exist_ok=True)
        cv2.imwrite("demo_output/mock_screenshot.png", screenshot)
        print("Mock screenshot saved to demo_output/mock_screenshot.png")
        
        cv2.imshow("Mock Screenshot", screenshot)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Poker Detection Demo")
    parser.add_argument("--mode", choices=[
        "full", "card", "text", "roi", "grab", "all"
    ], default="full", help="Demo mode to run")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--templates", type=str, default="card_templates", 
                        help="Path to card templates directory")
    
    args = parser.parse_args()
    
    # Create template directory if it doesn't exist
    if not os.path.exists(args.templates):
        print(f"Template directory {args.templates} not found, creating...")
        from card_template_generator import create_synthetic_card_templates
        create_synthetic_card_templates(args.templates)
    
    # Run the selected demo
    if args.mode == "full" or args.mode == "all":
        if args.image:
            demo_full_analysis(args.image, args.templates)
        else:
            print("Error: --image is required for full analysis demo")
    
    if args.mode == "card" or args.mode == "all":
        if args.image:
            demo_card_detection(args.image, args.templates)
        else:
            print("Error: --image is required for card detection demo")
    
    if args.mode == "text" or args.mode == "all":
        if args.image:
            demo_text_recognition(args.image)
        else:
            print("Error: --image is required for text recognition demo")
    
    if args.mode == "roi" or args.mode == "all":
        if args.image:
            demo_roi_calibration(args.image)
        else:
            print("Error: --image is required for ROI calibration demo")
    
    if args.mode == "grab" or args.mode == "all":
        demo_screen_grabber()

if __name__ == "__main__":
    main()