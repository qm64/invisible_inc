#!/usr/bin/env python3
"""
Alarm Level Extractor v3 - Corrected 2560x1440 coordinates
Based on actual frame measurements from frame_000079.png
"""

import cv2
import numpy as np
from typing import Optional
import pytesseract
import re


def extract_alarm_level(image: np.ndarray, debug: bool = False) -> Optional[int]:
    """
    Extract alarm level from the security level indicator in upper-right corner.
    
    Coordinates based on frame_000079.png at 2560x1440:
    - Security Level text: (2403,56) to (2573,78)
    - Clock circle: (2395,84) to (2538,222)
    - Alarm digit box: (2443,126) to (2493,180)
    
    Args:
        image: BGR image from cv2.imread
        debug: If True, save debug images
    
    Returns:
        Alarm level (0-6) or None if extraction failed
    """
    
    height, width = image.shape[:2]
    
    # Resolution-based positioning
    if width >= 2500:  # 2560x1440 or similar
        # Use measured coordinates from frame_000079.png
        # Alarm digit is at ~96.4% across, ~10.6% down
        search_x = int(width * 0.954)  # Start a bit earlier for safety
        search_y = int(height * 0.088)  # Start a bit higher
        search_width = 80   # Generous width
        search_height = 100 # Generous height
    else:  # Assume 2000x1125 or other lower res
        # Scale proportionally
        search_x = int(width * 0.954)
        search_y = int(height * 0.088)
        search_width = 60
        search_height = 80
    
    # Extract ROI
    roi = image[search_y:search_y+search_height, search_x:search_x+search_width]
    
    if roi.size == 0:
        if debug:
            print("❌ Empty ROI")
        return None
    
    if debug:
        print(f"Image size: {width}x{height}")
        print(f"Search region: ({search_x}, {search_y}) size {search_width}x{search_height}")
        print(f"ROI shape: {roi.shape}")
        print(f"ROI mean brightness: {np.mean(roi):.1f}")
        cv2.imwrite('/tmp/alarm_roi.png', roi)
    
    # Convert to grayscale
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    if debug:
        cv2.imwrite('/tmp/alarm_gray.png', gray)
    
    # The clock/alarm display has light text on dark background
    # Invert so we have dark text on light background for OCR
    inverted = cv2.bitwise_not(gray)
    
    if debug:
        cv2.imwrite('/tmp/alarm_inverted.png', inverted)
    
    # Apply threshold to get clean binary image
    _, binary = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)
    
    if debug:
        cv2.imwrite('/tmp/alarm_binary.png', binary)
    
    # Run OCR with digit-only whitelist
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(binary, config=custom_config).strip()
    
    if debug:
        print(f"OCR text: '{text}'")
    
    # Extract first digit found
    match = re.search(r'\d+', text)
    if match:
        level = int(match.group())
        # Validate alarm level range
        if 0 <= level <= 6:
            if debug:
                print(f"✓ Extracted alarm level: {level}")
            return level
        else:
            if debug:
                print(f"❌ Invalid alarm level: {level} (must be 0-6)")
            return None
    
    if debug:
        print("❌ No digit found in OCR text")
    
    return None


def extract_alarm_level_from_file(image_path: str, debug: bool = False) -> Optional[int]:
    """
    Load image and extract alarm level.
    
    Args:
        image_path: Path to PNG screenshot
        debug: If True, save debug images and print info
    
    Returns:
        Alarm level (0-6) or None if extraction failed
    """
    image = cv2.imread(image_path)
    if image is None:
        if debug:
            print(f"❌ Failed to load image: {image_path}")
        return None
    
    return extract_alarm_level(image, debug=debug)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python alarm_level_extractor_v3.py <image_path> [--debug]")
        print("Example: python alarm_level_extractor_v3.py frame_000079.png --debug")
        sys.exit(1)
    
    image_path = sys.argv[1]
    debug = '--debug' in sys.argv
    
    level = extract_alarm_level_from_file(image_path, debug=debug)
    
    if level is not None:
        print(f"Alarm level: {level}")
        sys.exit(0)
    else:
        print("Failed to extract alarm level")
        sys.exit(1)
