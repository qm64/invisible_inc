#!/usr/bin/env python3
"""
Alarm Level Extractor v5 - Handles dark red level 5/6 displays
Uses adaptive thresholding and multiple threshold values
"""

import cv2
import numpy as np
from typing import Optional
import pytesseract
import re


def try_ocr_on_image(binary: np.ndarray, debug: bool = False) -> Optional[int]:
    """Try multiple OCR strategies on a binary image"""
    
    strategies = [
        r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456',
        r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456',
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456',
    ]
    
    for i, config in enumerate(strategies):
        text = pytesseract.image_to_string(binary, config=config).strip()
        
        if debug:
            print(f"  OCR attempt {i+1}: '{text}'")
        
        # Extract first digit
        match = re.search(r'\d+', text)
        if match:
            level = int(match.group())
            if 0 <= level <= 6:
                return level
    
    return None


def extract_alarm_level(image: np.ndarray, debug: bool = False) -> Optional[int]:
    """
    Extract alarm level with support for dark red level 5/6 displays.
    
    V5 improvements:
    - Multiple threshold values (127, 100, 80) for dark displays
    - Adaptive thresholding as fallback
    - Tries each preprocessing method until success
    
    Args:
        image: BGR image from cv2.imread
        debug: If True, save debug images and print info
    
    Returns:
        Alarm level (0-6) or None if extraction failed
    """
    
    height, width = image.shape[:2]
    
    # Resolution-based positioning
    if width >= 2500:  # 2560x1440 or similar
        center_x = int(width * 0.964)
        center_y = int(height * 0.106)
        roi_size = 60
    else:  # Lower res
        center_x = int(width * 0.964)
        center_y = int(height * 0.106)
        roi_size = 50
    
    # Extract square ROI centered on digit
    half_size = roi_size // 2
    search_x = center_x - half_size
    search_y = center_y - half_size
    
    # Bounds check
    if search_x < 0 or search_y < 0:
        return None
    if search_x + roi_size > width or search_y + roi_size > height:
        return None
    
    # Extract ROI
    roi = image[search_y:search_y+roi_size, search_x:search_x+roi_size]
    
    if roi.size == 0:
        if debug:
            print("❌ Empty ROI")
        return None
    
    if debug:
        print(f"Image size: {width}x{height}")
        print(f"Center point: ({center_x}, {center_y})")
        print(f"Search region: ({search_x}, {search_y}) size {roi_size}x{roi_size}")
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
    
    # Skip very dark ROIs (non-game screens)
    mean_brightness = np.mean(gray)
    if mean_brightness < 5:
        if debug:
            print(f"❌ ROI too dark: {mean_brightness:.1f}")
        return None
    
    # Invert for OCR (light text on dark background)
    inverted = cv2.bitwise_not(gray)
    
    if debug:
        cv2.imwrite('/tmp/alarm_inverted.png', inverted)
    
    # Strategy 1: Try multiple fixed thresholds
    # Lower thresholds help with dark red displays (level 5/6)
    thresholds = [127, 100, 80, 60]
    
    for thresh_val in thresholds:
        _, binary = cv2.threshold(inverted, thresh_val, 255, cv2.THRESH_BINARY)
        
        if debug:
            cv2.imwrite(f'/tmp/alarm_binary_t{thresh_val}.png', binary)
            print(f"Trying threshold {thresh_val}:")
        
        # Check if binary is not all white (which indicates threshold too low)
        white_ratio = np.sum(binary == 255) / binary.size
        if white_ratio > 0.95:
            if debug:
                print(f"  Skipping - {white_ratio*100:.1f}% white")
            continue
        
        # Try OCR
        result = try_ocr_on_image(binary, debug=debug)
        if result is not None:
            if debug:
                print(f"✓ Success with threshold {thresh_val}: level {result}")
            return result
    
    # Strategy 2: Adaptive thresholding
    if debug:
        print("Trying adaptive threshold:")
    
    binary_adaptive = cv2.adaptiveThreshold(
        inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    if debug:
        cv2.imwrite('/tmp/alarm_binary_adaptive.png', binary_adaptive)
    
    result = try_ocr_on_image(binary_adaptive, debug=debug)
    if result is not None:
        if debug:
            print(f"✓ Success with adaptive threshold: level {result}")
        return result
    
    # Strategy 3: Otsu's method
    if debug:
        print("Trying Otsu's method:")
    
    _, binary_otsu = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if debug:
        cv2.imwrite('/tmp/alarm_binary_otsu.png', binary_otsu)
    
    result = try_ocr_on_image(binary_otsu, debug=debug)
    if result is not None:
        if debug:
            print(f"✓ Success with Otsu: level {result}")
        return result
    
    if debug:
        print("❌ All strategies failed")
    
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
        print("Usage: python alarm_level_extractor_v5.py <image_path> [--debug]")
        print("Example: python alarm_level_extractor_v5.py frame_000079.png --debug")
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
