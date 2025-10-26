"""
Alarm Level Extraction - Viewport-Relative Approach
Extracts the 0-6 alarm level digit from the center of the security clock
"""

import cv2
import numpy as np
import pytesseract
import re
from typing import Optional

def extract_alarm_level_viewport_relative(image: np.ndarray, viewport: tuple, debug: bool = False) -> Optional[int]:
    """
    Extract alarm level number (0-6) from security clock using image-relative positioning.
    
    The alarm level is always at a fixed position in the upper-right corner:
    - Approximately 2.3-3.5% from the right edge of the image
    - Approximately 9.7% from the top edge of the image
    
    This works with or without viewport information.
    
    Args:
        image: BGR image from OpenCV
        viewport: (x, y, w, h) tuple defining the game viewport (optional, can be None)
        debug: Enable debug output
        
    Returns:
        int 0-6 if detected, None if extraction fails
    """
    h, w = image.shape[:2]
    
    # Use image-relative positioning (works across resolutions)
    # Based on analysis of 2000x1125 frames: ~46-71px from right, ~109px from top
    # As percentages: 97.3% across, 9.7% down
    clock_center_x = int(w * 0.973)
    clock_center_y = int(h * 0.097)
    
    # Extract a region around the alarm level number
    # Use larger region to ensure we capture the full digit
    region_size = 50
    
    x1 = max(0, clock_center_x - region_size // 2)
    y1 = max(0, clock_center_y - region_size // 2)
    x2 = min(image.shape[1], clock_center_x + region_size // 2)
    y2 = min(image.shape[0], clock_center_y + region_size // 2)
    
    region = image[y1:y2, x1:x2]
    
    if debug:
        print(f"  Alarm level search region: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  Region size: {region.shape}")
    
    # Convert to HSV for color-based extraction
    # The alarm level is always yellow/orange/red depending on severity
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Try multiple color ranges to catch yellow, orange, and red
    methods = []
    
    # Method 1: Yellow (low alarm levels 0-2)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([45, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    methods.append(("yellow", mask_yellow))
    
    # Method 2: Orange (medium alarm levels 3-4)  
    lower_orange = np.array([8, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    methods.append(("orange", mask_orange))
    
    # Method 3: Red (high alarm levels 5-6)
    # Red wraps around HSV, need two ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    methods.append(("red", mask_red))
    
    # Method 4: Combined mask (any colored digit)
    mask_combined = cv2.bitwise_or(mask_yellow, cv2.bitwise_or(mask_orange, mask_red))
    methods.append(("combined", mask_combined))
    
    # Method 5: Grayscale threshold (fallback - works when color masks fail)
    # Scale the region first, then threshold
    region_scaled_gray = cv2.resize(region, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(region_scaled_gray, cv2.COLOR_BGR2GRAY)
    _, gray_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # Downsample back to original size for consistent processing
    gray_thresh_small = cv2.resize(gray_thresh, (region.shape[1], region.shape[0]), 
                                   interpolation=cv2.INTER_AREA)
    methods.append(("grayscale", gray_thresh_small))
    
    # Try OCR with each color extraction method
    # Try both PSM 6 (uniform block) and PSM 8 (single word)
    # PSM 8 works better for simple digits like 0,6 but PSM 6 works better for 4
    
    best_result = None
    best_confidence = 0
    
    for method_name, mask in methods:
        # Scale up the mask for better OCR
        scale = 8
        mask_scaled = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Try both PSM modes
        for psm in [6, 8]:
            config = f'--psm {psm} -c tessedit_char_whitelist=0123456'
            text = pytesseract.image_to_string(mask_scaled, config=config).strip()
            
            if debug:
                pixels = np.count_nonzero(mask)
                print(f"    {method_name} PSM{psm}: '{text}' ({pixels} colored pixels)")
            
            # Look for single digit 0-6
            match = re.search(r'[0-6]', text)
            if match:
                digit = int(match.group(0))
                # Confidence based on exact match and number of pixels found
                pixels = np.count_nonzero(mask)
                confidence = 1.0 if text == str(digit) else 0.7
                confidence += min(pixels / 1000, 0.3)  # Bonus for finding more pixels
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = (digit, f"{method_name}_PSM{psm}")
    
    if best_result:
        alarm_level, method = best_result
        if debug:
            print(f"  ✓ Alarm level: {alarm_level} (method: {method})")
        return alarm_level
    
    if debug:
        print(f"  ✗ Could not extract alarm level")
    
    return None


# Test function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python alarm_level_extractor.py <image_path>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error: Could not load image {img_path}")
        sys.exit(1)
    
    # For testing, use a typical 2560x1440 viewport
    # In production, this comes from the structural detector
    viewport = (23, 12, 2514, 1416)
    
    print(f"Testing alarm level extraction on: {img_path}")
    print(f"Viewport: {viewport}")
    
    alarm_level = extract_alarm_level_viewport_relative(img, viewport, debug=True)
    
    if alarm_level is not None:
        print(f"\n✓ ALARM LEVEL: {alarm_level}")
    else:
        print(f"\n✗ Failed to extract alarm level")
