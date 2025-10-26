"""
Alarm Level Extraction v2 - Multi-Resolution Support
Works with 2560x1440, 2540x1310, and 2000x1125 resolutions
"""

import cv2
import numpy as np
import pytesseract
import re
from typing import Optional

def extract_alarm_level(image: np.ndarray, debug: bool = False) -> Optional[int]:
    """
    Extract alarm level number (0-6) from security clock.
    
    Auto-detects resolution and uses appropriate positioning.
    Supports: 2560x1440, 2540x1310, 2000x1125
    
    Returns:
        int 0-6 if detected, None if extraction fails
    """
    h, w = image.shape[:2]
    
    if debug:
        print(f"Image resolution: {w}x{h}")
    
    # Resolution-specific positioning
    # Format: (x_percent, y_percent, region_size)
    if w >= 2500:  # 2560x1440 or 2540x1310
        # For 2560x1440: alarm level at ~98.3% across, varies 1.7-10% down
        # Use center at ~6% down with large search region
        clock_center_x = int(w * 0.983)
        clock_center_y = int(h * 0.06)
        region_size = 100  # Larger region to catch position variations
    elif w >= 1900:  # 2000x1125 (or uploaded/resized frames)
        # For 2000x1125: alarm level at ~97.7% across, ~9.7% down
        clock_center_x = int(w * 0.973)
        clock_center_y = int(h * 0.097)
        region_size = 50
    else:
        if debug:
            print(f"⚠ Unsupported resolution: {w}x{h}")
        return None
    
    # Extract region around alarm level
    x1 = max(0, clock_center_x - region_size // 2)
    y1 = max(0, clock_center_y - region_size // 2)
    x2 = min(w, clock_center_x + region_size // 2)
    y2 = min(h, clock_center_y + region_size // 2)
    
    region = image[y1:y2, x1:x2]
    
    if debug:
        print(f"  Search region: ({x1}, {y1}) to ({x2}, {y2}), size: {region.shape}")
    
    # Convert to HSV for color-based extraction
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Color masks for different alarm levels
    methods = []
    
    # Yellow (low alarm 0-2)
    mask_yellow = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([45, 255, 255]))
    methods.append(("yellow", mask_yellow))
    
    # Orange (medium alarm 3-4)
    mask_orange = cv2.inRange(hsv, np.array([8, 100, 100]), np.array([25, 255, 255]))
    methods.append(("orange", mask_orange))
    
    # Red (high alarm 5-6)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    )
    methods.append(("red", mask_red))
    
    # Combined
    mask_combined = cv2.bitwise_or(mask_yellow, cv2.bitwise_or(mask_orange, mask_red))
    methods.append(("combined", mask_combined))
    
    # Grayscale fallback
    region_scaled_gray = cv2.resize(region, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(region_scaled_gray, cv2.COLOR_BGR2GRAY)
    _, gray_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    gray_thresh_small = cv2.resize(gray_thresh, (region.shape[1], region.shape[0]), 
                                   interpolation=cv2.INTER_AREA)
    methods.append(("grayscale", gray_thresh_small))
    
    # Try OCR with PSM 6 and 8
    best_result = None
    best_confidence = 0
    
    for method_name, mask in methods:
        # Scale up for OCR
        scale = 8
        mask_scaled = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        for psm in [6, 8]:
            config = f'--psm {psm} -c tessedit_char_whitelist=0123456'
            text = pytesseract.image_to_string(mask_scaled, config=config).strip()
            
            match = re.search(r'[0-6]', text)
            if match:
                digit = int(match.group(0))
                pixels = np.count_nonzero(mask)
                confidence = 1.0 if text == str(digit) else 0.7
                confidence += min(pixels / 1000, 0.3)
                
                if debug and confidence > 0.5:
                    print(f"    {method_name} PSM{psm}: '{text}' (conf: {confidence:.2f})")
                
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
        print("Usage: python alarm_level_extractor_v2.py <image_path>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error: Could not load image {img_path}")
        sys.exit(1)
    
    print(f"Testing alarm level extraction on: {img_path}")
    
    alarm_level = extract_alarm_level(img, debug=True)
    
    if alarm_level is not None:
        print(f"\n✓ ALARM LEVEL: {alarm_level}")
    else:
        print(f"\n✗ Failed to extract alarm level")
