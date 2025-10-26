#!/usr/bin/env python3
"""
Resources Extractor v1 - Extract Power and Credits
Based on measurements from frame_000079.png at 2560x1440
"""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple
import pytesseract
import re


def extract_resources(image: np.ndarray, debug: bool = False) -> Dict[str, Optional[any]]:
    """
    Extract power and credits from top-left corner of game screen.
    
    Measurements from frame_000079.png (2560x1440):
    - Power "3/20 PWR": (10,13) to (123,45), size 113×32
    - Credits "650 CR": (136,15) to (245,47), size 109×32
    
    Args:
        image: BGR image from cv2.imread
        debug: If True, save debug images and print info
    
    Returns:
        Dictionary with:
        {
            'power_current': int or None,
            'power_max': int or None,
            'credits': int or None,
            'power_text': str or None,
            'credits_text': str or None
        }
    """
    
    height, width = image.shape[:2]
    
    result = {
        'power_current': None,
        'power_max': None,
        'credits': None,
        'power_text': None,
        'credits_text': None
    }
    
    if debug:
        print(f"Image size: {width}x{height}")
    
    # Resolution-based positioning
    if width >= 2500:  # 2560x1440 or similar
        # Use measured coordinates with small margin
        power_box = (8, 11, 120, 37)  # x, y, width, height (slightly expanded)
        credits_box = (134, 13, 115, 37)
    else:  # Scale for other resolutions
        scale = width / 2560.0
        power_box = (
            int(8 * scale),
            int(11 * scale),
            int(120 * scale),
            int(37 * scale)
        )
        credits_box = (
            int(134 * scale),
            int(13 * scale),
            int(115 * scale),
            int(37 * scale)
        )
    
    if debug:
        print(f"Power box: {power_box}")
        print(f"Credits box: {credits_box}")
    
    # Extract Power
    power_result = _extract_text_from_box(image, power_box, "power", debug)
    if power_result:
        result['power_text'] = power_result
        # Parse "3/20 PWR" format
        match = re.search(r'(\d+)\s*/\s*(\d+)', power_result)
        if match:
            result['power_current'] = int(match.group(1))
            result['power_max'] = int(match.group(2))
            if debug:
                print(f"✓ Power: {result['power_current']}/{result['power_max']}")
        else:
            if debug:
                print(f"✗ Could not parse power text: '{power_result}'")
    
    # Extract Credits
    credits_result = _extract_text_from_box(image, credits_box, "credits", debug)
    if credits_result:
        result['credits_text'] = credits_result
        # Parse "650 CR" format
        match = re.search(r'(\d+)', credits_result)
        if match:
            result['credits'] = int(match.group(1))
            if debug:
                print(f"✓ Credits: {result['credits']}")
        else:
            if debug:
                print(f"✗ Could not parse credits text: '{credits_result}'")
    
    return result


def _extract_text_from_box(
    image: np.ndarray, 
    box: Tuple[int, int, int, int], 
    name: str, 
    debug: bool
) -> Optional[str]:
    """
    Extract text from a bounding box using OCR.
    
    Args:
        image: Source image
        box: (x, y, width, height)
        name: Name for debug output
        debug: Whether to save debug images
    
    Returns:
        Extracted text or None if failed
    """
    x, y, w, h = box
    
    # Bounds check
    height, width = image.shape[:2]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        if debug:
            print(f"✗ {name}: Box out of bounds")
        return None
    
    # Extract ROI
    roi = image[y:y+h, x:x+w]
    
    if roi.size == 0:
        if debug:
            print(f"✗ {name}: Empty ROI")
        return None
    
    if debug:
        print(f"\n{name.upper()}:")
        print(f"  ROI shape: {roi.shape}")
        print(f"  ROI mean brightness: {np.mean(roi):.1f}")
        cv2.imwrite(f'/tmp/{name}_1_roi.png', roi)
    
    # Convert to grayscale
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    if debug:
        cv2.imwrite(f'/tmp/{name}_2_gray.png', gray)
    
    # Skip very dark ROIs (non-game screens)
    mean_brightness = np.mean(gray)
    if mean_brightness < 5:
        if debug:
            print(f"  ✗ Too dark: {mean_brightness:.1f}")
        return None
    
    # Invert for OCR (white text on dark background)
    inverted = cv2.bitwise_not(gray)
    
    if debug:
        cv2.imwrite(f'/tmp/{name}_3_inverted.png', inverted)
    
    # Try multiple thresholding strategies
    thresholds = [127, 100, 80]
    
    for thresh_val in thresholds:
        _, binary = cv2.threshold(inverted, thresh_val, 255, cv2.THRESH_BINARY)
        
        if debug:
            cv2.imwrite(f'/tmp/{name}_4_binary_t{thresh_val}.png', binary)
        
        # Skip if mostly white
        white_ratio = np.sum(binary == 255) / binary.size
        if white_ratio > 0.95:
            if debug:
                print(f"  Threshold {thresh_val}: skipping ({white_ratio*100:.1f}% white)")
            continue
        
        # OCR with alphanumeric + space + slash
        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/ '
        text = pytesseract.image_to_string(binary, config=config).strip()
        
        if debug:
            print(f"  Threshold {thresh_val}: '{text}'")
        
        if text:
            return text
    
    # Try adaptive thresholding
    if debug:
        print(f"  Trying adaptive threshold")
    
    binary_adaptive = cv2.adaptiveThreshold(
        inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    if debug:
        cv2.imwrite(f'/tmp/{name}_5_binary_adaptive.png', binary_adaptive)
    
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/ '
    text = pytesseract.image_to_string(binary_adaptive, config=config).strip()
    
    if debug:
        print(f"  Adaptive: '{text}'")
    
    if text:
        return text
    
    # Try Otsu's method
    if debug:
        print(f"  Trying Otsu's method")
    
    _, binary_otsu = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if debug:
        cv2.imwrite(f'/tmp/{name}_6_binary_otsu.png', binary_otsu)
    
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/ '
    text = pytesseract.image_to_string(binary_otsu, config=config).strip()
    
    if debug:
        print(f"  Otsu: '{text}'")
    
    return text if text else None


def extract_resources_from_file(image_path: str, debug: bool = False) -> Dict[str, Optional[any]]:
    """
    Load image and extract resources.
    
    Args:
        image_path: Path to PNG screenshot
        debug: If True, save debug images and print info
    
    Returns:
        Dictionary with power and credits data
    """
    image = cv2.imread(image_path)
    if image is None:
        if debug:
            print(f"❌ Failed to load image: {image_path}")
        return {
            'power_current': None,
            'power_max': None,
            'credits': None,
            'power_text': None,
            'credits_text': None
        }
    
    return extract_resources(image, debug=debug)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python resources_extractor.py <image_path> [--debug]")
        print("Example: python resources_extractor.py frame_000079.png --debug")
        sys.exit(1)
    
    image_path = sys.argv[1]
    debug = '--debug' in sys.argv
    
    result = extract_resources_from_file(image_path, debug=debug)
    
    print("\n" + "="*60)
    print("RESOURCES EXTRACTION RESULT")
    print("="*60)
    
    if result['power_current'] is not None:
        print(f"Power: {result['power_current']}/{result['power_max']}")
    else:
        print(f"Power: Failed (text: '{result['power_text']}')")
    
    if result['credits'] is not None:
        print(f"Credits: {result['credits']}")
    else:
        print(f"Credits: Failed (text: '{result['credits_text']}')")
    
    # Exit code: 0 if both succeeded, 1 otherwise
    success = (result['power_current'] is not None and 
               result['credits'] is not None)
    sys.exit(0 if success else 1)
    