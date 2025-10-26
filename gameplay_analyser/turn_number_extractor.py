#!/usr/bin/env python3
"""
Turn Number Extractor v1 - Extract turn number from top-right
Based on measurements from frame_000079.png at 2560x1440
"""

import cv2
import numpy as np
from typing import Optional, Dict
import pytesseract
import re


def extract_turn_number(image: np.ndarray, debug: bool = False) -> Dict[str, Optional[any]]:
    """
    Extract turn number from top-right of game screen.
    
    Measurements from frame_000079.png (2560x1440):
    - Turn "TURN 03": (2016,17) to (2083,31), size 67×14
    
    Args:
        image: BGR image from cv2.imread
        debug: If True, save debug images and print info
    
    Returns:
        Dictionary with:
        {
            'turn_number': int or None,
            'turn_text': str or None
        }
    """
    
    height, width = image.shape[:2]
    
    result = {
        'turn_number': None,
        'turn_text': None
    }
    
    if debug:
        print(f"Image size: {width}x{height}")
    
    # Resolution-based positioning
    if width >= 2500:  # 2560x1440 or similar
        # Use measured coordinates with margin (especially vertical for small text)
        turn_box = (2010, 14, 80, 22)  # x, y, width, height (expanded vertically)
    else:  # Scale for other resolutions
        scale = width / 2560.0
        turn_box = (
            int(2010 * scale),
            int(14 * scale),
            int(80 * scale),
            int(22 * scale)
        )
    
    if debug:
        print(f"Turn box: {turn_box}")
    
    # Extract turn number
    turn_text = _extract_turn_text(image, turn_box, debug)
    
    if turn_text:
        result['turn_text'] = turn_text
        # Parse "TURN 03" or "TURN03" or just "03"
        match = re.search(r'(\d+)', turn_text)
        if match:
            result['turn_number'] = int(match.group(1))
            if debug:
                print(f"✓ Turn number: {result['turn_number']}")
        else:
            if debug:
                print(f"✗ Could not parse turn text: '{turn_text}'")
    
    return result


def _extract_turn_text(image: np.ndarray, box: tuple, debug: bool) -> Optional[str]:
    """
    Extract turn text from bounding box.
    Small text needs special handling - upscale before OCR.
    """
    x, y, w, h = box
    
    # Bounds check
    height, width = image.shape[:2]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        if debug:
            print(f"✗ Turn box out of bounds")
        return None
    
    # Extract ROI
    roi = image[y:y+h, x:x+w]
    
    if roi.size == 0:
        if debug:
            print(f"✗ Empty ROI")
        return None
    
    if debug:
        print(f"\nTURN NUMBER:")
        print(f"  ROI shape: {roi.shape}")
        print(f"  ROI mean brightness: {np.mean(roi):.1f}")
        cv2.imwrite(f'/tmp/turn_1_roi.png', roi)
    
    # Convert to grayscale
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    if debug:
        cv2.imwrite(f'/tmp/turn_2_gray.png', gray)
    
    # Skip very dark ROIs
    mean_brightness = np.mean(gray)
    if mean_brightness < 5:
        if debug:
            print(f"  ✗ Too dark: {mean_brightness:.1f}")
        return None
    
    # UPSCALE for better OCR (text is only 14px tall!)
    scale_factor = 3
    gray_scaled = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, 
                            interpolation=cv2.INTER_CUBIC)
    
    if debug:
        print(f"  Upscaled to: {gray_scaled.shape}")
        cv2.imwrite(f'/tmp/turn_3_upscaled.png', gray_scaled)
    
    # Invert for OCR (white text on dark background)
    inverted = cv2.bitwise_not(gray_scaled)
    
    if debug:
        cv2.imwrite(f'/tmp/turn_4_inverted.png', inverted)
    
    # Try multiple thresholding strategies
    thresholds = [127, 100, 80]
    
    for thresh_val in thresholds:
        _, binary = cv2.threshold(inverted, thresh_val, 255, cv2.THRESH_BINARY)
        
        if debug:
            cv2.imwrite(f'/tmp/turn_5_binary_t{thresh_val}.png', binary)
        
        # Skip if mostly white
        white_ratio = np.sum(binary == 255) / binary.size
        if white_ratio > 0.95:
            if debug:
                print(f"  Threshold {thresh_val}: skipping ({white_ratio*100:.1f}% white)")
            continue
        
        # Try multiple PSM modes for small text
        psm_modes = [
            (7, 'single line'),
            (8, 'single word'),
            (6, 'single block')
        ]
        
        for psm, desc in psm_modes:
            config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789TURN '
            text = pytesseract.image_to_string(binary, config=config).strip()
            
            if debug:
                print(f"  Threshold {thresh_val}, PSM {psm} ({desc}): '{text}'")
            
            if text and any(c.isdigit() for c in text):
                return text
    
    # Try adaptive thresholding
    if debug:
        print(f"  Trying adaptive threshold")
    
    binary_adaptive = cv2.adaptiveThreshold(
        inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    if debug:
        cv2.imwrite(f'/tmp/turn_6_binary_adaptive.png', binary_adaptive)
    
    for psm, desc in [(7, 'line'), (8, 'word'), (6, 'block')]:
        config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789TURN '
        text = pytesseract.image_to_string(binary_adaptive, config=config).strip()
        
        if debug:
            print(f"  Adaptive, PSM {psm} ({desc}): '{text}'")
        
        if text and any(c.isdigit() for c in text):
            return text
    
    # Try Otsu's method
    if debug:
        print(f"  Trying Otsu's method")
    
    _, binary_otsu = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if debug:
        cv2.imwrite(f'/tmp/turn_7_binary_otsu.png', binary_otsu)
    
    for psm, desc in [(7, 'line'), (8, 'word'), (6, 'block')]:
        config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789TURN '
        text = pytesseract.image_to_string(binary_otsu, config=config).strip()
        
        if debug:
            print(f"  Otsu, PSM {psm} ({desc}): '{text}'")
        
        if text and any(c.isdigit() for c in text):
            return text
    
    return None


def extract_turn_number_from_file(image_path: str, debug: bool = False) -> Dict[str, Optional[any]]:
    """
    Load image and extract turn number.
    
    Args:
        image_path: Path to PNG screenshot
        debug: If True, save debug images and print info
    
    Returns:
        Dictionary with turn number data
    """
    image = cv2.imread(image_path)
    if image is None:
        if debug:
            print(f"❌ Failed to load image: {image_path}")
        return {
            'turn_number': None,
            'turn_text': None
        }
    
    return extract_turn_number(image, debug=debug)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python turn_number_extractor.py <image_path> [--debug]")
        print("Example: python turn_number_extractor.py frame_000079.png --debug")
        sys.exit(1)
    
    image_path = sys.argv[1]
    debug = '--debug' in sys.argv
    
    result = extract_turn_number_from_file(image_path, debug=debug)
    
    print("\n" + "="*60)
    print("TURN NUMBER EXTRACTION RESULT")
    print("="*60)
    
    if result['turn_number'] is not None:
        print(f"Turn: {result['turn_number']}")
    else:
        print(f"Turn: Failed (text: '{result['turn_text']}')")
    
    # Exit code: 0 if succeeded, 1 otherwise
    success = result['turn_number'] is not None
    sys.exit(0 if success else 1)
    