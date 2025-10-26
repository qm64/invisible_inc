#!/usr/bin/env python3
"""
Agent Action Point (AP) Extractor v3.3
Extracts action points for all visible agents from gameplay screenshots

Version 3 Changes:
- Increased ROI height from 25px to 35px (2x text height for proper OCR padding)
- Adjusted starting Y position to provide 50% padding above text
- Follows OCR best practice: text should have 50% padding above/below

Version 3.1 Changes:
- Lowered brightness threshold from 20 to 10 (ROI is mostly dark background)
- Added filename to output for easier identification

Version 3.2 Changes:
- Fixed starting Y position back to measured value (y=1120)
- v3.0/3.1 incorrectly moved start position above measured location
- Now uses correct measured position with taller ROI for padding

Version 3.3 Changes:
- Changed from relative to absolute pixel measurements
- Eliminates rounding errors (profile_h varies between 229-230 pixels)
- start_y = profile_y - 70 (measured: 1190 - 1120 = 70)
- spacing = 37 (measured: 1157 - 1120 = 37)

Usage:
    python agent_ap_extractor.py <image_path> [--debug]
"""

import sys
import cv2
import numpy as np
import pytesseract
from pathlib import Path

VERSION = "3.3.0"

def detect_profile_box(img):
    """
    Detect if agent profile box is visible in lower-left
    Profile is always present during planning phase
    
    Returns: (has_profile, mean_brightness)
    """
    height, width = img.shape[:2]
    
    # Profile box location (2560x1440 reference)
    profile_x = int(width * 0.0078125)  # 20/2560
    profile_y = int(height * 0.826389)  # 1190/1440
    profile_w = int(width * 0.076172)   # 195/2560
    profile_h = int(height * 0.159722)  # 230/1440
    
    # Sample the profile region
    profile_roi = img[profile_y:profile_y+profile_h, profile_x:profile_x+profile_w]
    
    if profile_roi.size == 0:
        return False, 0
    
    # Convert to grayscale for brightness check
    gray_profile = cv2.cvtColor(profile_roi, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray_profile)
    
    # Profile box is present if region is bright enough
    # (contains character portrait, UI elements)
    has_profile = mean_brightness > 40
    
    return has_profile, mean_brightness

def extract_ap_for_agent(img, profile_box, agent_idx, debug=False):
    """
    Extract AP value for a specific agent
    
    Args:
        img: Full screenshot
        profile_box: (x, y, w, h) of profile box
        agent_idx: Agent index (0 = first agent above profile)
        debug: Print debug info
    
    Returns: (ap_value, ocr_text) where ap_value is int or None
    """
    profile_x, profile_y, profile_w, profile_h = profile_box
    
    # AP box dimensions (relative to profile box)
    # v3: Increased height from 0.109 to 0.152 (25px -> 35px at 2560x1440)
    # This provides 50% padding above and below text for OCR
    ap_width = int(profile_w * 0.333)   # 65/195
    ap_height = int(profile_h * 0.152)  # 35/230 (was 25/230 in v2)
    
    # Horizontal position (43 pixels right of profile left edge)
    ap_x = profile_x + int(profile_w * 0.220)  # 43/195
    
    # Vertical position and spacing: Use absolute measurements
    # v3.3: Changed from relative to absolute to eliminate rounding errors
    # Measured from frame 81: Agent 0 at y=1120, profile at y=1190
    start_y = profile_y - 70  # 1190 - 1120 = 70 pixels above profile
    
    # Spacing between agents (measured: 1157 - 1120 = 37)
    spacing = 37
    
    # Calculate this agent's Y position
    ap_y = start_y + (agent_idx * spacing)
    
    # Extract ROI
    roi = img[ap_y:ap_y+ap_height, ap_x:ap_x+ap_width]
    
    if roi.size == 0:
        return None, ""
    
    # Check if we've gone beyond reasonable bounds
    # (agents shouldn't be more than ~300 pixels above profile)
    if ap_y < profile_y - 300:
        return None, ""
    
    # Calculate mean brightness
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray_roi)
    
    if debug:
        print(f"  Box: ({ap_x},{ap_y}) size {ap_width}×{ap_height}")
        print(f"  ROI mean brightness: {mean_brightness:.1f}")
    
    # Check if ROI is extremely dark (no agent at this position)
    # v3.1: Lowered from 20 to 10 - ROI is mostly dark background even with bright text
    if mean_brightness < 10:
        return None, ""
    
    # Preprocess for OCR
    # Threshold to isolate cyan/yellow text on dark background
    _, binary = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY)
    
    # Upscale 3x for better OCR (small text ~15-17px)
    scale_factor = 3
    binary_large = cv2.resize(binary, None, fx=scale_factor, fy=scale_factor, 
                             interpolation=cv2.INTER_CUBIC)
    
    if debug:
        print(f"  Threshold 127: '{cv2.imencode('.png', binary)[1].tobytes().hex()[:20]}...'")
    
    # OCR with configuration for small text
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789AP'
    text = pytesseract.image_to_string(binary_large, config=custom_config).strip()
    
    # Parse AP value
    ap_value = None
    if text:
        # Extract just the number (ignore "AP" suffix)
        digits = ''.join(c for c in text if c.isdigit())
        if digits:
            try:
                ap_value = int(digits)
            except ValueError:
                pass
    
    if debug:
        if ap_value is not None:
            print(f"  ✓ AP: {ap_value}")
        else:
            print(f"  ⚠ AP found but value unreadable (OCR: '{text}')")
    
    return ap_value, text

def extract_agent_ap(image_path, debug=False):
    """
    Extract action points for all visible agents
    
    Returns:
        {
            'profile_detected': bool,
            'agents': [
                {'index': 0, 'ap': int or None, 'ocr_text': str},
                ...
            ]
        }
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return {'error': f'Could not read image: {image_path}'}
    
    height, width = img.shape[:2]
    
    if debug:
        print(f"File: {Path(image_path).name}")
        print(f"Image size: {width}x{height}")
    
    # Get profile box dimensions
    profile_x = int(width * 0.0078125)
    profile_y = int(height * 0.826389)
    profile_w = int(width * 0.076172)
    profile_h = int(height * 0.159722)
    
    if debug:
        print(f"Profile box: ({profile_x}, {profile_y}, {profile_w}, {profile_h})")
        # Calculate AP search dimensions for reference
        ap_width = int(profile_w * 0.333)
        ap_height = int(profile_h * 0.152)
        ap_x = profile_x + int(profile_w * 0.220)
        print(f"AP search: x={ap_x}, width={ap_width}, height={ap_height}")
        start_y = profile_y - 70  # Absolute measurement
        spacing = 37  # Absolute measurement
        print(f"Searching from y={start_y} with spacing={spacing}")
        print()
    
    # Check for profile
    has_profile, profile_brightness = detect_profile_box(img)
    
    if debug:
        print("Profile detection:")
        print(f"  Mean brightness: {profile_brightness:.1f}")
        if has_profile:
            print("  ✓ Profile detected")
        else:
            print("  ✗ No profile (opponent turn or transition)")
        print()
    
    if not has_profile:
        return {
            'profile_detected': False,
            'agents': []
        }
    
    # Extract AP for each potential agent position (max 8 agents)
    agents = []
    for agent_idx in range(8):
        if debug:
            print(f"Agent {agent_idx}:")
        
        ap_value, ocr_text = extract_ap_for_agent(
            img, 
            (profile_x, profile_y, profile_w, profile_h),
            agent_idx,
            debug
        )
        
        # Stop if we found an empty slot
        if ap_value is None and not ocr_text:
            # Check if we'd overlap with profile box
            start_y = profile_y - 70  # Absolute measurement
            spacing = 37  # Absolute measurement
            ap_y = start_y + (agent_idx * spacing)
            
            if ap_y >= profile_y:
                if debug:
                    print(f"  Agent {agent_idx}: Would overlap profile, stopping")
                break
            
            # If dark/empty and above profile, stop
            if not ocr_text:
                if debug:
                    print(f"  Agent {agent_idx}: Empty slot, stopping")
                break
        
        # Record this agent (even if AP is None but we got some OCR text)
        if ap_value is not None or ocr_text:
            agents.append({
                'index': agent_idx,
                'ap': ap_value,
                'ocr_text': ocr_text
            })
    
    if debug:
        print(f"\n✓ Found {len(agents)} agents")
    
    return {
        'profile_detected': True,
        'agents': agents
    }

def main():
    if len(sys.argv) < 2:
        print(f"Agent AP Extractor v{VERSION}")
        print("Usage: python agent_ap_extractor.py <image_path> [--debug]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    debug = '--debug' in sys.argv
    
    result = extract_agent_ap(image_path, debug=debug)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    # Print summary
    print()
    print("=" * 60)
    print("AGENT AP EXTRACTION RESULT")
    print("=" * 60)
    print(f"File: {Path(image_path).name}")
    print(f"Profile detected: {'Yes' if result['profile_detected'] else 'No'}")
    print(f"Agents found: {len(result['agents'])}")
    print()
    
    for agent in result['agents']:
        ap_str = f"{agent['ap']} AP" if agent['ap'] is not None else "None AP"
        print(f"Agent {agent['index']}: {ap_str} (text: '{agent['ocr_text']}')")

if __name__ == "__main__":
    main()