#!/usr/bin/env python3
"""
Diagnostic script for alarm level extraction v2
Shows detailed debug info for first 10 frames to identify the issue
"""

import cv2
import numpy as np
import os
import sys
from typing import Optional
import pytesseract
import re

def extract_alarm_level_debug(image: np.ndarray) -> dict:
    """Extract alarm level with detailed debug information"""
    
    result = {
        'success': False,
        'alarm_level': None,
        'resolution': None,
        'search_config': None,
        'roi_stats': None,
        'ocr_text': None,
        'error': None
    }
    
    try:
        height, width = image.shape[:2]
        result['resolution'] = f"{width}x{height}"
        
        # Resolution-based search regions
        if width >= 2500:  # 2560x1440 or 2540x1310
            search_x_pct = 0.983
            search_y_pct = 0.06
            search_width = 100
            result['search_config'] = "2560x1440 config"
        else:  # 2000x1125 (resized images)
            search_x_pct = 0.973
            search_y_pct = 0.097
            search_width = 50
            result['search_config'] = "2000x1125 config"
        
        # Calculate search region
        search_x = int(width * search_x_pct)
        search_y = int(height * search_y_pct)
        
        # Extract ROI
        roi = image[search_y:search_y+50, search_x:search_x+search_width]
        
        if roi.size == 0:
            result['error'] = "Empty ROI"
            return result
        
        # ROI statistics
        result['roi_stats'] = {
            'shape': roi.shape,
            'mean': float(np.mean(roi)),
            'min': int(np.min(roi)),
            'max': int(np.max(roi))
        }
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # Invert for OCR (text is light on dark background)
        inverted = cv2.bitwise_not(gray)
        
        # Threshold
        _, binary = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)
        
        # OCR
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(binary, config=custom_config).strip()
        result['ocr_text'] = repr(text)  # Show exact text including whitespace
        
        # Extract number
        match = re.search(r'\d+', text)
        if match:
            level = int(match.group())
            if 0 <= level <= 6:
                result['success'] = True
                result['alarm_level'] = level
            else:
                result['error'] = f"Invalid level: {level}"
        else:
            result['error'] = "No digit found in OCR text"
            
    except Exception as e:
        result['error'] = str(e)
    
    return result


def diagnose_frames(session_dir: str, num_frames: int = 10):
    """Diagnose first N frames"""
    
    frames_dir = os.path.join(session_dir, "frames")
    if not os.path.exists(frames_dir):
        print(f"❌ Frames directory not found: {frames_dir}")
        return
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    if len(frame_files) == 0:
        print(f"❌ No PNG frames found in {frames_dir}")
        return
    
    print(f"Found {len(frame_files)} frames")
    print(f"Diagnosing first {min(num_frames, len(frame_files))} frames...\n")
    
    successes = 0
    
    for i, frame_file in enumerate(frame_files[:num_frames]):
        frame_path = os.path.join(frames_dir, frame_file)
        image = cv2.imread(frame_path)
        
        if image is None:
            print(f"❌ Frame {i+1}/{num_frames}: Failed to load {frame_file}")
            continue
        
        result = extract_alarm_level_debug(image)
        
        print(f"Frame {i+1}/{num_frames}: {frame_file}")
        print(f"  Resolution: {result['resolution']}")
        print(f"  Config: {result['search_config']}")
        print(f"  ROI shape: {result['roi_stats']['shape'] if result['roi_stats'] else 'N/A'}")
        if result['roi_stats']:
            print(f"  ROI stats: mean={result['roi_stats']['mean']:.1f}, min={result['roi_stats']['min']}, max={result['roi_stats']['max']}")
        print(f"  OCR text: {result['ocr_text']}")
        print(f"  Success: {result['success']}")
        if result['success']:
            print(f"  ✓ Alarm level: {result['alarm_level']}")
            successes += 1
        else:
            print(f"  ✗ Error: {result['error']}")
        print()
    
    print("=" * 70)
    print(f"Diagnostic summary: {successes}/{num_frames} successful ({successes/num_frames*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_alarm_v2.py <session_directory> [num_frames]")
        print("Example: python diagnose_alarm_v2.py captures/20251022_201216 10")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print("=" * 70)
    print("ALARM LEVEL EXTRACTION V2 - DIAGNOSTICS")
    print("=" * 70)
    print()
    
    diagnose_frames(session_dir, num_frames)
    