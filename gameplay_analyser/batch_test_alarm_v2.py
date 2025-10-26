#!/usr/bin/env python3
"""
Batch test alarm level extraction v2 - Multi-Resolution Support
"""

import cv2
import numpy as np
import os
import sys
from collections import Counter
import pytesseract
import re
from typing import Optional

def extract_alarm_level(image: np.ndarray, debug: bool = False) -> Optional[int]:
    """Extract alarm level with multi-resolution support"""
    h, w = image.shape[:2]
    
    if debug:
        print(f"Image resolution: {w}x{h}")
    
    # Resolution-specific positioning
    if w >= 2500:  # 2560x1440 or 2540x1310
        clock_center_x = int(w * 0.983)
        clock_center_y = int(h * 0.06)
        region_size = 100
    elif w >= 1900:  # 2000x1125
        clock_center_x = int(w * 0.973)
        clock_center_y = int(h * 0.097)
        region_size = 50
    else:
        return None
    
    x1 = max(0, clock_center_x - region_size // 2)
    y1 = max(0, clock_center_y - region_size // 2)
    x2 = min(w, clock_center_x + region_size // 2)
    y2 = min(h, clock_center_y + region_size // 2)
    
    region = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Color masks
    methods = []
    mask_yellow = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([45, 255, 255]))
    methods.append(("yellow", mask_yellow))
    
    mask_orange = cv2.inRange(hsv, np.array([8, 100, 100]), np.array([25, 255, 255]))
    methods.append(("orange", mask_orange))
    
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    )
    methods.append(("red", mask_red))
    
    mask_combined = cv2.bitwise_or(mask_yellow, cv2.bitwise_or(mask_orange, mask_red))
    methods.append(("combined", mask_combined))
    
    # Grayscale fallback
    region_scaled_gray = cv2.resize(region, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(region_scaled_gray, cv2.COLOR_BGR2GRAY)
    _, gray_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    gray_thresh_small = cv2.resize(gray_thresh, (region.shape[1], region.shape[0]), 
                                   interpolation=cv2.INTER_AREA)
    methods.append(("grayscale", gray_thresh_small))
    
    # OCR
    best_result = None
    best_confidence = 0
    
    for method_name, mask in methods:
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
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = (digit, f"{method_name}_PSM{psm}")
    
    if best_result:
        return best_result[0]
    
    return None


def test_session(session_dir):
    frames_dir = os.path.join(session_dir, 'frames')
    
    if not os.path.exists(frames_dir):
        print(f"Error: {frames_dir} not found")
        sys.exit(1)
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.png')])
    
    print(f"Found {len(frame_files)} frames in {frames_dir}")
    
    # Check resolutions
    resolutions = Counter()
    for frame in frame_files[:20]:
        img = cv2.imread(os.path.join(frames_dir, frame))
        if img is not None:
            resolutions[f"{img.shape[1]}x{img.shape[0]}"] += 1
    
    print(f"Resolution sample: {dict(resolutions)}")
    print("Processing...\n")
    
    results = {
        'success': 0,
        'failed': 0,
        'levels': Counter(),
        'failed_frames': []
    }
    
    sample_rate = 1  # Test all frames
    
    for i, frame_file in enumerate(frame_files[::sample_rate]):
        frame_path = os.path.join(frames_dir, frame_file)
        img = cv2.imread(frame_path)
        
        if img is None:
            continue
        
        alarm_level = extract_alarm_level(img, debug=False)
        
        if alarm_level is not None:
            results['success'] += 1
            results['levels'][alarm_level] += 1
        else:
            results['failed'] += 1
            results['failed_frames'].append(frame_file)
        
        if (i + 1) % 50 == 0:
            current_rate = results['success'] / (results['success'] + results['failed']) * 100
            print(f"Processed {i + 1} frames... Success rate: {current_rate:.1f}%")
    
    return results, len(frame_files[::sample_rate])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_test_alarm_v2.py <session_directory>")
        print("Example: python batch_test_alarm_v2.py captures/20251022_201216")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    
    print("=" * 70)
    print("ALARM LEVEL EXTRACTION V2 - BATCH TEST (Multi-Resolution)")
    print("=" * 70)
    print()
    
    results, total = test_session(session_dir)
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total frames tested: {total}")
    print(f"Successful extractions: {results['success']} ({results['success']/total*100:.1f}%)")
    print(f"Failed extractions: {results['failed']} ({results['failed']/total*100:.1f}%)")
    print()
    print("Alarm level distribution:")
    for level in sorted(results['levels'].keys()):
        count = results['levels'][level]
        pct = count / results['success'] * 100
        print(f"  Level {level}: {count} frames ({pct:.1f}%)")
    print()
    
    if results['failed'] > 0 and results['failed'] < 20:
        print("Failed frames:")
        for frame in results['failed_frames'][:20]:
            print(f"  - {frame}")
        if len(results['failed_frames']) > 20:
            print(f"  ... and {len(results['failed_frames']) - 20} more")
        print()
    
    expected_rate = 97.0
    actual_rate = results['success'] / total * 100
    
    print("=" * 70)
    if actual_rate >= expected_rate - 10:
        print(f"✓ GOOD: {actual_rate:.1f}% (target was ~{expected_rate}%)")
    else:
        print(f"⚠ BELOW TARGET: {actual_rate:.1f}% (target was ~{expected_rate}%)")
    print("=" * 70)
