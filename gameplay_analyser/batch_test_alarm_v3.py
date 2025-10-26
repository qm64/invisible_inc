#!/usr/bin/env python3
"""
Batch test alarm level extraction v3 - Corrected coordinates
"""

import cv2
import numpy as np
import os
import sys
from collections import Counter
from typing import Optional
import pytesseract
import re


def extract_alarm_level(image: np.ndarray, debug: bool = False) -> Optional[int]:
    """Extract alarm level from security indicator (v3 corrected coordinates)"""
    
    height, width = image.shape[:2]
    
    # Resolution-based positioning
    if width >= 2500:  # 2560x1440 or similar
        search_x = int(width * 0.954)
        search_y = int(height * 0.088)
        search_width = 80
        search_height = 100
    else:  # Lower res
        search_x = int(width * 0.954)
        search_y = int(height * 0.088)
        search_width = 60
        search_height = 80
    
    # Extract ROI
    roi = image[search_y:search_y+search_height, search_x:search_x+search_width]
    
    if roi.size == 0:
        return None
    
    # Convert to grayscale
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    # Invert for OCR (light text on dark background)
    inverted = cv2.bitwise_not(gray)
    
    # Threshold
    _, binary = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)
    
    # OCR
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(binary, config=custom_config).strip()
    
    # Extract number
    match = re.search(r'\d+', text)
    if match:
        level = int(match.group())
        if 0 <= level <= 6:
            return level
    
    return None


def test_session(session_dir: str, sample_rate: int = 1):
    """Test alarm level extraction on all frames in a session"""
    
    frames_dir = os.path.join(session_dir, "frames")
    if not os.path.exists(frames_dir):
        print(f"❌ Frames directory not found: {frames_dir}")
        return None, 0
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    if len(frame_files) == 0:
        print(f"❌ No PNG frames found in {frames_dir}")
        return None, 0
    
    print(f"Found {len(frame_files)} frames in {frames_dir}")
    
    # Check resolution diversity
    resolutions = Counter()
    for i in range(min(20, len(frame_files))):
        img = cv2.imread(os.path.join(frames_dir, frame_files[i]))
        if img is not None:
            resolutions[f"{img.shape[1]}x{img.shape[0]}"] += 1
    print(f"Resolution sample: {dict(resolutions)}")
    
    print("Processing...")
    print()
    
    results = {
        'success': 0,
        'failed': 0,
        'levels': Counter(),
        'failed_frames': []
    }
    
    for i, frame_file in enumerate(frame_files[::sample_rate]):
        frame_path = os.path.join(frames_dir, frame_file)
        image = cv2.imread(frame_path)
        
        if image is None:
            results['failed'] += 1
            results['failed_frames'].append(frame_file)
            continue
        
        level = extract_alarm_level(image)
        
        if level is not None:
            results['success'] += 1
            results['levels'][level] += 1
        else:
            results['failed'] += 1
            results['failed_frames'].append(frame_file)
        
        # Progress updates
        if (i + 1) % 50 == 0:
            total = i + 1
            current_rate = results['success'] / total * 100
            print(f"Processed {total} frames... Success rate: {current_rate:.1f}%")
    
    return results, len(frame_files[::sample_rate])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_test_alarm_v3.py <session_directory>")
        print("Example: python batch_test_alarm_v3.py captures/20251022_201216")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    
    print("=" * 70)
    print("ALARM LEVEL EXTRACTION V3 - BATCH TEST (Corrected Coordinates)")
    print("=" * 70)
    print()
    
    results, total = test_session(session_dir)
    
    if results is None:
        sys.exit(1)
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total frames tested: {total}")
    print(f"Successful extractions: {results['success']} ({results['success']/total*100:.1f}%)")
    print(f"Failed extractions: {results['failed']} ({results['failed']/total*100:.1f}%)")
    print()
    
    if results['success'] > 0:
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
        print()
    
    # Compare to target
    expected_rate = 97.0
    actual_rate = results['success'] / total * 100
    
    print("=" * 70)
    if actual_rate >= expected_rate - 5:
        print(f"✓ SUCCESS: {actual_rate:.1f}% is close to target (~{expected_rate}%)")
    else:
        print(f"⚠ BELOW TARGET: {actual_rate:.1f}% (target was ~{expected_rate}%)")
    print("=" * 70)
