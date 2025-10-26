#!/usr/bin/env python3
"""
Batch test alarm level extraction v5 - Handles dark red displays
"""

import cv2
import numpy as np
import os
import sys
from collections import Counter
from typing import Optional
import pytesseract
import re


def try_ocr_on_image(binary: np.ndarray) -> Optional[int]:
    """Try multiple OCR strategies on a binary image"""
    
    strategies = [
        r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456',
        r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456',
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456',
    ]
    
    for config in strategies:
        text = pytesseract.image_to_string(binary, config=config).strip()
        
        # Extract first digit
        match = re.search(r'\d+', text)
        if match:
            level = int(match.group())
            if 0 <= level <= 6:
                return level
    
    return None


def extract_alarm_level(image: np.ndarray) -> Optional[int]:
    """Extract alarm level with v5 multi-threshold approach"""
    
    height, width = image.shape[:2]
    
    # Resolution-based positioning
    if width >= 2500:
        center_x = int(width * 0.964)
        center_y = int(height * 0.106)
        roi_size = 60
    else:
        center_x = int(width * 0.964)
        center_y = int(height * 0.106)
        roi_size = 50
    
    # Extract square ROI
    half_size = roi_size // 2
    search_x = center_x - half_size
    search_y = center_y - half_size
    
    # Bounds check
    if search_x < 0 or search_y < 0:
        return None
    if search_x + roi_size > width or search_y + roi_size > height:
        return None
    
    roi = image[search_y:search_y+roi_size, search_x:search_x+roi_size]
    
    if roi.size == 0:
        return None
    
    # Convert to grayscale
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    # Skip very dark ROIs
    if np.mean(gray) < 5:
        return None
    
    # Invert for OCR
    inverted = cv2.bitwise_not(gray)
    
    # Strategy 1: Multiple fixed thresholds
    thresholds = [127, 100, 80, 60]
    
    for thresh_val in thresholds:
        _, binary = cv2.threshold(inverted, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Skip if mostly white
        white_ratio = np.sum(binary == 255) / binary.size
        if white_ratio > 0.95:
            continue
        
        result = try_ocr_on_image(binary)
        if result is not None:
            return result
    
    # Strategy 2: Adaptive thresholding
    binary_adaptive = cv2.adaptiveThreshold(
        inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    result = try_ocr_on_image(binary_adaptive)
    if result is not None:
        return result
    
    # Strategy 3: Otsu's method
    _, binary_otsu = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    result = try_ocr_on_image(binary_otsu)
    if result is not None:
        return result
    
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
        print("Usage: python batch_test_alarm_v5.py <session_directory>")
        print("Example: python batch_test_alarm_v5.py captures/20251022_201216")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    
    print("=" * 70)
    print("ALARM LEVEL EXTRACTION V5 - BATCH TEST (Multi-Threshold)")
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
