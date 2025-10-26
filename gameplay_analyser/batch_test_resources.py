#!/usr/bin/env python3
"""
Batch test resources extraction - PARALLELIZED
Tests Power and Credits extraction across all frames
"""

import cv2
import numpy as np
import os
import sys
from collections import Counter
from typing import Optional, Dict, Tuple
import pytesseract
import re
from multiprocessing import Pool
from functools import partial


def extract_resources(image: np.ndarray) -> Dict[str, Optional[any]]:
    """Extract power and credits (simplified for batch testing)"""
    
    height, width = image.shape[:2]
    
    result = {
        'power_current': None,
        'power_max': None,
        'credits': None
    }
    
    # Resolution-based positioning
    if width >= 2500:  # 2560x1440 or similar
        power_box = (8, 11, 120, 37)
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
    
    # Extract Power
    power_text = _extract_text_from_box(image, power_box)
    if power_text:
        match = re.search(r'(\d+)\s*/\s*(\d+)', power_text)
        if match:
            result['power_current'] = int(match.group(1))
            result['power_max'] = int(match.group(2))
    
    # Extract Credits
    credits_text = _extract_text_from_box(image, credits_box)
    if credits_text:
        match = re.search(r'(\d+)', credits_text)
        if match:
            result['credits'] = int(match.group(1))
    
    return result


def _extract_text_from_box(image: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[str]:
    """Extract text from a bounding box using OCR"""
    x, y, w, h = box
    
    # Bounds check
    height, width = image.shape[:2]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return None
    
    roi = image[y:y+h, x:x+w]
    
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
    
    # Try multiple thresholds
    thresholds = [127, 100, 80]
    
    for thresh_val in thresholds:
        _, binary = cv2.threshold(inverted, thresh_val, 255, cv2.THRESH_BINARY)
        
        white_ratio = np.sum(binary == 255) / binary.size
        if white_ratio > 0.95:
            continue
        
        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/ '
        text = pytesseract.image_to_string(binary, config=config).strip()
        
        if text:
            return text
    
    # Try adaptive
    binary_adaptive = cv2.adaptiveThreshold(
        inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/ '
    text = pytesseract.image_to_string(binary_adaptive, config=config).strip()
    
    if text:
        return text
    
    # Try Otsu
    _, binary_otsu = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/ '
    text = pytesseract.image_to_string(binary_otsu, config=config).strip()
    
    return text if text else None


def process_frame(frame_info: Tuple[str, str]) -> Tuple[str, Dict]:
    """
    Process a single frame (worker function for multiprocessing)
    
    Returns:
        (frame_file, result_dict)
    """
    frame_file, frame_path = frame_info
    
    image = cv2.imread(frame_path)
    if image is None:
        return (frame_file, {
            'power_current': None,
            'power_max': None,
            'credits': None
        })
    
    result = extract_resources(image)
    return (frame_file, result)


def test_session(session_dir: str, sample_rate: int = 1, num_workers: int = 5):
    """Test resources extraction on all frames using parallel processing"""
    
    frames_dir = os.path.join(session_dir, "frames")
    if not os.path.exists(frames_dir):
        print(f"❌ Frames directory not found: {frames_dir}")
        return None, 0
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    if len(frame_files) == 0:
        print(f"❌ No PNG frames found in {frames_dir}")
        return None, 0
    
    # Apply sampling
    frame_files = frame_files[::sample_rate]
    
    print(f"Found {len(frame_files)} frames in {frames_dir}")
    
    # Check resolution diversity
    resolutions = Counter()
    for i in range(min(20, len(frame_files))):
        img = cv2.imread(os.path.join(frames_dir, frame_files[i]))
        if img is not None:
            resolutions[f"{img.shape[1]}x{img.shape[0]}"] += 1
    print(f"Resolution sample: {dict(resolutions)}")
    
    print(f"Processing with {num_workers} workers...")
    print()
    
    # Prepare frame info list
    frame_infos = [(f, os.path.join(frames_dir, f)) for f in frame_files]
    
    # Track results
    results = {
        'power_success': 0,
        'credits_success': 0,
        'both_success': 0,
        'power_failed': 0,
        'credits_failed': 0,
        'both_failed': 0,
        'failed_frames': [],
        'power_values': Counter(),
        'credits_values': Counter()
    }
    
    processed_count = 0
    batch_size = 50
    
    with Pool(processes=num_workers) as pool:
        # Process in batches for progress updates
        for i in range(0, len(frame_infos), batch_size):
            batch = frame_infos[i:i+batch_size]
            
            # Process batch
            batch_results = pool.map(process_frame, batch)
            
            # Collect results
            for frame_file, res in batch_results:
                processed_count += 1
                
                power_ok = res['power_current'] is not None
                credits_ok = res['credits'] is not None
                
                if power_ok:
                    results['power_success'] += 1
                    results['power_values'][f"{res['power_current']}/{res['power_max']}"] += 1
                else:
                    results['power_failed'] += 1
                
                if credits_ok:
                    results['credits_success'] += 1
                    results['credits_values'][res['credits']] += 1
                else:
                    results['credits_failed'] += 1
                
                if power_ok and credits_ok:
                    results['both_success'] += 1
                elif not power_ok and not credits_ok:
                    results['both_failed'] += 1
                    results['failed_frames'].append(frame_file)
            
            # Progress update
            power_rate = results['power_success'] / processed_count * 100
            credits_rate = results['credits_success'] / processed_count * 100
            both_rate = results['both_success'] / processed_count * 100
            print(f"Processed {processed_count} frames... "
                  f"Power: {power_rate:.1f}%, Credits: {credits_rate:.1f}%, Both: {both_rate:.1f}%")
    
    return results, len(frame_files)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_test_resources.py <session_directory> [num_workers]")
        print("Example: python batch_test_resources.py captures/20251022_201216 5")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print("=" * 70)
    print(f"RESOURCES EXTRACTION - BATCH TEST (Parallel, {num_workers} workers)")
    print("=" * 70)
    print()
    
    import time
    start_time = time.time()
    
    results, total = test_session(session_dir, num_workers=num_workers)
    
    elapsed = time.time() - start_time
    
    if results is None:
        sys.exit(1)
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total frames tested: {total}")
    print()
    print(f"Power extractions: {results['power_success']} ({results['power_success']/total*100:.1f}%)")
    print(f"Credits extractions: {results['credits_success']} ({results['credits_success']/total*100:.1f}%)")
    print(f"Both successful: {results['both_success']} ({results['both_success']/total*100:.1f}%)")
    print()
    print(f"Processing time: {elapsed:.1f} seconds ({total/elapsed:.1f} frames/sec)")
    print()
    
    if results['power_success'] > 0:
        print("Power value distribution (top 10):")
        for value, count in results['power_values'].most_common(10):
            pct = count / results['power_success'] * 100
            print(f"  {value}: {count} frames ({pct:.1f}%)")
        print()
    
    if results['credits_success'] > 0:
        print("Credits value distribution (top 10):")
        for value, count in results['credits_values'].most_common(10):
            pct = count / results['credits_success'] * 100
            print(f"  {value}: {count} frames ({pct:.1f}%)")
        print()
    
    if results['both_failed'] > 0 and results['both_failed'] < 20:
        print("Frames where both failed:")
        for frame in results['failed_frames'][:20]:
            print(f"  - {frame}")
        print()
    
    # Compare to target
    expected_rate = 97.0
    actual_rate = results['both_success'] / total * 100
    
    print("=" * 70)
    if actual_rate >= expected_rate - 5:
        print(f"✓ SUCCESS: {actual_rate:.1f}% is close to target (~{expected_rate}%)")
    else:
        print(f"⚠ BELOW TARGET: {actual_rate:.1f}% (target was ~{expected_rate}%)")
    print("=" * 70)
    