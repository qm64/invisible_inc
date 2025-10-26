#!/usr/bin/env python3
"""
Batch test alarm level extraction v5 - PARALLELIZED
Uses 5 workers like turn_phase_detector for ~5-8x speedup
"""

import cv2
import numpy as np
import os
import sys
from collections import Counter
from typing import Optional, Tuple, List
import pytesseract
import re
from multiprocessing import Pool, cpu_count
from functools import partial


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


def process_frame(frame_info: Tuple[str, str]) -> Tuple[str, Optional[int]]:
    """
    Process a single frame (worker function for multiprocessing)
    
    Args:
        frame_info: (frame_file, frame_path)
    
    Returns:
        (frame_file, alarm_level or None)
    """
    frame_file, frame_path = frame_info
    
    image = cv2.imread(frame_path)
    if image is None:
        return (frame_file, None)
    
    level = extract_alarm_level(image)
    return (frame_file, level)


def test_session(session_dir: str, sample_rate: int = 1, num_workers: int = 5):
    """
    Test alarm level extraction on all frames using parallel processing
    
    Args:
        session_dir: Path to session directory
        sample_rate: Process every Nth frame
        num_workers: Number of parallel workers
    """
    
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
    
    # Process frames in parallel
    results = {
        'success': 0,
        'failed': 0,
        'levels': Counter(),
        'failed_frames': []
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
            for frame_file, level in batch_results:
                processed_count += 1
                
                if level is not None:
                    results['success'] += 1
                    results['levels'][level] += 1
                else:
                    results['failed'] += 1
                    results['failed_frames'].append(frame_file)
            
            # Progress update
            current_rate = results['success'] / processed_count * 100
            print(f"Processed {processed_count} frames... Success rate: {current_rate:.1f}%")
    
    return results, len(frame_files)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_test_alarm_v5_parallel.py <session_directory> [num_workers]")
        print("Example: python batch_test_alarm_v5_parallel.py captures/20251022_201216 5")
        print()
        print("Default: 5 workers (recommended)")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print("=" * 70)
    print(f"ALARM LEVEL EXTRACTION V5 - BATCH TEST (Parallel, {num_workers} workers)")
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
    print(f"Successful extractions: {results['success']} ({results['success']/total*100:.1f}%)")
    print(f"Failed extractions: {results['failed']} ({results['failed']/total*100:.1f}%)")
    print(f"Processing time: {elapsed:.1f} seconds ({total/elapsed:.1f} frames/sec)")
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
