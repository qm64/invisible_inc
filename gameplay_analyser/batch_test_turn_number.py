#!/usr/bin/env python3
"""
Batch test turn number extraction - PARALLELIZED
Tests turn number extraction across all frames
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


def extract_turn_number(image: np.ndarray) -> Optional[int]:
    """Extract turn number (simplified for batch testing)"""
    
    height, width = image.shape[:2]
    
    # Resolution-based positioning
    if width >= 2500:  # 2560x1440 or similar
        turn_box = (2010, 14, 80, 22)
    else:  # Scale for other resolutions
        scale = width / 2560.0
        turn_box = (
            int(2010 * scale),
            int(14 * scale),
            int(80 * scale),
            int(22 * scale)
        )
    
    # Extract turn text
    turn_text = _extract_turn_text(image, turn_box)
    
    if turn_text:
        # Parse number
        match = re.search(r'(\d+)', turn_text)
        if match:
            return int(match.group(1))
    
    return None


def _extract_turn_text(image: np.ndarray, box: tuple) -> Optional[str]:
    """Extract turn text from bounding box with upscaling"""
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
    
    # Upscale for better OCR (text is small!)
    scale_factor = 3
    gray_scaled = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, 
                            interpolation=cv2.INTER_CUBIC)
    
    # Invert for OCR
    inverted = cv2.bitwise_not(gray_scaled)
    
    # Try multiple thresholds
    thresholds = [127, 100, 80]
    
    for thresh_val in thresholds:
        _, binary = cv2.threshold(inverted, thresh_val, 255, cv2.THRESH_BINARY)
        
        white_ratio = np.sum(binary == 255) / binary.size
        if white_ratio > 0.95:
            continue
        
        # Try multiple PSM modes
        for psm in [7, 8, 6]:
            config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789TURN '
            text = pytesseract.image_to_string(binary, config=config).strip()
            
            if text and any(c.isdigit() for c in text):
                return text
    
    # Try adaptive
    binary_adaptive = cv2.adaptiveThreshold(
        inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    for psm in [7, 8, 6]:
        config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789TURN '
        text = pytesseract.image_to_string(binary_adaptive, config=config).strip()
        
        if text and any(c.isdigit() for c in text):
            return text
    
    # Try Otsu
    _, binary_otsu = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    for psm in [7, 8, 6]:
        config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789TURN '
        text = pytesseract.image_to_string(binary_otsu, config=config).strip()
        
        if text and any(c.isdigit() for c in text):
            return text
    
    return None


def process_frame(frame_info: Tuple[str, str]) -> Tuple[str, Optional[int]]:
    """
    Process a single frame (worker function for multiprocessing)
    
    Returns:
        (frame_file, turn_number or None)
    """
    frame_file, frame_path = frame_info
    
    image = cv2.imread(frame_path)
    if image is None:
        return (frame_file, None)
    
    turn = extract_turn_number(image)
    return (frame_file, turn)


def test_session(session_dir: str, sample_rate: int = 1, num_workers: int = 5):
    """Test turn number extraction on all frames using parallel processing"""
    
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
        'success': 0,
        'failed': 0,
        'turn_values': Counter(),
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
            for frame_file, turn in batch_results:
                processed_count += 1
                
                if turn is not None:
                    results['success'] += 1
                    results['turn_values'][turn] += 1
                else:
                    results['failed'] += 1
                    results['failed_frames'].append(frame_file)
            
            # Progress update
            success_rate = results['success'] / processed_count * 100
            print(f"Processed {processed_count} frames... Success rate: {success_rate:.1f}%")
    
    return results, len(frame_files)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_test_turn_number.py <session_directory> [num_workers]")
        print("Example: python batch_test_turn_number.py captures/20251022_201216 5")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print("=" * 70)
    print(f"TURN NUMBER EXTRACTION - BATCH TEST (Parallel, {num_workers} workers)")
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
    print()
    print(f"Processing time: {elapsed:.1f} seconds ({total/elapsed:.1f} frames/sec)")
    print()
    
    if results['success'] > 0:
        print("Turn number distribution:")
        for turn in sorted(results['turn_values'].keys()):
            count = results['turn_values'][turn]
            pct = count / results['success'] * 100
            print(f"  Turn {turn}: {count} frames ({pct:.1f}%)")
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
    