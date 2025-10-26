#!/usr/bin/env python3
"""
Batch test agent AP extraction - PARALLELIZED
Tests agent AP extraction across all frames
"""

import cv2
import numpy as np
import os
import sys
from collections import Counter
from typing import Optional, Dict, List, Tuple
import pytesseract
import re
from multiprocessing import Pool


def extract_agent_aps(image: np.ndarray) -> Dict[str, any]:
    """Extract agent APs (simplified for batch testing)"""
    
    height, width = image.shape[:2]
    
    result = {
        'agents': [],
        'profile_detected': False
    }
    
    # Resolution-based positioning
    if width >= 2500:  # 2560x1440 or similar
        profile_box = (20, 1190, 195, 230)
        ap_start_x = 63
        ap_width = 65
        ap_height = 25
        ap_start_y = 1120
        ap_spacing = 37
        max_agents = 8
    else:  # Scale for other resolutions
        scale = width / 2560.0
        profile_box = (
            int(20 * scale),
            int(1190 * scale),
            int(195 * scale),
            int(230 * scale)
        )
        ap_start_x = int(63 * scale)
        ap_width = int(65 * scale)
        ap_height = int(25 * scale)
        ap_start_y = int(1120 * scale)
        ap_spacing = int(37 * scale)
        max_agents = 8
    
    # Try to detect large profile
    result['profile_detected'] = _detect_profile(image, profile_box)
    
    if not result['profile_detected']:
        return result
    
    # Search for agent APs vertically
    for agent_idx in range(max_agents):
        y_pos = ap_start_y + (agent_idx * ap_spacing)
        
        # Stop if would overlap profile
        if y_pos + ap_height > profile_box[1]:
            break
        
        ap_box = (ap_start_x, y_pos, ap_width, ap_height)
        ap_data = _extract_ap_from_box(image, ap_box, agent_idx)
        
        if ap_data:
            result['agents'].append(ap_data)
        else:
            # If we miss one, probably no more agents
            break
    
    return result


def _detect_profile(image: np.ndarray, box: tuple) -> bool:
    """Detect if the large agent profile is present"""
    x, y, w, h = box
    
    # Bounds check
    height, width = image.shape[:2]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return False
    
    roi = image[y:y+h, x:x+w]
    
    if roi.size == 0:
        return False
    
    # Profile area should have brightness > 15
    return np.mean(roi) >= 15


def _extract_ap_from_box(image: np.ndarray, box: tuple, agent_idx: int) -> Optional[Dict]:
    """Extract AP value from a bounding box"""
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
    if np.mean(gray) < 10:
        return None
    
    # Upscale for better OCR
    scale_factor = 2
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
        
        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789AP '
        text = pytesseract.image_to_string(binary, config=config).strip()
        
        if text:
            match = re.search(r'(\d+)', text)
            if match:
                ap_value = int(match.group(1))
                if 0 <= ap_value <= 99:
                    return {
                        'index': agent_idx,
                        'ap': ap_value,
                        'ap_text': text
                    }
    
    # Try adaptive
    binary_adaptive = cv2.adaptiveThreshold(
        inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789AP '
    text = pytesseract.image_to_string(binary_adaptive, config=config).strip()
    
    if text:
        match = re.search(r'(\d+)', text)
        if match:
            ap_value = int(match.group(1))
            if 0 <= ap_value <= 99:
                return {
                    'index': agent_idx,
                    'ap': ap_value,
                    'ap_text': text
                }
    
    return None


def process_frame(frame_info: Tuple[str, str]) -> Tuple[str, Dict]:
    """Process a single frame (worker function for multiprocessing)"""
    frame_file, frame_path = frame_info
    
    image = cv2.imread(frame_path)
    if image is None:
        return (frame_file, {
            'agents': [],
            'profile_detected': False
        })
    
    result = extract_agent_aps(image)
    return (frame_file, result)


def test_session(session_dir: str, sample_rate: int = 1, num_workers: int = 5):
    """Test agent AP extraction on all frames using parallel processing"""
    
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
        'frames_with_profile': 0,
        'frames_without_profile': 0,
        'frames_with_agents': 0,
        'total_agents_found': 0,
        'ap_values': Counter(),
        'agent_counts': Counter()
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
                
                if res['profile_detected']:
                    results['frames_with_profile'] += 1
                    
                    num_agents = len(res['agents'])
                    if num_agents > 0:
                        results['frames_with_agents'] += 1
                        results['agent_counts'][num_agents] += 1
                        results['total_agents_found'] += num_agents
                        
                        # Collect AP values
                        for agent in res['agents']:
                            results['ap_values'][agent['ap']] += 1
                else:
                    results['frames_without_profile'] += 1
            
            # Progress update
            profile_rate = results['frames_with_profile'] / processed_count * 100
            agent_rate = results['frames_with_agents'] / processed_count * 100
            print(f"Processed {processed_count} frames... "
                  f"Profile: {profile_rate:.1f}%, With agents: {agent_rate:.1f}%")
    
    return results, len(frame_files)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_test_agent_ap.py <session_directory> [num_workers]")
        print("Example: python batch_test_agent_ap.py captures/20251022_201216 5")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print("=" * 70)
    print(f"AGENT AP EXTRACTION - BATCH TEST (Parallel, {num_workers} workers)")
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
    print(f"Frames with profile: {results['frames_with_profile']} "
          f"({results['frames_with_profile']/total*100:.1f}%)")
    print(f"Frames without profile: {results['frames_without_profile']} "
          f"({results['frames_without_profile']/total*100:.1f}%)")
    print()
    print(f"Frames with agents detected: {results['frames_with_agents']} "
          f"({results['frames_with_agents']/total*100:.1f}%)")
    print(f"Total agents found: {results['total_agents_found']}")
    print()
    print(f"Processing time: {elapsed:.1f} seconds ({total/elapsed:.1f} frames/sec)")
    print()
    
    if results['agent_counts']:
        print("Agent count distribution:")
        for count in sorted(results['agent_counts'].keys()):
            frames = results['agent_counts'][count]
            pct = frames / results['frames_with_profile'] * 100
            print(f"  {count} agents: {frames} frames ({pct:.1f}% of profile frames)")
        print()
    
    if results['ap_values']:
        print("AP value distribution (top 15):")
        for ap_val, count in results['ap_values'].most_common(15):
            pct = count / results['total_agents_found'] * 100
            print(f"  {ap_val} AP: {count} occurrences ({pct:.1f}%)")
        print()
    
    print("=" * 70)
    print("NOTE: Agent AP is only visible during planning phase (when profile is shown)")
    print("Action phases will show 'Frames without profile'")
    print("=" * 70)
    