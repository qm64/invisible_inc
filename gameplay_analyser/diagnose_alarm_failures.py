#!/usr/bin/env python3
"""
Diagnostic for failed alarm level extractions
Samples failed frames and saves ROI images for inspection
"""

import cv2
import numpy as np
import os
import sys
from typing import Optional
import pytesseract
import re
import random


def extract_alarm_level_with_images(image: np.ndarray, output_prefix: str) -> tuple[Optional[int], dict]:
    """Extract alarm level and save intermediate images for inspection"""
    
    height, width = image.shape[:2]
    
    # Same positioning as v3
    search_x = int(width * 0.954)
    search_y = int(height * 0.088)
    search_width = 80
    search_height = 100
    
    info = {
        'resolution': f"{width}x{height}",
        'search_pos': (search_x, search_y),
        'roi_shape': None,
        'roi_mean': None,
        'ocr_text': None
    }
    
    # Extract ROI
    roi = image[search_y:search_y+search_height, search_x:search_x+search_width]
    
    if roi.size == 0:
        return None, info
    
    info['roi_shape'] = roi.shape
    info['roi_mean'] = float(np.mean(roi))
    
    # Save original ROI
    cv2.imwrite(f"{output_prefix}_1_roi.png", roi)
    
    # Convert to grayscale
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    cv2.imwrite(f"{output_prefix}_2_gray.png", gray)
    
    # Invert
    inverted = cv2.bitwise_not(gray)
    cv2.imwrite(f"{output_prefix}_3_inverted.png", inverted)
    
    # Threshold
    _, binary = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"{output_prefix}_4_binary.png", binary)
    
    # OCR
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(binary, config=custom_config).strip()
    info['ocr_text'] = repr(text)
    
    # Extract number
    match = re.search(r'\d+', text)
    if match:
        level = int(match.group())
        if 0 <= level <= 6:
            return level, info
    
    return None, info


def diagnose_failures(session_dir: str, num_samples: int = 20):
    """Sample failed frames and save diagnostic images"""
    
    frames_dir = os.path.join(session_dir, "frames")
    if not os.path.exists(frames_dir):
        print(f"❌ Frames directory not found: {frames_dir}")
        return
    
    output_dir = os.path.join(session_dir, "alarm_diagnostics")
    os.makedirs(output_dir, exist_ok=True)
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    print(f"Found {len(frame_files)} frames")
    print(f"Analyzing all frames to find failures...")
    print()
    
    # Find failed frames
    failed_frames = []
    successful_frames = []
    
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        image = cv2.imread(frame_path)
        
        if image is None:
            continue
        
        height, width = image.shape[:2]
        search_x = int(width * 0.954)
        search_y = int(height * 0.088)
        search_width = 80
        search_height = 100
        
        roi = image[search_y:search_y+search_height, search_x:search_x+search_width]
        
        if roi.size == 0:
            failed_frames.append(frame_file)
            continue
        
        # Quick extraction
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        inverted = cv2.bitwise_not(gray)
        _, binary = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)
        
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(binary, config=custom_config).strip()
        
        match = re.search(r'\d+', text)
        if match and 0 <= int(match.group()) <= 6:
            successful_frames.append(frame_file)
        else:
            failed_frames.append(frame_file)
    
    print(f"✓ Successful: {len(successful_frames)} frames")
    print(f"✗ Failed: {len(failed_frames)} frames")
    print()
    
    # Sample failures
    if len(failed_frames) == 0:
        print("No failed frames to diagnose!")
        return
    
    sample_size = min(num_samples, len(failed_frames))
    sampled_failures = random.sample(failed_frames, sample_size)
    
    print(f"Sampling {sample_size} failed frames for detailed analysis...")
    print(f"Output directory: {output_dir}")
    print()
    
    for i, frame_file in enumerate(sampled_failures, 1):
        frame_path = os.path.join(frames_dir, frame_file)
        image = cv2.imread(frame_path)
        
        if image is None:
            continue
        
        # Save full frame for context
        cv2.imwrite(os.path.join(output_dir, f"fail_{i:02d}_0_full.png"), image)
        
        # Extract with diagnostic images
        output_prefix = os.path.join(output_dir, f"fail_{i:02d}")
        level, info = extract_alarm_level_with_images(image, output_prefix)
        
        print(f"Failed frame {i}/{sample_size}: {frame_file}")
        print(f"  Resolution: {info['resolution']}")
        print(f"  Search pos: {info['search_pos']}")
        print(f"  ROI shape: {info['roi_shape']}")
        print(f"  ROI mean: {info['roi_mean']:.1f}")
        print(f"  OCR text: {info['ocr_text']}")
        print(f"  Extracted: {level}")
        print()
    
    print("=" * 70)
    print(f"Diagnostic images saved to: {output_dir}")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Check the full frame images (fail_XX_0_full.png) to see context")
    print("2. Check the ROI images (fail_XX_1_roi.png) to see what was extracted")
    print("3. Check the binary images (fail_XX_4_binary.png) to see what OCR saw")
    print("4. Look for patterns in what's causing failures")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_alarm_failures.py <session_directory> [num_samples]")
        print("Example: python diagnose_alarm_failures.py captures/20251022_201216 20")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    print("=" * 70)
    print("ALARM LEVEL EXTRACTION - FAILURE DIAGNOSTICS")
    print("=" * 70)
    print()
    
    diagnose_failures(session_dir, num_samples)
