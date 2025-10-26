#!/usr/bin/env python3
"""
Analyze v5 failures to understand what's still failing
Categorizes failures by type to guide next improvements
"""

import cv2
import numpy as np
import os
import sys
from collections import Counter
from typing import Optional, Tuple, Dict
import pytesseract
import re


def analyze_frame_type(image: np.ndarray) -> Dict:
    """
    Analyze frame characteristics to understand why it might fail
    
    Returns:
        Dictionary with frame characteristics
    """
    height, width = image.shape[:2]
    
    info = {
        'resolution': f"{width}x{height}",
        'overall_brightness': float(np.mean(image)),
        'is_very_dark': np.mean(image) < 10,
        'is_very_bright': np.mean(image) > 200,
    }
    
    # Check alarm region specifically
    if width >= 2500:
        center_x = int(width * 0.964)
        center_y = int(height * 0.106)
        roi_size = 60
    else:
        center_x = int(width * 0.964)
        center_y = int(height * 0.106)
        roi_size = 50
    
    half_size = roi_size // 2
    search_x = center_x - half_size
    search_y = center_y - half_size
    
    if search_x >= 0 and search_y >= 0 and search_x + roi_size <= width and search_y + roi_size <= height:
        roi = image[search_y:search_y+roi_size, search_x:search_x+roi_size]
        
        if roi.size > 0:
            info['roi_brightness'] = float(np.mean(roi))
            info['roi_very_dark'] = np.mean(roi) < 5
            info['roi_dim'] = 5 <= np.mean(roi) < 15  # Dimmed UI range
            info['roi_normal'] = np.mean(roi) >= 15
            
            # Check red channel dominance (for level 5/6)
            if len(roi.shape) == 3:
                red_mean = np.mean(roi[:, :, 2])
                green_mean = np.mean(roi[:, :, 1])
                blue_mean = np.mean(roi[:, :, 0])
                
                info['roi_red_dominant'] = red_mean > green_mean * 1.5 and red_mean > blue_mean * 1.5
                info['roi_color_balance'] = {
                    'red': float(red_mean),
                    'green': float(green_mean),
                    'blue': float(blue_mean)
                }
    
    return info


def categorize_failure(info: Dict) -> str:
    """Categorize why a frame might have failed"""
    
    if info.get('is_very_dark', False):
        return "black_screen"
    
    if info.get('is_very_bright', False):
        return "white_screen"
    
    if info.get('roi_very_dark', False):
        return "roi_black"
    
    if info.get('roi_dim', False):
        return "roi_dimmed"
    
    if info.get('roi_red_dominant', False):
        return "roi_red_level5_6"
    
    if info.get('roi_normal', False):
        return "roi_normal_but_failed"
    
    return "unknown"


def diagnose_failures(session_dir: str, max_samples: int = 30):
    """Analyze failed frames to understand failure patterns"""
    
    frames_dir = os.path.join(session_dir, "frames")
    if not os.path.exists(frames_dir):
        print(f"❌ Frames directory not found: {frames_dir}")
        return
    
    output_dir = os.path.join(session_dir, "alarm_diagnostics_v5")
    os.makedirs(output_dir, exist_ok=True)
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    print(f"Found {len(frame_files)} frames")
    print(f"Analyzing all frames to find v5 failures...")
    print()
    
    # Find failed frames with analysis
    failed_analyses = []
    successful_count = 0
    
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        image = cv2.imread(frame_path)
        
        if image is None:
            continue
        
        # Quick v5 extraction (simplified)
        height, width = image.shape[:2]
        
        if width >= 2500:
            center_x = int(width * 0.964)
            center_y = int(height * 0.106)
            roi_size = 60
        else:
            center_x = int(width * 0.964)
            center_y = int(height * 0.106)
            roi_size = 50
        
        half_size = roi_size // 2
        search_x = center_x - half_size
        search_y = center_y - half_size
        
        if search_x < 0 or search_y < 0 or search_x + roi_size > width or search_y + roi_size > height:
            failed_analyses.append((frame_file, analyze_frame_type(image)))
            continue
        
        roi = image[search_y:search_y+roi_size, search_x:search_x+roi_size]
        
        if roi.size == 0 or np.mean(roi) < 5:
            failed_analyses.append((frame_file, analyze_frame_type(image)))
            continue
        
        # Try extraction (simplified - just check if we'd succeed)
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        inverted = cv2.bitwise_not(gray)
        
        # Try main thresholds
        extracted = False
        for thresh_val in [127, 100, 80, 60]:
            _, binary = cv2.threshold(inverted, thresh_val, 255, cv2.THRESH_BINARY)
            white_ratio = np.sum(binary == 255) / binary.size
            
            if white_ratio <= 0.95:
                config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456'
                text = pytesseract.image_to_string(binary, config=config).strip()
                match = re.search(r'\d+', text)
                
                if match and 0 <= int(match.group()) <= 6:
                    extracted = True
                    successful_count += 1
                    break
        
        if not extracted:
            # Try adaptive
            binary_adaptive = cv2.adaptiveThreshold(
                inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456'
            text = pytesseract.image_to_string(binary_adaptive, config=config).strip()
            match = re.search(r'\d+', text)
            
            if match and 0 <= int(match.group()) <= 6:
                extracted = True
                successful_count += 1
        
        if not extracted:
            failed_analyses.append((frame_file, analyze_frame_type(image)))
    
    print(f"✓ Successful: {successful_count} frames")
    print(f"✗ Failed: {len(failed_analyses)} frames")
    print()
    
    # Categorize failures
    failure_categories = Counter()
    for frame_file, info in failed_analyses:
        category = categorize_failure(info)
        failure_categories[category] += 1
    
    print("=" * 70)
    print("FAILURE CATEGORIES")
    print("=" * 70)
    for category, count in failure_categories.most_common():
        pct = count / len(failed_analyses) * 100
        print(f"  {category}: {count} frames ({pct:.1f}%)")
    print()
    
    # Sample each category
    if len(failed_analyses) == 0:
        print("No failures to diagnose!")
        return
    
    print(f"Sampling up to {max_samples} failed frames across categories...")
    print(f"Output directory: {output_dir}")
    print()
    
    # Sample from each category
    category_samples = {}
    for frame_file, info in failed_analyses:
        category = categorize_failure(info)
        if category not in category_samples:
            category_samples[category] = []
        category_samples[category].append((frame_file, info))
    
    sample_count = 0
    for category in sorted(category_samples.keys()):
        samples = category_samples[category][:max(3, max_samples // len(category_samples))]
        
        print(f"Category: {category} (showing {len(samples)} of {len(category_samples[category])})")
        
        for frame_file, info in samples:
            sample_count += 1
            frame_path = os.path.join(frames_dir, frame_file)
            image = cv2.imread(frame_path)
            
            if image is None:
                continue
            
            # Save full frame
            prefix = f"{category}_{sample_count:02d}"
            cv2.imwrite(os.path.join(output_dir, f"{prefix}_full.png"), image)
            
            # Save ROI if available
            height, width = image.shape[:2]
            if width >= 2500:
                center_x = int(width * 0.964)
                center_y = int(height * 0.106)
                roi_size = 60
            else:
                center_x = int(width * 0.964)
                center_y = int(height * 0.106)
                roi_size = 50
            
            half_size = roi_size // 2
            search_x = center_x - half_size
            search_y = center_y - half_size
            
            if 0 <= search_x and 0 <= search_y and search_x + roi_size <= width and search_y + roi_size <= height:
                roi = image[search_y:search_y+roi_size, search_x:search_x+roi_size]
                if roi.size > 0:
                    cv2.imwrite(os.path.join(output_dir, f"{prefix}_roi.png"), roi)
            
            print(f"  {frame_file}:")
            print(f"    Overall brightness: {info['overall_brightness']:.1f}")
            if 'roi_brightness' in info:
                print(f"    ROI brightness: {info['roi_brightness']:.1f}")
            if 'roi_color_balance' in info:
                print(f"    ROI colors: R={info['roi_color_balance']['red']:.1f} "
                      f"G={info['roi_color_balance']['green']:.1f} "
                      f"B={info['roi_color_balance']['blue']:.1f}")
            print()
            
            if sample_count >= max_samples:
                break
        
        if sample_count >= max_samples:
            break
    
    print("=" * 70)
    print(f"Diagnostic images saved to: {output_dir}")
    print("=" * 70)
    print()
    print("Next steps based on failure categories:")
    print("  - black_screen / white_screen: Non-game frames (OK to fail)")
    print("  - roi_black: Very dark ROI (non-game or extreme dim)")
    print("  - roi_dimmed: Dialog overlay dimming (may need lower thresholds)")
    print("  - roi_red_level5_6: Dark red display still failing (need better handling)")
    print("  - roi_normal_but_failed: Unexpected failures (investigate)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_alarm_v5_failures.py <session_directory> [max_samples]")
        print("Example: python diagnose_alarm_v5_failures.py captures/20251022_201216 30")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    print("=" * 70)
    print("ALARM LEVEL EXTRACTION V5 - FAILURE ANALYSIS")
    print("=" * 70)
    print()
    
    diagnose_failures(session_dir, max_samples)
