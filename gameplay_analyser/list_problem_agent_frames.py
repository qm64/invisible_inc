#!/usr/bin/env python3
"""
List frames where profile is detected but no agents found
Output: Simple list for web viewer inspection
"""

import cv2
import numpy as np
import os
import sys
from typing import Optional, Dict, Tuple
import pytesseract
import re


def extract_agent_aps(image: np.ndarray) -> Dict[str, any]:
    """Extract agent APs (simplified)"""
    
    height, width = image.shape[:2]
    
    result = {
        'agents': [],
        'profile_detected': False
    }
    
    # Resolution-based positioning
    if width >= 2500:
        profile_box = (20, 1190, 195, 230)
        ap_start_x = 63
        ap_width = 65
        ap_height = 25
        ap_start_y = 1120
        ap_spacing = 37
        max_agents = 8
    else:
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
    
    # Detect profile
    result['profile_detected'] = _detect_profile(image, profile_box)
    
    if not result['profile_detected']:
        return result
    
    # Search for agent APs
    for agent_idx in range(max_agents):
        y_pos = ap_start_y + (agent_idx * ap_spacing)
        
        if y_pos + ap_height > profile_box[1]:
            break
        
        ap_box = (ap_start_x, y_pos, ap_width, ap_height)
        ap_data = _extract_ap_from_box(image, ap_box, agent_idx)
        
        if ap_data:
            result['agents'].append(ap_data)
        else:
            break
    
    return result


def _detect_profile(image: np.ndarray, box: tuple) -> bool:
    """Detect if the large agent profile is present"""
    x, y, w, h = box
    
    height, width = image.shape[:2]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return False
    
    roi = image[y:y+h, x:x+w]
    
    if roi.size == 0:
        return False
    
    return np.mean(roi) >= 15


def _extract_ap_from_box(image: np.ndarray, box: tuple, agent_idx: int) -> Optional[Dict]:
    """Extract AP value from a bounding box"""
    x, y, w, h = box
    
    height, width = image.shape[:2]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return None
    
    roi = image[y:y+h, x:x+w]
    
    if roi.size == 0:
        return None
    
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    if np.mean(gray) < 10:
        return None
    
    scale_factor = 2
    gray_scaled = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, 
                            interpolation=cv2.INTER_CUBIC)
    
    inverted = cv2.bitwise_not(gray_scaled)
    
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


def find_problem_frames(session_dir: str):
    """Find frames with profile but no agents detected"""
    
    frames_dir = os.path.join(session_dir, "frames")
    if not os.path.exists(frames_dir):
        print(f"❌ Frames directory not found: {frames_dir}")
        return
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    if len(frame_files) == 0:
        print(f"❌ No PNG frames found in {frames_dir}")
        return
    
    print(f"Analyzing {len(frame_files)} frames...")
    print()
    
    problem_frames = []
    
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_file)
        image = cv2.imread(frame_path)
        
        if image is None:
            continue
        
        result = extract_agent_aps(image)
        
        # Profile detected but no agents found
        if result['profile_detected'] and len(result['agents']) == 0:
            # Extract frame number
            frame_num = int(frame_file.replace('frame_', '').replace('.png', ''))
            problem_frames.append(frame_num)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(frame_files)}...")
    
    print()
    print("=" * 70)
    print(f"FRAMES WITH PROFILE BUT NO AGENTS: {len(problem_frames)}")
    print("=" * 70)
    print()
    
    if problem_frames:
        # Output as comma-separated list for easy copying
        print("Frame numbers (comma-separated):")
        print(",".join(str(f) for f in problem_frames))
        print()
        
        # Also output as ranges for readability
        print("Frame ranges:")
        ranges = []
        start = problem_frames[0]
        prev = start
        
        for frame_num in problem_frames[1:]:
            if frame_num == prev + 1:
                prev = frame_num
            else:
                if start == prev:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{prev}")
                start = frame_num
                prev = frame_num
        
        # Add final range
        if start == prev:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{prev}")
        
        print(", ".join(ranges))
        print()
        
        # Save to file
        output_file = os.path.join(session_dir, "problem_frames.txt")
        with open(output_file, 'w') as f:
            f.write("Frames with profile detected but no agents found\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total: {len(problem_frames)} frames\n\n")
            f.write("Frame numbers:\n")
            for frame_num in problem_frames:
                f.write(f"{frame_num}\n")
        
        print(f"✓ Saved to {output_file}")
    else:
        print("No problem frames found!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python list_problem_agent_frames.py <session_directory>")
        print("Example: python list_problem_agent_frames.py captures/20251022_201216")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    
    print("=" * 70)
    print("FIND FRAMES: Profile Detected, No Agents Found")
    print("=" * 70)
    print()
    
    find_problem_frames(session_dir)
    