#!/usr/bin/env python3
"""
Debug tool for framework-based alarm level detector.
Visualizes detection process and saves debug images.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pytesseract

from alarm_level_detector import HamburgerMenuDetector, AlarmLevelDetector


def debug_frame(frame_path: Path, save_dir: Path = None):
    """
    Debug alarm detection with visualization.
    
    Args:
        frame_path: Path to frame image
        save_dir: Directory to save debug images (default: debug_output)
    """
    # Load frame
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"Error: Could not load {frame_path}")
        return
    
    # Create save directory
    if save_dir is None:
        save_dir = Path('debug_output')
    save_dir.mkdir(exist_ok=True)
    
    frame_name = frame_path.stem
    
    print(f"Debugging: {frame_path.name}")
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
    print(f"Saving to: {save_dir}")
    
    # Initialize detectors
    hamburger_detector = HamburgerMenuDetector()
    alarm_detector = AlarmLevelDetector()
    
    # Step 1: Detect hamburger menu
    print("\n=== Step 1: Hamburger Menu Detection ===")
    hamburger_result = hamburger_detector.detect(frame)
    
    print(f"Success: {hamburger_result.success}")
    print(f"Confidence: {hamburger_result.confidence:.2f}")
    
    if hamburger_result.success:
        hamburger_pos = hamburger_result.data['position']
        print(f"Position: {hamburger_pos}")
    else:
        print("Hamburger menu not detected - cannot proceed")
        return
    
    # Step 2: Calculate expected clock position
    print("\n=== Step 2: Clock Position Calculation ===")
    clock_pos = alarm_detector._calculate_clock_position(hamburger_pos)
    print(f"Expected clock center: {clock_pos}")
    print(f"Offset from hamburger: dx={clock_pos[0]-hamburger_pos[0]}, dy={clock_pos[1]-hamburger_pos[1]}")
    
    # Step 3: Detect alarm level
    print("\n=== Step 3: Alarm Level Detection ===")
    alarm_result = alarm_detector.detect(frame, {'hamburger_menu': hamburger_result})
    
    print(f"Success: {alarm_result.success}")
    print(f"Confidence: {alarm_result.confidence:.2f}")
    
    if alarm_result.success:
        data = alarm_result.data
        
        if 'clock_center' in data:
            print(f"Clock center: {data['clock_center']}")
            print(f"Clock radius: {data['clock_radius']}")
        
        if 'major_alarm' in data:
            print(f"Major alarm: {data['major_alarm']}")
        else:
            print(f"Major alarm: FAILED")
        
        if 'minor_alarm' in data:
            print(f"Minor alarm: {data['minor_alarm']}/5 segments")
        else:
            print(f"Minor alarm: FAILED")
        
        # Extract center region for OCR visualization
        if 'clock_center' in data:
            cx, cy = data['clock_center']
            r = data['clock_radius']
            
            inner_radius = int(r * 0.7)
            x1 = max(0, cx - inner_radius)
            y1 = max(0, cy - inner_radius)
            x2 = min(frame.shape[1], cx + inner_radius)
            y2 = min(frame.shape[0], cy + inner_radius)
            
            center_region = frame[y1:y2, x1:x2].copy()
            
            if center_region.size > 0:
                print(f"\n=== OCR Center Region ===")
                print(f"Region size: {center_region.shape[1]}x{center_region.shape[0]}")
                
                # Save center region
                center_4x = cv2.resize(center_region, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(save_dir / f'{frame_name}_center_original.png'), center_4x)
                
                # Show OCR preprocessing
                hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
                gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
                
                # Color mask
                yellow_mask = cv2.inRange(hsv, alarm_detector.yellow_lower, alarm_detector.yellow_upper)
                orange_mask = cv2.inRange(hsv, alarm_detector.orange_lower, alarm_detector.orange_upper)
                red_mask = cv2.inRange(hsv, alarm_detector.red_lower, alarm_detector.red_upper)
                color_mask = cv2.bitwise_or(yellow_mask, orange_mask)
                color_mask = cv2.bitwise_or(color_mask, red_mask)
                
                # Morphological closing
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                color_mask_closed = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
                
                # Upscale and test OCR
                scale = 5
                color_mask_big = cv2.resize(color_mask_closed, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                text = pytesseract.image_to_string(color_mask_big, config=alarm_detector.tesseract_config).strip()
                
                print(f"  OCR (color mask): '{text}'")
                cv2.imwrite(str(save_dir / f'{frame_name}_ocr_color_mask.png'), color_mask_big)
    else:
        print(f"Error: {alarm_result.data.get('error', 'Unknown')}")
    
    # Step 4: Create visualization
    print("\n=== Creating Visualization ===")
    vis_frame = frame.copy()
    
    # Draw hamburger menu position
    hx, hy = hamburger_pos
    cv2.circle(vis_frame, (hx, hy), 10, (0, 165, 255), 2)  # Orange circle
    cv2.putText(vis_frame, "HAMBURGER", (hx + 15, hy), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    
    # Draw calculated clock position
    cv2.circle(vis_frame, clock_pos, 5, (255, 0, 255), -1)  # Magenta dot
    
    # Draw connection line
    cv2.line(vis_frame, hamburger_pos, clock_pos, (128, 128, 128), 1, cv2.LINE_AA)
    
    # If alarm detected, draw clock details
    if alarm_result.success and 'clock_center' in alarm_result.data:
        cx, cy = alarm_result.data['clock_center']
        r = alarm_result.data['clock_radius']
        
        # Draw clock circle
        cv2.circle(vis_frame, (cx, cy), r, (0, 255, 0), 2)  # Green outer circle
        cv2.circle(vis_frame, (cx, cy), 3, (0, 255, 0), -1)  # Green center dot
        
        # Draw inner region (70% for digit)
        inner_r = int(r * 0.7)
        cv2.circle(vis_frame, (cx, cy), inner_r, (0, 255, 255), 1)  # Cyan inner circle
        
        # Draw segment sample points (90% radius, 5 segments with 5 samples each)
        segment_centers = [-90, -18, 54, 126, 198]
        colors = [(255, 255, 0), (255, 192, 0), (255, 128, 0), (255, 64, 0), (255, 0, 0)]  # Blue gradient
        
        for seg_idx, center_angle in enumerate(segment_centers):
            for offset in [-25, -12, 0, 12, 25]:
                angle_deg = center_angle + offset
                angle_rad = np.radians(angle_deg)
                sample_radius = int(r * 0.90)
                sample_x = int(cx + sample_radius * np.cos(angle_rad))
                sample_y = int(cy + sample_radius * np.sin(angle_rad))
                
                # Draw sample point (different color per segment)
                cv2.circle(vis_frame, (sample_x, sample_y), 2, colors[seg_idx], -1)
        
        # Add labels
        if 'major_alarm' in alarm_result.data:
            cv2.putText(vis_frame, f"MAJOR: {alarm_result.data['major_alarm']}", 
                       (cx + r + 10, cy - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if 'minor_alarm' in alarm_result.data:
            cv2.putText(vis_frame, f"MINOR: {alarm_result.data['minor_alarm']}/5", 
                       (cx + r + 10, cy + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save visualization
    cv2.imwrite(str(save_dir / f'{frame_name}_detection.png'), vis_frame)
    
    print(f"\nDebug images saved to {save_dir}/")
    print(f"  - {frame_name}_detection.png (main visualization)")
    print(f"  - {frame_name}_center_original.png (center region)")
    print(f"  - {frame_name}_ocr_color_mask.png (OCR preprocessing)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_framework_alarm.py <frame_path> [frame_path2 ...] [--output-dir <dir>]")
        print("       Saves debug images to output_dir (default: debug_output)")
        sys.exit(1)
    
    # Parse arguments
    frame_paths = []
    output_dir = Path('debug_output')
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--output-dir':
            if i + 1 < len(sys.argv):
                output_dir = Path(sys.argv[i + 1])
                i += 2
            else:
                print("Error: --output-dir requires a directory path")
                sys.exit(1)
        else:
            frame_paths.append(sys.argv[i])
            i += 1
    
    if not frame_paths:
        print("Error: No frame paths provided")
        sys.exit(1)
    
    for frame_path_str in frame_paths:
        frame_path = Path(frame_path_str)
        if not frame_path.exists():
            print(f"Error: Frame not found: {frame_path}")
            continue
        
        debug_frame(frame_path, output_dir)
        print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
