#!/usr/bin/env python3
"""
Debug script for alarm level detection.
Shows intermediate steps and visualizations to diagnose issues.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pytesseract

from alarm_level_extractor import AlarmLevelExtractor


def debug_frame(frame_path: Path, save_dir: Path = None):
    """
    Debug alarm detection on a single frame with detailed visualization.
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
    
    extractor = AlarmLevelExtractor()
    
    # Step 1: Color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    yellow_mask = cv2.inRange(hsv, extractor.yellow_lower, extractor.yellow_upper)
    orange_mask = cv2.inRange(hsv, extractor.orange_lower, extractor.orange_upper)
    red_mask = cv2.inRange(hsv, extractor.red_lower, extractor.red_upper)
    
    color_mask = cv2.bitwise_or(yellow_mask, orange_mask)
    color_mask = cv2.bitwise_or(color_mask, red_mask)
    
    # Apply region mask
    height, width = frame.shape[:2]
    search_x_start = int(width * 0.85)  # Match extractor: 85%
    search_y_end = int(height * 0.20)   # Match extractor: 20% to capture full clock
    
    region_mask = np.zeros(color_mask.shape, dtype=np.uint8)
    region_mask[0:search_y_end, search_x_start:width] = 255
    color_mask_region = cv2.bitwise_and(color_mask, region_mask)
    
    print(f"Search region: x>{search_x_start}, y<{search_y_end}")
    print(f"Colored pixels in region: {np.count_nonzero(color_mask_region)}")
    
    # Step 2: Circle detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,      # Match extractor
        param1=50,
        param2=20,       # Match extractor
        minRadius=extractor.min_radius,
        maxRadius=extractor.max_radius
    )
    
    if circles is not None:
        print(f"Found {len(circles[0])} circles")
        circles_np = np.uint16(np.around(circles))
        
        # Show all circles and which would be selected (rightmost in region with color)
        best_x = 0
        selected_idx = -1
        for i, circle in enumerate(circles_np[0][:10]):  # Show first 10
            cx, cy, r = circle
            in_region = (cx >= search_x_start and cy <= search_y_end)
            
            # Check color ratio
            circle_mask = np.zeros(color_mask_region.shape, dtype=np.uint8)
            cv2.circle(circle_mask, (int(cx), int(cy)), int(r), 255, -1)
            circle_colored = cv2.bitwise_and(color_mask_region, circle_mask)
            color_pixels = np.count_nonzero(circle_colored)
            circle_pixels = np.count_nonzero(circle_mask)
            color_ratio = color_pixels / circle_pixels if circle_pixels > 0 else 0
            
            if in_region and color_ratio > 0.03 and cx > best_x:
                best_x = cx
                selected_idx = i
            
            marker = " <- SELECTED (rightmost with color)" if i == selected_idx else ""
            print(f"  Circle {i}: center=({cx},{cy}), radius={r}, in_region={in_region}, color_ratio={color_ratio:.2f}{marker}")
    else:
        print("No circles found")
    
    # Step 3: Run full detection
    result = extractor.detect(frame)
    
    print(f"\nDetection Result:")
    print(f"  Success: {result['success']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    
    if 'clock_center' in result['data']:
        cx, cy = result['data']['clock_center']
        r = result['data']['clock_radius']
        print(f"  Clock: center=({cx},{cy}), radius={r}")
        
        # Extract and show center region
        inner_radius = int(r * 0.7)
        x1 = max(0, cx - inner_radius)
        y1 = max(0, cy - inner_radius)
        x2 = min(frame.shape[1], cx + inner_radius)
        y2 = min(frame.shape[0], cy + inner_radius)
        
        center_region = frame[y1:y2, x1:x2].copy()
        print(f"  Center region: {center_region.shape[1]}x{center_region.shape[0]}")
        
        # Try OCR with different preprocessing
        gray_center = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
        
        # Get HSV for color-based preprocessing
        hsv_center = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        
        print(f"\n  OCR attempts:")
        
        scale = 5
        
        # Method 1: Color mask (yellow/orange/red)
        yellow_mask = cv2.inRange(hsv_center, extractor.yellow_lower, extractor.yellow_upper)
        orange_mask = cv2.inRange(hsv_center, extractor.orange_lower, extractor.orange_upper)
        red_mask = cv2.inRange(hsv_center, extractor.red_lower, extractor.red_upper)
        color_mask = cv2.bitwise_or(yellow_mask, orange_mask)
        color_mask = cv2.bitwise_or(color_mask, red_mask)
        
        # Apply morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        color_mask_closed = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        color_mask_big = cv2.resize(color_mask_closed, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        text1 = pytesseract.image_to_string(color_mask_big, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456').strip()
        print(f"    Color mask (closed): '{text1}'")
        cv2.imwrite(str(save_dir / f'{frame_name}_ocr_1_color_mask.png'), color_mask_big)
        
        # Method 2: Color mask inverted
        color_mask_inv_big = cv2.resize(cv2.bitwise_not(color_mask_closed), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        text2 = pytesseract.image_to_string(color_mask_inv_big, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456').strip()
        print(f"    Color mask inverse: '{text2}'")
        cv2.imwrite(str(save_dir / f'{frame_name}_ocr_2_color_inv.png'), color_mask_inv_big)
        
        # Method 3: Otsu threshold
        _, thresh1 = cv2.threshold(gray_center, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh1_big = cv2.resize(thresh1, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        text3 = pytesseract.image_to_string(thresh1_big, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456').strip()
        print(f"    Otsu: '{text3}'")
        cv2.imwrite(str(save_dir / f'{frame_name}_ocr_3_otsu.png'), thresh1_big)
        
        # Method 4: Otsu inverse
        _, thresh2 = cv2.threshold(gray_center, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh2_big = cv2.resize(thresh2, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        text4 = pytesseract.image_to_string(thresh2_big, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456').strip()
        print(f"    Otsu inverse: '{text4}'")
        cv2.imwrite(str(save_dir / f'{frame_name}_ocr_4_otsu_inv.png'), thresh2_big)
        
        # Method 5: Adaptive
        thresh3 = cv2.adaptiveThreshold(gray_center, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh3_big = cv2.resize(thresh3, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        text5 = pytesseract.image_to_string(thresh3_big, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456').strip()
        print(f"    Adaptive: '{text5}'")
        cv2.imwrite(str(save_dir / f'{frame_name}_ocr_5_adaptive.png'), thresh3_big)
        
        # Method 6: Adaptive inverse
        thresh4 = cv2.bitwise_not(cv2.adaptiveThreshold(gray_center, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
        thresh4_big = cv2.resize(thresh4, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        text6 = pytesseract.image_to_string(thresh4_big, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456').strip()
        print(f"    Adaptive inverse: '{text6}'")
        cv2.imwrite(str(save_dir / f'{frame_name}_ocr_6_adaptive_inv.png'), thresh4_big)
        
        # Save center region images
        center_4x = cv2.resize(center_region, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        gray_4x = cv2.resize(gray_center, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(save_dir / f'{frame_name}_center_original.png'), center_4x)
        cv2.imwrite(str(save_dir / f'{frame_name}_center_gray.png'), gray_4x)
    
    if 'major_alarm' in result['data']:
        print(f"  Major alarm: {result['data']['major_alarm']}")
    else:
        print(f"  Major alarm: FAILED")
    
    if 'minor_alarm' in result['data']:
        print(f"  Minor alarm: {result['data']['minor_alarm']}/5 segments")
    
    # Create visualization
    vis_frame = frame.copy()
    
    # Draw search region
    cv2.rectangle(vis_frame, (search_x_start, 0), (width, search_y_end), (255, 0, 0), 2)
    
    # Draw all detected circles
    if circles is not None:
        circles_np = np.uint16(np.around(circles))
        for circle in circles_np[0]:
            cx, cy, r = circle
            cv2.circle(vis_frame, (int(cx), int(cy)), int(r), (128, 128, 128), 1)
    
    # Draw selected clock
    if 'clock_center' in result['data']:
        cx, cy = result['data']['clock_center']
        r = result['data']['clock_radius']
        cv2.circle(vis_frame, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(vis_frame, (cx, cy), 3, (0, 255, 0), -1)
        
        # Draw inner region (70% of radius)
        inner_r = int(r * 0.7)
        cv2.circle(vis_frame, (cx, cy), inner_r, (0, 255, 255), 1)
        
        # Draw segment sample points
        for angle_deg in [-90, -18, 54, 126, 198]:
            angle_rad = np.radians(angle_deg)
            sample_radius = int(r * 0.7)
            sample_x = int(cx + sample_radius * np.cos(angle_rad))
            sample_y = int(cy + sample_radius * np.sin(angle_rad))
            cv2.circle(vis_frame, (sample_x, sample_y), 3, (255, 0, 255), -1)
    
    # Show visualizations
    cv2.imwrite(str(save_dir / f'{frame_name}_color_mask.png'), color_mask)
    cv2.imwrite(str(save_dir / f'{frame_name}_color_mask_region.png'), color_mask_region)
    cv2.imwrite(str(save_dir / f'{frame_name}_detection.png'), vis_frame)
    
    print(f"\nDebug images saved to {save_dir}/")
    print(f"  - {frame_name}_*.png")


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_alarm.py <frame_path> [frame_path2 ...] [--output-dir <dir>]")
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
