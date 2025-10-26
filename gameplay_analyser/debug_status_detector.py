#!/usr/bin/env python3
"""
Debug utility for calibrating game status detection.
Helps visualize regions and test OCR on sample frames.
"""

import cv2
import numpy as np
from pathlib import Path
import pytesseract
import argparse


def draw_regions(frame: np.ndarray, viewport: tuple, regions: dict) -> np.ndarray:
    """Draw all detection regions on the frame"""
    vx, vy, vw, vh = viewport
    annotated = frame.copy()
    
    colors = {
        'power_credits': (0, 255, 0),      # Green
        'turn_number': (255, 0, 0),        # Blue
        'alarm': (0, 0, 255),              # Red
        'agent_panel': (255, 255, 0),      # Cyan
        'incognita_switch': (255, 0, 255), # Magenta
    }
    
    for name, (rx, ry, rw, rh) in regions.items():
        # Convert relative to absolute coordinates
        x = int(vx + rx * vw)
        y = int(vy + ry * vh)
        w = int((rw - rx) * vw)
        h = int((rh - ry) * vh)
        
        color = colors.get(name, (128, 128, 128))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        
        # Add label
        label = name.replace('_', ' ').title()
        cv2.putText(annotated, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return annotated


def test_ocr_region(frame: np.ndarray, viewport: tuple, region_def: tuple, name: str):
    """Test OCR on a specific region and show results"""
    vx, vy, vw, vh = viewport
    rx, ry, rw, rh = region_def
    
    # Extract region
    x = int(vx + rx * vw)
    y = int(vy + ry * vh)
    w = int((rw - rx) * vw)
    h = int((rh - ry) * vh)
    
    region = frame[y:y+h, x:x+w]
    
    # Preprocess
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    denoised = cv2.fastNlMeansDenoising(thresh)
    
    # Upscale
    h_orig, w_orig = denoised.shape
    scale = 3
    upscaled = cv2.resize(denoised, (w_orig * scale, h_orig * scale), 
                         interpolation=cv2.INTER_CUBIC)
    
    # Run OCR
    ocr_config = '--psm 7 -c tessedit_char_whitelist=0123456789/'
    text = pytesseract.image_to_string(upscaled, config=ocr_config).strip()
    
    print(f"\n{name.upper()}")
    print(f"  Region: ({x}, {y}, {w}, {h})")
    print(f"  OCR Result: '{text}'")
    
    # Save debug images
    output_dir = Path('/tmp/status_debug')
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / f'{name}_original.png'), region)
    cv2.imwrite(str(output_dir / f'{name}_processed.png'), upscaled)
    
    print(f"  Saved: {output_dir}/{name}_*.png")


def detect_viewport_visual(frame: np.ndarray) -> tuple:
    """Detect viewport with visual feedback"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        frame_h, frame_w = frame.shape[:2]
        return (0, 0, frame_w, frame_h)
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Validate
    frame_h, frame_w = frame.shape[:2]
    if w < frame_w * 0.5 or h < frame_h * 0.5:
        return (0, 0, frame_w, frame_h)
    
    return (x, y, w, h)


def analyze_color_regions(frame: np.ndarray, viewport: tuple):
    """Analyze color distribution in key regions"""
    vx, vy, vw, vh = viewport
    
    # Test alarm region for red detection
    rx, ry, rw, rh = (0.85, 0.0, 1.0, 0.08)
    x = int(vx + rx * vw)
    y = int(vy + ry * vh)
    w = int((rw - rx) * vw)
    h = int((rh - ry) * vh)
    
    alarm_region = frame[y:y+h, x:x+w]
    hsv = cv2.cvtColor(alarm_region, cv2.COLOR_BGR2HSV)
    
    # Red masks
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = alarm_region.shape[0] * alarm_region.shape[1]
    red_ratio = red_pixels / total_pixels
    
    print(f"\nALARM COLOR ANALYSIS")
    print(f"  Red pixels: {red_pixels}/{total_pixels} ({red_ratio*100:.1f}%)")
    
    # Estimate alarm level
    if red_ratio < 0.05:
        level = 0
    elif red_ratio < 0.15:
        level = 1
    elif red_ratio < 0.25:
        level = 2
    elif red_ratio < 0.35:
        level = 3
    elif red_ratio < 0.45:
        level = 4
    elif red_ratio < 0.55:
        level = 5
    else:
        level = 6
    
    print(f"  Estimated alarm level: {level}")
    
    # Save debug image
    output_dir = Path('/tmp/status_debug')
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(output_dir / 'alarm_region.png'), alarm_region)
    cv2.imwrite(str(output_dir / 'alarm_red_mask.png'), red_mask)
    print(f"  Saved: {output_dir}/alarm_*.png")


def main():
    parser = argparse.ArgumentParser(description='Debug game status detection')
    parser.add_argument('frame_path', type=Path, help='Path to a frame image')
    parser.add_argument('--show-regions', action='store_true', help='Show all detection regions')
    parser.add_argument('--test-ocr', action='store_true', help='Test OCR on all text regions')
    parser.add_argument('--analyze-colors', action='store_true', help='Analyze color regions (alarm, etc)')
    parser.add_argument('--all', action='store_true', help='Run all debug tests')
    
    args = parser.parse_args()
    
    if not args.frame_path.exists():
        print(f"Error: Frame not found: {args.frame_path}")
        return 1
    
    # Load frame
    frame = cv2.imread(str(args.frame_path))
    if frame is None:
        print(f"Error: Could not load frame: {args.frame_path}")
        return 1
    
    print(f"Loaded frame: {args.frame_path}")
    print(f"Dimensions: {frame.shape[1]}x{frame.shape[0]}")
    
    # Detect viewport
    viewport = detect_viewport_visual(frame)
    vx, vy, vw, vh = viewport
    print(f"Detected viewport: ({vx}, {vy}) size {vw}x{vh}")
    
    # Region definitions
    regions = {
        'power_credits': (0.0, 0.0, 0.15, 0.08),
        'turn_number': (0.42, 0.0, 0.58, 0.06),
        'alarm': (0.85, 0.0, 1.0, 0.08),
        'agent_panel': (0.0, 0.85, 0.25, 1.0),
        'incognita_switch': (0.0, 0.3, 0.08, 0.5),
    }
    
    # Run requested tests
    run_all = args.all
    
    if args.show_regions or run_all:
        print("\n=== SHOWING REGIONS ===")
        annotated = draw_regions(frame, viewport, regions)
        output_path = Path('/tmp/status_debug/regions_annotated.png')
        output_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved annotated frame: {output_path}")
    
    if args.test_ocr or run_all:
        print("\n=== TESTING OCR ===")
        test_ocr_region(frame, viewport, regions['power_credits'], 'power_credits')
        test_ocr_region(frame, viewport, regions['turn_number'], 'turn_number')
    
    if args.analyze_colors or run_all:
        print("\n=== ANALYZING COLORS ===")
        analyze_color_regions(frame, viewport)
    
    if not any([args.show_regions, args.test_ocr, args.analyze_colors, args.all]):
        print("\nNo tests specified. Use --help to see options.")
        print("Quick start: --all")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
