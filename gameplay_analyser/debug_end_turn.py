"""
Debug script to find the correct HSV values for the "END TURN" button.
Run this on a frame where you know the button is visible (like frame 114).

Usage:
    python debug_end_turn.py captures/20251021_200738/frames/frame_0114.png
"""

import sys
from pathlib import Path
import cv2
import numpy as np

def analyze_end_turn_region(frame_path: Path):
    """Analyze the lower-right region to find END TURN button colors"""
    
    img = cv2.imread(str(frame_path))
    if img is None:
        print(f"Error: Could not load {frame_path}")
        return
    
    height, width = img.shape[:2]
    print(f"Image size: {width}x{height}")
    
    # Lower-right region
    h_start = int(height * 0.85)
    w_start = int(width * 0.80)
    
    region = img[h_start:, w_start:]
    region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    print(f"\nLower-right region: {region.shape[1]}x{region.shape[0]} pixels")
    print(f"Position: ({w_start}, {h_start}) to ({width}, {height})")
    
    # Get all unique colors in the region
    pixels = region_hsv.reshape(-1, 3)
    
    # Find the most common colors
    print("\n=== Color Analysis ===")
    print("Looking for teal/cyan button colors...")
    
    # Try different hue ranges
    ranges = [
        ("Cyan (80-100)", 80, 100),
        ("Teal (160-180)", 160, 180),
        ("Blue-Cyan (90-110)", 90, 110),
        ("Green-Cyan (70-90)", 70, 90),
    ]
    
    for name, h_low, h_high in ranges:
        mask = cv2.inRange(region_hsv,
                          np.array([h_low, 50, 50]),
                          np.array([h_high, 255, 255]))
        pixel_count = np.sum(mask > 0)
        percentage = (pixel_count / (region.shape[0] * region.shape[1])) * 100
        print(f"{name:20s}: {pixel_count:6d} pixels ({percentage:5.2f}%)")
    
    # Find pixels with high saturation (likely UI elements)
    high_sat_mask = region_hsv[:, :, 1] > 100
    high_sat_pixels = region_hsv[high_sat_mask]
    
    if len(high_sat_pixels) > 0:
        print(f"\n=== High Saturation Pixels (S > 100) ===")
        print(f"Count: {len(high_sat_pixels)}")
        print(f"Hue range: {high_sat_pixels[:, 0].min()}-{high_sat_pixels[:, 0].max()}")
        print(f"Sat range: {high_sat_pixels[:, 1].min()}-{high_sat_pixels[:, 1].max()}")
        print(f"Val range: {high_sat_pixels[:, 2].min()}-{high_sat_pixels[:, 2].max()}")
        
        # Histogram of hues
        hue_hist, _ = np.histogram(high_sat_pixels[:, 0], bins=18, range=(0, 180))
        print("\nHue distribution (0-180 in 10° bins):")
        for i, count in enumerate(hue_hist):
            if count > 100:  # Only show significant bins
                h_start = i * 10
                h_end = (i + 1) * 10
                print(f"  {h_start:3d}-{h_end:3d}°: {'█' * (count // 100)} ({count})")
    
    # Save debug images
    output_dir = frame_path.parent.parent / "debug"
    output_dir.mkdir(exist_ok=True)
    
    # Save the region
    cv2.imwrite(str(output_dir / "end_turn_region.png"), region)
    print(f"\n✓ Saved region to: {output_dir / 'end_turn_region.png'}")
    
    # Save HSV channels
    h, s, v = cv2.split(region_hsv)
    cv2.imwrite(str(output_dir / "hue.png"), h)
    cv2.imwrite(str(output_dir / "saturation.png"), s)
    cv2.imwrite(str(output_dir / "value.png"), v)
    print(f"✓ Saved HSV channels to: {output_dir}/")
    
    # Try to find the button by looking for teal rectangles
    print("\n=== Trying different detection strategies ===")
    
    # Strategy 1: Broad cyan range
    cyan_mask = cv2.inRange(region_hsv,
                           np.array([75, 80, 80]),
                           np.array([105, 255, 255]))
    cyan_pixels = np.sum(cyan_mask > 0)
    print(f"Broad cyan (H:75-105, S:80+, V:80+): {cyan_pixels} pixels ({(cyan_pixels/region.size*300):.2f}%)")
    
    # Strategy 2: Dark teal
    dark_teal_mask = cv2.inRange(region_hsv,
                                 np.array([75, 60, 40]),
                                 np.array([105, 255, 120]))
    dark_teal_pixels = np.sum(dark_teal_mask > 0)
    print(f"Dark teal (H:75-105, S:60+, V:40-120): {dark_teal_pixels} pixels ({(dark_teal_pixels/region.size*300):.2f}%)")
    
    cv2.imwrite(str(output_dir / "cyan_mask.png"), cyan_mask)
    cv2.imwrite(str(output_dir / "dark_teal_mask.png"), dark_teal_mask)
    print(f"✓ Saved masks to: {output_dir}/")

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_end_turn.py <frame_path>")
        print("\nExample:")
        print("  python debug_end_turn.py captures/20251021_200738/frames/frame_0114.png")
        sys.exit(1)
    
    frame_path = Path(sys.argv[1])
    if not frame_path.exists():
        print(f"Error: Frame not found: {frame_path}")
        sys.exit(1)
    
    print("="*60)
    print("END TURN Button Detection Debug")
    print("="*60)
    print(f"\nAnalyzing: {frame_path.name}")
    
    analyze_end_turn_region(frame_path)
    
    print("\n" + "="*60)
    print("Analysis complete! Check the debug/ folder for images.")
    print("="*60)

if __name__ == "__main__":
    main()