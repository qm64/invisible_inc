#!/usr/bin/env python3
"""
Batch process frames with structural detector
Properly handles temporal viewport inference across sequential frames

Usage:
    python batch_structural_detector.py <frames_directory> [output_dir]
"""

import sys
import cv2
from pathlib import Path
from structural_detector import StructuralDetector
import json

def process_frame_sequence(frames_dir: Path, output_dir: Path = None, debug: bool = False):
    """
    Process a sequence of frames maintaining temporal consistency
    
    Args:
        frames_dir: Directory containing frame_*.png files
        output_dir: Directory for output (default: frames_dir/detections)
        debug: Enable debug output
    """
    
    if output_dir is None:
        output_dir = frames_dir / "detections"
    
    output_dir.mkdir(exist_ok=True)
    
    # Get sorted list of frame files
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    
    if not frame_files:
        print(f"No frame files found in {frames_dir}")
        return
    
    print(f"Processing {len(frame_files)} frames from {frames_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Reset temporal cache at start of session
    StructuralDetector.reset_temporal_cache()
    
    # Create detector instance
    detector = StructuralDetector(debug=debug)
    
    # Track statistics
    stats = {
        'total_frames': len(frame_files),
        'viewport_detected': 0,
        'viewport_from_power': 0,
        'viewport_from_anchors': 0,
        'viewport_temporal': 0,
        'viewport_failed': 0
    }
    
    for i, frame_path in enumerate(frame_files):
        frame_num = frame_path.stem.replace('frame_', '')
        
        if debug:
            print(f"\n{'='*70}")
            print(f"Processing {frame_path.name} ({i+1}/{len(frame_files)})")
            print(f"{'='*70}")
        elif i % 50 == 0:
            print(f"Progress: {i}/{len(frame_files)} frames...")
        
        # Load image
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"ERROR: Could not load {frame_path}")
            continue
        
        # Detect elements
        elements = detector.detect_anchors(image)
        
        # Infer viewport
        viewport = detector.infer_viewport()
        
        if viewport:
            stats['viewport_detected'] += 1
            
            # Determine which method was used (check debug output)
            if "power_text" in elements:
                stats['viewport_from_power'] += 1
            elif "hamburger_menu" in elements and "tactical_view" in elements:
                stats['viewport_from_anchors'] += 1
            else:
                stats['viewport_temporal'] += 1
        else:
            stats['viewport_failed'] += 1
            if not debug:
                print(f"  WARNING: Frame {frame_num} - viewport detection failed")
        
        # Save visualization
        vis_image = detector.visualize_detections(image)
        vis_path = output_dir / f"{frame_path.stem}_detected.png"
        cv2.imwrite(str(vis_path), vis_image)
        
        # Save JSON
        json_path = output_dir / f"{frame_path.stem}_detected.json"
        detector.save_detections(str(json_path))
    
    # Print summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Viewport detected: {stats['viewport_detected']}/{stats['total_frames']} ({stats['viewport_detected']/stats['total_frames']*100:.1f}%)")
    print(f"  - From power/credits: {stats['viewport_from_power']}")
    print(f"  - From hamburger+tactical: {stats['viewport_from_anchors']}")
    print(f"  - From temporal cache: {stats['viewport_temporal']}")
    print(f"Viewport failed: {stats['viewport_failed']}")
    print(f"\nOutput saved to: {output_dir}")
    print(f"{'='*70}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_structural_detector.py <frames_directory> [output_dir] [--debug]")
        print("Example: python batch_structural_detector.py captures/20251022_201216/frames")
        sys.exit(1)
    
    frames_dir = Path(sys.argv[1])
    
    if not frames_dir.exists():
        print(f"ERROR: Directory not found: {frames_dir}")
        sys.exit(1)
    
    output_dir = None
    debug = False
    
    if len(sys.argv) > 2:
        if sys.argv[2] == '--debug':
            debug = True
        else:
            output_dir = Path(sys.argv[2])
            if len(sys.argv) > 3 and sys.argv[3] == '--debug':
                debug = True
    
    process_frame_sequence(frames_dir, output_dir, debug)

if __name__ == "__main__":
    main()
