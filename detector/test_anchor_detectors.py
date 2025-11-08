"""
Test script for Priority #1 Anchor Detectors

Demonstrates how to register and use the persistent UI anchor detectors
with the modular detector framework.

Usage:
    python test_anchor_detectors.py <image_path>
    python test_anchor_detectors.py --session <session_path>
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from detector_framework import DetectorRegistry
from anchor_detectors import (
    EndTurnDetector,
    HamburgerMenuDetector,
    TacticalViewDetector,
    PowerCreditsAnchorDetector,
    SecurityClockDetector
)


def visualize_results(image: np.ndarray, results: dict, output_path: str = None) -> np.ndarray:
    """Draw bounding boxes on image for all detected anchors"""
    
    vis_image = image.copy()
    
    # Color scheme for different detectors
    colors = {
        'end_turn': (255, 255, 0),        # Cyan
        'hamburger_menu': (0, 255, 255),  # Yellow
        'tactical_view': (255, 0, 255),    # Magenta
        'power_credits_anchor': (0, 255, 0),  # Green
        'security_clock': (0, 128, 255)    # Orange
    }
    
    labels = {
        'end_turn': 'END TURN',
        'hamburger_menu': 'MENU',
        'tactical_view': 'TACTICAL',
        'power_credits_anchor': 'PWR/CREDITS',
        'security_clock': 'SECURITY CLOCK'
    }
    
    for name, result in results.items():
        if not result.success:
            continue
        
        bbox = result.data.get('bbox')
        if bbox is None:
            continue
        
        x, y, w, h = bbox
        color = colors.get(name, (255, 255, 255))
        label = labels.get(name, name)
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
        
        # Draw label with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Background rectangle for text
        cv2.rectangle(vis_image, (x, y - text_h - 8), (x + text_w + 4, y), color, -1)
        
        # Text
        cv2.putText(vis_image, label, (x + 2, y - 4), font, font_scale, 
                    (0, 0, 0), thickness, cv2.LINE_AA)
        
        # Draw confidence score
        conf_text = f"{result.confidence:.2f}"
        cv2.putText(vis_image, conf_text, (x, y + h + 15), font, 0.4, 
                    color, 1, cv2.LINE_AA)
    
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"\nVisualization saved to: {output_path}")
    
    return vis_image


def print_results(results: dict):
    """Print detection results summary"""
    
    print("\n" + "="*60)
    print("ANCHOR DETECTION RESULTS")
    print("="*60)
    
    success_count = sum(1 for r in results.values() if r.success)
    total_count = len(results)
    
    for name, result in results.items():
        status = "✓" if result.success else "✗"
        conf = f"{result.confidence:.2f}" if result.success else "0.00"
        
        print(f"{status} {name:25s} confidence: {conf}")
        
        if result.success and 'bbox' in result.data:
            x, y, w, h = result.data['bbox']
            print(f"  └─ bbox: ({x}, {y}) size: {w}×{h}")
        elif not result.success and result.error:
            print(f"  └─ {result.error}")
    
    print("-"*60)
    print(f"Success rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")
    print("="*60 + "\n")


def test_single_image(image_path: str, debug: bool = False):
    """Test anchor detectors on a single image"""
    
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Image size: {image.shape[1]}×{image.shape[0]}")
    
    # Create registry and register all anchor detectors
    registry = DetectorRegistry()
    registry.register(EndTurnDetector())
    registry.register(HamburgerMenuDetector())
    registry.register(TacticalViewDetector())
    registry.register(PowerCreditsAnchorDetector())
    registry.register(SecurityClockDetector())
    
    print(f"\nRegistered {len(registry._detectors)} anchor detectors")
    print("Running detection...")
    
    # Run all detectors
    results = registry.detect_all(image, debug=debug)
    
    # Print results
    print_results(results)
    
    # Create visualization
    output_path = Path(image_path).stem + "_anchors.png"
    visualize_results(image, results, output_path)
    
    return results


def test_session(session_path: str, max_frames: int = 50, debug: bool = False):
    """Test anchor detectors on frames from a capture session"""
    
    session_dir = Path(session_path)
    frames_dir = session_dir / "frames"
    
    if not frames_dir.exists():
        print(f"Error: Frames directory not found: {frames_dir}")
        return
    
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    
    if not frame_files:
        print(f"Error: No frame files found in {frames_dir}")
        return
    
    # Limit frames for testing
    frame_files = frame_files[:max_frames]
    
    print(f"Testing on {len(frame_files)} frames from {session_path}")
    
    # Create registry
    registry = DetectorRegistry()
    registry.register(EndTurnDetector())
    registry.register(HamburgerMenuDetector())
    registry.register(TacticalViewDetector())
    registry.register(PowerCreditsAnchorDetector())
    registry.register(SecurityClockDetector())
    
    # Track success rates
    success_counts = {name: 0 for name in registry._detectors.keys()}
    total_frames = len(frame_files)
    
    print(f"\nProcessing frames...")
    
    for i, frame_path in enumerate(frame_files):
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        
        results = registry.detect_all(image, debug=False)
        
        for name, result in results.items():
            if result.success:
                success_counts[name] += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{total_frames} frames...")
    
    # Print summary
    print("\n" + "="*60)
    print(f"ANCHOR DETECTION SUMMARY ({total_frames} frames)")
    print("="*60)
    
    for name, count in success_counts.items():
        rate = 100 * count / total_frames
        print(f"{name:25s} {count:4d}/{total_frames} ({rate:5.1f}%)")
    
    print("="*60 + "\n")


def main():
    """Main entry point"""
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image:  python test_anchor_detectors.py <image_path>")
        print("  Session:       python test_anchor_detectors.py --session <session_path>")
        print("  Debug mode:    python test_anchor_detectors.py <image_path> --debug")
        return
    
    debug = '--debug' in sys.argv
    
    if '--session' in sys.argv:
        session_idx = sys.argv.index('--session')
        if session_idx + 1 < len(sys.argv):
            session_path = sys.argv[session_idx + 1]
            test_session(session_path, debug=debug)
        else:
            print("Error: --session requires a path argument")
    else:
        image_path = sys.argv[1]
        test_single_image(image_path, debug=debug)


if __name__ == '__main__':
    main()
