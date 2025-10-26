#!/usr/bin/env python3
"""
Test script showing alarm level extraction integration
Demonstrates both standalone and structural_detector integration patterns
"""

import cv2
import sys

# Method 1: Standalone usage
def test_standalone():
    from alarm_level_extractor import extract_alarm_level_viewport_relative
    
    test_frames = [
        ('frame_000046.png', 0),  # Expected: level 0 (yellow)
        ('frame_000384.png', 4),  # Expected: level 4 (orange)
        ('frame_000478.png', 6),  # Expected: level 6 (red)
    ]
    
    print("=" * 60)
    print("STANDALONE ALARM LEVEL EXTRACTION TEST")
    print("=" * 60)
    
    for filename, expected in test_frames:
        img = cv2.imread(f'/mnt/user-data/uploads/{filename}')
        if img is None:
            print(f"⚠ Could not load {filename}")
            continue
        
        # Works without viewport
        alarm_level = extract_alarm_level_viewport_relative(img, viewport=None, debug=False)
        
        status = "✓" if alarm_level == expected else "✗"
        print(f"{status} {filename}: Level {alarm_level} (expected {expected})")
    
    print()


# Method 2: Integrated with structural_detector (simulated)
def test_integrated():
    """
    Shows how to use with structural_detector.py
    
    In your actual code:
    detector = StructuralDetector(debug=True)
    detector.detect_anchors(image)
    viewport = detector.infer_viewport()
    alarm_level = detector.extract_alarm_level(image)  # New method
    """
    
    print("=" * 60)
    print("INTEGRATED USAGE EXAMPLE")
    print("=" * 60)
    print("""
# In your game analysis loop:

import cv2
from structural_detector import StructuralDetector

detector = StructuralDetector(debug=True)

for frame_path in frame_paths:
    image = cv2.imread(frame_path)
    
    # Detect UI elements
    detector.detect_anchors(image)
    viewport = detector.infer_viewport()
    
    # Extract game state
    power = detector.extract_power_value(image)
    credits = detector.extract_credits_value(image)
    alarm_level = detector.extract_alarm_level(image)  # ← NEW METHOD
    ap_values = detector.extract_agent_ap_values(image)
    
    print(f"Frame: {frame_path}")
    print(f"  Power: {power}")
    print(f"  Credits: {credits}")
    print(f"  Alarm Level: {alarm_level}")  # 0-6
    print(f"  AP Values: {ap_values}")
""")
    print()


# Method 3: Batch processing example
def test_batch():
    from alarm_level_extractor import extract_alarm_level_viewport_relative
    
    print("=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)
    
    test_frames = [
        '/mnt/user-data/uploads/frame_000046.png',
        '/mnt/user-data/uploads/frame_000384.png',
        '/mnt/user-data/uploads/frame_000478.png',
    ]
    
    results = []
    
    for frame_path in test_frames:
        img = cv2.imread(frame_path)
        if img is None:
            continue
        
        alarm_level = extract_alarm_level_viewport_relative(img, viewport=None, debug=False)
        results.append({
            'frame': frame_path.split('/')[-1],
            'alarm_level': alarm_level,
        })
    
    print("\nBatch Results:")
    for result in results:
        print(f"  {result['frame']}: Level {result['alarm_level']}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ALARM LEVEL EXTRACTION - INTEGRATION TESTS")
    print("=" * 60 + "\n")
    
    try:
        test_standalone()
        test_integrated()
        test_batch()
        
        print("=" * 60)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Copy extract_alarm_level() method into structural_detector.py")
        print("2. Update version to 1.3.1 with changelog entry")
        print("3. Test on your full capture session (711 frames)")
        print("4. Monitor success rate (should be ~97% matching viewport rate)")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
