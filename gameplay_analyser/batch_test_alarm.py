#!/usr/bin/env python3
"""
Batch test alarm level extraction on full capture session
"""

import cv2
import os
import sys
from collections import Counter
from alarm_level_extractor import extract_alarm_level_viewport_relative

def test_session(session_dir):
    frames_dir = os.path.join(session_dir, 'frames')
    
    if not os.path.exists(frames_dir):
        print(f"Error: {frames_dir} not found")
        sys.exit(1)
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.png')])
    
    print(f"Found {len(frame_files)} frames in {frames_dir}")
    print("Processing...\n")
    
    results = {
        'success': 0,
        'failed': 0,
        'levels': Counter(),
        'failed_frames': []
    }
    
    # Process every 10th frame for speed (or all frames for accuracy)
    sample_rate = 1  # Change to 10 for faster testing
    
    for i, frame_file in enumerate(frame_files[::sample_rate]):
        frame_path = os.path.join(frames_dir, frame_file)
        img = cv2.imread(frame_path)
        
        if img is None:
            continue
        
        alarm_level = extract_alarm_level_viewport_relative(img, viewport=None, debug=False)
        
        if alarm_level is not None:
            results['success'] += 1
            results['levels'][alarm_level] += 1
        else:
            results['failed'] += 1
            results['failed_frames'].append(frame_file)
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            current_rate = results['success'] / (results['success'] + results['failed']) * 100
            print(f"Processed {i + 1} frames... Success rate: {current_rate:.1f}%")
    
    return results, len(frame_files[::sample_rate])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_test_alarm.py <session_directory>")
        print("Example: python batch_test_alarm.py ~/invisibleinc_analysis/captures/session_20241023_225441")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    
    print("=" * 70)
    print("ALARM LEVEL EXTRACTION - BATCH TEST")
    print("=" * 70)
    print()
    
    results, total = test_session(session_dir)
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total frames tested: {total}")
    print(f"Successful extractions: {results['success']} ({results['success']/total*100:.1f}%)")
    print(f"Failed extractions: {results['failed']} ({results['failed']/total*100:.1f}%)")
    print()
    print("Alarm level distribution:")
    for level in sorted(results['levels'].keys()):
        count = results['levels'][level]
        pct = count / results['success'] * 100
        print(f"  Level {level}: {count} frames ({pct:.1f}%)")
    print()
    
    if results['failed'] > 0 and results['failed'] < 20:
        print("Failed frames:")
        for frame in results['failed_frames']:
            print(f"  - {frame}")
        print()
    
    # Compare to viewport detection rate
    expected_rate = 97.0
    actual_rate = results['success'] / total * 100
    
    print("=" * 70)
    if actual_rate >= expected_rate - 5:
        print(f"✓ SUCCESS: {actual_rate:.1f}% is close to viewport rate (~{expected_rate}%)")
    else:
        print(f"⚠ NOTE: {actual_rate:.1f}% is below viewport rate (~{expected_rate}%)")
    print("=" * 70)
