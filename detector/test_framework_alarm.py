#!/usr/bin/env python3
"""
Test script for framework-based alarm level detector.
Analyzes complete capture session with parallel processing.
"""

import argparse
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple
import cv2

from alarm_level_detector import HamburgerMenuDetector, AlarmLevelDetector


def process_frame(args: Tuple[Path, int]) -> Dict:
    """
    Process a single frame with framework-based detectors.
    
    Args:
        args: Tuple of (frame_path, frame_number)
        
    Returns:
        Dictionary with detection results
    """
    frame_path, frame_num = args
    
    # Load frame
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return {
            'frame_num': frame_num,
            'error': 'Failed to load frame'
        }
    
    # Initialize detectors (per-process)
    hamburger_detector = HamburgerMenuDetector()
    alarm_detector = AlarmLevelDetector()
    
    # Detect hamburger menu (dependency)
    hamburger_result = hamburger_detector.detect(frame)
    
    # Detect alarm level
    alarm_result = alarm_detector.detect(frame, {'hamburger_menu': hamburger_result})
    
    return {
        'frame_num': frame_num,
        'hamburger_detected': hamburger_result.success,
        'hamburger_confidence': hamburger_result.confidence,
        'alarm_detected': alarm_result.success,
        'alarm_confidence': alarm_result.confidence,
        'major_alarm': alarm_result.data.get('major_alarm'),
        'minor_alarm': alarm_result.data.get('minor_alarm'),
        'error': alarm_result.data.get('error') if not alarm_result.success else None
    }


def analyze_session(session_dir: Path, num_workers: int) -> None:
    """
    Analyze all frames in session directory with parallel processing.
    
    Args:
        session_dir: Path to session directory with frame_*.png files
        num_workers: Number of parallel workers
    """
    # Find all frame files
    frame_files = sorted(session_dir.glob('frame_*.png'))
    
    if not frame_files:
        print(f"Error: No frame files found in {session_dir}")
        return
    
    print(f"Analyzing {len(frame_files)} frames using {num_workers} workers...")
    
    # Prepare arguments
    frame_args = [(frame_path, int(frame_path.stem.split('_')[1])) 
                  for frame_path in frame_files]
    
    # Process in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_frame, frame_args)
    
    # Aggregate statistics
    total_frames = len(results)
    hamburger_detected = sum(1 for r in results if r.get('hamburger_detected', False))
    alarm_detected = sum(1 for r in results if r.get('alarm_detected', False))
    major_extracted = sum(1 for r in results if r.get('major_alarm') is not None)
    minor_extracted = sum(1 for r in results if r.get('minor_alarm') is not None)
    both_extracted = sum(1 for r in results 
                        if r.get('major_alarm') is not None 
                        and r.get('minor_alarm') is not None)
    
    # Print summary
    print("\nSession Summary:")
    print(f"Total frames:           {total_frames}")
    print(f"Hamburger detected:     {hamburger_detected:4d} ({100*hamburger_detected/total_frames:.1f}%)")
    print(f"Alarm detected:         {alarm_detected:4d} ({100*alarm_detected/total_frames:.1f}%)")
    print(f"Major extracted:        {major_extracted:4d} ({100*major_extracted/total_frames:.1f}%)")
    print(f"Minor extracted:        {minor_extracted:4d} ({100*minor_extracted/total_frames:.1f}%)")
    print(f"Both extracted:         {both_extracted:4d} ({100*both_extracted/total_frames:.1f}%)")
    
    # Major alarm distribution
    major_values = [r.get('major_alarm') for r in results if r.get('major_alarm') is not None]
    if major_values:
        print(f"\nMajor Alarm Distribution:")
        for level in range(7):
            count = major_values.count(level)
            if count > 0:
                print(f"  Level {level}: {count:4d} frames ({100*count/len(major_values):.1f}%)")
    
    # Minor alarm distribution
    minor_values = [r.get('minor_alarm') for r in results if r.get('minor_alarm') is not None]
    if minor_values:
        print(f"\nMinor Alarm Distribution:")
        for segments in range(6):
            count = minor_values.count(segments)
            if count > 0:
                print(f"  {segments} segments: {count:4d} frames ({100*count/len(minor_values):.1f}%)")
    
    # Average confidence
    alarm_confidences = [r['alarm_confidence'] for r in results 
                        if r.get('alarm_detected', False)]
    if alarm_confidences:
        avg_confidence = sum(alarm_confidences) / len(alarm_confidences)
        print(f"\nAverage confidence: {avg_confidence:.3f}")
    
    # Error analysis
    error_types = {}
    for r in results:
        if r.get('error'):
            error = r['error']
            error_types[error] = error_types.get(error, 0) + 1
    
    if error_types:
        print(f"\nError Analysis:")
        for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error}: {count} frames")
    
    # Hamburger failures
    hamburger_failures = [r for r in results if not r.get('hamburger_detected', False)]
    if hamburger_failures:
        print(f"\nHamburger menu not detected: {len(hamburger_failures)} frames")
        print("Sample frames:")
        for r in hamburger_failures[:10]:
            print(f"  Frame {r['frame_num']:06d}")


def main():
    parser = argparse.ArgumentParser(
        description='Test framework-based alarm level detector'
    )
    parser.add_argument(
        'session',
        type=Path,
        help='Path to session directory with frame_*.png files'
    )
    parser.add_argument(
        '--cpus',
        type=int,
        default=None,
        help=f'Number of parallel workers (default: {cpu_count()-2}, max: {cpu_count()})'
    )
    
    args = parser.parse_args()
    
    # Validate session directory
    if not args.session.exists():
        print(f"Error: Session directory not found: {args.session}")
        return 1
    
    if not args.session.is_dir():
        print(f"Error: Not a directory: {args.session}")
        return 1
    
    # Determine number of workers
    max_cpus = cpu_count()
    default_cpus = max(1, max_cpus - 2)
    
    if args.cpus is None:
        num_workers = default_cpus
    else:
        if args.cpus < 1:
            print(f"Error: --cpus must be at least 1")
            return 1
        if args.cpus > max_cpus:
            print(f"Warning: --cpus {args.cpus} exceeds available CPUs ({max_cpus}), using {max_cpus}")
            num_workers = max_cpus
        else:
            num_workers = args.cpus
    
    print(f"Using {num_workers} of {max_cpus} available CPUs")
    
    # Analyze session
    analyze_session(args.session, num_workers)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
