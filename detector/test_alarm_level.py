#!/usr/bin/env python3
"""
Test script for alarm level extractor with parallel processing.
Analyzes a complete capture session and reports extraction success rates.
"""

import argparse
import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alarm_level_extractor import AlarmLevelExtractor


def process_frame(args: Tuple[Path, int]) -> Dict:
    """
    Process a single frame and extract alarm level.
    
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
    
    # Initialize extractor (per-process)
    extractor = AlarmLevelExtractor()
    
    # Extract alarm level
    result = extractor.detect(frame)
    
    return {
        'frame_num': frame_num,
        'success': result['success'],
        'confidence': result['confidence'],
        'major_alarm': result['data'].get('major_alarm'),
        'minor_alarm': result['data'].get('minor_alarm'),
        'clock_found': 'clock_center' in result['data'],
    }


def analyze_session(session_dir: Path, num_workers: int) -> None:
    """
    Analyze all frames in a session directory using parallel processing.
    
    Args:
        session_dir: Path to session directory containing frame_*.png files
        num_workers: Number of parallel workers to use
    """
    # Find all frame files
    frame_files = sorted(session_dir.glob('frame_*.png'))
    
    if not frame_files:
        print(f"Error: No frame files found in {session_dir}")
        return
    
    print(f"Analyzing {len(frame_files)} frames using {num_workers} workers...")
    
    # Prepare arguments for parallel processing
    frame_args = [(frame_path, int(frame_path.stem.split('_')[1])) 
                  for frame_path in frame_files]
    
    # Process frames in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_frame, frame_args)
    
    # Aggregate statistics
    total_frames = len(results)
    clock_found = sum(1 for r in results if r.get('clock_found', False))
    major_extracted = sum(1 for r in results if r.get('major_alarm') is not None)
    minor_extracted = sum(1 for r in results if r.get('minor_alarm') is not None)
    both_extracted = sum(1 for r in results 
                        if r.get('major_alarm') is not None 
                        and r.get('minor_alarm') is not None)
    success = sum(1 for r in results if r.get('success', False))
    
    # Print summary
    print("\nSession Summary:")
    print(f"Total frames:        {total_frames}")
    print(f"Clock found:         {clock_found:4d} ({100*clock_found/total_frames:.1f}%)")
    print(f"Major extracted:     {major_extracted:4d} ({100*major_extracted/total_frames:.1f}%)")
    print(f"Minor extracted:     {minor_extracted:4d} ({100*minor_extracted/total_frames:.1f}%)")
    print(f"Both extracted:      {both_extracted:4d} ({100*both_extracted/total_frames:.1f}%)")
    print(f"Success (major+):    {success:4d} ({100*success/total_frames:.1f}%)")
    
    # Analyze major alarm distribution
    major_values = [r.get('major_alarm') for r in results if r.get('major_alarm') is not None]
    if major_values:
        print(f"\nMajor Alarm Distribution:")
        for level in range(7):
            count = major_values.count(level)
            if count > 0:
                print(f"  Level {level}: {count:4d} frames ({100*count/len(major_values):.1f}%)")
    
    # Analyze minor alarm distribution
    minor_values = [r.get('minor_alarm') for r in results if r.get('minor_alarm') is not None]
    if minor_values:
        print(f"\nMinor Alarm Distribution:")
        for segments in range(6):  # 0-5 segments
            count = minor_values.count(segments)
            if count > 0:
                print(f"  {segments} segments: {count:4d} frames ({100*count/len(minor_values):.1f}%)")
    
    # Average confidence
    confidences = [r['confidence'] for r in results if 'confidence' in r]
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        print(f"\nAverage confidence: {avg_confidence:.3f}")
    
    # Find frames with errors
    error_frames = [r for r in results if 'error' in r]
    if error_frames:
        print(f"\nFrames with errors: {len(error_frames)}")
        for r in error_frames[:5]:  # Show first 5
            print(f"  Frame {r['frame_num']:06d}: {r['error']}")
    
    # Find frames where clock found but extraction failed
    extraction_failures = [r for r in results 
                          if r.get('clock_found', False) 
                          and r.get('major_alarm') is None]
    
    if extraction_failures:
        print(f"\nClock found but major extraction failed: {len(extraction_failures)} frames")
        print("Sample frames:")
        for r in extraction_failures[:10]:  # Show first 10
            print(f"  Frame {r['frame_num']:06d}")


def main():
    parser = argparse.ArgumentParser(
        description='Test alarm level extractor with parallel processing'
    )
    parser.add_argument(
        'session',
        type=Path,
        help='Path to session directory containing frame_*.png files'
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
