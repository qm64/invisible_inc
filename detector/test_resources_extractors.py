#!/usr/bin/env python3
"""
Test script for power and credits extractors with parallel processing.
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

from detection.power_credits_anchor_detector import PowerCreditsAnchorDetector
from extraction.power_extractor import PowerExtractor
from extraction.credits_extractor import CreditsExtractor


def process_frame(args: Tuple[Path, int]) -> Dict:
    """
    Process a single frame and extract resources.
    
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
    anchor_detector = PowerCreditsAnchorDetector()
    power_extractor = PowerExtractor()
    credits_extractor = CreditsExtractor()
    
    # Detect anchor
    anchor_result = anchor_detector.detect(frame)
    
    # Extract power and credits
    power_result = power_extractor.detect(frame)
    credits_result = credits_extractor.detect(frame)
    
    return {
        'frame_num': frame_num,
        'anchor_detected': anchor_result.success,
        'anchor_confidence': anchor_result.confidence,
        'power_detected': power_result.success,
        'power_value': power_result.data.get('power_text') if power_result.success else None,
        'credits_detected': credits_result.success,
        'credits_value': credits_result.data.get('credits') if credits_result.success else None,
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
    anchor_detected = sum(1 for r in results if r.get('anchor_detected', False))
    power_extracted = sum(1 for r in results if r.get('power_detected', False))
    credits_extracted = sum(1 for r in results if r.get('credits_detected', False))
    both_extracted = sum(1 for r in results 
                        if r.get('power_detected', False) and r.get('credits_detected', False))
    
    # Print summary
    print("\nSession Summary:")
    print(f"Total frames:        {total_frames}")
    print(f"Anchor detected:     {anchor_detected:4d} ({100*anchor_detected/total_frames:.1f}%)")
    print(f"Power extracted:     {power_extracted:4d} ({100*power_extracted/total_frames:.1f}%)")
    print(f"Credits extracted:   {credits_extracted:4d} ({100*credits_extracted/total_frames:.1f}%)")
    print(f"Both extracted:      {both_extracted:4d} ({100*both_extracted/total_frames:.1f}%)")
    
    # Find frames with errors
    error_frames = [r for r in results if 'error' in r]
    if error_frames:
        print(f"\nFrames with errors: {len(error_frames)}")
        for r in error_frames[:5]:  # Show first 5
            print(f"  Frame {r['frame_num']:06d}: {r['error']}")
    
    # Find frames where anchor detected but extraction failed
    extraction_failures = [r for r in results 
                          if r.get('anchor_detected', False) 
                          and (not r.get('power_detected', False) 
                               or not r.get('credits_detected', False))]
    
    if extraction_failures:
        print(f"\nAnchor detected but extraction failed: {len(extraction_failures)} frames")
        print("Sample frames:")
        for r in extraction_failures[:5]:  # Show first 5
            status = []
            if not r.get('power_detected', False):
                status.append("power failed")
            if not r.get('credits_detected', False):
                status.append("credits failed")
            print(f"  Frame {r['frame_num']:06d}: {', '.join(status)}")


def main():
    parser = argparse.ArgumentParser(
        description='Test power and credits extractors with parallel processing'
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
