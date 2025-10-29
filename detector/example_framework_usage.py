"""
Example Usage of Modular Detector Framework

Demonstrates how to:
1. Register multiple detectors
2. Run them on single frames
3. Process batches of frames
4. Handle dependencies between detectors

Usage:
    python example_framework_usage.py <image_path>
    python example_framework_usage.py --session <session_folder>
"""

import sys
from pathlib import Path
import cv2
import json

from detector_framework import (
    DetectorRegistry,
    DetectorPipeline,
    DetectorConfig,
    DetectorType
)

from detector_adapters import (
    StructuralDetectorAdapter,
    TurnPhaseDetectorAdapter,
    ViewportDetectorAdapter
)


def example_single_frame(image_path: Path):
    """Example: Analyze a single frame with multiple detectors"""
    
    print("="*60)
    print("SINGLE FRAME ANALYSIS")
    print("="*60)
    print(f"Image: {image_path}")
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
    
    print(f"Size: {img.shape[1]}x{img.shape[0]}")
    
    # Create registry and register detectors
    registry = DetectorRegistry()
    
    # Register turn phase detector
    turn_detector = TurnPhaseDetectorAdapter()
    registry.register(turn_detector)
    
    # Register structural detector (with debug output)
    structural_detector = StructuralDetectorAdapter(debug=False)
    registry.register(structural_detector)
    
    # Register viewport detector (depends on structural)
    viewport_detector = ViewportDetectorAdapter()
    registry.register(viewport_detector)
    
    print(f"\nRegistered detectors: {registry.list_detectors()}")
    
    # Run all detectors
    print("\nRunning detection...")
    results = registry.detect_all(img)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for name, result in results.items():
        print(f"\n[{name}]")
        print(f"  Success: {result.success}")
        print(f"  Confidence: {result.confidence:.2f}")
        
        if result.success:
            if name == "turn_phase":
                phase = result.data.get('phase', 'unknown')
                print(f"  Phase: {phase}")
                print(f"  END TURN score: {result.data.get('end_turn_score', 0):.3f}")
                print(f"  Profile score: {result.data.get('profile_score', 0):.3f}")
            
            elif name == "structural":
                elements = result.data.get('elements', {})
                print(f"  Elements detected: {len(elements)}")
                for elem_name in list(elements.keys())[:5]:  # Show first 5
                    elem = elements[elem_name]
                    print(f"    - {elem_name}: bbox={elem['bbox']}, conf={elem['confidence']:.2f}")
                if len(elements) > 5:
                    print(f"    ... and {len(elements) - 5} more")
            
            elif name == "viewport":
                vp = result.data.get('viewport', {})
                print(f"  Viewport: ({vp.get('x', 0)}, {vp.get('y', 0)}) "
                      f"{vp.get('width', 0)}x{vp.get('height', 0)}")
        
        else:
            print(f"  Error: {result.error}")
    
    print("\n" + "="*60)


def example_session_analysis(session_dir: Path):
    """Example: Analyze a full session with pipeline"""
    
    print("="*60)
    print("SESSION ANALYSIS")
    print("="*60)
    print(f"Session: {session_dir}")
    
    frames_dir = session_dir / "frames"
    if not frames_dir.exists():
        print(f"Error: Frames directory not found: {frames_dir}")
        return
    
    # Get frame list
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    print(f"Frames: {len(frame_files)}")
    
    if len(frame_files) == 0:
        print("No frames found!")
        return
    
    # Create registry
    registry = DetectorRegistry()
    
    # Register detectors (no debug output for batch processing)
    registry.register(TurnPhaseDetectorAdapter())
    registry.register(StructuralDetectorAdapter(debug=False))
    
    print(f"Detectors: {registry.list_detectors()}")
    
    # Create pipeline
    pipeline = DetectorPipeline(registry)
    
    # Process session
    print("\nProcessing frames...")
    results_path = session_dir / "detection_results.json"
    results = pipeline.process_session(frame_files, save_to=results_path)
    
    # Generate summary
    summary = pipeline.get_summary()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total frames: {summary['total_frames']}")
    
    for detector_name, stats in summary['detectors'].items():
        print(f"\n[{detector_name}]")
        print(f"  Success rate: {stats['success_rate']*100:.1f}%")
        print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
    
    # Analyze turn phases if available
    if 'turn_phase' in results[0]:
        phase_counts = {}
        for frame_result in results:
            phase = frame_result['turn_phase'].data.get('phase', 'unknown')
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        print("\n[Turn Phase Distribution]")
        for phase, count in sorted(phase_counts.items()):
            pct = count / len(results) * 100
            print(f"  {phase:15s}: {count:4d} frames ({pct:5.1f}%)")
    
    print("\n" + "="*60)


def example_selective_detection(image_path: Path):
    """Example: Run only specific detectors"""
    
    print("="*60)
    print("SELECTIVE DETECTION")
    print("="*60)
    
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
    
    # Create registry with all detectors
    registry = DetectorRegistry()
    registry.register(TurnPhaseDetectorAdapter())
    registry.register(StructuralDetectorAdapter(debug=False))
    registry.register(ViewportDetectorAdapter())
    
    # Run only turn phase detection
    print("\nRunning only 'turn_phase' detector...")
    results = registry.detect_all(img, detectors=['turn_phase'])
    
    print(f"Results: {list(results.keys())}")
    phase = results['turn_phase'].data.get('phase', 'unknown')
    conf = results['turn_phase'].confidence
    print(f"Phase: {phase} (confidence: {conf:.2f})")
    
    print("\n" + "="*60)


def example_detector_management(image_path: Path):
    """Example: Enable/disable detectors dynamically"""
    
    print("="*60)
    print("DETECTOR MANAGEMENT")
    print("="*60)
    
    img = cv2.imread(str(image_path))
    if img is None:
        return
    
    registry = DetectorRegistry()
    
    # Register detectors
    turn_det = TurnPhaseDetectorAdapter()
    struct_det = StructuralDetectorAdapter(debug=False)
    
    registry.register(turn_det)
    registry.register(struct_det)
    
    # Run with both enabled
    print("\n1. Both detectors enabled:")
    results = registry.detect_all(img)
    print(f"   Results: {list(results.keys())}")
    
    # Disable structural
    struct_det.disable()
    print("\n2. Structural detector disabled:")
    results = registry.detect_all(img)
    print(f"   Results: {list(results.keys())}")
    
    # Re-enable
    struct_det.enable()
    print("\n3. Structural detector re-enabled:")
    results = registry.detect_all(img)
    print(f"   Results: {list(results.keys())}")
    
    print("\n" + "="*60)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExamples:")
        print("  python example_framework_usage.py test_image.png")
        print("  python example_framework_usage.py --session captures/20251021_200738")
        sys.exit(1)
    
    if sys.argv[1] == '--session':
        if len(sys.argv) < 3:
            print("Error: Session directory required")
            sys.exit(1)
        session_dir = Path(sys.argv[2])
        example_session_analysis(session_dir)
    
    else:
        image_path = Path(sys.argv[1])
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)
        
        # Run all examples
        example_single_frame(image_path)
        print("\n\n")
        example_selective_detection(image_path)
        print("\n\n")
        example_detector_management(image_path)


if __name__ == "__main__":
    main()
