"""
Generate missing metadata files for incomplete capture sessions

Usage:
    python generate_missing_metadata.py captures/20251021_143022
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def generate_metadata(session_dir: Path):
    """Generate missing metadata.json file"""
    
    frames_dir = session_dir / "frames"
    if not frames_dir.exists():
        print(f"✗ Error: No frames directory found at {frames_dir}")
        return False
    
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        print(f"✗ Error: No frame files found in {frames_dir}")
        return False
    
    # Count input events if available
    inputs_file = session_dir / "inputs.jsonl"
    total_events = 0
    if inputs_file.exists():
        with open(inputs_file) as f:
            total_events = sum(1 for _ in f)
    
    # Estimate duration from frame count (assuming 2 fps default)
    duration = len(frame_files) / 2.0
    
    metadata = {
        'session_id': session_dir.name,
        'platform': 'unknown',
        'platform_release': 'unknown',
        'fps': 2.0,
        'title_patterns': ["Invisible", "Inc"],
        'process_patterns': ["Invisible", "InvisibleInc"],
        'start_time': 'unknown',
        'end_time': 'unknown',
        'total_frames': len(frame_files),
        'total_events': total_events,
        'duration_seconds': duration,
        'note': 'Generated metadata - original capture incomplete'
    }
    
    metadata_file = session_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Generated metadata.json")
    print(f"  Total frames: {len(frame_files)}")
    print(f"  Total events: {total_events}")
    print(f"  Est. duration: {duration:.1f}s")
    
    return True


def generate_frames_metadata(session_dir: Path):
    """Generate missing frames_metadata.jsonl file"""
    
    frames_dir = session_dir / "frames"
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    
    if not frame_files:
        return False
    
    frames_meta_file = session_dir / "frames_metadata.jsonl"
    
    # Get first frame timestamp from inputs if available
    base_timestamp = int(datetime.now().timestamp() * 1000)
    inputs_file = session_dir / "inputs.jsonl"
    if inputs_file.exists():
        with open(inputs_file) as f:
            first_line = f.readline()
            if first_line:
                first_event = json.loads(first_line)
                base_timestamp = first_event.get('timestamp', base_timestamp)
    
    # Generate metadata for each frame (2 fps = 500ms per frame)
    interval_ms = 500
    
    # Import PIL to read image dimensions
    try:
        from PIL import Image
        can_read_dimensions = True
    except ImportError:
        print("  Warning: PIL not available, dimensions will be 'unknown'")
        can_read_dimensions = False
    
    with open(frames_meta_file, 'w') as f:
        for i, frame_file in enumerate(frame_files):
            # Try to read actual image dimensions
            resolution = 'unknown'
            if can_read_dimensions:
                try:
                    with Image.open(frame_file) as img:
                        resolution = f"{img.width}x{img.height}"
                except:
                    pass
            
            metadata = {
                'frame_id': i,
                'timestamp': base_timestamp + (i * interval_ms),
                'filename': frame_file.name,
                'capture_mode': 'unknown',
                'window_title': 'unknown',
                'is_fullscreen': True,
                'resolution': resolution,
                'detection_method': 'regenerated'
            }
            f.write(json.dumps(metadata) + '\n')
    
    print(f"✓ Generated frames_metadata.jsonl")
    print(f"  Generated metadata for {len(frame_files)} frames")
    
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_missing_metadata.py <session_directory>")
        print("\nExample:")
        print("  python generate_missing_metadata.py captures/20251021_143022")
        sys.exit(1)
    
    session_dir = Path(sys.argv[1])
    if not session_dir.exists():
        print(f"✗ Error: Directory not found: {session_dir}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Generating Missing Metadata")
    print(f"{'='*60}\n")
    print(f"Session: {session_dir}\n")
    
    # Check what's missing
    metadata_file = session_dir / "metadata.json"
    frames_meta_file = session_dir / "frames_metadata.jsonl"
    
    metadata_exists = metadata_file.exists()
    frames_meta_exists = frames_meta_file.exists()
    
    print(f"Current status:")
    print(f"  metadata.json: {'✓ EXISTS' if metadata_exists else '✗ MISSING'}")
    print(f"  frames_metadata.jsonl: {'✓ EXISTS' if frames_meta_exists else '✗ MISSING'}")
    print()
    
    if metadata_exists and frames_meta_exists:
        print("✓ All metadata files exist. Nothing to do!")
        sys.exit(0)
    
    # Generate what's missing
    if not metadata_exists:
        print("Generating metadata.json...")
        if not generate_metadata(session_dir):
            sys.exit(1)
        print()
    
    if not frames_meta_exists:
        print("Generating frames_metadata.jsonl...")
        if not generate_frames_metadata(session_dir):
            sys.exit(1)
        print()
    
    print(f"{'='*60}")
    print("✓ Done! You can now use the viewer:")
    print(f"  python web_viewer.py {session_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()