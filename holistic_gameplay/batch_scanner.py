#!/usr/bin/env env python3
"""
Batch Frame Scanner for Invisible Inc Gameplay Analysis
Version 1.0.0

Scans thousands of gameplay screenshots to extract color signatures,
spatial patterns, and structural features for holistic game state analysis.
"""

import argparse
import sys
import sqlite3
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
import time

# ============================================================================
# DATABASE SCHEMA
# ============================================================================

def init_database(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite database with schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Frames table - one row per frame
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS frames (
            path TEXT PRIMARY KEY,
            scan_timestamp TEXT,
            width INTEGER,
            height INTEGER,
            aspect_ratio REAL,
            is_game_frame INTEGER,
            mean_brightness REAL,
            has_cyan INTEGER,
            has_yellow INTEGER,
            has_red INTEGER,
            edge_density REAL
        )
    """)
    
    # Color signatures - HSV histogram data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS color_signatures (
            path TEXT,
            hue_bin INTEGER,
            sat_bin INTEGER,
            val_bin INTEGER,
            pixel_count INTEGER,
            FOREIGN KEY(path) REFERENCES frames(path)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_color_path ON color_signatures(path)
    """)
    
    # Spatial data - 10x10 grid color presence
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS spatial_data (
            path TEXT,
            grid_x INTEGER,
            grid_y INTEGER,
            has_cyan INTEGER,
            has_yellow INTEGER,
            has_orange INTEGER,
            has_red INTEGER,
            has_white INTEGER,
            edge_density REAL,
            FOREIGN KEY(path) REFERENCES frames(path)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_spatial_path ON spatial_data(path)
    """)
    
    # Metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    
    conn.commit()
    return conn

# ============================================================================
# COLOR DETECTION
# ============================================================================

def detect_color_presence(hsv_image: np.ndarray) -> Dict[str, bool]:
    """Detect presence of key UI colors in image."""
    
    # Color ranges in HSV
    color_ranges = {
        'cyan': [(85, 100, 100), (100, 255, 255)],
        'yellow': [(20, 100, 100), (30, 255, 255)],
        'orange': [(10, 100, 100), (20, 255, 255)],
        'red': [(0, 100, 100), (10, 255, 255)],
        'white': [(0, 0, 200), (180, 30, 255)]
    }
    
    presence = {}
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
        pixel_count = cv2.countNonZero(mask)
        # Consider present if >0.1% of pixels
        presence[color_name] = pixel_count > (hsv_image.shape[0] * hsv_image.shape[1] * 0.001)
    
    return presence

def detect_spatial_colors(image: np.ndarray, hsv_image: np.ndarray, grid_size: int = 10) -> List[Dict]:
    """Detect color presence in grid cells."""
    h, w = image.shape[:2]
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    spatial_data = []
    
    for grid_y in range(grid_size):
        for grid_x in range(grid_size):
            y1 = grid_y * cell_h
            y2 = min((grid_y + 1) * cell_h, h)
            x1 = grid_x * cell_w
            x2 = min((grid_x + 1) * cell_w, w)
            
            cell_hsv = hsv_image[y1:y2, x1:x2]
            cell_gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            
            # Detect colors in this cell
            presence = detect_color_presence(cell_hsv)
            
            # Edge density in this cell
            edges = cv2.Canny(cell_gray, 50, 150)
            edge_density = cv2.countNonZero(edges) / (edges.shape[0] * edges.shape[1])
            
            spatial_data.append({
                'grid_x': grid_x,
                'grid_y': grid_y,
                'has_cyan': presence['cyan'],
                'has_yellow': presence['yellow'],
                'has_orange': presence['orange'],
                'has_red': presence['red'],
                'has_white': presence['white'],
                'edge_density': edge_density
            })
    
    return spatial_data

# ============================================================================
# HSV HISTOGRAM
# ============================================================================

def compute_hsv_histogram(hsv_image: np.ndarray, h_bins: int = 36, s_bins: int = 3, v_bins: int = 3) -> List[Dict]:
    """Compute HSV histogram with binning."""
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [h_bins, s_bins, v_bins], 
                        [0, 180, 0, 256, 0, 256])
    
    histogram_data = []
    for h in range(h_bins):
        for s in range(s_bins):
            for v in range(v_bins):
                count = int(hist[h, s, v])
                if count > 0:  # Only store non-zero bins
                    histogram_data.append({
                        'hue_bin': h,
                        'sat_bin': s,
                        'val_bin': v,
                        'pixel_count': count
                    })
    
    return histogram_data

# ============================================================================
# FRAME CLASSIFICATION
# ============================================================================

def classify_frame(image: np.ndarray, hsv_image: np.ndarray) -> Tuple[bool, Dict]:
    """Classify if frame is a game window and extract metrics."""
    h, w = image.shape[:2]
    
    # Basic metrics
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    aspect_ratio = w / h
    
    # Color presence
    color_presence = detect_color_presence(hsv_image)
    
    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = cv2.countNonZero(edges) / (edges.shape[0] * edges.shape[1])
    
    # Classification heuristics
    # Game frames typically:
    # - Have aspect ratio 1.5-2.5
    # - Mean brightness 20-80 (not black screen, not bright desktop)
    # - Some cyan present (UI elements)
    # - Edge density > 0.02 (has UI structure)
    
    is_game = (
        1.5 < aspect_ratio < 2.5 and
        20 < mean_brightness < 80 and
        edge_density > 0.02
    )
    
    metrics = {
        'mean_brightness': mean_brightness,
        'aspect_ratio': aspect_ratio,
        'has_cyan': color_presence['cyan'],
        'has_yellow': color_presence['yellow'],
        'has_red': color_presence['red'],
        'edge_density': edge_density
    }
    
    return is_game, metrics

# ============================================================================
# FRAME PROCESSOR
# ============================================================================

def process_frame(frame_path: str) -> Optional[Dict]:
    """Process a single frame and extract all features."""
    try:
        # Load image
        image = cv2.imread(frame_path)
        if image is None:
            return None
        
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        
        # Classify frame
        is_game, metrics = classify_frame(image, hsv_image)
        
        # Extract features
        frame_data = {
            'path': frame_path,
            'width': w,
            'height': h,
            'is_game_frame': is_game,
            **metrics
        }
        
        # HSV histogram
        color_signature = compute_hsv_histogram(hsv_image)
        
        # Spatial data
        spatial_data = detect_spatial_colors(image, hsv_image)
        
        return {
            'frame': frame_data,
            'color_signature': color_signature,
            'spatial_data': spatial_data
        }
        
    except Exception as e:
        print(f"Error processing {frame_path}: {e}", file=sys.stderr)
        return None

# ============================================================================
# BATCH PROCESSING
# ============================================================================

def get_unprocessed_frames(conn: sqlite3.Connection, all_frames: List[str], force: bool) -> List[str]:
    """Get list of frames that haven't been processed yet."""
    if force:
        return all_frames
    
    cursor = conn.cursor()
    cursor.execute("SELECT path FROM frames")
    processed = set(row[0] for row in cursor.fetchall())
    
    return [f for f in all_frames if f not in processed]

def batch_process(frame_paths: List[str], db_path: str, workers: int, force: bool):
    """Process frames in parallel and store results."""
    conn = init_database(db_path)
    
    # Get unprocessed frames
    to_process = get_unprocessed_frames(conn, frame_paths, force)
    
    if not to_process:
        print(f"All {len(frame_paths)} frames already processed.")
        return
    
    print(f"Processing {len(to_process)} frames with {workers} workers...")
    start_time = time.time()
    
    processed_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_frame, path): path for path in to_process}
        
        for future in as_completed(futures):
            result = future.result()
            
            if result is None:
                error_count += 1
                continue
            
            # Store in database
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            
            # Insert frame data
            frame = result['frame']
            cursor.execute("""
                INSERT OR REPLACE INTO frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                frame['path'], timestamp, frame['width'], frame['height'],
                frame['aspect_ratio'], frame['is_game_frame'],
                frame['mean_brightness'], frame['has_cyan'],
                frame['has_yellow'], frame['has_red'], frame['edge_density']
            ))
            
            # Insert color signature
            for entry in result['color_signature']:
                cursor.execute("""
                    INSERT INTO color_signatures VALUES (?, ?, ?, ?, ?)
                """, (frame['path'], entry['hue_bin'], entry['sat_bin'],
                      entry['val_bin'], entry['pixel_count']))
            
            # Insert spatial data
            for entry in result['spatial_data']:
                cursor.execute("""
                    INSERT INTO spatial_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (frame['path'], entry['grid_x'], entry['grid_y'],
                      entry['has_cyan'], entry['has_yellow'], entry['has_orange'],
                      entry['has_red'], entry['has_white'], entry['edge_density']))
            
            conn.commit()
            processed_count += 1
            
            if processed_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = processed_count / elapsed
                print(f"Processed {processed_count}/{len(to_process)} frames ({fps:.1f} fps)")
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({processed_count/elapsed:.1f} fps)")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors: {error_count}")
    
    # Update metadata
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO metadata VALUES ('last_scan', ?)
    """, (datetime.now().isoformat(),))
    cursor.execute("""
        INSERT OR REPLACE INTO metadata VALUES ('total_frames', ?)
    """, (str(processed_count + error_count),))
    conn.commit()
    conn.close()

# ============================================================================
# INPUT PARSING
# ============================================================================

def collect_frame_paths(args) -> List[str]:
    """Collect frame paths from all input sources."""
    paths = []
    
    # From file list
    if args.file_list:
        if args.file_list == '-':
            # Read from stdin
            for line in sys.stdin:
                paths.append(line.strip())
        else:
            # Read from file
            with open(args.file_list, 'r') as f:
                for line in f:
                    paths.append(line.strip())
    
    # From direct arguments
    if args.frames:
        paths.extend(args.frames)
    
    # Filter out empty lines and validate files exist
    valid_paths = []
    for path in paths:
        path = path.strip()
        if path and Path(path).exists():
            valid_paths.append(path)
    
    return valid_paths

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Batch scan gameplay frames for holistic pattern analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From file list
  %(prog)s -f frame_list.txt
  
  # From stdin
  find captures -name "frame_*.png" | %(prog)s -f -
  
  # Direct arguments
  %(prog)s frame_001.png frame_002.png
  
  # Combined
  %(prog)s -f list.txt frame_extra.png
  
  # Force reprocess all
  %(prog)s -f frame_list.txt --force
        """
    )
    
    parser.add_argument('-f', '--file-list', metavar='FILE',
                        help='File containing frame paths (one per line), or "-" for stdin')
    parser.add_argument('frames', nargs='*',
                        help='Frame paths as direct arguments')
    parser.add_argument('-o', '--output', default='analysis.db',
                        help='Output database path (default: analysis.db)')
    parser.add_argument('-w', '--workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    parser.add_argument('--force', action='store_true',
                        help='Reprocess all frames (ignore existing data)')
    
    args = parser.parse_args()
    
    # Collect frame paths
    frame_paths = collect_frame_paths(args)
    
    if not frame_paths:
        print("Error: No valid frame paths provided", file=sys.stderr)
        print("Use -f to specify a file list, or provide paths as arguments", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(frame_paths)} frames to analyze")
    
    # Process frames
    batch_process(frame_paths, args.output, args.workers, args.force)
    
    print(f"\nResults written to {args.output}")
    print("\nNext steps:")
    print("  1. Run generate_reports.py to create visualizations")
    print("  2. Query analysis.db directly for custom analysis")

if __name__ == '__main__':
    main()
