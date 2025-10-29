"""
Turn Phase Detector - Parallel Processing Version

Analyzes Invisible Inc gameplay captures to detect turn phases.
Uses multiprocessing for faster frame analysis.

Usage:
    python turn_phase_detector.py <session_folder>
"""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

import cv2
import numpy as np


def analyze_single_frame(frame_path: Path, frame_idx: int) -> Dict:
    """
    Analyze a single frame for turn phase detection.
    Primary strategy: Profile pic in lower-left indicates player control.
    This function is called in parallel by worker processes.
    """
    try:
        img = cv2.imread(str(frame_path))
        if img is None:
            return {
                'frame': frame_idx,
                'phase': 'unknown',
                'confidence': 0.0,
                'features': {}
            }
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height, width = img.shape[:2]
        
        features = {}
        
        # PRIMARY INDICATOR: "END TURN" button in lower-right corner
        # ROI: Bottom 15% of height, right 20% of width
        end_turn_h_start = int(height * 0.85)
        end_turn_w_start = int(width * 0.80)
        end_turn_region = img[end_turn_h_start:, end_turn_w_start:]
        end_turn_hsv = hsv[end_turn_h_start:, end_turn_w_start:]
        
        # Detect cyan button color (H: 85-100, high S and V)
        # The button shows up strongly in the 90-100° hue range
        cyan_mask = cv2.inRange(end_turn_hsv,
                               np.array([85, 80, 80]),   # Cyan hue range
                               np.array([100, 255, 255]))
        cyan_pixels = np.sum(cyan_mask > 0)
        end_turn_area = end_turn_region.shape[0] * end_turn_region.shape[1]
        features['end_turn_cyan_ratio'] = cyan_pixels / end_turn_area if end_turn_area > 0 else 0
        
        # Also check for darker cyan (the button background)
        dark_cyan_mask = cv2.inRange(end_turn_hsv,
                                    np.array([85, 60, 40]),
                                    np.array([100, 255, 120]))
        dark_cyan_pixels = np.sum(dark_cyan_mask > 0)
        features['end_turn_dark_ratio'] = dark_cyan_pixels / end_turn_area if end_turn_area > 0 else 0
        
        # Combined "END TURN" presence score
        end_turn_score = features['end_turn_cyan_ratio'] + features['end_turn_dark_ratio']
        # Button is ~10% of region, so threshold at 3% to be safe
        features['end_turn_present'] = end_turn_score > 0.03
        
        # SECONDARY INDICATOR: Profile pic in lower-left corner (for player_normal vs player_action)
        # ROI: Bottom 20% of height, left 15% of width
        profile_h_start = int(height * 0.80)
        profile_w_end = int(width * 0.15)
        profile_region = img[profile_h_start:, :profile_w_end]
        profile_hsv = hsv[profile_h_start:, :profile_w_end]
        
        # Detect skin tones (orange/brown hues typical in character portraits)
        skin_mask = cv2.inRange(profile_hsv,
                               np.array([0, 20, 50]),    # Low saturation browns
                               np.array([30, 255, 255])) # Orange tones
        skin_pixels = np.sum(skin_mask > 0)
        profile_area = profile_region.shape[0] * profile_region.shape[1]
        features['profile_skin_ratio'] = skin_pixels / profile_area if profile_area > 0 else 0
        
        # Detect high saturation colors (colorful UI elements in portraits)
        sat_mask = cv2.inRange(profile_hsv,
                              np.array([0, 100, 50]),
                              np.array([180, 255, 255]))
        sat_pixels = np.sum(sat_mask > 0)
        features['profile_color_ratio'] = sat_pixels / profile_area if profile_area > 0 else 0
        
        # Edge density in profile region (portraits have defined edges)
        profile_gray = cv2.cvtColor(profile_region, cv2.COLOR_BGR2GRAY)
        profile_edges = cv2.Canny(profile_gray, 50, 150)
        features['profile_edge_ratio'] = np.sum(profile_edges > 0) / profile_area if profile_area > 0 else 0
        
        # Overall brightness in profile area
        features['profile_brightness'] = float(np.mean(profile_gray))
        
        # Profile presence score (combine multiple signals)
        profile_score = (
            features['profile_skin_ratio'] * 2.0 +      # Skin tones weighted heavily
            features['profile_color_ratio'] * 1.5 +     # Colorful UI
            features['profile_edge_ratio'] * 1.0        # Defined edges
        )
        features['profile_presence_score'] = profile_score
        
        # SECONDARY INDICATORS for disambiguating action vs opponent
        
        # Check for "Enemy Activity" banner (center screen)
        banner_h_start = int(height * 0.4)
        banner_h_end = int(height * 0.6)
        banner_region = img[banner_h_start:banner_h_end, :]
        banner_hsv = hsv[banner_h_start:banner_h_end, :]
        
        # Red banner typically has high red saturation
        red_mask1 = cv2.inRange(banner_hsv,
                               np.array([0, 100, 100]),
                               np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(banner_hsv,
                               np.array([170, 100, 100]),
                               np.array([180, 255, 255]))
        banner_red = cv2.bitwise_or(red_mask1, red_mask2)
        banner_area = banner_region.shape[0] * banner_region.shape[1]
        features['banner_red_ratio'] = np.sum(banner_red > 0) / banner_area if banner_area > 0 else 0
        
        # Action indicators (yellow highlights, movement trails)
        yellow_mask = cv2.inRange(hsv,
                                 np.array([20, 100, 100]),
                                 np.array([30, 255, 255]))
        features['yellow_ratio'] = np.sum(yellow_mask > 0) / (height * width)
        
        # Overall edge density (actions typically have more visual activity)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (height * width)
        
        # PHASE CLASSIFICATION
        phase = 'unknown'
        confidence = 0.0
        
        # "END TURN" button visible = Player's turn
        if features['end_turn_present']:
            # Profile present = Player planning
            if profile_score > 0.10:
                phase = 'player_normal'
                confidence = min(end_turn_score * 5 + profile_score * 2, 1.0)
            
            # Profile absent = Agent executing action
            else:
                phase = 'player_action'
                confidence = min(end_turn_score * 5, 0.9)
        
        # "END TURN" button absent = Opponent's turn
        else:
            phase = 'opponent'
            # Higher confidence if we see typical opponent indicators
            if features['banner_red_ratio'] > 0.05:
                confidence = min(features['banner_red_ratio'] * 10, 1.0)
            else:
                confidence = 0.7  # Reasonable default for opponent turn
        
        return {
            'frame': frame_idx,
            'phase': phase,
            'confidence': float(confidence),
            'features': features
        }
        
    except Exception as e:
        return {
            'frame': frame_idx,
            'phase': 'unknown',
            'confidence': 0.0,
            'features': {},
            'error': str(e)
        }


class TurnPhaseDetector:
    """Detects turn phases from gameplay captures"""
    
    def __init__(self, num_workers: Optional[int] = None):
        if num_workers is None:
            # Conservative: use half the cores
            num_workers = max(1, multiprocessing.cpu_count() // 2)
        self.num_workers = num_workers
        print(f"Using {self.num_workers} worker processes")
    
    def convert_numpy_types(self, obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def analyze_session(self, session_dir: Path) -> Dict:
        """Analyze a complete capture session in parallel"""
        
        frames_dir = session_dir / "frames"
        if not frames_dir.exists():
            raise ValueError(f"Frames directory not found: {frames_dir}")
        
        # Load session metadata
        metadata_file = session_dir / "metadata.json"
        session_data = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                session_data = json.load(f)
        
        # Get all frame files
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        total_frames = len(frame_files)
        
        if total_frames == 0:
            raise ValueError(f"No frames found in {frames_dir}")
        
        print(f"\nAnalyzing {total_frames} frames...")
        
        # Load input events if available
        input_file = session_dir / "input_events.json"
        input_events = []
        if input_file.exists():
            with open(input_file) as f:
                input_events = json.load(f)
            print(f"Input events: {len(input_events)}")
        
        # Parallel frame analysis
        frame_results = self.analyze_frames_parallel(frame_files)
        
        print("✓ Analysis complete!")
        
        # Post-process: smooth phase transitions
        smoothed_phases = self.smooth_phase_transitions(frame_results)
        
        # Generate phase sequences
        sequences = self.generate_phase_sequences(smoothed_phases)
        
        # Calculate statistics
        stats = self.calculate_statistics(smoothed_phases, sequences)
        
        return {
            'session_dir': str(session_dir),
            'total_frames': total_frames,
            'frame_results': frame_results,
            'smoothed_phases': smoothed_phases,
            'sequences': sequences,
            'statistics': stats,
            'input_events': len(input_events)
        }
    
    def analyze_frames_parallel(self, frame_files: List[Path]) -> List[Dict]:
        """Analyze frames in parallel using ProcessPoolExecutor"""
        
        results = []
        total = len(frame_files)
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all frames for processing
            future_to_idx = {
                executor.submit(analyze_single_frame, frame_file, idx): idx
                for idx, frame_file in enumerate(frame_files)
            }
            
            # Collect results as they complete (with progress)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                    
                    # Progress indicator
                    if len(results) % 50 == 0 or len(results) == total:
                        print(f"  Processing frame {len(results)}/{total}...", end='\r')
                        
                except Exception as e:
                    print(f"\n  Error processing frame {idx}: {e}")
                    results.append((idx, {
                        'frame': idx,
                        'phase': 'unknown',
                        'confidence': 0.0,
                        'features': {},
                        'error': str(e)
                    }))
        
        print()  # New line after progress
        
        # Sort results by frame index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def smooth_phase_transitions(self, frame_results: List[Dict], 
                                 window_size: int = 3) -> List[Dict]:
        """Smooth phase transitions using a sliding window"""
        
        smoothed = []
        n = len(frame_results)
        
        for i in range(n):
            # Get window of frames
            start = max(0, i - window_size // 2)
            end = min(n, i + window_size // 2 + 1)
            window = frame_results[start:end]
            
            # Count phases in window
            phase_counts = {}
            for frame in window:
                phase = frame['phase']
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
            
            # Most common phase wins
            dominant_phase = max(phase_counts.items(), key=lambda x: x[1])[0]
            
            smoothed.append({
                'frame': i,
                'phase': dominant_phase,
                'confidence': frame_results[i]['confidence'],
                'original_phase': frame_results[i]['phase']
            })
        
        return smoothed
    
    def generate_phase_sequences(self, smoothed_phases: List[Dict]) -> List[Dict]:
        """Generate sequences of contiguous phase blocks"""
        
        if not smoothed_phases:
            return []
        
        sequences = []
        current_phase = smoothed_phases[0]['phase']
        start_frame = 0
        
        for i, frame_data in enumerate(smoothed_phases):
            if frame_data['phase'] != current_phase:
                # End current sequence
                sequences.append({
                    'phase': current_phase,
                    'start_frame': start_frame,
                    'end_frame': i - 1,
                    'duration_frames': i - start_frame,
                    'duration_seconds': (i - start_frame) * 0.5  # Assuming 2 fps
                })
                
                # Start new sequence
                current_phase = frame_data['phase']
                start_frame = i
        
        # Add final sequence
        sequences.append({
            'phase': current_phase,
            'start_frame': start_frame,
            'end_frame': len(smoothed_phases) - 1,
            'duration_frames': len(smoothed_phases) - start_frame,
            'duration_seconds': (len(smoothed_phases) - start_frame) * 0.5
        })
        
        return sequences
    
    def calculate_statistics(self, smoothed_phases: List[Dict], 
                            sequences: List[Dict]) -> Dict:
        """Calculate statistics about phase distribution"""
        
        phase_counts = {}
        for frame_data in smoothed_phases:
            phase = frame_data['phase']
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        total_frames = len(smoothed_phases)
        
        stats = {
            'total_frames': total_frames,
            'phase_distribution': {
                phase: {
                    'count': count,
                    'percentage': (count / total_frames) * 100
                }
                for phase, count in sorted(phase_counts.items())
            },
            'sequence_count': len(sequences),
            'average_sequence_length': np.mean([s['duration_frames'] for s in sequences])
        }
        
        return stats
    
    def print_summary(self, results: Dict):
        """Print a readable summary of the analysis"""
        
        print("\n" + "="*60)
        print("TURN PHASE DETECTION SUMMARY")
        print("="*60)
        
        stats = results['statistics']
        sequences = results['sequences']
        
        print(f"\nPhase Distribution:")
        for phase, data in stats['phase_distribution'].items():
            count = data['count']
            pct = data['percentage']
            print(f"  {phase:20s}: {count:4d} frames ({pct:5.1f}%)")
        
        print(f"\nPhase Sequences: {stats['sequence_count']} transitions")
        
        print(f"\nSequence Timeline:")
        for seq in sequences:
            duration = seq['duration_seconds']
            print(f"  Frames {seq['start_frame']:3d}-{seq['end_frame']:3d} "
                  f"({duration:5.1f}s): {seq['phase']}")
    
    def save_results(self, results: Dict, output_file: Path):
        """Save analysis results to JSON"""
        # Convert numpy types before saving
        results_clean = self.convert_numpy_types(results)
        
        with open(output_file, 'w') as f:
            json.dump(results_clean, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python turn_phase_detector.py <session_folder>")
        print("\nExample:")
        print("  python turn_phase_detector.py captures/20251021_200738")
        sys.exit(1)
    
    session_path = Path(sys.argv[1])
    
    if not session_path.exists():
        print(f"Error: Session folder not found: {session_path}")
        sys.exit(1)
    
    print("="*60)
    print("Turn Phase Detector")
    print("="*60)
    print(f"\nSession: {session_path.name}")
    
    # Create detector with 5 workers
    detector = TurnPhaseDetector(num_workers=5)
    
    # Analyze the session
    results = detector.analyze_session(session_path)
    
    # Print summary
    detector.print_summary(results)
    
    # Save detailed results
    output_file = session_path / "turn_phase_analysis.json"
    detector.save_results(results, output_file)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()