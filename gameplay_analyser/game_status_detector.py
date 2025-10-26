#!/usr/bin/env python3
"""
Game Status Detector for Invisible Inc
Extracts detailed game state information from captured gameplay frames.

Detects:
- Resources (Power, Credits)
- Turn information (Turn number, Alarm level)
- Agent status (AP, inventory, augments)
- Incognita programs and cooldowns
"""

import cv2
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import pytesseract
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


@dataclass
class Resources:
    """Power and Credits"""
    power_current: Optional[int] = None
    power_max: Optional[int] = None
    credits: Optional[int] = None


@dataclass
class AlarmStatus:
    """Alarm level and tracker status"""
    level: Optional[int] = None  # 0-6
    ticks: Optional[int] = None  # Progress to next level
    tracker_count: Optional[int] = None  # Number of active trackers


@dataclass
class AgentStatus:
    """Individual agent state"""
    name: Optional[str] = None
    ap_current: Optional[int] = None
    ap_max: Optional[int] = None
    inventory_items: List[str] = None
    augments: List[str] = None
    is_visible: bool = False
    is_selected: bool = False
    
    def __post_init__(self):
        if self.inventory_items is None:
            self.inventory_items = []
        if self.augments is None:
            self.augments = []


@dataclass
class IncognitaStatus:
    """Incognita programs and PWR availability"""
    active_programs: List[Dict[str, Any]] = None
    available_pwr: Optional[int] = None
    
    def __post_init__(self):
        if self.active_programs is None:
            self.active_programs = []


@dataclass
class GameStatus:
    """Complete game state for a single frame"""
    frame_number: int
    turn_number: Optional[int] = None
    resources: Resources = None
    alarm: AlarmStatus = None
    agents: List[AgentStatus] = None
    incognita: IncognitaStatus = None
    
    def __post_init__(self):
        if self.resources is None:
            self.resources = Resources()
        if self.alarm is None:
            self.alarm = AlarmStatus()
        if self.agents is None:
            self.agents = []
        if self.incognita is None:
            self.incognita = IncognitaStatus()
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'frame_number': self.frame_number,
            'turn_number': self.turn_number,
            'resources': asdict(self.resources),
            'alarm': asdict(self.alarm),
            'agents': [asdict(agent) for agent in self.agents],
            'incognita': asdict(self.incognita)
        }


class GameStatusDetector:
    """Extracts game status information from Invisible Inc screenshots"""
    
    # UI region coordinates (relative to detected game viewport)
    # These will need calibration based on resolution
    REGIONS = {
        'power_credits': (0.0, 0.0, 0.15, 0.08),  # Top-left corner
        'turn_number': (0.42, 0.0, 0.58, 0.06),   # Top-center
        'alarm': (0.85, 0.0, 1.0, 0.08),          # Top-right
        'agent_panel': (0.0, 0.85, 0.25, 1.0),    # Bottom-left
        'incognita_switch': (0.0, 0.3, 0.08, 0.5), # Left side
    }
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        
        # OCR configuration for better digit recognition
        self.ocr_config = '--psm 7 -c tessedit_char_whitelist=0123456789/'
        
    def detect_viewport(self, frame: np.ndarray) -> Optional[tuple]:
        """
        Detect the game viewport boundaries within the screenshot.
        Returns (x, y, width, height) or None if not found.
        
        The game has a distinctive dark border and consistent UI elements.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # The game viewport typically has very dark borders (near black)
        # and the playable area is lighter
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours of bright regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour (should be the game viewport)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Validate reasonable viewport size (should be significant portion of screen)
        frame_h, frame_w = frame.shape[:2]
        if w < frame_w * 0.5 or h < frame_h * 0.5:
            # Viewport too small, likely incorrect detection
            # Fall back to full frame
            return (0, 0, frame_w, frame_h)
        
        return (x, y, w, h)
    
    def extract_region(self, frame: np.ndarray, viewport: tuple, region_name: str) -> np.ndarray:
        """Extract a specific UI region from the frame"""
        vx, vy, vw, vh = viewport
        rx, ry, rw, rh = self.REGIONS[region_name]
        
        # Convert relative coordinates to absolute
        x = int(vx + rx * vw)
        y = int(vy + ry * vh)
        w = int((rw - rx) * vw)
        h = int((rh - ry) * vh)
        
        return frame[y:y+h, x:x+w]
    
    def preprocess_for_ocr(self, region: np.ndarray, invert: bool = False) -> np.ndarray:
        """Preprocess image region for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast
        gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
        
        # Apply threshold
        if invert:
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Resize for better OCR (upscale small text)
        scale_factor = 3
        h, w = denoised.shape
        resized = cv2.resize(denoised, (w * scale_factor, h * scale_factor), 
                            interpolation=cv2.INTER_CUBIC)
        
        return resized
    
    def extract_power_credits(self, frame: np.ndarray, viewport: tuple) -> Resources:
        """
        Extract power and credits from top-left region.
        Format: "XX/YY PWR  ZZZZ"
        """
        region = self.extract_region(frame, viewport, 'power_credits')
        
        # The text is typically white/cyan on dark background
        processed = self.preprocess_for_ocr(region, invert=False)
        
        if self.debug:
            cv2.imwrite('/tmp/debug_power_credits.png', processed)
        
        # Run OCR
        text = pytesseract.image_to_string(processed, config=self.ocr_config).strip()
        
        resources = Resources()
        
        try:
            # Parse format: "XX/YY PWR  ZZZZ"
            parts = text.split()
            
            # Find power (format: XX/YY)
            for part in parts:
                if '/' in part:
                    power_parts = part.split('/')
                    resources.power_current = int(power_parts[0])
                    resources.power_max = int(power_parts[1])
                elif part.isdigit() and len(part) >= 3:
                    # Likely credits (typically 3+ digits)
                    resources.credits = int(part)
        except (ValueError, IndexError) as e:
            if self.debug:
                print(f"Failed to parse power/credits: {text} - {e}")
        
        return resources
    
    def extract_turn_number(self, frame: np.ndarray, viewport: tuple) -> Optional[int]:
        """Extract turn number from top-center region"""
        region = self.extract_region(frame, viewport, 'turn_number')
        processed = self.preprocess_for_ocr(region, invert=False)
        
        if self.debug:
            cv2.imwrite('/tmp/debug_turn.png', processed)
        
        text = pytesseract.image_to_string(processed, config=self.ocr_config).strip()
        
        try:
            # Turn number is typically just digits
            return int(''.join(filter(str.isdigit, text)))
        except ValueError:
            if self.debug:
                print(f"Failed to parse turn number: {text}")
            return None
    
    def extract_alarm_status(self, frame: np.ndarray, viewport: tuple) -> AlarmStatus:
        """
        Extract alarm level and tracker status from top-right region.
        Alarm is shown as filled bars/icons and tracker count.
        """
        region = self.extract_region(frame, viewport, 'alarm')
        
        alarm = AlarmStatus()
        
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Alarm indicators are typically red/orange when active
        # Red range in HSV
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Count red pixels as proxy for alarm level
        red_pixels = cv2.countNonZero(red_mask)
        total_pixels = region.shape[0] * region.shape[1]
        red_ratio = red_pixels / total_pixels
        
        # Rough alarm level estimation (0-6)
        if red_ratio < 0.05:
            alarm.level = 0
        elif red_ratio < 0.15:
            alarm.level = 1
        elif red_ratio < 0.25:
            alarm.level = 2
        elif red_ratio < 0.35:
            alarm.level = 3
        elif red_ratio < 0.45:
            alarm.level = 4
        elif red_ratio < 0.55:
            alarm.level = 5
        else:
            alarm.level = 6
        
        if self.debug:
            print(f"Alarm red ratio: {red_ratio:.3f} -> level {alarm.level}")
            cv2.imwrite('/tmp/debug_alarm_red.png', red_mask)
        
        # TODO: Extract tracker count via OCR or icon detection
        
        return alarm
    
    def detect_agent_panel(self, frame: np.ndarray, viewport: tuple) -> List[AgentStatus]:
        """
        Detect agent status from bottom-left panel.
        This is complex as it shows different info based on selection.
        """
        region = self.extract_region(frame, viewport, 'agent_panel')
        
        agents = []
        
        # TODO: Implement agent detection
        # This requires:
        # 1. Detect if agent panel is visible (vs Incognita panel in mainframe mode)
        # 2. Extract agent name
        # 3. Extract AP (current/max)
        # 4. Detect inventory icons
        # 5. Detect augment icons
        
        # For now, return empty list
        return agents
    
    def detect_incognita_programs(self, frame: np.ndarray, viewport: tuple) -> IncognitaStatus:
        """
        Detect active Incognita programs and available PWR.
        Programs are shown on left side below the Incognita switch.
        """
        incognita = IncognitaStatus()
        
        # TODO: Implement Incognita program detection
        # This requires:
        # 1. Detect program icons
        # 2. Extract cooldown numbers
        # 3. Extract variable PWR costs
        
        return incognita
    
    def analyze_frame(self, frame_path: Path, frame_number: int) -> GameStatus:
        """Analyze a single frame and extract all game status information"""
        # Load frame
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return GameStatus(frame_number=frame_number)
        
        # Detect viewport
        viewport = self.detect_viewport(frame)
        if viewport is None:
            if self.debug:
                print(f"Frame {frame_number}: Could not detect viewport")
            return GameStatus(frame_number=frame_number)
        
        # Extract all status information
        status = GameStatus(frame_number=frame_number)
        
        try:
            status.resources = self.extract_power_credits(frame, viewport)
            status.turn_number = self.extract_turn_number(frame, viewport)
            status.alarm = self.extract_alarm_status(frame, viewport)
            status.agents = self.detect_agent_panel(frame, viewport)
            status.incognita = self.detect_incognita_programs(frame, viewport)
        except Exception as e:
            if self.debug:
                print(f"Frame {frame_number}: Error extracting status - {e}")
        
        return status


def analyze_session_parallel(session_dir: Path, debug: bool = False, workers: int = None) -> Dict[str, Any]:
    """
    Analyze all frames in a session using parallel processing.
    
    Args:
        session_dir: Path to session directory
        debug: Enable debug output
        workers: Number of parallel workers (default: CPU count)
    
    Returns:
        Dictionary containing analysis results
    """
    frames_dir = session_dir / 'frames'
    if not frames_dir.exists():
        raise ValueError(f"Frames directory not found: {frames_dir}")
    
    # Get all frame files
    frame_files = sorted(frames_dir.glob('frame_*.png'))
    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir}")
    
    print(f"Analyzing {len(frame_files)} frames from {session_dir.name}...")
    
    # Determine number of workers
    if workers is None:
        workers = max(1, mp.cpu_count() - 1)
    
    print(f"Using {workers} parallel workers")
    
    detector = GameStatusDetector(debug=debug)
    
    # Process frames in parallel
    frame_statuses = []
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_frame = {}
        for frame_file in frame_files:
            frame_num = int(frame_file.stem.split('_')[1])
            future = executor.submit(detector.analyze_frame, frame_file, frame_num)
            future_to_frame[future] = (frame_num, frame_file)
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_frame):
            frame_num, frame_file = future_to_frame[future]
            try:
                status = future.result()
                frame_statuses.append(status)
                completed += 1
                
                if completed % 50 == 0:
                    print(f"  Processed {completed}/{len(frame_files)} frames...")
            except Exception as e:
                print(f"  Error processing frame {frame_num}: {e}")
    
    # Sort by frame number
    frame_statuses.sort(key=lambda x: x.frame_number)
    
    print(f"Completed analysis of {len(frame_statuses)} frames")
    
    # Compile results
    results = {
        'session': session_dir.name,
        'total_frames': len(frame_statuses),
        'frame_statuses': [status.to_dict() for status in frame_statuses],
        'summary': compile_summary(frame_statuses)
    }
    
    return results


def compile_summary(frame_statuses: List[GameStatus]) -> Dict[str, Any]:
    """Compile summary statistics from frame statuses"""
    summary = {
        'turn_range': None,
        'resource_stats': {},
        'alarm_stats': {},
        'detection_rates': {}
    }
    
    # Find turn range
    turns = [s.turn_number for s in frame_statuses if s.turn_number is not None]
    if turns:
        summary['turn_range'] = {'min': min(turns), 'max': max(turns)}
    
    # Resource statistics
    powers = [s.resources.power_current for s in frame_statuses 
              if s.resources.power_current is not None]
    credits = [s.resources.credits for s in frame_statuses 
               if s.resources.credits is not None]
    
    if powers:
        summary['resource_stats']['power'] = {
            'min': min(powers),
            'max': max(powers),
            'avg': sum(powers) / len(powers)
        }
    
    if credits:
        summary['resource_stats']['credits'] = {
            'min': min(credits),
            'max': max(credits),
            'avg': sum(credits) / len(credits)
        }
    
    # Alarm statistics
    alarms = [s.alarm.level for s in frame_statuses if s.alarm.level is not None]
    if alarms:
        summary['alarm_stats'] = {
            'min': min(alarms),
            'max': max(alarms),
            'avg': sum(alarms) / len(alarms)
        }
    
    # Detection success rates
    total = len(frame_statuses)
    summary['detection_rates'] = {
        'turn_number': sum(1 for s in frame_statuses if s.turn_number is not None) / total,
        'power': sum(1 for s in frame_statuses if s.resources.power_current is not None) / total,
        'credits': sum(1 for s in frame_statuses if s.resources.credits is not None) / total,
        'alarm': sum(1 for s in frame_statuses if s.alarm.level is not None) / total,
    }
    
    return summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect game status from Invisible Inc gameplay captures')
    parser.add_argument('session_dir', type=Path, help='Path to session directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug output and save debug images')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--output', type=Path, default=None, help='Output JSON file (default: session_dir/game_status.json)')
    
    args = parser.parse_args()
    
    if not args.session_dir.exists():
        print(f"Error: Session directory not found: {args.session_dir}")
        return 1
    
    # Analyze session
    results = analyze_session_parallel(args.session_dir, debug=args.debug, workers=args.workers)
    
    # Save results
    output_file = args.output or (args.session_dir / 'game_status.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    summary = results['summary']
    print("\n=== Summary ===")
    if summary['turn_range']:
        print(f"Turn range: {summary['turn_range']['min']} - {summary['turn_range']['max']}")
    
    print("\nDetection success rates:")
    for key, rate in summary['detection_rates'].items():
        print(f"  {key}: {rate*100:.1f}%")
    
    if summary['resource_stats']:
        print("\nResource stats:")
        for resource, stats in summary['resource_stats'].items():
            print(f"  {resource}: min={stats['min']}, max={stats['max']}, avg={stats['avg']:.1f}")
    
    if summary['alarm_stats']:
        alarm = summary['alarm_stats']
        print(f"\nAlarm: min={alarm['min']}, max={alarm['max']}, avg={alarm['avg']:.1f}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
