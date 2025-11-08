#!/usr/bin/env python3
"""
Alarm Level Detector for Invisible Inc - Framework Integration

Detects security alarm level (major digit 0-6 and minor segments 0-5)
by using hamburger menu anchor as spatial reference.
"""

import cv2
import numpy as np
import pytesseract
from typing import Dict, Optional, Tuple
from pathlib import Path


class DetectionResult:
    """Standardized detection result format."""
    
    def __init__(self, success: bool, confidence: float = 0.0, data: Dict = None):
        self.success = success
        self.confidence = confidence
        self.data = data if data is not None else {}


class BaseDetector:
    """Base class for all detectors in the framework."""
    
    def __init__(self):
        self.dependencies = []
    
    def detect(self, frame: np.ndarray, dependency_results: Dict = None) -> DetectionResult:
        """
        Detect feature in frame.
        
        Args:
            frame: BGR image
            dependency_results: Dictionary of {detector_name: DetectionResult} from dependencies
            
        Returns:
            DetectionResult with success status, confidence, and extracted data
        """
        raise NotImplementedError


class HamburgerMenuDetector(BaseDetector):
    """Detects the hamburger menu in top-right corner."""
    
    def __init__(self):
        super().__init__()
        self.dependencies = []
    
    def detect(self, frame: np.ndarray, dependency_results: Dict = None) -> DetectionResult:
        """
        Detect hamburger menu using horizontal line pattern matching.
        
        Returns position of hamburger menu icon.
        """
        height, width = frame.shape[:2]
        
        # Search in top-right corner
        search_x_start = int(width * 0.90)
        search_y_end = int(height * 0.10)
        
        # Convert to grayscale
        gray = frame[0:search_y_end, search_x_start:width]
        
        # Look for three horizontal lines (hamburger pattern)
        # Use edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find horizontal lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=15, maxLineGap=5)
        
        if lines is None or len(lines) < 3:
            return DetectionResult(success=False, confidence=0.0)
        
        # Find horizontal lines (nearly 0 or 180 degrees)
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 10 or angle > 170:  # Horizontal
                horizontal_lines.append((x1, y1, x2, y2))
        
        if len(horizontal_lines) < 3:
            return DetectionResult(success=False, confidence=0.0)
        
        # Find center position of hamburger (average of line centers)
        centers_x = [(x1 + x2) / 2 + search_x_start for x1, y1, x2, y2 in horizontal_lines[:3]]
        centers_y = [(y1 + y2) / 2 for x1, y1, x2, y2 in horizontal_lines[:3]]
        
        menu_x = int(np.mean(centers_x))
        menu_y = int(np.mean(centers_y))
        
        return DetectionResult(
            success=True,
            confidence=0.8,
            data={
                'position': (menu_x, menu_y),
                'lines_detected': len(horizontal_lines)
            }
        )


class AlarmLevelDetector(BaseDetector):
    """
    Detects security alarm level using hamburger menu as spatial anchor.
    
    The alarm clock is positioned at a fixed offset from the hamburger menu:
    - Horizontally: Slightly left of hamburger
    - Vertically: Below hamburger
    """
    
    def __init__(self):
        super().__init__()
        self.dependencies = ['hamburger_menu']
        
        # HSV ranges for colored segments (yellow/orange/red)
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([60, 255, 255])
        self.orange_lower = np.array([0, 100, 100])
        self.orange_upper = np.array([20, 255, 255])
        self.red_lower = np.array([160, 100, 100])
        self.red_upper = np.array([180, 255, 255])
        
        # Clock parameters
        self.expected_radius = 40
        
        # OCR configuration
        self.tesseract_config = '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456'
    
    def detect(self, frame: np.ndarray, dependency_results: Dict = None) -> DetectionResult:
        """
        Detect alarm level using hamburger menu anchor.
        
        Args:
            frame: BGR image
            dependency_results: Must contain 'hamburger_menu' result
            
        Returns:
            DetectionResult with major_alarm (0-6) and minor_alarm (0-5)
        """
        # Check dependency
        if not dependency_results or 'hamburger_menu' not in dependency_results:
            return DetectionResult(success=False, confidence=0.0, 
                                  data={'error': 'Missing hamburger_menu dependency'})
        
        hamburger_result = dependency_results['hamburger_menu']
        if not hamburger_result.success:
            return DetectionResult(success=False, confidence=0.0,
                                  data={'error': 'Hamburger menu not detected'})
        
        # Calculate clock position from hamburger
        hamburger_pos = hamburger_result.data['position']
        clock_center = self._calculate_clock_position(hamburger_pos)
        
        # Verify clock is present at calculated position
        clock_info = self._verify_clock(frame, clock_center, self.expected_radius)
        if clock_info is None:
            return DetectionResult(success=False, confidence=0.0,
                                  data={'error': 'Clock not found at expected position'})
        
        center, radius = clock_info
        
        # Extract major alarm (center digit)
        major_alarm = self._extract_center_digit(frame, center, radius)
        
        # Extract minor alarm (segment count)
        minor_alarm = self._count_filled_segments(frame, center, radius)
        
        # Build result
        result_data = {
            'clock_center': center,
            'clock_radius': radius,
            'hamburger_position': hamburger_pos
        }
        
        if major_alarm is not None:
            result_data['major_alarm'] = major_alarm
        if minor_alarm is not None:
            result_data['minor_alarm'] = minor_alarm
        
        # Success if we got at least major alarm
        success = major_alarm is not None
        confidence = 0.0
        if major_alarm is not None:
            confidence += 0.5
        if minor_alarm is not None:
            confidence += 0.3
        confidence = min(1.0, confidence)
        
        return DetectionResult(success=success, confidence=confidence, data=result_data)
    
    def _calculate_clock_position(self, hamburger_pos: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate expected clock position from hamburger menu position.
        
        Spatial relationship (from visual verification):
        - Clock is ~30 pixels LEFT of hamburger
        - Clock is ~130 pixels BELOW hamburger
        
        Args:
            hamburger_pos: (x, y) position of hamburger menu
            
        Returns:
            (x, y) expected position of clock center
        """
        hx, hy = hamburger_pos
        
        # Clock is to the LEFT and well below hamburger
        clock_x = hx - 30
        clock_y = hy + 130
        
        return (clock_x, clock_y)
    
    def _verify_clock(
        self, 
        frame: np.ndarray, 
        expected_center: Tuple[int, int], 
        expected_radius: int
    ) -> Optional[Tuple[Tuple[int, int], int]]:
        """
        Verify clock is present at expected position by checking for colored pixels.
        
        Args:
            frame: BGR image
            expected_center: Expected (x, y) center of clock
            expected_radius: Expected radius of clock
            
        Returns:
            ((center_x, center_y), radius) if verified, None otherwise
        """
        cx, cy = expected_center
        
        # Check bounds
        if (cx < 0 or cy < 0 or 
            cx >= frame.shape[1] or cy >= frame.shape[0]):
            return None
        
        # Extract region around expected clock position (smaller margin for precision)
        margin = expected_radius
        x1 = max(0, cx - margin)
        y1 = max(0, cy - margin)
        x2 = min(frame.shape[1], cx + margin)
        y2 = min(frame.shape[0], cy + margin)
        
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return None
        
        # Convert to HSV and check for colored pixels
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        orange_mask = cv2.inRange(hsv, self.orange_lower, self.orange_upper)
        red_mask = cv2.inRange(hsv, self.red_lower, self.red_upper)
        
        color_mask = cv2.bitwise_or(yellow_mask, orange_mask)
        color_mask = cv2.bitwise_or(color_mask, red_mask)
        
        # Check if enough colored pixels present (very lenient - just needs some color)
        colored_pixels = np.count_nonzero(color_mask)
        
        # Require at least 20 colored pixels (clock has digit + segments)
        if colored_pixels < 20:
            return None
        
        # Clock verified at expected position
        return (expected_center, expected_radius)
    
    def _extract_center_digit(
        self, 
        frame: np.ndarray, 
        center: Tuple[int, int], 
        radius: int
    ) -> Optional[int]:
        """Extract the digit (0-6) from center of clock using OCR."""
        cx, cy = center
        
        # Extract center region (inner 70% of radius)
        inner_radius = int(radius * 0.7)
        x1 = max(0, cx - inner_radius)
        y1 = max(0, cy - inner_radius)
        x2 = min(frame.shape[1], cx + inner_radius)
        y2 = min(frame.shape[0], cy + inner_radius)
        
        center_region = frame[y1:y2, x1:x2].copy()
        
        # Reject if region is too small (clock partially off-screen)
        if center_region.size == 0 or center_region.shape[0] < 10 or center_region.shape[1] < 10:
            return None
        
        # Convert to HSV
        hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        
        # Create color mask for digit
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        orange_mask = cv2.inRange(hsv, self.orange_lower, self.orange_upper)
        red_mask = cv2.inRange(hsv, self.red_lower, self.red_upper)
        
        color_mask = cv2.bitwise_or(yellow_mask, orange_mask)
        color_mask = cv2.bitwise_or(color_mask, red_mask)
        
        # Apply morphological closing to connect fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        # Also prepare grayscale
        gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
        
        # Try multiple preprocessing methods
        preprocessing_methods = [
            ('color_mask', color_mask),
            ('color_mask_inv', cv2.bitwise_not(color_mask)),
            ('threshold', cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ('threshold_inv', cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
        ]
        
        # Upscale for better OCR
        scale = 5
        for method_name, processed in preprocessing_methods:
            if not isinstance(processed, np.ndarray):
                continue
            
            upscaled = cv2.resize(processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            
            # Run OCR
            text = pytesseract.image_to_string(upscaled, config=self.tesseract_config).strip()
            
            # Parse as digit
            if text and text.isdigit():
                digit = int(text)
                if 0 <= digit <= 6:
                    return digit
        
        return None
    
    def _count_filled_segments(
        self,
        frame: np.ndarray,
        center: Tuple[int, int],
        radius: int
    ) -> Optional[int]:
        """Count filled segments (0-5) by sampling arc regions."""
        cx, cy = center
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create combined color mask
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        orange_mask = cv2.inRange(hsv, self.orange_lower, self.orange_upper)
        red_mask = cv2.inRange(hsv, self.red_lower, self.red_upper)
        color_mask = cv2.bitwise_or(yellow_mask, orange_mask)
        color_mask = cv2.bitwise_or(color_mask, red_mask)
        
        # 5 segment centers (72° each)
        segment_centers = [-90, -18, 54, 126, 198]
        
        filled_segments = 0
        
        # Sample each segment across its arc
        for center_angle in segment_centers:
            # Sample 5 points across ±25° from center
            sample_angles = [center_angle + offset for offset in [-25, -12, 0, 12, 25]]
            colored_samples = 0
            
            for angle_deg in sample_angles:
                angle_rad = np.radians(angle_deg)
                
                # Sample at 90% radius (segments on outer ring)
                sample_radius = int(radius * 0.90)
                sample_x = int(cx + sample_radius * np.cos(angle_rad))
                sample_y = int(cy + sample_radius * np.sin(angle_rad))
                
                # Check bounds
                if (0 <= sample_x < frame.shape[1] and 
                    0 <= sample_y < frame.shape[0]):
                    
                    # Check 7x7 region
                    region_size = 7
                    x1 = max(0, sample_x - region_size // 2)
                    y1 = max(0, sample_y - region_size // 2)
                    x2 = min(color_mask.shape[1], sample_x + region_size // 2 + 1)
                    y2 = min(color_mask.shape[0], sample_y + region_size // 2 + 1)
                    
                    region = color_mask[y1:y2, x1:x2]
                    
                    # Count if majority colored
                    if region.size > 0 and np.mean(region) > 127:
                        colored_samples += 1
            
            # Segment filled if 3+ of 5 samples colored
            if colored_samples >= 3:
                filled_segments += 1
        
        return filled_segments


def main():
    """Test the alarm level detector on a single frame."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python alarm_level_detector.py <frame_path>")
        sys.exit(1)
    
    frame_path = Path(sys.argv[1])
    if not frame_path.exists():
        print(f"Error: Frame not found: {frame_path}")
        sys.exit(1)
    
    # Load frame
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"Error: Could not load frame: {frame_path}")
        sys.exit(1)
    
    # Detect hamburger menu
    hamburger_detector = HamburgerMenuDetector()
    hamburger_result = hamburger_detector.detect(frame)
    
    print(f"Hamburger menu: {hamburger_result.success}")
    if hamburger_result.success:
        print(f"  Position: {hamburger_result.data['position']}")
    
    # Detect alarm level
    alarm_detector = AlarmLevelDetector()
    alarm_result = alarm_detector.detect(frame, {'hamburger_menu': hamburger_result})
    
    print(f"\nAlarm detection: {alarm_result.success}")
    print(f"Confidence: {alarm_result.confidence:.2f}")
    
    if alarm_result.success:
        data = alarm_result.data
        if 'major_alarm' in data:
            print(f"Major alarm: {data['major_alarm']}")
        if 'minor_alarm' in data:
            print(f"Minor alarm: {data['minor_alarm']}/5 segments")
        if 'clock_center' in data:
            print(f"Clock position: {data['clock_center']}")
    else:
        print(f"Error: {alarm_result.data.get('error', 'Unknown')}")


if __name__ == '__main__':
    main()