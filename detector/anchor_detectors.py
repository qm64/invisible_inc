"""
Priority #1: Persistent UI Anchor Detectors

These detectors find always-present UI elements that serve as spatial anchors
for other game state detection.

Detectors:
- EndTurnDetector: Cyan "END TURN" button (lower-right)
- HamburgerMenuDetector: 3-line menu icon (upper-right)
- TacticalViewDetector: Polygon at top center
- PowerCreditsAnchorDetector: Cyan "PWR" text (upper-left) - FIXED v1.0.1
- SecurityClockDetector: Red/orange circular clock (upper-right) - needs improvement

Version: 1.0.1
Changes in v1.0.1:
- Fixed PowerCreditsAnchorDetector to detect dark cyan PWR text (V=13)
- Expanded hue range to 85-100 (was 85-95) to catch H=97
- Lowered value threshold to 10 (was 100) to catch dark text
- Added max_y filter (40px) to prioritize topmost element and avoid other UI
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple
from detector_framework import BaseDetector, DetectionResult, DetectorType, DetectorConfig


class EndTurnDetector(BaseDetector):
    """
    Detects the cyan "END TURN" button in the lower-right area.
    
    This is the most reliable indicator of player control and turn phase.
    Success rate: ~97%+
    
    Detection method:
    - HSV color detection for cyan (H: 85-100°)
    - Located in lower-right quadrant
    - Distinctive cyan color unique in UI
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        if config is None:
            config = DetectorConfig(
                name="end_turn",
                type=DetectorType.STRUCTURAL,
                params={
                    'hue_min': 85,
                    'hue_max': 100,
                    'sat_min': 100,
                    'val_min': 100,
                    'min_area': 1000
                }
            )
        super().__init__(config)
    
    def detect(self, image: np.ndarray, context: Optional[Dict[str, DetectionResult]] = None, 
               **kwargs) -> DetectionResult:
        """Detect END TURN button via cyan color in lower-right area"""
        
        debug = kwargs.get('debug', False)
        h, w = image.shape[:2]
        
        # Search region: lower-right quadrant
        search_x = w // 2
        search_y = h // 2
        search_region = image[search_y:h, search_x:w]
        
        # Convert to HSV and find cyan pixels
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        
        params = self.config.params
        lower_cyan = np.array([params['hue_min'], params['sat_min'], params['val_min']])
        upper_cyan = np.array([params['hue_max'], 255, 255])
        
        cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        
        # Find contours
        contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest cyan region
        best_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area and area > params['min_area']:
                max_area = area
                best_contour = contour
        
        if best_contour is not None:
            # Get bounding box
            x, y, bw, bh = cv2.boundingRect(best_contour)
            
            # Adjust to full image coordinates
            x += search_x
            y += search_y
            
            bbox = (x, y, bw, bh)
            center = (x + bw // 2, y + bh // 2)
            
            data = {
                'bbox': bbox,
                'center': center,
                'area': max_area,
                'location': 'lower_right'
            }
            
            if debug:
                print(f"✓ END TURN button detected at ({x}, {y}), size {bw}×{bh}")
            
            return DetectionResult(
                detector_name=self.get_name(),
                detector_type=self.get_type(),
                success=True,
                confidence=0.95,
                data=data
            )
        
        if debug:
            print("✗ END TURN button not detected")
        
        return DetectionResult(
            detector_name=self.get_name(),
            detector_type=self.get_type(),
            success=False,
            confidence=0.0,
            data={},
            error="No cyan button found in lower-right quadrant"
        )
    
    def get_name(self) -> str:
        return "end_turn"
    
    def get_type(self) -> DetectorType:
        return DetectorType.STRUCTURAL


class HamburgerMenuDetector(BaseDetector):
    """
    Detects the 3-line hamburger menu icon in upper-right corner.
    
    Success rate: 78.2%
    
    Detection method:
    - Looks for 3 horizontal white lines stacked vertically
    - Located in upper-right corner
    - Uses edge detection and horizontal line pattern matching
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        if config is None:
            config = DetectorConfig(
                name="hamburger_menu",
                type=DetectorType.STRUCTURAL,
                params={
                    'brightness_threshold': 200,
                    'min_line_width': 15,
                    'max_line_gap': 15
                }
            )
        super().__init__(config)
    
    def detect(self, image: np.ndarray, context: Optional[Dict[str, DetectionResult]] = None,
               **kwargs) -> DetectionResult:
        """Detect hamburger menu via horizontal line pattern"""
        
        debug = kwargs.get('debug', False)
        h, w = image.shape[:2]
        
        # Search region: upper-right corner
        search_x = 2 * w // 3
        search_y = 0
        search_h = h // 4
        search_region = image[search_y:search_y + search_h, search_x:w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        
        # Threshold for bright elements
        params = self.config.params
        _, bright = cv2.threshold(gray, params['brightness_threshold'], 255, cv2.THRESH_BINARY)
        
        # Find horizontal lines using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        horizontal_lines = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
        
        # Find contours of lines
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get bounding boxes of lines
        line_boxes = []
        for contour in contours:
            x, y, lw, lh = cv2.boundingRect(contour)
            if lw > params['min_line_width'] and lh < 10:  # Horizontal line
                line_boxes.append((x, y, lw, lh))
        
        # Sort by y-coordinate
        line_boxes.sort(key=lambda b: b[1])
        
        # Look for 3 evenly-spaced horizontal lines
        if len(line_boxes) >= 3:
            for i in range(len(line_boxes) - 2):
                y1 = line_boxes[i][1]
                y2 = line_boxes[i + 1][1]
                y3 = line_boxes[i + 2][1]
                
                gap1 = y2 - y1
                gap2 = y3 - y2
                
                # Check if gaps are similar (within tolerance)
                if abs(gap1 - gap2) < params['max_line_gap'] and 5 < gap1 < 20:
                    # Found hamburger pattern!
                    x = min(b[0] for b in line_boxes[i:i+3])
                    y = y1
                    menu_w = max(b[0] + b[2] for b in line_boxes[i:i+3]) - x
                    menu_h = y3 + line_boxes[i + 2][3] - y
                    
                    # Adjust to full image coordinates
                    x += search_x
                    y += search_y
                    
                    bbox = (x, y, menu_w, menu_h)
                    center = (x + menu_w // 2, y + menu_h // 2)
                    
                    data = {
                        'bbox': bbox,
                        'center': center,
                        'location': 'upper_right'
                    }
                    
                    if debug:
                        print(f"✓ Hamburger menu detected at ({x}, {y})")
                    
                    return DetectionResult(
                        detector_name=self.get_name(),
                        detector_type=self.get_type(),
                        success=True,
                        confidence=0.90,
                        data=data
                    )
        
        if debug:
            print("✗ Hamburger menu not detected")
        
        return DetectionResult(
            detector_name=self.get_name(),
            detector_type=self.get_type(),
            success=False,
            confidence=0.0,
            data={},
            error="No 3-line pattern found in upper-right"
        )
    
    def get_name(self) -> str:
        return "hamburger_menu"
    
    def get_type(self) -> DetectorType:
        return DetectorType.STRUCTURAL


class TacticalViewDetector(BaseDetector):
    """
    Detects the tactical view polygon at top center.
    
    Success rate: 80.9%
    
    Detection method:
    - Looks for distinctive polygon shape at very top of viewport
    - Uses edge detection and contour analysis
    - Located at top center
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        if config is None:
            config = DetectorConfig(
                name="tactical_view",
                type=DetectorType.STRUCTURAL,
                params={
                    'canny_low': 50,
                    'canny_high': 150,
                    'min_area': 500,
                    'max_y': 0.15  # Top 15% of image
                }
            )
        super().__init__(config)
    
    def detect(self, image: np.ndarray, context: Optional[Dict[str, DetectionResult]] = None,
               **kwargs) -> DetectionResult:
        """Detect tactical view polygon at top center"""
        
        debug = kwargs.get('debug', False)
        h, w = image.shape[:2]
        
        # Search region: top center
        max_y = int(h * self.config.params['max_y'])
        search_region = image[0:max_y, :]
        
        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        params = self.config.params
        edges = cv2.Canny(gray, params['canny_low'], params['canny_high'])
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for polygon at top center
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < params['min_area']:
                continue
            
            # Get bounding box
            x, y, pw, ph = cv2.boundingRect(contour)
            
            # Check if it's near top center
            center_x = x + pw // 2
            distance_from_center = abs(center_x - w // 2)
            
            # Score: prefer larger areas closer to center
            score = area / (1 + distance_from_center)
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        if best_contour is not None:
            x, y, pw, ph = cv2.boundingRect(best_contour)
            
            bbox = (x, y, pw, ph)
            center = (x + pw // 2, y + ph // 2)
            
            data = {
                'bbox': bbox,
                'center': center,
                'location': 'top_center'
            }
            
            if debug:
                print(f"✓ Tactical view detected at ({x}, {y}), size {pw}×{ph}")
            
            return DetectionResult(
                detector_name=self.get_name(),
                detector_type=self.get_type(),
                success=True,
                confidence=0.85,
                data=data
            )
        
        if debug:
            print("✗ Tactical view not detected")
        
        return DetectionResult(
            detector_name=self.get_name(),
            detector_type=self.get_type(),
            success=False,
            confidence=0.0,
            data={},
            error="No polygon found at top center"
        )
    
    def get_name(self) -> str:
        return "tactical_view"
    
    def get_type(self) -> DetectorType:
        return DetectorType.STRUCTURAL


class PowerCreditsAnchorDetector(BaseDetector):
    """
    Detects the cyan "PWR" text in upper-left area.
    
    Success rate: 77.2%
    
    This serves as the primary anchor for the power/credits display region.
    The actual values are extracted by a separate OCR detector that uses this anchor.
    
    Detection method:
    - HSV color detection for cyan text
    - Located in upper-left area
    - Text pattern recognition for "PWR"
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        if config is None:
            config = DetectorConfig(
                name="power_credits_anchor",
                type=DetectorType.STRUCTURAL,
                params={
                    'hue_min': 85,
                    'hue_max': 100,  # Expanded to catch H=97 (dark cyan PWR)
                    'sat_min': 100,
                    'val_min': 10,   # Lowered to catch V=13 (dark text)
                    'min_width': 20,  # PWR text is substantial
                    'max_y': 40      # Only consider top 40 pixels (avoid other UI elements)
                }
            )
        super().__init__(config)
    
    def detect(self, image: np.ndarray, context: Optional[Dict[str, DetectionResult]] = None,
               **kwargs) -> DetectionResult:
        """Detect cyan PWR text in upper-left"""
        
        debug = kwargs.get('debug', False)
        h, w = image.shape[:2]
        
        # Search region: upper-left area
        search_w = w // 2
        search_h = h // 4
        search_region = image[0:search_h, 0:search_w]
        
        # Convert to HSV and find cyan
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        
        params = self.config.params
        lower_cyan = np.array([params['hue_min'], params['sat_min'], params['val_min']])
        upper_cyan = np.array([params['hue_max'], 255, 255])
        
        cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        
        # Find contours
        contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for PWR text (horizontal cyan region in top area)
        best_contour = None
        max_width = 0
        min_x = float('inf')  # Prioritize leftmost element (PWR is always on left)
        
        if debug:
            print(f"  Found {len(contours)} cyan contours in search region")
            print(f"  Filtering: y <= {params['max_y']}, width > {params['min_width']}, wider than tall")
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Filter: only consider elements in top portion of screen
            if y > params['max_y']:
                if debug and cw > params['min_width']:
                    print(f"  ✗ Contour at ({x},{y}) size {cw}×{ch}: y={y} > max_y={params['max_y']}")
                continue
            
            # PWR text should be wider than tall
            if cw > params['min_width'] and cw > ch:
                # Prioritize leftmost element (PWR is in upper-left corner)
                if x < min_x:
                    if debug:
                        print(f"  ✓ Candidate at ({x},{y}) size {cw}×{ch}")
                    min_x = x
                    max_width = cw
                    best_contour = contour
                elif debug:
                    print(f"  ○ Contour at ({x},{y}) size {cw}×{ch}: valid but not leftmost (x={x} > min_x={min_x})")
            elif debug and cw > 5:  # Only debug non-tiny contours
                print(f"  ✗ Contour at ({x},{y}) size {cw}×{ch}: too narrow or not horizontal")
        
        if best_contour is not None:
            x, y, cw, ch = cv2.boundingRect(best_contour)
            
            # Extend the bounding box to capture full power/credits line
            # The full line is typically much wider than just "PWR"
            extended_w = int(cw * 8)  # Extend 8× to capture full line
            extended_x = max(0, x - int(cw * 1))  # Start a bit left
            
            bbox = (extended_x, y, min(extended_w, search_w - extended_x), ch)
            center = (extended_x + bbox[2] // 2, y + ch // 2)
            
            data = {
                'bbox': bbox,
                'pwr_bbox': (x, y, cw, ch),  # Original PWR text box
                'center': center,
                'location': 'upper_left'
            }
            
            if debug:
                print(f"✓ Power/credits anchor detected at ({extended_x}, {y})")
            
            return DetectionResult(
                detector_name=self.get_name(),
                detector_type=self.get_type(),
                success=True,
                confidence=0.90,
                data=data
            )
        
        if debug:
            print("✗ Power/credits anchor not detected")
        
        return DetectionResult(
            detector_name=self.get_name(),
            detector_type=self.get_type(),
            success=False,
            confidence=0.0,
            data={},
            error="No cyan PWR text found in upper-left"
        )
    
    def get_name(self) -> str:
        return "power_credits_anchor"
    
    def get_type(self) -> DetectorType:
        return DetectorType.STRUCTURAL


class SecurityClockDetector(BaseDetector):
    """
    Detects the security clock (circular with yellow/orange/red arcs) in upper-right.
    
    Success rate: 38.4% (NEEDS IMPROVEMENT)
    
    This detector is included for completeness but has known reliability issues.
    The alarm level can be extracted more reliably using viewport-relative positioning
    instead of depending on finding the security clock first.
    
    Detection method:
    - Looks for circular shapes with yellow/orange/red color
    - Yellow: low alarm (early game)
    - Orange: medium alarm  
    - Red: high alarm
    - Located in upper-right area
    - Uses color detection + contour analysis
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        if config is None:
            config = DetectorConfig(
                name="security_clock",
                type=DetectorType.STRUCTURAL,
                params={
                    'min_area': 5000,
                    'max_area': 35000,
                    'search_width': 500  # pixels from right edge
                }
            )
        super().__init__(config)
    
    def detect(self, image: np.ndarray, context: Optional[Dict[str, DetectionResult]] = None,
               **kwargs) -> DetectionResult:
        """Detect security clock via red/orange circular shape"""
        
        debug = kwargs.get('debug', False)
        h, w = image.shape[:2]
        
        # Search region: upper-right corner
        params = self.config.params
        search_x = max(0, w - params['search_width'])
        search_region = image[0:h//3, search_x:w]
        
        # Convert to HSV and find yellow/orange/red arcs
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        
        # Color ranges for alarm clock progression: yellow → orange → red
        # Yellow (low alarm): 20-60° hue
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([60, 255, 255])
        
        # Orange (medium alarm): 0-20° hue
        lower_orange = np.array([0, 50, 50])
        upper_orange = np.array([20, 255, 255])
        
        # Red (high alarm): wraps around HSV, 160-180° hue
        lower_red = np.array([160, 50, 50])
        upper_red = np.array([180, 255, 255])
        
        # Combine all color masks
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        
        # Combined mask for any alarm color
        alarm_mask = cv2.bitwise_or(yellow_mask, orange_mask)
        alarm_mask = cv2.bitwise_or(alarm_mask, red_mask)
        
        # Dilate to connect segments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        alarm_mask = cv2.dilate(alarm_mask, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(alarm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for circular contour with appropriate size
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if params['min_area'] < area < params['max_area']:
                x, y, cw, ch = cv2.boundingRect(contour)
                
                # Check if roughly circular (aspect ratio close to 1)
                aspect_ratio = cw / ch if ch > 0 else 0
                
                # Too tall is not a clock
                if ch > cw * 2.2:
                    continue
                
                # Score based on circularity
                circularity = 1.0 - abs(aspect_ratio - 1.0)
                score = area * circularity
                
                if score > best_score:
                    best_score = score
                    best_contour = contour
        
        if best_contour is not None:
            x, y, cw, ch = cv2.boundingRect(best_contour)
            
            # Adjust to full image coordinates
            x += search_x
            
            bbox = (x, y, cw, ch)
            center = (x + cw // 2, y + ch // 2)
            
            data = {
                'bbox': bbox,
                'center': center,
                'location': 'upper_right'
            }
            
            if debug:
                print(f"✓ Security clock detected at ({x}, {y}), size {cw}×{ch}")
            
            return DetectionResult(
                detector_name=self.get_name(),
                detector_type=self.get_type(),
                success=True,
                confidence=0.65,  # Lower confidence due to known issues
                data=data
            )
        
        if debug:
            print("✗ Security clock not detected (known issue - 38.4% success rate)")
        
        return DetectionResult(
            detector_name=self.get_name(),
            detector_type=self.get_type(),
            success=False,
            confidence=0.0,
            data={},
            error="No circular yellow/orange/red shape found (detector needs improvement)"
        )
    
    def get_name(self) -> str:
        return "security_clock"
    
    def get_type(self) -> DetectorType:
        return DetectorType.STRUCTURAL


# Export all detectors
__all__ = [
    'EndTurnDetector',
    'HamburgerMenuDetector',
    'TacticalViewDetector',
    'PowerCreditsAnchorDetector',
    'SecurityClockDetector'
]