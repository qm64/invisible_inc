"""
Color-Based Game State Detector
Uses color signatures to detect UI elements regardless of window size/aspect ratio
"""

import cv2
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List


@dataclass
class BoundingBox:
    """Represents a detected UI element's position"""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def top_left(self) -> Tuple[int, int]:
        return (self.x, self.y)
    
    @property
    def bottom_right(self) -> Tuple[int, int]:
        return (self.x + self.width, self.y + self.height)


class ColorBasedDetector:
    """Detects game UI elements using color signatures"""
    
    def __init__(self, signatures_path: str = 'color_signatures_merged.json'):
        """Load color signatures from JSON file"""
        with open(signatures_path, 'r') as f:
            self.signatures = json.load(f)
        
        self.image = None
        self.hsv_image = None
        self.detected_elements = {}
        
    def load_image(self, image_path: str):
        """Load an image for analysis"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.detected_elements = {}
        
    def detect_by_color(self, signature_name: str, search_region: Optional[BoundingBox] = None,
                       tolerance_multiplier: float = 1.0) -> Optional[BoundingBox]:
        """
        Detect an element by its color signature
        
        Args:
            signature_name: Name of the signature to search for
            search_region: Optional region to limit search (None = whole image)
            tolerance_multiplier: Multiply HSV ranges by this factor (>1 = more tolerant)
        
        Returns:
            BoundingBox if found, None otherwise
        """
        # Find signature in any category
        sig = self._find_signature(signature_name)
        if sig is None:
            print(f"Warning: Signature '{signature_name}' not found")
            return None
        
        # Get HSV detection range
        hsv_range = sig['hsv']['detection_range']
        
        # Apply tolerance multiplier
        h_range = hsv_range['h']
        s_range = hsv_range['s']
        v_range = hsv_range['v']
        
        if tolerance_multiplier != 1.0:
            # Expand ranges around mean
            h_mean = sig['hsv']['mean'][0]
            s_mean = sig['hsv']['mean'][1]
            v_mean = sig['hsv']['mean'][2]
            
            h_width = (h_range[1] - h_range[0]) * tolerance_multiplier / 2
            s_width = (s_range[1] - s_range[0]) * tolerance_multiplier / 2
            v_width = (v_range[1] - v_range[0]) * tolerance_multiplier / 2
            
            h_range = [max(0, int(h_mean - h_width)), min(179, int(h_mean + h_width))]
            s_range = [max(0, int(s_mean - s_width)), min(255, int(s_mean + s_width))]
            v_range = [max(0, int(v_mean - v_width)), min(255, int(v_mean + v_width))]
        
        # Create color mask
        lower = np.array([h_range[0], s_range[0], v_range[0]])
        upper = np.array([h_range[1], s_range[1], v_range[1]])
        
        # Determine search area
        if search_region:
            hsv_search = self.hsv_image[
                search_region.y:search_region.y + search_region.height,
                search_region.x:search_region.x + search_region.width
            ]
            offset_x, offset_y = search_region.x, search_region.y
        else:
            hsv_search = self.hsv_image
            offset_x, offset_y = 0, 0
        
        # Apply color threshold
        mask = cv2.inRange(hsv_search, lower, upper)
        
        # Find contours of matching regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour (most likely match)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate confidence based on size match
        expected_size = sig['region_size']
        size_ratio = (w * h) / (expected_size['width'] * expected_size['height'])
        confidence = min(1.0, size_ratio) if size_ratio <= 1.0 else min(1.0, 1.0 / size_ratio)
        
        return BoundingBox(
            x=x + offset_x,
            y=y + offset_y,
            width=w,
            height=h,
            confidence=confidence
        )
    
    def _find_signature(self, name: str) -> Optional[Dict]:
        """Find a signature by name in any category"""
        for category_name, category in self.signatures.items():
            if category_name.startswith('_') or category_name == 'metadata':
                continue
            if isinstance(category, dict) and name in category:
                return category[name]
        return None
    
    def detect_viewport(self) -> Optional[BoundingBox]:
        """
        Detect the game viewport using always-present anchors
        Returns the bounding box of the viewport
        """
        # Try to detect key anchors (those most likely to succeed)
        anchors = {}
        
        # Top anchors
        tactical = self.detect_by_color('tactical_view', tolerance_multiplier=1.2)
        if tactical:
            anchors['tactical_view'] = tactical
        
        # Right anchors
        menu = self.detect_by_color('menu_hamburger', tolerance_multiplier=1.2)
        if menu:
            anchors['menu_hamburger'] = menu
        
        security = self.detect_by_color('security_clock', tolerance_multiplier=1.2)
        if security:
            anchors['security_clock'] = security
        
        # Bottom anchor
        end_turn = self.detect_by_color('end_turn', tolerance_multiplier=1.2)
        if end_turn:
            anchors['end_turn'] = end_turn
        
        # Left anchor (may be absent)
        incognita = self.detect_by_color('incognita_upper', tolerance_multiplier=1.2)
        if incognita:
            anchors['incognita_upper'] = incognita
        
        if len(anchors) < 3:
            print(f"Warning: Only found {len(anchors)} anchors, need at least 3")
            return None
        
        # Calculate viewport bounds from anchors
        # Left edge: from incognita if present, otherwise conservative estimate
        if 'incognita_upper' in anchors:
            left = anchors['incognita_upper'].x
        elif 'tactical_view' in anchors:
            # Estimate based on typical layout
            left = max(0, anchors['tactical_view'].x - 200)
        else:
            left = 0
        
        # Top edge: from tactical view or incognita
        if 'tactical_view' in anchors:
            top = anchors['tactical_view'].y
        elif 'incognita_upper' in anchors:
            top = anchors['incognita_upper'].y
        else:
            top = 0
        
        # Right edge: from menu or security
        if 'menu_hamburger' in anchors:
            right = anchors['menu_hamburger'].x + anchors['menu_hamburger'].width
        elif 'security_clock' in anchors:
            right = anchors['security_clock'].x + anchors['security_clock'].width
        else:
            right = self.image.shape[1]
        
        # Bottom edge: from end_turn or security
        if 'end_turn' in anchors:
            bottom = anchors['end_turn'].y + anchors['end_turn'].height
        elif 'security_clock' in anchors:
            bottom = anchors['security_clock'].y + anchors['security_clock'].height
        else:
            bottom = self.image.shape[0]
        
        viewport = BoundingBox(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
            confidence=len(anchors) / 5.0  # Max 5 anchors
        )
        
        self.detected_elements['viewport'] = viewport
        self.detected_elements['anchors'] = anchors
        
        return viewport
    
    def detect_mode(self) -> str:
        """
        Detect the current game mode
        Returns: 'mainframe', 'agent_full', 'agent_moving', 'drone', or 'blank'
        """
        # Check for mainframe mode
        if self.detect_by_color('incognita_programs', tolerance_multiplier=1.3):
            return 'mainframe'
        
        # Check for drone mode
        if self.detect_by_color('lower_left_drone', tolerance_multiplier=1.3):
            return 'drone'
        
        # Check for agent moving mode
        if self.detect_by_color('lower_left_agent_moving', tolerance_multiplier=1.3):
            return 'agent_moving'
        
        # Check for full agent mode
        if self.detect_by_color('agent_profile', tolerance_multiplier=1.3):
            return 'agent_full'
        
        # Check for agent icons (alternative for full mode)
        if self.detect_by_color('agent_icons', tolerance_multiplier=1.3):
            return 'agent_full'
        
        return 'blank'
    
    def detect_all_elements(self) -> Dict[str, BoundingBox]:
        """
        Detect all UI elements based on current mode
        Returns dictionary of element_name -> BoundingBox
        """
        results = {}
        
        # First detect viewport
        viewport = self.detect_viewport()
        if viewport:
            results['viewport'] = viewport
        
        # Detect mode
        mode = self.detect_mode()
        results['mode'] = mode
        
        # Detect mode-specific elements
        if mode == 'mainframe':
            elements = ['incognita_profile', 'incognita_programs', 'lower_left_mainframe']
        elif mode == 'agent_full':
            elements = ['agent_profile', 'agent_icons', 'actions', 'augments', 'inventory']
        elif mode == 'agent_moving':
            elements = ['lower_left_agent_moving']
        elif mode == 'drone':
            elements = ['lower_left_drone']
        else:
            elements = []
        
        # Detect each element
        for element in elements:
            bbox = self.detect_by_color(element, tolerance_multiplier=1.3)
            if bbox:
                results[element] = bbox
        
        # Always try to detect always-present elements
        always_present = ['power', 'credits', 'security_clock', 'alarm_level', 
                         'objectives', 'end_turn']
        for element in always_present:
            if element not in results:
                bbox = self.detect_by_color(element, tolerance_multiplier=1.3)
                if bbox:
                    results[element] = bbox
        
        self.detected_elements = results
        return results
    
    def visualize_detections(self, output_path: Optional[str] = None) -> np.ndarray:
        """
        Draw detected elements on the image
        Returns annotated image
        """
        if self.image is None:
            raise ValueError("No image loaded")
        
        annotated = self.image.copy()
        
        # Color scheme for different element types
        colors = {
            'viewport': (0, 255, 0),      # Green
            'anchors': (255, 0, 0),       # Blue
            'agent': (0, 255, 255),       # Yellow
            'mainframe': (255, 0, 255),   # Magenta
            'default': (0, 165, 255)      # Orange
        }
        
        for name, element in self.detected_elements.items():
            if name == 'mode':
                continue  # Skip mode string
            
            if isinstance(element, dict):  # anchors dict
                for anchor_name, bbox in element.items():
                    cv2.rectangle(annotated, bbox.top_left, bbox.bottom_right, 
                                colors['anchors'], 2)
                    cv2.putText(annotated, anchor_name, 
                              (bbox.x, bbox.y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['anchors'], 1)
                continue
            
            # Determine color based on element name
            if 'viewport' in name:
                color = colors['viewport']
            elif 'agent' in name or 'drone' in name:
                color = colors['agent']
            elif 'mainframe' in name or 'incognita' in name:
                color = colors['mainframe']
            else:
                color = colors['default']
            
            # Draw bounding box
            cv2.rectangle(annotated, element.top_left, element.bottom_right, color, 2)
            
            # Draw label with confidence
            label = f"{name} ({element.confidence:.2f})"
            cv2.putText(annotated, label, (element.x, element.y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw mode text
        if 'mode' in self.detected_elements:
            mode_text = f"Mode: {self.detected_elements['mode']}"
            cv2.putText(annotated, mode_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, annotated)
            print(f"✓ Saved visualization to {output_path}")
        
        return annotated


def main():
    """Test the detector on sample images"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python color_detector.py <image_path>")
        print("\nExample: python color_detector.py window_test.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Create detector
    detector = ColorBasedDetector()
    
    # Load image
    print(f"Loading image: {image_path}")
    detector.load_image(image_path)
    
    # Detect all elements
    print("\nDetecting UI elements...")
    results = detector.detect_all_elements()
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Detection Results")
    print(f"{'='*60}")
    print(f"Mode: {results.get('mode', 'unknown')}")
    print(f"\nDetected {len(results)} elements:")
    for name, element in results.items():
        if name == 'mode':
            continue
        if isinstance(element, dict):
            print(f"\n  {name}:")
            for k, v in element.items():
                print(f"    - {k}: {v.center} (conf: {v.confidence:.2f})")
        else:
            print(f"  - {name}: {element.center} (conf: {element.confidence:.2f})")
    
    # Visualize
    output_name = Path(image_path).stem + "_detected.png"
    print(f"\nCreating visualization...")
    detector.visualize_detections(output_name)
    
    print(f"\n{'='*60}")
    print("Detection complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
    