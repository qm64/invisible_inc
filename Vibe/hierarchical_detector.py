"""
Hierarchical Game State Detector for Invisible Inc
Top-down detection strategy with fallbacks and mode detection
"""

import cv2
import numpy as np
import pytesseract
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class DisplayMode(Enum):
    """Game display mode"""
    AGENT = "agent"
    MAINFRAME = "mainframe"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Represents a rectangular region"""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def x2(self):
        return self.x + self.width
    
    @property
    def y2(self):
        return self.y + self.height
    
    def __repr__(self):
        return f"BoundingBox({self.x}, {self.y}, {self.width}x{self.height})"


@dataclass
class GameState:
    """Complete game state"""
    # Always available
    power: Optional[int] = None
    power_max: Optional[int] = None
    credits: Optional[int] = None
    turn: Optional[int] = None
    day: Optional[int] = None
    mission_type: Optional[str] = None
    alarm_level: Optional[int] = None
    
    # Mode-dependent
    mode: DisplayMode = DisplayMode.UNKNOWN
    agent_name: Optional[str] = None
    action_points: Optional[int] = None
    
    # Metadata
    viewport_size: Optional[Tuple[int, int]] = None
    lower_left_visible: bool = False


class TemplateManager:
    """Manages template images for matching"""
    
    def __init__(self):
        self.templates = {}
        self.debug = True
    
    def load_template(self, name: str, path: str):
        """Load a template image"""
        template = cv2.imread(path)
        if template is not None:
            # Convert to RGB to match our viewport format
            template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
            self.templates[name] = template
            if self.debug:
                print(f"Loaded template '{name}': {template.shape}")
        else:
            print(f"Warning: Failed to load template '{name}' from {path}")
    
    def match_template(self, image, template_name: str, threshold=0.8) -> Optional[BoundingBox]:
        """
        Find template in image using normalized cross-correlation
        Tries multiple scales if initial match fails
        Returns best match above threshold, or None
        """
        if template_name not in self.templates:
            return None
        
        template = self.templates[template_name]
        
        # Try exact scale first
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= threshold:
            x, y = max_loc
            h, w = template.shape[:2]
            if self.debug:
                print(f"      Matched '{template_name}' at scale 1.0 (score: {max_val:.3f})")
            return BoundingBox(x, y, w, h)
        
        # Try different scales (80% to 120%)
        best_match = None
        best_score = threshold
        
        for scale in [0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2]:
            scaled_template = cv2.resize(template, None, fx=scale, fy=scale, 
                                        interpolation=cv2.INTER_LINEAR)
            
            # Skip if scaled template is larger than image
            if scaled_template.shape[0] > image.shape[0] or scaled_template.shape[1] > image.shape[1]:
                continue
            
            result = cv2.matchTemplate(image, scaled_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                x, y = max_loc
                h, w = scaled_template.shape[:2]
                best_match = (BoundingBox(x, y, w, h), scale)
        
        if best_match:
            bbox, scale = best_match
            if self.debug:
                print(f"      Matched '{template_name}' at scale {scale:.2f} (score: {best_score:.3f})")
            return bbox
        
        if self.debug:
            print(f"      No match for '{template_name}' (best score: {max_val:.3f})")
        
        return None


class ViewportDetector:
    """Detects game viewport boundaries using multiple anchors"""
    
    def __init__(self, template_manager: TemplateManager):
        self.templates = template_manager
        self.debug = True
    
    def find_viewport_with_all_anchors(self, window_img) -> Optional[Tuple[BoundingBox, Dict[str, BoundingBox]]]:
        """
        Primary strategy: Find viewport using all four edges
        Returns: (viewport_bbox, anchor_positions) or None
        """
        h, w = window_img.shape[:2]
        anchors = {}
        
        # Top: Tactical view polygon (try template matching first, fallback to color)
        tactical_view = self._find_tactical_view(window_img)
        if tactical_view:
            anchors['tactical_view'] = tactical_view
        
        # Right: Menu hamburger
        menu = self._find_menu_hamburger(window_img)
        if menu:
            anchors['menu'] = menu
        
        # Bottom right: End turn button
        end_turn = self._find_end_turn_button(window_img)
        if end_turn:
            anchors['end_turn'] = end_turn
        
        # Left: Agent icons or Incognita profile
        left_anchor = self._find_left_anchor(window_img)
        if left_anchor:
            anchors['left_anchor'] = left_anchor
        
        if len(anchors) >= 3:  # Need at least 3 anchors
            viewport = self._compute_viewport_from_anchors(anchors, w, h)
            return viewport, anchors
        
        return None
    
    def find_viewport_fallback(self, window_img) -> Optional[Tuple[BoundingBox, Dict[str, BoundingBox]]]:
        """
        Fallback strategy: Find viewport without left side
        Returns: (viewport_bbox, anchor_positions) or None
        """
        h, w = window_img.shape[:2]
        anchors = {}
        
        # Must have top, right, and bottom
        tactical_view = self._find_tactical_view(window_img)
        if tactical_view:
            anchors['tactical_view'] = tactical_view
        
        menu = self._find_menu_hamburger(window_img)
        if menu:
            anchors['menu'] = menu
        
        end_turn = self._find_end_turn_button(window_img)
        if end_turn:
            anchors['end_turn'] = end_turn
        
        if len(anchors) >= 3:
            viewport = self._compute_viewport_from_anchors(anchors, w, h, infer_left=True)
            return viewport, anchors
        
        return None
    
    def _find_tactical_view(self, image) -> Optional[BoundingBox]:
        """Find tactical view polygon at top center"""
        # Try template matching first
        tactical = self.templates.match_template(image, 'tactical_view', threshold=0.75)
        if tactical:
            return tactical
        
        # Fallback: Look for distinctive shape/color at top center
        h, w = image.shape[:2]
        search_top = 0
        search_bottom = int(h * 0.08)
        search_left = int(w * 0.3)
        search_right = int(w * 0.7)
        
        search_area = image[search_top:search_bottom, search_left:search_right]
        
        # Look for bright/yellow text "TACTICAL VIEW" or distinctive shape
        hsv = cv2.cvtColor(search_area, cv2.COLOR_RGB2HSV)
        lower = np.array([20, 50, 150])
        upper = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        if np.any(mask):
            rows = np.any(mask > 0, axis=1)
            cols = np.any(mask > 0, axis=0)
            if np.any(rows) and np.any(cols):
                top = np.argmax(rows)
                bottom = len(rows) - np.argmax(rows[::-1])
                left = np.argmax(cols)
                right = len(cols) - np.argmax(cols[::-1])
                
                return BoundingBox(
                    search_left + left,
                    search_top + top,
                    right - left,
                    bottom - top
                )
        
        return None
    
    def _find_menu_hamburger(self, image) -> Optional[BoundingBox]:
        """Find menu hamburger in top right corner"""
        # Try template matching
        menu = self.templates.match_template(image, 'menu_hamburger', threshold=0.75)
        if menu:
            return menu
        
        # Fallback: Look for 3 horizontal lines in top right
        h, w = image.shape[:2]
        search_top = 0
        search_bottom = int(h * 0.06)
        search_left = int(w * 0.95)
        search_right = w
        
        search_area = image[search_top:search_bottom, search_left:search_right]
        
        # Look for bright lines
        gray = cv2.cvtColor(search_area, cv2.COLOR_RGB2GRAY)
        _, bright = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        if np.any(bright):
            rows = np.any(bright > 0, axis=1)
            cols = np.any(bright > 0, axis=0)
            if np.any(rows) and np.any(cols):
                top = np.argmax(rows)
                bottom = len(rows) - np.argmax(rows[::-1])
                left = np.argmax(cols)
                right = len(cols) - np.argmax(cols[::-1])
                
                return BoundingBox(
                    search_left + left,
                    search_top + top,
                    right - left,
                    bottom - top
                )
        
        return None
    
    def _find_end_turn_button(self, image) -> Optional[BoundingBox]:
        """Find end turn polygon in bottom right"""
        # Try template matching
        end_turn = self.templates.match_template(image, 'end_turn', threshold=0.75)
        if end_turn:
            return end_turn
        
        # Fallback: Look for distinctive polygon shape in bottom right
        h, w = image.shape[:2]
        search_top = int(h * 0.85)
        search_bottom = h
        search_left = int(w * 0.80)
        search_right = w
        
        search_area = image[search_top:search_bottom, search_left:search_right]
        
        # Look for bright/cyan polygon
        hsv = cv2.cvtColor(search_area, cv2.COLOR_RGB2HSV)
        lower = np.array([80, 100, 100])
        upper = np.array([100, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        if np.any(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w_box, h_box = cv2.boundingRect(largest)
                return BoundingBox(
                    search_left + x,
                    search_top + y,
                    w_box,
                    h_box
                )
        
        return None
    
    def _find_left_anchor(self, image) -> Optional[BoundingBox]:
        """Find left side anchor (agent icons or Incognita profile)"""
        # Try Incognita profile template
        incognita = self.templates.match_template(image, 'incognita_upper', threshold=0.75)
        if incognita:
            return incognita
        
        # Fallback: Look for cyan/white in upper left
        h, w = image.shape[:2]
        search_top = 0
        search_bottom = int(h * 0.15)
        search_left = 0
        search_right = int(w * 0.15)
        
        search_area = image[search_top:search_bottom, search_left:search_right]
        
        # Cyan/white detection (Incognita colors)
        red = search_area[:, :, 0]
        green = search_area[:, :, 1]
        blue = search_area[:, :, 2]
        
        cyan_mask = ((green > 100) & (blue > 100) & (red < 80)).astype(np.uint8) * 255
        white_mask = ((red > 180) & (green > 180) & (blue > 180)).astype(np.uint8) * 255
        combined = cv2.bitwise_or(cyan_mask, white_mask)
        
        if np.any(combined):
            rows = np.any(combined > 0, axis=1)
            cols = np.any(combined > 0, axis=0)
            if np.any(rows) and np.any(cols):
                top = np.argmax(rows)
                bottom = len(rows) - np.argmax(rows[::-1])
                left = np.argmax(cols)
                right = len(cols) - np.argmax(cols[::-1])
                
                return BoundingBox(left, top, right - left, bottom - top)
        
        return None
    
    def _compute_viewport_from_anchors(self, anchors: Dict[str, BoundingBox], 
                                      window_w: int, window_h: int,
                                      infer_left: bool = False) -> BoundingBox:
        """Compute viewport bounds from detected anchors"""
        # Start with conservative bounds
        left = 0
        top = 0
        right = window_w
        bottom = window_h
        
        # Refine using anchors
        if 'tactical_view' in anchors:
            # Use TOP edge of tactical view (not bottom), since power/credits are above it
            top = 0  # Start from very top of window
        
        if 'incognita_upper' in anchors:
            # Left edge is the left edge of Incognita box
            left = anchors['incognita_upper'].x
        elif 'left_anchor' in anchors:
            left = anchors['left_anchor'].x
        
        if 'menu' in anchors:
            # Right edge: menu_hamburger is at far right, but we need room for security clock/daemons
            # Security clock extends about 180-200px left of the menu
            # So right edge is approximately menu.x (left edge of menu) + some padding
            right = anchors['menu'].x + 50  # Small padding past menu hamburger
        
        if 'end_turn' in anchors:
            bottom = anchors['end_turn'].y2 + 10  # Add small padding for agent profile
        
        if infer_left:
            # Infer left edge from right edge and aspect ratio
            # Game viewport is roughly 16:9 or 16:10
            viewport_width = right - left
            expected_width = int((bottom - top) * 1.6)  # Assume 16:10
            if viewport_width > expected_width * 1.2:
                # Too wide, move left edge in
                left = right - expected_width
        
        return BoundingBox(left, top, right - left, bottom - top)


class RegionDetector:
    """Base class for region-specific detectors"""
    
    def __init__(self, template_manager: TemplateManager):
        self.templates = template_manager
        self.debug = True
    
    def preprocess_for_ocr(self, image, color_channel='green', threshold=100):
        """Preprocess image for OCR"""
        if color_channel == 'green':
            channel = image[:, :, 1]
        elif color_channel == 'gray':
            channel = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            channel = image
        
        # Simple threshold (works better than adaptive for bright UI text)
        _, thresh = cv2.threshold(channel, threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up with morphology
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Scale up for better OCR
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        return thresh
    
    def read_text(self, image, config='--psm 7'):
        """Read text from preprocessed image"""
        try:
            text = pytesseract.image_to_string(image, config=config)
            return text.strip()
        except:
            return None


class UpperLeftDetector(RegionDetector):
    """Detects power and credits in upper left"""
    
    def detect(self, viewport, anchors: Dict[str, BoundingBox]) -> Dict:
        """
        Detect power and credits
        Strategy: Find PWR text, read left (power) and right (credits) of it
        """
        vh, vw = viewport.shape[:2]
        
        # Constrain search to upper-left quadrant
        search_region = viewport[0:int(vh*0.1), 0:int(vw*0.2)]
        
        if self.debug:
            print(f"    Searching upper-left region: {search_region.shape[1]}x{search_region.shape[0]}")
        
        # Find "PWR" text using color detection (cyan/green text)
        pwr_box = self._find_pwr_text(search_region)
        
        if self.debug:
            print(f"    PWR text location: {pwr_box}")
        
        power_current, power_max = None, None
        credits = None
        
        if pwr_box:
            # Read power (left of PWR)
            power_current, power_max = self._read_power_left_of_pwr(search_region, pwr_box)
            if self.debug:
                print(f"    Power OCR result: {power_current}/{power_max}")
            
            # Read credits (right of PWR)
            credits = self._read_credits_right_of_pwr(search_region, pwr_box)
            if self.debug:
                print(f"    Credits OCR result: {credits}")
        else:
            if self.debug:
                print(f"    PWR text not found")
        
        return {
            'power': power_current,
            'power_max': power_max,
            'credits': credits
        }
    
    def _find_pwr_text(self, search_area) -> Optional[BoundingBox]:
        """Find 'PWR' text in search area using color detection"""
        # Look for green/cyan text
        green = search_area[:, :, 1]
        _, mask = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        
        if self.debug:
            cv2.imwrite('debug_pwr_search_mask.png', mask)
        
        # Find text regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Look for rightmost text group (PWR is typically right of numbers)
        # Also filter by size - PWR is small and compact
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # PWR is roughly 20-40 pixels wide, 10-20 pixels tall
            if 15 < w < 50 and 8 < h < 25:
                candidates.append((x + w, BoundingBox(x, y, w, h)))  # Sort by right edge
        
        if not candidates:
            # Fallback: just use rightmost
            rightmost = max(contours, key=lambda c: cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2])
            x, y, w, h = cv2.boundingRect(rightmost)
            return BoundingBox(x, y, w, h)
        
        # Return the rightmost candidate
        candidates.sort(reverse=True)
        return candidates[0][1]
    
    def _read_power_left_of_pwr(self, search_region, pwr_box: BoundingBox) -> Tuple[Optional[int], Optional[int]]:
        """Read power value left of PWR text"""
        # Define region left of PWR
        left = 0
        right = pwr_box.x - 5  # Small gap before PWR
        top = max(0, pwr_box.y - 5)
        bottom = min(search_region.shape[0], pwr_box.y + pwr_box.height + 5)
        
        if right <= left:
            return None, None
        
        region = search_region[top:bottom, left:right]
        
        if self.debug:
            cv2.imwrite('debug_power_region.png', cv2.cvtColor(region, cv2.COLOR_RGB2BGR))
        
        preprocessed = self.preprocess_for_ocr(region)
        
        if self.debug:
            cv2.imwrite('debug_power_preprocessed.png', preprocessed)
        
        text = self.read_text(preprocessed, config='--psm 7')
        if self.debug:
            print(f"      Power raw OCR: '{text}'")
        
        if text and '/' in text:
            try:
                parts = text.split('/')
                current = int(''.join(filter(str.isdigit, parts[0])))
                maximum = int(''.join(filter(str.isdigit, parts[1])))
                return current, maximum
            except:
                pass
        
        return None, None
    
    def _read_credits_right_of_pwr(self, search_region, pwr_box: BoundingBox) -> Optional[int]:
        """Read credits value right of PWR text"""
        # Define region right of PWR
        left = pwr_box.x + pwr_box.width + 5  # Small gap after PWR
        right = search_region.shape[1]
        top = max(0, pwr_box.y - 5)
        bottom = min(search_region.shape[0], pwr_box.y + pwr_box.height + 5)
        
        if right <= left:
            return None
        
        region = search_region[top:bottom, left:right]
        
        if self.debug:
            cv2.imwrite('debug_credits_region.png', cv2.cvtColor(region, cv2.COLOR_RGB2BGR))
        
        preprocessed = self.preprocess_for_ocr(region)
        
        if self.debug:
            cv2.imwrite('debug_credits_preprocessed.png', preprocessed)
        
        text = self.read_text(preprocessed, config='--psm 7 digits')
        if self.debug:
            print(f"      Credits raw OCR: '{text}'")
        
        digits = ''.join(filter(str.isdigit, text))
        
        try:
            return int(digits) if digits else None
        except:
            return None
    
    def _find_pwr_text(self, search_area) -> Optional[BoundingBox]:
        """Find 'PWR' text in search area"""
        # Look for green/cyan text
        green = search_area[:, :, 1]
        _, mask = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        
        # Find text regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Look for rightmost text group (PWR is typically right of numbers)
        rightmost_x = 0
        pwr_contour = None
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x + w > rightmost_x:
                rightmost_x = x + w
                pwr_contour = contour
        
        if pwr_contour is not None:
            x, y, w, h = cv2.boundingRect(pwr_contour)
            return BoundingBox(x, y, w, h)
        
        return None
    
    def _read_power_from_region(self, viewport, bbox: BoundingBox) -> Tuple[Optional[int], Optional[int]]:
        """Read power value from template-matched region"""
        region = viewport[bbox.y:bbox.y2, bbox.x:bbox.x2]
        
        if self.debug:
            cv2.imwrite('debug_power_region.png', cv2.cvtColor(region, cv2.COLOR_RGB2BGR))
        
        preprocessed = self.preprocess_for_ocr(region)
        
        if self.debug:
            cv2.imwrite('debug_power_preprocessed.png', preprocessed)
        
        text = self.read_text(preprocessed, config='--psm 7')
        if self.debug:
            print(f"      Power raw OCR: '{text}'")
        
        if text and '/' in text:
            try:
                parts = text.split('/')
                current = int(''.join(filter(str.isdigit, parts[0])))
                maximum = int(''.join(filter(str.isdigit, parts[1])))
                return current, maximum
            except:
                pass
        
        return None, None
    
    def _read_credits_from_region(self, viewport, bbox: BoundingBox) -> Optional[int]:
        """Read credits value from template-matched region"""
        region = viewport[bbox.y:bbox.y2, bbox.x:bbox.x2]
        
        if self.debug:
            cv2.imwrite('debug_credits_region.png', cv2.cvtColor(region, cv2.COLOR_RGB2BGR))
        
        preprocessed = self.preprocess_for_ocr(region)
        
        if self.debug:
            cv2.imwrite('debug_credits_preprocessed.png', preprocessed)
        
        text = self.read_text(preprocessed, config='--psm 7 digits')
        if self.debug:
            print(f"      Credits raw OCR: '{text}'")
        
        digits = ''.join(filter(str.isdigit, text))
        
        try:
            return int(digits) if digits else None
        except:
            return None
        """Find 'PWR' text in search area"""
        # Look for green/cyan text
        green = search_area[:, :, 1]
        _, mask = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        
        # Find text regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Look for rightmost text group (PWR is typically right of numbers)
        rightmost_x = 0
        pwr_contour = None
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x + w > rightmost_x:
                rightmost_x = x + w
                pwr_contour = contour
        
        if pwr_contour is not None:
            x, y, w, h = cv2.boundingRect(pwr_contour)
            return BoundingBox(x, y, w, h)
        
        return None
    
    def _read_power(self, viewport, left, right, y, height) -> Tuple[Optional[int], Optional[int]]:
        """Read power value in format XX/YY"""
        if right <= left:
            return None, None
        
        region = viewport[y-5:y+height+5, left:right-5]
        preprocessed = self.preprocess_for_ocr(region)
        
        text = self.read_text(preprocessed, config='--psm 7')
        if text and '/' in text:
            try:
                parts = text.split('/')
                current = int(''.join(filter(str.isdigit, parts[0])))
                maximum = int(''.join(filter(str.isdigit, parts[1])))
                return current, maximum
            except:
                pass
        
        return None, None
    
    def _read_credits(self, viewport, left, right, y, height) -> Optional[int]:
        """Read credits value (digits only)"""
        if right <= left:
            return None
        
        region = viewport[y-5:y+height+5, left:right]
        preprocessed = self.preprocess_for_ocr(region)
        
        text = self.read_text(preprocessed, config='--psm 7 digits')
        digits = ''.join(filter(str.isdigit, text))
        
        try:
            return int(digits) if digits else None
        except:
            return None


class UpperRightDetector(RegionDetector):
    """Detects turn info and alarm level in upper right"""
    
    def detect(self, viewport, anchors: Dict[str, BoundingBox]) -> Dict:
        """Detect turn, day, mission type, and alarm level"""
        vh, vw = viewport.shape[:2]
        
        # Search area: right side, top portion
        if 'menu' in anchors:
            search_right = anchors['menu'].x
        else:
            search_right = vw
        
        search_left = int(vw * 0.6)
        search_top = 0
        search_bottom = int(vh * 0.3)
        
        # Find alarm level (most distinctive)
        alarm = self._find_alarm_level(viewport, search_left, search_right, search_top, search_bottom)
        
        # TODO: Parse turn/day/mission from status string
        
        return {
            'alarm_level': alarm,
            'turn': None,  # TODO
            'day': None,   # TODO
            'mission_type': None  # TODO
        }
    
    def _find_alarm_level(self, viewport, left, right, top, bottom) -> Optional[int]:
        """Find and read alarm level digit in center of security clock"""
        search_area = viewport[top:bottom, left:right]
        
        # Look for very bright digit (white/yellow)
        gray = cv2.cvtColor(search_area, cv2.COLOR_RGB2GRAY)
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find largest bright region (the digit)
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Extract and read digit
        digit_img = search_area[y:y+h, x:x+w]
        preprocessed = self.preprocess_for_ocr(digit_img, color_channel='gray')
        
        text = self.read_text(preprocessed, config='--psm 10')
        digits = ''.join(filter(str.isdigit, text))
        
        try:
            return int(digits) if digits else None
        except:
            return None
    
    def _read_alarm_from_region(self, viewport, bbox: BoundingBox) -> Optional[int]:
        """Read alarm level from template-matched region"""
        region = viewport[bbox.y:bbox.y2, bbox.x:bbox.x2]
        
        if self.debug:
            cv2.imwrite('debug_alarm_region.png', cv2.cvtColor(region, cv2.COLOR_RGB2BGR))
        
        preprocessed = self.preprocess_for_ocr(region, color_channel='gray', threshold=200)
        
        if self.debug:
            cv2.imwrite('debug_alarm_preprocessed.png', preprocessed)
        
        text = self.read_text(preprocessed, config='--psm 10')
        if self.debug:
            print(f"      Alarm raw OCR: '{text}'")
        
        digits = ''.join(filter(str.isdigit, text))
        
        try:
            return int(digits) if digits else None
        except:
            return None
        """Find and read alarm level digit in center of security clock"""
        search_area = viewport[top:bottom, left:right]
        
        # Look for very bright digit (white/yellow)
        gray = cv2.cvtColor(search_area, cv2.COLOR_RGB2GRAY)
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find largest bright region (the digit)
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Extract and read digit
        digit_img = search_area[y:y+h, x:x+w]
        preprocessed = self.preprocess_for_ocr(digit_img, color_channel='gray')
        
        text = self.read_text(preprocessed, config='--psm 10')
        digits = ''.join(filter(str.isdigit, text))
        
        try:
            return int(digits) if digits else None
        except:
            return None


class LowerLeftDetector(RegionDetector):
    """Detects agent/Incognita profile and related elements (optional)"""
    
    def detect(self, viewport, anchors: Dict[str, BoundingBox]) -> Dict:
        """
        Detect lower left elements
        Returns: agent name, AP, mode indicators
        """
        # Check if lower left is visible
        if 'left_anchor' not in anchors:
            return {'visible': False}
        
        vh, vw = viewport.shape[:2]
        
        # Search bottom left quadrant
        search_left = 0
        search_right = int(vw * 0.3)
        search_top = int(vh * 0.6)
        search_bottom = vh
        
        # TODO: Implement agent name, AP detection
        
        return {
            'visible': True,
            'agent_name': None,  # TODO
            'action_points': None  # TODO
        }


class LowerRightDetector(RegionDetector):
    """Detects objectives and end turn button in lower right"""
    
    def detect(self, viewport, anchors: Dict[str, BoundingBox]) -> Dict:
        """Detect objectives list"""
        # TODO: Implement objectives detection
        
        return {
            'objectives': []  # TODO
        }


class HierarchicalGameStateDetector:
    """
    Main detector that coordinates all region detectors
    Uses hierarchical top-down approach with fallbacks
    """
    
    def __init__(self, window_detector=None):
        self.window_detector = window_detector
        self.templates = TemplateManager()
        self.viewport_detector = ViewportDetector(self.templates)
        
        # Region detectors
        self.upper_left = UpperLeftDetector(self.templates)
        self.upper_right = UpperRightDetector(self.templates)
        self.lower_left = LowerLeftDetector(self.templates)
        self.lower_right = LowerRightDetector(self.templates)
        
        self.debug = True
        self.debug_images = []
    
    def load_templates(self, template_dir: str):
        """Load all template images from directory"""
        import os
        
        # Load all templates that match the pattern *_template.png
        if not os.path.exists(template_dir):
            print(f"Warning: Template directory not found: {template_dir}")
            return
        
        loaded_count = 0
        for filename in os.listdir(template_dir):
            if filename.endswith('_template.png'):
                # Extract template name (remove _template.png suffix)
                name = filename[:-13]  # Remove '_template.png'
                path = os.path.join(template_dir, filename)
                self.templates.load_template(name, path)
                loaded_count += 1
        
        print(f"Loaded {loaded_count} templates from {template_dir}")
        
        # List which key anchors were found
        key_anchors = ['tactical_view', 'menu_hamburger', 'end_turn', 'incognita_upper']
        found_anchors = [a for a in key_anchors if a in self.templates.templates]
        print(f"Key anchors loaded: {found_anchors}")
    
    def get_game_state(self) -> Optional[GameState]:
        """
        Main entry point: Capture window and extract game state
        """
        if self.window_detector is None:
            print("Error: No window detector configured")
            return None
        
        # Capture game window
        window_img = self.window_detector.capture_game_window(auto_focus=True)
        if window_img is None:
            print("Failed to capture game window")
            return None
        
        # Find viewport (try primary, then fallback)
        viewport_result = self.viewport_detector.find_viewport_with_all_anchors(window_img)
        
        if viewport_result is None:
            print("Primary viewport detection failed, trying fallback...")
            viewport_result = self.viewport_detector.find_viewport_fallback(window_img)
        
        if viewport_result is None:
            print("All viewport detection strategies failed")
            return None
        
        viewport_bbox, anchors = viewport_result
        viewport = window_img[viewport_bbox.y:viewport_bbox.y2, 
                             viewport_bbox.x:viewport_bbox.x2]
        
        vh, vw = viewport.shape[:2]
        print(f"Viewport detected: {vw}x{vh}")
        print(f"Anchors found: {list(anchors.keys())}")
        
        # Create game state
        state = GameState(viewport_size=(vw, vh))
        
        # Detect regions (each is independent)
        print("\nDetecting regions...")
        
        # Upper left: Power & Credits
        print("  Upper left (power/credits)...")
        upper_left_data = self.upper_left.detect(viewport, anchors)
        state.power = upper_left_data.get('power')
        state.power_max = upper_left_data.get('power_max')
        state.credits = upper_left_data.get('credits')
        
        # Upper right: Turn info & Alarm
        print("  Upper right (alarm)...")
        upper_right_data = self.upper_right.detect(viewport, anchors)
        state.alarm_level = upper_right_data.get('alarm_level')
        state.turn = upper_right_data.get('turn')
        state.day = upper_right_data.get('day')
        state.mission_type = upper_right_data.get('mission_type')
        
        # Lower left: Agent/Profile info (optional)
        print("  Lower left (agent)...")
        lower_left_data = self.lower_left.detect(viewport, anchors)
        state.lower_left_visible = lower_left_data.get('visible', False)
        state.agent_name = lower_left_data.get('agent_name')
        state.action_points = lower_left_data.get('action_points')
        
        # Lower right: Objectives
        print("  Lower right (objectives)...")
        lower_right_data = self.lower_right.detect(viewport, anchors)
        
        # Determine mode
        state.mode = self._determine_mode(lower_left_data)
        
        if self.debug:
            self._visualize_detections(viewport, anchors, state)
        
        return state
    
    def _determine_mode(self, lower_left_data: Dict) -> DisplayMode:
        """Determine if we're in agent or mainframe mode"""
        # TODO: Implement proper mode detection
        # For now, assume agent mode if lower left is visible
        if lower_left_data.get('visible'):
            return DisplayMode.AGENT
        return DisplayMode.UNKNOWN
    
    def _visualize_detections(self, viewport, anchors, state):
        """Create debug visualization"""
        vis = viewport.copy()
        
        # Draw anchors
        for name, bbox in anchors.items():
            cv2.rectangle(vis, (bbox.x, bbox.y), (bbox.x2, bbox.y2), (255, 0, 255), 2)
            cv2.putText(vis, name, (bbox.x, bbox.y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # Draw state info
        info_lines = [
            f"Power: {state.power}/{state.power_max}",
            f"Credits: {state.credits}",
            f"Alarm: {state.alarm_level}",
            f"Mode: {state.mode.value}",
        ]
        
        y_offset = 20
        for line in info_lines:
            cv2.putText(vis, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_offset += 20
        
        filename = 'debug_hierarchical_detection.png'
        cv2.imwrite(filename, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        self.debug_images.append(filename)
        print(f"\nDebug visualization: {filename}")


# Test harness
if __name__ == "__main__":
    print("="*60)
    print("HIERARCHICAL GAME STATE DETECTOR")
    print("="*60)
    
    # You'll need to provide your WindowDetector
    # from window_detector import WindowDetector
    # detector = HierarchicalGameStateDetector(WindowDetector())
    
    # Load templates (you'll need to create these)
    # detector.load_templates('./templates')
    
    # state = detector.get_game_state()
    # if state:
    #     print(f"\nDetected state:")
    #     print(f"  Power: {state.power}/{state.power_max}")
    #     print(f"  Credits: {state.credits}")
    #     print(f"  Alarm: {state.alarm_level}")
    #     print(f"  Mode: {state.mode}")
    
    print("\nNote: This is the framework. You'll need to:")
    print("  1. Integrate your WindowDetector")
    print("  2. Create template images for key UI elements")
    print("  3. Test and refine each detector")