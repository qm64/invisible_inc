"""
Structural Detector for Invisible Inc Game State
Uses edge detection, shape analysis, and spatial relationships to find UI anchors

Version: 1.3.0
Changes in v1.3.0:
- MAJOR: Improved viewport inference for opponent turn frames
- Added temporal consistency: uses last known viewport when primary anchors missing
- New viewport detection strategy using hamburger + tactical_view (always present)
- Relaxed fallback requirements: only need 2 edges instead of 3
- Added reset_temporal_cache() method for new session handling
- Now handles opponent turns much better (previously failed viewport detection)
Changes in v1.2.5:
- Fixed AP value extraction to work with all agent AP values (0-99)
- Expanded AP label bounding boxes vertically (±2px) to prevent digit squashing
- Added 7 different preprocessing methods for robust OCR
- Lower thresholds (60, 80) to capture fainter text
- Successfully extracts all agent AP values
Changes in v1.2.4:
- Changed AP value extraction to use brightness threshold instead of HSV cyan filtering
- More robust AP number detection (threshold at gray > 100)
Changes in v1.2.3:
- Fixed agent icon X position: now at viewport_x + 2
- Added extensive debug output for AP value extraction
- Store AP labels as DetectedElements for OCR extraction
Changes in v1.2.2:
- Store AP label bbox as DetectedElement (ap_label_1, ap_label_2, etc.)
- Updated AP value extraction to search LEFT of detected "AP" text for numbers
- Much more accurate AP value reading using actual detected positions
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import json

VERSION = "1.3.0"


@dataclass
class DetectedElement:
    """Represents a detected UI element"""
    name: str
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[int, int]
    confidence: float
    method: str  # How it was detected


class StructuralDetector:
    """Detects UI elements using structural/geometric features"""
    
    # Class-level variable for temporal consistency across frames
    _last_known_viewport = None
    
    def __init__(self, debug=False):
        self.debug = debug
        self.detected_elements = {}
        self.img_width = 0
        self.img_height = 0
    
    @classmethod
    def reset_temporal_cache(cls):
        """Reset the temporal viewport cache (call when starting new session)"""
        cls._last_known_viewport = None
        
    def detect_anchors(self, image: np.ndarray) -> dict:
        """
        Detect all level 1 anchor elements in the image
        
        Primary anchors:
        1. Power/Credit text (upper left) - PRIMARY ANCHOR
        2. Menu hamburger (upper right)
        3. Security clock (upper right)
        4. Tactical view polygon (top center)
        5. End turn polygon (lower right)
        
        Optional:
        6. Profile rectangle (lower left - absent during movement)
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            dict of DetectedElement objects keyed by element name
        """
        self.detected_elements = {}
        self.img_height, self.img_width = image.shape[:2]
        
        # Detect in order of reliability - PWR/CREDITS FIRST (primary anchor)
        self._detect_power_credit_text(image)
        self._detect_hamburger_menu(image)
        self._detect_security_clock(image)
        self._detect_tactical_view(image)
        self._detect_end_turn_polygon(image)
        
        # Additional UI elements
        self._detect_alarm(image)
        self._detect_daemons(image)
        self._detect_objectives(image)
        
        # Profile rectangle must be detected BEFORE agent icons (anchor dependency)
        self._detect_profile_rectangle(image)
        
        # Agent icons depend on profile rectangle as anchor
        self._detect_agent_icons(image)
        
        # Agent-specific panels (when agent selected)
        self._detect_augments(image)
        self._detect_inventory(image)
        self._detect_quick_actions(image)
        
        return self.detected_elements
    
    def _detect_power_credit_text(self, image: np.ndarray):
        """Detect full power/credits region in upper left - PRIMARY ANCHOR"""
        h, w = image.shape[:2]
        
        # Phase 1: Find PWR anchor (cyan text)
        search_region = image[0:h//4, 0:w//3]
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        
        # Cyan color range for PWR label
        lower_cyan = np.array([85, 100, 100])
        upper_cyan = np.array([95, 255, 255])
        cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        
        # Dilate slightly to connect text characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        cyan_mask = cv2.dilate(cyan_mask, kernel, iterations=1)
        
        # Find PWR anchor
        contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pwr_anchor = None
        if contours:
            text_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:
                    x, y, tw, th = cv2.boundingRect(contour)
                    if tw > th and tw > 30:  # Horizontal text
                        text_regions.append((x, y, tw, th))
            
            if text_regions:
                text_regions.sort(key=lambda r: (r[1], r[0]))
                pwr_anchor = text_regions[0]
        
        if not pwr_anchor:
            if self.debug:
                print("✗ PWR anchor not found")
            return
        
        pwr_x, pwr_y, pwr_w, pwr_h = pwr_anchor
        
        if self.debug:
            print(f"  Found PWR anchor at ({pwr_x}, {pwr_y})")
        
        # Phase 2: Expand search region around PWR to capture full text
        expand_left = 100
        expand_right = 200
        expand_vertical = 20
        
        search_x = max(0, pwr_x - expand_left)
        search_y = max(0, pwr_y - expand_vertical)
        search_w = min(search_region.shape[1] - search_x, expand_left + pwr_w + expand_right)
        search_h = min(search_region.shape[0] - search_y, pwr_h + 2 * expand_vertical)
        
        expanded_region = search_region[search_y:search_y+search_h, 
                                       search_x:search_x+search_w]
        
        # Phase 3: Find ALL text in expanded region
        gray = cv2.cvtColor(expanded_region, cv2.COLOR_BGR2GRAY)
        _, text_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        hsv_expanded = cv2.cvtColor(expanded_region, cv2.COLOR_BGR2HSV)
        
        # Cyan range
        cyan_mask_expanded = cv2.inRange(hsv_expanded, 
                                         np.array([85, 100, 100]), 
                                         np.array([95, 255, 255]))
        
        # Green range (for credits)
        green_mask = cv2.inRange(hsv_expanded,
                                np.array([40, 100, 100]),
                                np.array([80, 255, 255]))
        
        # Combine all text masks
        combined_mask = cv2.bitwise_or(text_mask, cyan_mask_expanded)
        combined_mask = cv2.bitwise_or(combined_mask, green_mask)
        
        # Dilate to connect characters
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        combined_mask = cv2.dilate(combined_mask, kernel_h, iterations=2)
        
        # Find all text regions
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            if self.debug:
                print("✗ No text found in expanded region")
            return
        
        # Phase 4: Find horizontal text regions near PWR vertical position
        text_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 30:
                x, y, tw, th = cv2.boundingRect(contour)
                
                if tw > th:
                    text_center_y = y + th // 2
                    pwr_center_y = (pwr_y - search_y) + pwr_h // 2
                    
                    if abs(text_center_y - pwr_center_y) < 15:
                        text_boxes.append((x, y, tw, th))
        
        if not text_boxes:
            if self.debug:
                print("✗ No horizontal text found near PWR")
            return
        
        # Phase 5: Create bounding box from leftmost to rightmost
        leftmost = min(box[0] for box in text_boxes)
        rightmost = max(box[0] + box[2] for box in text_boxes)
        topmost = min(box[1] for box in text_boxes)
        bottommost = max(box[1] + box[3] for box in text_boxes)
        
        # Convert back to full image coordinates
        final_x = search_x + leftmost
        final_y = search_y + topmost
        final_w = rightmost - leftmost
        final_h = bottommost - topmost
        
        element = DetectedElement(
            name="power_text",
            bbox=(final_x, final_y, final_w, final_h),
            center=(final_x + final_w // 2, final_y + final_h // 2),
            confidence=0.9,
            method="power_credits_full_text"
        )
        self.detected_elements["power_text"] = element
        
        if self.debug:
            print(f"✓ Found power/credits at ({final_x}, {final_y}) size {final_w}×{final_h}")
    
    def _detect_hamburger_menu(self, image: np.ndarray):
        """Detect hamburger menu (3 horizontal white lines)"""
        h, w = image.shape[:2]
        
        region_x_offset = 2 * w // 3
        region_y_offset = 0
        search_region = image[0:h//3, region_x_offset:w]
        
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        lines = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) >= 3:
            line_boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 15 and h < 5:
                    line_boxes.append((x, y, w, h))
            
            line_boxes.sort(key=lambda b: b[1])
            
            if len(line_boxes) >= 3:
                y_diffs = [line_boxes[i+1][1] - line_boxes[i][1] for i in range(len(line_boxes)-1)]
                
                if len(y_diffs) >= 2:
                    avg_spacing = np.mean(y_diffs[:3])
                    if 5 < avg_spacing < 15:
                        local_x = min(b[0] for b in line_boxes[:3])
                        local_y = line_boxes[0][1]
                        local_w = max(b[0] + b[2] for b in line_boxes[:3]) - local_x
                        local_h = line_boxes[2][1] + line_boxes[2][3] - local_y
                        
                        x = local_x + region_x_offset
                        y = local_y + region_y_offset
                        
                        element = DetectedElement(
                            name="hamburger_menu",
                            bbox=(x, y, local_w, local_h),
                            center=(x + local_w // 2, y + local_h // 2),
                            confidence=0.95,
                            method="horizontal_lines_pattern"
                        )
                        self.detected_elements["hamburger_menu"] = element
                        
                        if self.debug:
                            print(f"✓ Found hamburger menu at ({x}, {y})")
                        return
        
        if self.debug:
            print("✗ Hamburger menu not found")
    
    def _detect_security_clock(self, image: np.ndarray):
        """Detect security clock (circular with red/orange arc)"""
        h, w = image.shape[:2]
        
        region_x_offset = max(0, w - 500)
        region_y_offset = 0
        search_region = image[0:h//3, region_x_offset:w]
        
        if self.debug:
            print(f"  Security clock search region: x={region_x_offset}, y=0, w={w-region_x_offset}, h={h//3}")
        
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([20, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        if self.debug:
            red_pixel_count = np.count_nonzero(red_mask)
            print(f"  Found {red_pixel_count} red pixels in search region")
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if self.debug:
            print(f"  Found {len(contours)} contours")
        
        best_candidate = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if 5000 < area < 35000:
                x, y, cw, ch = cv2.boundingRect(contour)
                
                if ch > cw * 2.2:
                    if self.debug:
                        print(f"    Rejected contour at ({x},{y}) size {cw}x{ch}: too tall")
                    continue
                
                size_match = 1.0 - abs(cw - 155) / 155 if cw < 300 else 0
                size_match *= 1.0 - abs(ch - 170) / 170 if ch < 300 else 0
                size_match = max(0, size_match)
                
                aspect_ratio = cw / ch if ch > 0 else 0
                
                if 0.5 < aspect_ratio < 2.0:
                    circularity_score = 1.0 - abs(aspect_ratio - 1.0)
                    vertical_score = 1.0 - (y / (h // 3))
                    size_score = min(area / 15000, 1.0)
                    
                    score = (circularity_score * 0.3 + 
                            vertical_score * 0.3 + 
                            size_score * 0.2 +
                            size_match * 0.2)
                    
                    if self.debug:
                        print(f"    Contour at ({x},{y}) size {cw}x{ch} area={area:.0f} score={score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = (x, y, cw, ch)
        
        if best_candidate and best_score > 0.3:
            x, y, cw, ch = best_candidate
            x += region_x_offset
            y += region_y_offset
            
            element = DetectedElement(
                name="security_clock",
                bbox=(x, y, cw, ch),
                center=(x + cw // 2, y + ch // 2),
                confidence=min(0.85, best_score),
                method="red_circular_shape"
            )
            self.detected_elements["security_clock"] = element
            
            if self.debug:
                print(f"✓ Found security clock at ({x}, {y}) with score {best_score:.3f}")
        elif self.debug:
            print(f"✗ Security clock not found (best score: {best_score:.3f})")
    
    def _detect_tactical_view(self, image: np.ndarray):
        """Detect tactical view polygon at top center"""
        h, w = image.shape[:2]
        
        region_x_offset = w // 4
        region_y_offset = 0
        search_region = image[0:h//8, region_x_offset:3*w//4]
        
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        
        lower_teal = np.array([80, 50, 50])
        upper_teal = np.array([100, 255, 255])
        teal_mask = cv2.inRange(hsv, lower_teal, upper_teal)
        
        contours, _ = cv2.findContours(teal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        best_score = 0
        center_x = search_region.shape[1] // 2
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 5000:
                x, y, pw, ph = cv2.boundingRect(contour)
                
                contour_center_x = x + pw // 2
                horizontal_distance = abs(contour_center_x - center_x) / center_x
                vertical_score = 1.0 - (y / (h // 8))
                
                score = vertical_score * (1.0 - horizontal_distance)
                
                if score > best_score:
                    best_score = score
                    best_candidate = (x, y, pw, ph)
        
        if best_candidate:
            x, y, pw, ph = best_candidate
            x += region_x_offset
            y += region_y_offset
            
            element = DetectedElement(
                name="tactical_view",
                bbox=(x, y, pw, ph),
                center=(x + pw // 2, y + ph // 2),
                confidence=0.8,
                method="teal_polygon_top_center"
            )
            self.detected_elements["tactical_view"] = element
            
            if self.debug:
                print(f"✓ Found tactical view at ({x}, {y})")
        elif self.debug:
            print("✗ Tactical view not found")
    
    def _detect_end_turn_polygon(self, image: np.ndarray):
        """Detect end turn polygon (cyan, lower right)"""
        h, w = image.shape[:2]
        
        region_x_offset = 2 * w // 3
        region_y_offset = 2 * h // 3
        search_region = image[region_y_offset:h, region_x_offset:w]
        
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        lower_cyan = np.array([85, 100, 100])
        upper_cyan = np.array([95, 255, 255])
        cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        
        contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, ew, eh = cv2.boundingRect(contour)
                
                x += region_x_offset
                y += region_y_offset
                
                element = DetectedElement(
                    name="end_turn",
                    bbox=(x, y, ew, eh),
                    center=(x + ew // 2, y + eh // 2),
                    confidence=0.9,
                    method="cyan_polygon_bottom_right"
                )
                self.detected_elements["end_turn"] = element
                
                if self.debug:
                    print(f"✓ Found end turn at ({x}, {y})")
                return
        
        if self.debug:
            print("✗ End turn not found")
    
    def _detect_alarm(self, image: np.ndarray):
        """Detect alarm indicator (red icon in upper right)"""
        h, w = image.shape[:2]
        
        expected_x = w - 230
        expected_y = 132
        
        search_x = max(0, expected_x - 50)
        search_y = max(0, expected_y - 50)
        search_w = min(w - search_x, 150)
        search_h = min(h - search_y, 150)
        
        search_region = image[search_y:search_y+search_h, search_x:search_x+search_w]
        
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 2000:
                x, y, aw, ah = cv2.boundingRect(contour)
                
                x += search_x
                y += search_y
                
                element = DetectedElement(
                    name="alarm",
                    bbox=(x, y, aw, ah),
                    center=(x + aw // 2, y + ah // 2),
                    confidence=0.8,
                    method="red_icon_upper_right"
                )
                self.detected_elements["alarm"] = element
                
                if self.debug:
                    print(f"✓ Found alarm at ({x}, {y})")
                return
        
        if self.debug:
            print("⚠ Alarm not found (may be inactive)")
    
    def _detect_daemons(self, image: np.ndarray):
        """Detect daemon indicators panel (below security clock)"""
        h, w = image.shape[:2]
        
        expected_x = w - 290
        expected_y = 230
        
        search_x = max(0, expected_x - 50)
        search_y = max(0, expected_y - 50)
        search_w = min(w - search_x, 250)
        search_h = min(h - search_y, 250)
        
        search_region = image[search_y:search_y+search_h, search_x:search_x+search_w]
        
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([20, 255, 255])
        lower_red2 = np.array([160, 80, 80])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 2000 < area < 25000:
                x, y, dw, dh = cv2.boundingRect(contour)
                
                x += search_x
                y += search_y
                
                element = DetectedElement(
                    name="daemons",
                    bbox=(x, y, dw, dh),
                    center=(x + dw // 2, y + dh // 2),
                    confidence=0.75,
                    method="red_panel_below_clock"
                )
                self.detected_elements["daemons"] = element
                
                if self.debug:
                    print(f"✓ Found daemons panel at ({x}, {y})")
                return
        
        if self.debug:
            print("⚠ Daemons panel not found")
    
    def _detect_objectives(self, image: np.ndarray):
        """Detect objectives panel (lower right)"""
        h, w = image.shape[:2]
        
        expected_x = w - 422
        expected_y = h - 242
        
        search_x = max(0, expected_x - 50)
        search_y = max(0, expected_y - 50)
        search_w = min(w - search_x, 400)
        search_h = min(h - search_y, 300)
        
        search_region = image[search_y:search_y+search_h, search_x:search_x+search_w]
        
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        _, text_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        lower_cyan = np.array([85, 100, 100])
        upper_cyan = np.array([95, 255, 255])
        cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        
        combined = cv2.bitwise_or(text_mask, cyan_mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        combined = cv2.dilate(combined, kernel, iterations=2)
        
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        best_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:
                x, y, ow, oh = cv2.boundingRect(contour)
                
                if y > search_h // 3:
                    if area > best_area:
                        best_area = area
                        best_candidate = (x, y, ow, oh)
        
        if best_candidate:
            x, y, ow, oh = best_candidate
            x += search_x
            y += search_y
            
            element = DetectedElement(
                name="objectives",
                bbox=(x, y, ow, oh),
                center=(x + ow // 2, y + oh // 2),
                confidence=0.7,
                method="text_panel_lower_right"
            )
            self.detected_elements["objectives"] = element
            
            if self.debug:
                print(f"✓ Found objectives panel at ({x}, {y})")
        elif self.debug:
            print("⚠ Objectives panel not found")
    
    def _detect_profile_rectangle(self, image: np.ndarray):
        """Detect profile rectangle in lower left (OPTIONAL)"""
        h, w = image.shape[:2]
        
        region_x_offset = 0
        region_y_offset = 2 * h // 3
        search_region = image[region_y_offset:h, 0:w//3]
        
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:
                x, y, pw, ph = cv2.boundingRect(contour)
                aspect_ratio = pw / ph if ph > 0 else 0
                
                if 0.8 < aspect_ratio < 1.5:
                    x += region_x_offset
                    y += region_y_offset
                    
                    element = DetectedElement(
                        name="profile_rectangle",
                        bbox=(x, y, pw, ph),
                        center=(x + pw // 2, y + ph // 2),
                        confidence=0.7,
                        method="rectangular_region_lower_left"
                    )
                    self.detected_elements["profile_rectangle"] = element
                    
                    if self.debug:
                        print(f"✓ Found profile rectangle at ({x}, {y})")
                    return
        
        if self.debug:
            print("⚠ Profile rectangle not found (may be absent)")
    
    def _detect_agent_icons(self, image: np.ndarray):
        """Detect agent portrait icons by finding their AP (action points) labels"""
        if "power_text" not in self.detected_elements:
            if self.debug:
                print("⚠ Cannot detect agent icons without viewport")
            return
        
        if "profile_rectangle" not in self.detected_elements:
            if self.debug:
                print("⚠ Cannot detect agent icons without profile rectangle")
            return
        
        viewport_x = self.detected_elements["power_text"].bbox[0]
        profile = self.detected_elements["profile_rectangle"]
        profile_top_y = profile.bbox[1]
        
        h, w = image.shape[:2]
        
        # Search region: middle of viewport down to profile (avoids PWR/credits at top)
        # AP text appears to the right of icons at x ~190-220
        search_x = viewport_x + 40  # Start after icons
        search_w = 80  # Width to capture "XX AP" text
        search_y = h // 2  # Start from middle of screen (avoids top PWR/credits)
        search_h = profile_top_y - search_y - 20  # Down to just above profile
        
        if search_h < 50:
            if self.debug:
                print("⚠ Search region too small (profile too close to middle)")
            return
        
        if self.debug:
            print(f"  Searching for AP labels in region: ({search_x},{search_y},{search_w},{search_h})")
        
        search_region = image[search_y:search_y+search_h, search_x:search_x+search_w]
        
        # Find cyan/teal "AP" text
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        
        # Cyan/teal range for "AP" text
        lower_cyan = np.array([85, 100, 100])
        upper_cyan = np.array([95, 255, 255])
        cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        
        # Dilate horizontally to connect "A" and "P"
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        cyan_mask = cv2.dilate(cyan_mask, kernel_h, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if self.debug:
            print(f"  Found {len(contours)} cyan text regions")
        
        # Find "AP" text regions
        ap_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 20:  # Small text
                x, y, tw, th = cv2.boundingRect(contour)
                
                # "AP" should be roughly horizontal text
                if tw > th and tw > 8:
                    # Use FIXED width to ensure we always capture "XX AP"
                    fixed_width = 65
                    expand_left = 25
                    
                    expanded_x = max(0, x - expand_left)
                    
                    # Expand vertically by 2 pixels up and down
                    expand_vertical = 2
                    expanded_y = max(0, y - expand_vertical)
                    expanded_h = th + 2 * expand_vertical
                    
                    center_y = y + th // 2
                    full_y = search_y + center_y
                    full_x = search_x + expanded_x
                    
                    ap_regions.append({
                        'center_y': full_y,
                        'bbox': (full_x, search_y + expanded_y, fixed_width, expanded_h),
                        'local_y': y
                    })
        
        if not ap_regions:
            if self.debug:
                print("⚠ No AP labels found")
            return
        
        # Sort AP regions by Y position (top to bottom)
        ap_regions.sort(key=lambda r: r['center_y'])
        
        if self.debug:
            print(f"  Found {len(ap_regions)} AP labels")
        
        # For each AP label, infer the icon position AND store the AP label
        for idx, ap_region in enumerate(ap_regions, 1):
            ap_center_y = ap_region['center_y']
            ap_bbox = ap_region['bbox']
            
            # Store the AP label as a DetectedElement for later OCR extraction
            ap_element = DetectedElement(
                name=f"ap_label_{idx}",
                bbox=ap_bbox,
                center=(ap_bbox[0] + ap_bbox[2] // 2, ap_bbox[1] + ap_bbox[3] // 2),
                confidence=0.8,
                method="cyan_text_ap_label"
            )
            self.detected_elements[f"ap_label_{idx}"] = ap_element
            
            # Icon is approximately at the left edge of viewport
            # Icons are 36x36, starting at roughly viewport_x + 2px
            icon_x = viewport_x + 2
            icon_y = ap_center_y - 18  # Center icon vertically on AP
            icon_w = 36
            icon_h = 36
            icon_center_x = icon_x + 18
            icon_center_y = ap_center_y
            
            element = DetectedElement(
                name=f"agent_icon_{idx}",
                bbox=(icon_x, icon_y, icon_w, icon_h),
                center=(icon_center_x, icon_center_y),
                confidence=0.75,
                method="inferred_from_ap_label"
            )
            self.detected_elements[f"agent_icon_{idx}"] = element
            
            if self.debug:
                print(f"✓ Inferred agent icon {idx} at ({icon_x}, {icon_y}) from AP at y={ap_center_y}")
        
        if self.debug:
            print(f"✓ Detected {len(ap_regions)} agent icons via AP labels")
    
    def _detect_augments(self, image: np.ndarray):
        """Detect augments panel (lower left, when agent selected)"""
        if "power_text" not in self.detected_elements:
            return
        
        viewport_x = self.detected_elements["power_text"].bbox[0]
        h, w = image.shape[:2]
        
        expected_x = viewport_x + 184
        expected_y = h - 171
        
        search_x = max(0, expected_x - 30)
        search_y = max(0, expected_y - 30)
        search_w = min(w - search_x, 250)
        search_h = min(h - search_y, 120)
        
        search_region = image[search_y:search_y+search_h, search_x:search_x+search_w]
        
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        _, text_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        text_mask = cv2.dilate(text_mask, kernel, iterations=2)
        
        contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:
                x, y, aw, ah = cv2.boundingRect(contour)
                
                x += search_x
                y += search_y
                
                element = DetectedElement(
                    name="augments",
                    bbox=(x, y, aw, ah),
                    center=(x + aw // 2, y + ah // 2),
                    confidence=0.6,
                    method="text_panel_lower_left"
                )
                self.detected_elements["augments"] = element
                
                if self.debug:
                    print(f"✓ Found augments panel at ({x}, {y})")
                return
        
        if self.debug:
            print("⚠ Augments panel not found (agent not selected)")
    
    def _detect_inventory(self, image: np.ndarray):
        """Detect inventory panel (lower left, when agent selected)"""
        if "power_text" not in self.detected_elements:
            return
        
        viewport_x = self.detected_elements["power_text"].bbox[0]
        h, w = image.shape[:2]
        
        expected_x = viewport_x + 393
        expected_y = h - 171
        
        search_x = max(0, expected_x - 30)
        search_y = max(0, expected_y - 30)
        search_w = min(w - search_x, 350)
        search_h = min(h - search_y, 120)
        
        search_region = image[search_y:search_y+search_h, search_x:search_x+search_w]
        
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        _, text_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        lower_items = np.array([20, 50, 50])
        upper_items = np.array([100, 255, 255])
        items_mask = cv2.inRange(hsv, lower_items, upper_items)
        
        combined = cv2.bitwise_or(text_mask, items_mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined = cv2.dilate(combined, kernel, iterations=2)
        
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 3000:
                x, y, iw, ih = cv2.boundingRect(contour)
                
                x += search_x
                y += search_y
                
                element = DetectedElement(
                    name="inventory",
                    bbox=(x, y, iw, ih),
                    center=(x + iw // 2, y + ih // 2),
                    confidence=0.6,
                    method="items_panel_lower_left"
                )
                self.detected_elements["inventory"] = element
                
                if self.debug:
                    print(f"✓ Found inventory panel at ({x}, {y})")
                return
        
        if self.debug:
            print("⚠ Inventory panel not found (agent not selected)")
    
    def _detect_quick_actions(self, image: np.ndarray):
        """Detect quick actions panel (lower left, when agent selected)"""
        if "power_text" not in self.detected_elements:
            return
        
        viewport_x = self.detected_elements["power_text"].bbox[0]
        h, w = image.shape[:2]
        
        expected_x = viewport_x + 183
        expected_y = h - 91
        
        search_x = max(0, expected_x - 80)
        search_y = max(0, expected_y - 30)
        search_w = min(w - search_x, 300)
        search_h = min(h - search_y, 120)
        
        search_region = image[search_y:search_y+search_h, search_x:search_x+search_w]
        
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        _, text_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 5))
        text_mask = cv2.dilate(text_mask, kernel, iterations=2)
        
        contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:
                x, y, qw, qh = cv2.boundingRect(contour)
                
                x += search_x
                y += search_y
                
                element = DetectedElement(
                    name="quick_actions",
                    bbox=(x, y, qw, qh),
                    center=(x + qw // 2, y + qh // 2),
                    confidence=0.6,
                    method="actions_panel_lower_left"
                )
                self.detected_elements["quick_actions"] = element
                
                if self.debug:
                    print(f"✓ Found quick actions panel at ({x}, {y})")
                return
        
        if self.debug:
            print("⚠ Quick actions panel not found (agent not selected)")
    
    def _ocr_power_credits_region(self, image: np.ndarray) -> Optional[str]:
        """Internal helper: OCR the power/credits text region"""
        if "power_text" not in self.detected_elements:
            return None
        
        try:
            import pytesseract
        except ImportError:
            if self.debug:
                print("⚠ pytesseract not installed, cannot extract text values")
            return None
        
        bbox = self.detected_elements["power_text"].bbox
        x, y, w, h = bbox
        
        pad = 5
        region_y = max(0, y - pad)
        region_h = min(h + 2*pad, image.shape[0] - region_y)
        
        region = image[region_y:region_y+region_h, x:x+w]
        
        if self.debug:
            print(f"  OCR region: {w}×{region_h}px")
        
        scale = 5
        region_scaled = cv2.resize(region, None, fx=scale, fy=scale, 
                                   interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(region_scaled, cv2.COLOR_BGR2GRAY)
        
        _, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        
        config = '--psm 7 -c tessedit_char_whitelist=0123456789/PWR '
        
        results = []
        for i, thresh in enumerate([thresh1, thresh2, thresh3], 1):
            text = pytesseract.image_to_string(thresh, config=config).strip()
            if text:
                results.append(text)
                if self.debug:
                    print(f"  OCR attempt {i}: '{text}'")
        
        import re
        for text in results:
            if re.search(r'\d+\s*/\s*\d+\s+PWR\s+\d{4,6}', text):
                return text
        
        return results[0] if results else None
    
    def extract_power_value(self, image: np.ndarray) -> Optional[str]:
        """Extract power value from power/credits region"""
        text = self._ocr_power_credits_region(image)
        if not text:
            return None
        
        import re
        match = re.search(r'(\d+)\s*/\s*(\d+)\s+PWR\s+(\d+)', text)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
        
        match = re.search(r'(\d+)\s*/\s*(\d+)', text)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
        
        return None
    
    def extract_credits_value(self, image: np.ndarray) -> Optional[str]:
        """Extract credits value from power/credits region"""
        text = self._ocr_power_credits_region(image)
        if not text:
            return None
        
        import re
        match = re.search(r'(\d+)\s*/\s*(\d+)\s+PWR\s+(\d+)', text)
        if match:
            credits = match.group(3)
            if self.debug:
                print(f"  ✓ Parsed: power={match.group(1)}/{match.group(2)}, credits={credits}")
            return credits
        
        all_numbers = re.findall(r'\d+', text)
        
        if self.debug:
            print(f"  Fallback - all numbers found: {all_numbers}")
        
        likely_credits = [n for n in all_numbers if len(n) >= 3 or int(n) > 20]
        
        if likely_credits:
            return max(likely_credits, key=len)
        
        if all_numbers:
            return all_numbers[-1]
        
        return None
    
    def extract_agent_ap_values(self, image: np.ndarray) -> dict:
        """Extract AP (action points) values for each detected agent"""
        try:
            import pytesseract
        except ImportError:
            if self.debug:
                print("⚠ pytesseract not installed, cannot extract AP values")
            return {}
        
        ap_values = {}
        ap_labels = [name for name in self.detected_elements.keys() if name.startswith("ap_label_")]
        
        if self.debug:
            print(f"\n  Extracting AP values for {len(ap_labels)} labels: {sorted(ap_labels)}")
        
        for ap_name in sorted(ap_labels):
            ap_label = self.detected_elements[ap_name]
            ap_x, ap_y, ap_w, ap_h = ap_label.bbox
            
            if self.debug:
                print(f"  Processing {ap_name}: AP bbox at ({ap_x}, {ap_y}, {ap_w}, {ap_h})")
            
            region = image[ap_y:ap_y+ap_h, ap_x:ap_x+ap_w]
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # Scale up first
            scale = 6
            gray_scaled = cv2.resize(gray, None, fx=scale, fy=scale, 
                                    interpolation=cv2.INTER_CUBIC)
            
            # Try multiple preprocessing methods
            preprocessed_images = []
            
            # Method 1: Lower threshold to capture fainter text
            _, method1 = cv2.threshold(gray_scaled, 80, 255, cv2.THRESH_BINARY)
            preprocessed_images.append(("binary_80", method1))
            
            # Method 1b: Even lower threshold
            _, method1b = cv2.threshold(gray_scaled, 60, 255, cv2.THRESH_BINARY)
            preprocessed_images.append(("binary_60", method1b))
            
            # Method 2: Otsu's threshold
            _, method2 = cv2.threshold(gray_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_images.append(("otsu", method2))
            
            # Method 3: Adaptive threshold with larger block size
            method3 = cv2.adaptiveThreshold(gray_scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 15, 2)
            preprocessed_images.append(("adaptive_15", method3))
            
            # Method 4: Inverted binary with lower threshold
            _, method4 = cv2.threshold(gray_scaled, 80, 255, cv2.THRESH_BINARY_INV)
            preprocessed_images.append(("inverted_80", method4))
            
            # Method 5: Morphological closing with lower threshold and larger kernel
            _, method5 = cv2.threshold(gray_scaled, 80, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            method5 = cv2.morphologyEx(method5, cv2.MORPH_CLOSE, kernel)
            preprocessed_images.append(("morpho_80", method5))
            
            # Method 6: Original binary threshold (100) for comparison
            _, method6 = cv2.threshold(gray_scaled, 100, 255, cv2.THRESH_BINARY)
            preprocessed_images.append(("binary_100", method6))
            
            # Try OCR on each preprocessing method
            import re
            best_result = None
            best_confidence = 0
            
            config = '--psm 7 -c tessedit_char_whitelist=0123456789AP '
            
            for method_name, img in preprocessed_images:
                text = pytesseract.image_to_string(img, config=config).strip()
                match = re.search(r'(\d+)\s*AP', text)
                
                if match:
                    ap_num = int(match.group(1))
                    # Confidence heuristic: prefer results in valid range 0-99
                    if 0 <= ap_num <= 99:
                        confidence = 1.0
                        # Prefer results where we found both digit and AP
                        if "AP" in text:
                            confidence += 0.5
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_result = (ap_num, method_name, text)
                
                if self.debug:
                    print(f"    {method_name}: '{text}'")
            
            if best_result:
                ap_value, method_name, ocr_text = best_result
                icon_idx = ap_name.replace("ap_label_", "")
                icon_name = f"agent_icon_{icon_idx}"
                ap_values[icon_name] = ap_value
                if self.debug:
                    print(f"  ✓ {icon_name}: {ap_value} AP (method: {method_name})")
            elif self.debug:
                print(f"  ✗ {ap_name}: Could not extract AP value")
        
        return ap_values
    
    def extract_security_level(self, image: np.ndarray) -> Optional[str]:
        """Extract security level number from security clock"""
        if "security_clock" not in self.detected_elements:
            return None
        
        try:
            import pytesseract
        except ImportError:
            return None
        
        bbox = self.detected_elements["security_clock"].bbox
        x, y, w, h = bbox
        
        region = image[y:y+h, x:x+w]
        
        center_x = w // 2
        center_y = h // 2
        size = min(w, h) // 3
        
        center_region = region[center_y-size:center_y+size, 
                              center_x-size:center_x+size]
        
        center_region = cv2.resize(center_region, None, fx=4, fy=4, 
                                   interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        config = '--psm 10 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(thresh, config=config).strip()
        
        import re
        match = re.search(r'\d+', text)
        if match:
            return match.group(0)
        
        return None
    
    def infer_viewport(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Infer viewport boundaries from detected anchors
        
        Strategy:
        1. If power_text present (player frames): Use it as primary anchor
        2. If not (opponent frames): Use hamburger + tactical_view + security_clock
        3. Fallback: Use last known viewport (temporal consistency)
        4. Final fallback: Estimate from always-present elements
        """
        if not self.detected_elements:
            return None
        
        # Method 1: Power/Credits text (most reliable when present)
        if "power_text" in self.detected_elements:
            power = self.detected_elements["power_text"]
            
            viewport_x = power.bbox[0]
            viewport_y = max(0, power.bbox[1] - 10)
            viewport_w = self.img_width - viewport_x * 2
            viewport_h = self.img_height - viewport_y * 2
            
            viewport = (viewport_x, viewport_y, viewport_w, viewport_h)
            
            # Store for temporal consistency
            StructuralDetector._last_known_viewport = viewport
            
            if self.debug:
                print(f"✓ Inferred viewport from power/credits: {viewport}")
            
            return viewport
        
        # Method 2: Use always-present elements (hamburger + tactical_view)
        # These are reliable even on opponent turns
        if "hamburger_menu" in self.detected_elements and "tactical_view" in self.detected_elements:
            hamburger = self.detected_elements["hamburger_menu"]
            tactical = self.detected_elements["tactical_view"]
            
            # Hamburger is in upper-right corner
            # Its left edge approximates right edge of viewport
            hamburger_right = hamburger.bbox[0] + hamburger.bbox[2]
            viewport_right = hamburger_right
            
            # Tactical view is top-center
            # Its top edge approximates top of viewport
            tactical_top = tactical.bbox[1]
            viewport_top = tactical_top
            
            # Security clock can help refine if present
            if "security_clock" in self.detected_elements:
                clock = self.detected_elements["security_clock"]
                # Clock is upper-right, below hamburger
                # Can use to verify/refine right edge
                clock_right = clock.bbox[0] + clock.bbox[2]
                viewport_right = max(viewport_right, clock_right)
            
            # Use symmetry to estimate left edge
            # Viewport is typically centered with equal margins
            viewport_left = self.img_width - viewport_right
            
            # Estimate bottom from aspect ratio or last known viewport
            if StructuralDetector._last_known_viewport:
                # Use last known height
                _, _, _, last_h = StructuralDetector._last_known_viewport
                viewport_bottom = viewport_top + last_h
            else:
                # Estimate: typically ~1400px height for 1440p gameplay
                # Or use image height with small margin
                viewport_bottom = self.img_height - viewport_top
            
            viewport_w = viewport_right - viewport_left
            viewport_h = viewport_bottom - viewport_top
            
            # Sanity check
            if viewport_w > 0 and viewport_h > 0:
                viewport = (viewport_left, viewport_top, viewport_w, viewport_h)
                
                # Store for temporal consistency
                StructuralDetector._last_known_viewport = viewport
                
                if self.debug:
                    print(f"✓ Inferred viewport from hamburger + tactical_view: {viewport}")
                
                return viewport
        
        # Method 3: Use last known viewport (temporal consistency)
        # During opponent turns, viewport doesn't change frame-to-frame
        if StructuralDetector._last_known_viewport:
            if self.debug:
                print(f"✓ Using last known viewport (temporal): {StructuralDetector._last_known_viewport}")
            
            return StructuralDetector._last_known_viewport
        
        # Method 4: Fallback - attempt edge-based detection
        left = right = top = bottom = None
        
        # Try to find edges from any available elements
        candidates = []
        for name in ["tactical_view", "hamburger_menu"]:
            if name in self.detected_elements:
                candidates.append(self.detected_elements[name].bbox[1])
        if candidates:
            top = min(candidates)
        
        if "profile_rectangle" in self.detected_elements:
            left = self.detected_elements["profile_rectangle"].bbox[0]
        
        candidates = []
        for name in ["end_turn", "hamburger_menu"]:
            if name in self.detected_elements:
                bbox = self.detected_elements[name].bbox
                candidates.append(bbox[0] + bbox[2])
        if candidates:
            right = max(candidates)
        
        candidates = []
        for name in ["end_turn", "profile_rectangle"]:
            if name in self.detected_elements:
                bbox = self.detected_elements[name].bbox
                candidates.append(bbox[1] + bbox[3])
        if candidates:
            bottom = max(candidates)
        
        edges_found = sum(x is not None for x in [left, right, top, bottom])
        
        # Relaxed requirement: only need 2 edges if we have top and right
        if edges_found >= 2 and top is not None and right is not None:
            if left is None:
                left = self.img_width - right  # Symmetry assumption
            if bottom is None:
                bottom = self.img_height - top  # Use remaining space
            
            width = right - left
            height = bottom - top
            
            if width > 0 and height > 0:
                viewport = (left, top, width, height)
                
                # Store for temporal consistency
                StructuralDetector._last_known_viewport = viewport
                
                if self.debug:
                    print(f"✓ Inferred viewport (fallback with {edges_found} edges): {viewport}")
                
                return viewport
        
        if self.debug:
            print(f"✗ Insufficient anchors for viewport (only {edges_found} edges)")
        
        return None
    
    def visualize_detections(self, image: np.ndarray) -> np.ndarray:
        """Draw detected elements on image for debugging"""
        vis_image = image.copy()
        
        colors = {
            "power_text": (255, 255, 0),
            "hamburger_menu": (255, 255, 255),
            "security_clock": (0, 0, 255),
            "tactical_view": (0, 255, 0),
            "end_turn": (255, 255, 0),
            "profile_rectangle": (255, 0, 255)
        }
        
        for name, element in self.detected_elements.items():
            x, y, w, h = element.bbox
            color = colors.get(name, (128, 128, 128))
            
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            label = f"{name} ({element.confidence:.2f})"
            cv2.putText(vis_image, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        viewport = self.infer_viewport()
        if viewport:
            vx, vy, vw, vh = viewport
            cv2.rectangle(vis_image, (vx, vy), (vx + vw, vy + vh), (0, 255, 255), 3)
            cv2.putText(vis_image, "VIEWPORT", (vx + 10, vy + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return vis_image
    
    def save_detections(self, filepath: str):
        """Save detected elements to JSON file"""
        data = {
            name: {
                "bbox": element.bbox,
                "center": element.center,
                "confidence": element.confidence,
                "method": element.method
            }
            for name, element in self.detected_elements.items()
        }
        
        viewport = self.infer_viewport()
        if viewport:
            data["viewport"] = {
                "bbox": viewport,
                "inferred": True
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        if self.debug:
            print(f"✓ Saved detections to {filepath}")


def test_detector():
    """Test the structural detector on sample images"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python structural_detector.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image {image_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Testing Structural Detector v{VERSION} on: {image_path}")
    print(f"{'='*60}\n")
    
    detector = StructuralDetector(debug=True)
    elements = detector.detect_anchors(image)
    
    print(f"\n{'='*60}")
    print(f"Detection Summary:")
    print(f"{'='*60}")
    print(f"Found {len(elements)} elements:")
    for name, element in elements.items():
        print(f"  - {name}: confidence={element.confidence:.2f}, method={element.method}")
    
    viewport = detector.infer_viewport()
    if viewport:
        print(f"\nViewport: {viewport}")
    else:
        print("\nViewport: Could not infer")
    
    print(f"\n{'='*60}")
    print(f"Extracted Values:")
    print(f"{'='*60}")
    
    power_value = detector.extract_power_value(image)
    if power_value:
        print(f"Power: {power_value}")
    else:
        print("Power: Could not extract")
    
    credits_value = detector.extract_credits_value(image)
    if credits_value:
        print(f"Credits: {credits_value}")
    else:
        print("Credits: Could not extract")
    
    security_level = detector.extract_security_level(image)
    if security_level:
        print(f"Security Level: {security_level}")
    else:
        print("Security Level: Could not extract")
    
    # Extract agent AP values
    ap_values = detector.extract_agent_ap_values(image)
    if ap_values:
        print("\nAgent AP Values:")
        for agent, ap in sorted(ap_values.items()):
            print(f"  {agent}: {ap} AP")
    
    vis_image = detector.visualize_detections(image)
    output_path = image_path.replace('.png', '_detected.png')
    cv2.imwrite(output_path, vis_image)
    print(f"\n✓ Saved visualization to {output_path}")
    
    json_path = image_path.replace('.png', '_detected.json')
    detector.save_detections(json_path)


if __name__ == "__main__":
    test_detector()