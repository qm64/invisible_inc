from window_detector import WindowDetector
import cv2
import numpy as np
import pytesseract

class SequentialAdaptiveDetector:
    """
    Sequential search: Find each element relative to previous ones
    Order: Power → Credits → Alarm
    """
    
    def __init__(self):
        self.detector = WindowDetector()
        self.debug = True
    
    def find_viewport(self, window_img):
        """Find actual game content area"""
        green = window_img[:, :, 1]
        green_threshold = cv2.threshold(green, 80, 255, cv2.THRESH_BINARY)[1]
        
        rows_with_green = np.any(green_threshold > 0, axis=1)
        cols_with_green = np.any(green_threshold > 0, axis=0)
        
        if not np.any(rows_with_green) or not np.any(cols_with_green):
            return None, (0, 0)
        
        top = np.argmax(rows_with_green)
        bottom = len(rows_with_green) - np.argmax(rows_with_green[::-1])
        left = np.argmax(cols_with_green)
        right = len(cols_with_green) - np.argmax(cols_with_green[::-1])
        
        viewport = window_img[top:bottom, left:right]
        offset = (left, top)
        
        return viewport, offset
    
    def find_power(self, viewport):
        """
        Find power display in top-left
        Returns: (x, y, width, height) or None
        """
        vh, vw = viewport.shape[:2]
        
        # Search top-left area for green text
        search_w = int(vw * 0.15)  # First 15% of width
        search_h = int(vh * 0.08)  # First 8% of height
        search_area = viewport[0:search_h, 0:search_w]
        
        # Find green text (cyan)
        green = search_area[:, :, 1]
        _, mask = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        
        # Find bounding box
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        top = np.argmax(rows)
        bottom = len(rows) - np.argmax(rows[::-1])
        left = np.argmax(cols)
        right = len(cols) - np.argmax(cols[::-1])
        
        # Find where "PWR" text ends (look for gap in columns)
        # Scan from left to find end of power text
        power_right = left
        for x in range(left, right):
            col = mask[:, x]
            if np.sum(col) > 0:
                power_right = x
            elif power_right > left and (x - power_right) > 5:  # 5px gap = end of text
                break
        
        # Add small padding
        x = max(0, left - 2)
        y = max(0, top - 2)
        width = power_right - left + 15  # Extra for "PWR"
        height = bottom - top + 4
        
        return (x, y, width, height)
    
    def find_credits(self, viewport, power_region):
        """
        Find credits display - starts right after power
        Returns: (x, y, width, height) or None
        """
        if power_region is None:
            return None
        
        vh, vw = viewport.shape[:2]
        px, py, pw, ph = power_region
        
        # Search starts where power ends
        search_left = px + pw + 5  # 5px gap after power
        search_right = int(vw * 0.25)  # Up to 25% of viewport width
        search_top = py - 5  # Slightly above power (same line)
        search_bottom = py + ph + 5  # Slightly below power
        
        if search_left >= search_right or search_top >= search_bottom:
            return None
        
        search_area = viewport[search_top:search_bottom, search_left:search_right]
        
        # Find green text
        green = search_area[:, :, 1]
        _, mask = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        
        # Find bounding box
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        top = np.argmax(rows)
        bottom = len(rows) - np.argmax(rows[::-1])
        left = np.argmax(cols)
        right = len(cols) - np.argmax(cols[::-1])
        
        # Convert to viewport coordinates
        x = search_left + left - 2
        y = search_top + top - 2
        width = right - left + 10  # Extra for "CR"
        height = bottom - top + 4
        
        return (x, y, width, height)
    
    def find_alarm(self, viewport):
        """
        Find alarm level by locating "SECURITY LEVEL" text then the ring
        Returns: (x, y, width, height) of just the digit, or None
        """
        vh, vw = viewport.shape[:2]
        
        # Step 1: Find "SECURITY LEVEL" text in top-right
        search_left = int(vw * 0.80)  # Right 20% of viewport
        search_top = 0
        search_right = vw
        search_bottom = int(vh * 0.15)  # Top 15%
        
        search_area = viewport[search_top:search_bottom, search_left:search_right]
        
        # Look for yellow text (high R, G, low B)
        red = search_area[:, :, 0]
        green = search_area[:, :, 1]
        blue = search_area[:, :, 2]
        
        # Yellow has high red AND green
        yellow_mask = ((red > 150) & (green > 150) & (blue < 100)).astype(np.uint8) * 255
        
        # Find "SECURITY LEVEL" bounding box
        rows = np.any(yellow_mask > 0, axis=1)
        cols = np.any(yellow_mask > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # Fallback: search whole top-right for any bright text
            gray = cv2.cvtColor(search_area, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            rows = np.any(mask > 0, axis=1)
            cols = np.any(mask > 0, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                return None
        
        sec_top = np.argmax(rows)
        sec_bottom = len(rows) - np.argmax(rows[::-1])
        
        # Step 2: Look below "SECURITY LEVEL" for the alarm ring/digit
        # The digit should be centered below the text
        ring_search_top = search_top + sec_bottom + 5
        ring_search_bottom = min(vh, ring_search_top + int(vh * 0.15))  # ~15% height for ring
        ring_search_area = viewport[ring_search_top:ring_search_bottom, search_left:search_right]
        
        # Find the brightest/largest digit (the alarm number)
        gray = cv2.cvtColor(ring_search_area, cv2.COLOR_RGB2GRAY)
        _, digit_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)  # High threshold for bright digit
        
        # Find digit bounding box
        rows = np.any(digit_mask > 0, axis=1)
        cols = np.any(digit_mask > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        top = np.argmax(rows)
        bottom = len(rows) - np.argmax(rows[::-1])
        left = np.argmax(cols)
        right = len(cols) - np.argmax(cols[::-1])
        
        # Convert to viewport coordinates
        x = search_left + left
        y = ring_search_top + top
        width = right - left
        height = bottom - top
        
        # Add small padding
        x = max(0, x - 3)
        y = max(0, y - 3)
        width = min(width + 6, vw - x)
        height = min(height + 6, vh - y)
        
        return (x, y, width, height)
    
    def read_power(self, viewport, region):
        """Read power from region"""
        if region is None:
            return None, None
        
        x, y, w, h = region
        power_img = viewport[y:y+h, x:x+w]
        
        if self.debug:
            cv2.imwrite('debug_power_region.png', cv2.cvtColor(power_img, cv2.COLOR_RGB2BGR))
        
        green = power_img[:, :, 1]
        _, thresh = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        if self.debug:
            cv2.imwrite('debug_power_thresh.png', thresh)
        
        text = pytesseract.image_to_string(thresh, config='--psm 7')
        
        try:
            parts = text.split('/')
            if len(parts) >= 2:
                current = int(''.join(filter(str.isdigit, parts[0])))
                maximum = int(''.join(filter(str.isdigit, parts[1])))
                return current, maximum
        except:
            pass
        
        return None, None
    
    def read_credits(self, viewport, region):
        """Read credits from region"""
        if region is None:
            return None
        
        x, y, w, h = region
        credits_img = viewport[y:y+h, x:x+w]
        
        if self.debug:
            cv2.imwrite('debug_credits_region.png', cv2.cvtColor(credits_img, cv2.COLOR_RGB2BGR))
        
        green = credits_img[:, :, 1]
        _, thresh = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        if self.debug:
            cv2.imwrite('debug_credits_thresh.png', thresh)
        
        text = pytesseract.image_to_string(thresh, config='--psm 7 digits')
        digits = ''.join(filter(str.isdigit, text))
        
        try:
            return int(digits) if digits else None
        except:
            return None
    
    def read_alarm(self, viewport, region):
        """Read alarm level from region"""
        if region is None:
            return None
        
        x, y, w, h = region
        alarm_img = viewport[y:y+h, x:x+w]
        
        if self.debug:
            cv2.imwrite('debug_alarm_region.png', cv2.cvtColor(alarm_img, cv2.COLOR_RGB2BGR))
        
        gray = cv2.cvtColor(alarm_img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)  # High threshold for bright digit
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        if self.debug:
            cv2.imwrite('debug_alarm_thresh.png', thresh)
        
        text = pytesseract.image_to_string(thresh, config='--psm 7 digits')
        digits = ''.join(filter(str.isdigit, text))
        
        try:
            return int(digits) if digits else None
        except:
            return None
    
    def visualize_detections(self, viewport, power_region, credits_region, alarm_region):
        """Create annotated image"""
        vis = viewport.copy()
        
        if power_region:
            x, y, w, h = power_region
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis, "POWER", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if credits_region:
            x, y, w, h = credits_region
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis, "CREDITS", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if alarm_region:
            x, y, w, h = alarm_region
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis, "ALARM", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis
    
    def get_game_state(self):
        """Get game state with sequential detection"""
        window_img = self.detector.capture_game_window(auto_focus=True)
        if window_img is None:
            return None
        
        viewport, offset = self.find_viewport(window_img)
        if viewport is None:
            return None
        
        vh, vw = viewport.shape[:2]
        print(f"Viewport: {vw}x{vh}")
        
        # Sequential search
        print("Finding power...")
        power_region = self.find_power(viewport)
        print(f"  Power region: {power_region}")
        
        print("Finding credits...")
        credits_region = self.find_credits(viewport, power_region)
        print(f"  Credits region: {credits_region}")
        
        print("Finding alarm...")
        alarm_region = self.find_alarm(viewport)
        print(f"  Alarm region: {alarm_region}")
        
        # Read values
        power_current, power_max = self.read_power(viewport, power_region)
        credits = self.read_credits(viewport, credits_region)
        alarm = self.read_alarm(viewport, alarm_region)
        
        # Visualize
        if self.debug:
            vis = self.visualize_detections(viewport, power_region, credits_region, alarm_region)
            cv2.imwrite('debug_visualization.png', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        state = {
            'power': power_current,
            'power_max': power_max,
            'credits': credits,
            'alarm_level': alarm,
            'viewport_size': (vw, vh),
        }
        
        return state

# Test
if __name__ == "__main__":
    print("="*60)
    print("SEQUENTIAL ADAPTIVE DETECTOR")
    print("="*60)
    
    detector = SequentialAdaptiveDetector()
    
    print("\nExpected: Power=10/20, Credits=72314, Alarm=2")
    input("\nPress ENTER to test...")
    
    state = detector.get_game_state()
    
    if state:
        print(f"\n✓ Results:")
        print(f"  Power: {state['power']}/{state['power_max']}")
        print(f"  Credits: {state['credits']}")
        print(f"  Alarm: {state['alarm_level']}")
        
        print("\nDebug images:")
        print("open debug_visualization.png debug_power_region.png debug_credits_region.png debug_alarm_region.png")
    else:
        print("✗ Failed")
        