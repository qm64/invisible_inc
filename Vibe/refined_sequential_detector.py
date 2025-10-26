from window_detector import WindowDetector
import cv2
import numpy as np
import pytesseract

class RefinedSequentialDetector:
    """
    Refined sequential search with better boundary detection
    """
    
    def __init__(self):
        self.detector = WindowDetector()
        self.debug = True
        self.debug_files = []  # Track which debug files were created
    
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
        Find power display - stop at first large gap
        Returns: (x, y, width, height) or None
        """
        vh, vw = viewport.shape[:2]
        
        # Search top-left
        search_w = int(vw * 0.20)  # Wider search
        search_h = int(vh * 0.08)
        search_area = viewport[0:search_h, 0:search_w]
        
        # Find green text
        green = search_area[:, :, 1]
        _, mask = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        
        # Find overall bounding box
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        top = np.argmax(rows)
        bottom = len(rows) - np.argmax(rows[::-1])
        left = np.argmax(cols)
        right = len(cols) - np.argmax(cols[::-1])
        
        # Find where power text ends by looking for gap in horizontal projection
        h_projection = np.sum(mask[top:bottom, :], axis=0)
        
        # Find first significant gap (15+ pixels wide with no content)
        gap_threshold = 0
        gap_width = 15
        in_gap = False
        gap_start = left
        gap_count = 0
        power_end = right
        
        for x in range(left, right):
            if h_projection[x] <= gap_threshold:
                if not in_gap:
                    gap_start = x
                    in_gap = True
                gap_count += 1
                
                if gap_count >= gap_width:  # Found substantial gap
                    power_end = gap_start
                    break
            else:
                in_gap = False
                gap_count = 0
        
        # Build region
        x = max(0, left - 2)
        y = max(0, top - 2)
        width = power_end - left + 4
        height = bottom - top + 4
        
        return (x, y, width, height)
    
    def find_credits(self, viewport, power_region):
        """
        Find credits - search after power gap
        """
        if power_region is None:
            return None
        
        vh, vw = viewport.shape[:2]
        px, py, pw, ph = power_region
        
        # Start search after power ends + gap
        search_left = px + pw + 10  # Skip the gap
        search_right = int(vw * 0.30)  # Search up to 30% of viewport
        search_top = py - 5
        search_bottom = py + ph + 5
        
        if search_left >= search_right or search_left >= vw:
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
        
        # Stop at first gap (before "CR" probably)
        h_projection = np.sum(mask[top:bottom, :], axis=0)
        
        # Find end of number text (gap before "CR")
        gap_width = 8
        gap_count = 0
        credits_end = right
        
        for x in range(left, right):
            if h_projection[x] <= 0:
                gap_count += 1
                if gap_count >= gap_width:
                    credits_end = x - gap_width
                    break
            else:
                gap_count = 0
        
        # Convert to viewport coordinates
        x = search_left + left - 2
        y = search_top + top - 2
        width = credits_end - left + 10
        height = bottom - top + 4
        
        return (x, y, width, height)
    
    def find_security_level_text(self, viewport):
        """
        Find "SECURITY LEVEL" text in top-right
        Returns: (x, y, width, height) or None
        """
        vh, vw = viewport.shape[:2]
        
        # Search top-right corner
        search_left = int(vw * 0.85)
        search_right = vw
        search_top = 0
        search_bottom = int(vh * 0.10)
        
        search_area = viewport[search_top:search_bottom, search_left:search_right]
        
        # Look for yellow text (SECURITY LEVEL is yellow)
        # Yellow = high red + high green + low blue
        hsv = cv2.cvtColor(search_area, cv2.COLOR_RGB2HSV)
        
        # Yellow hue range in HSV
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Also try detecting bright text
        gray = cv2.cvtColor(search_area, cv2.COLOR_RGB2GRAY)
        _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(yellow_mask, bright_mask)
        
        if self.debug:
            filename = 'debug_security_level_mask.png'
            cv2.imwrite(filename, combined_mask)
            self.debug_files.append(filename)
        
        # Find bounding box
        rows = np.any(combined_mask > 0, axis=1)
        cols = np.any(combined_mask > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        top = np.argmax(rows)
        bottom = len(rows) - np.argmax(rows[::-1])
        left = np.argmax(cols)
        right = len(cols) - np.argmax(cols[::-1])
        
        # Convert to viewport coordinates
        x = search_left + left
        y = search_top + top
        width = right - left
        height = bottom - top
        
        return (x, y, width, height)
    
    def find_alarm(self, viewport):
        """
        Find alarm digit by locating SECURITY LEVEL then searching ring below
        """
        vh, vw = viewport.shape[:2]
        
        # Step 1: Find "SECURITY LEVEL" text
        sec_region = self.find_security_level_text(viewport)
        
        if sec_region is None:
            # Fallback: search whole top-right area
            search_left = int(vw * 0.85)
            ring_search_top = int(vh * 0.05)
            ring_search_bottom = int(vh * 0.20)
        else:
            sx, sy, sw, sh = sec_region
            # Search below the text for the ring
            search_left = max(0, sx - int(sw * 0.2))  # Slightly left of text
            ring_search_top = sy + sh + 5  # Just below text
            ring_search_bottom = min(vh, ring_search_top + int(vh * 0.15))
        
        ring_search_right = vw
        ring_search_area = viewport[ring_search_top:ring_search_bottom, search_left:ring_search_right]
        
        if self.debug:
            filename = 'debug_alarm_search_area.png'
            cv2.imwrite(filename, cv2.cvtColor(ring_search_area, cv2.COLOR_RGB2BGR))
            self.debug_files.append(filename)
        
        # Find the bright digit in center of ring
        # The alarm number is VERY bright (near white)
        gray = cv2.cvtColor(ring_search_area, cv2.COLOR_RGB2GRAY)
        _, digit_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # Very high threshold
        
        if self.debug:
            filename = 'debug_alarm_digit_mask.png'
            cv2.imwrite(filename, digit_mask)
            self.debug_files.append(filename)
        
        # Find contours to isolate the digit
        contours, _ = cv2.findContours(digit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour (should be the digit)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Filter by size (digit should be reasonable size)
        if w < 10 or h < 15 or w > 100 or h > 100:
            # Try with lower threshold
            _, digit_mask2 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            contours2, _ = cv2.findContours(digit_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours2:
                largest_contour = max(contours2, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Convert to viewport coordinates
        x = search_left + x - 3
        y = ring_search_top + y - 3
        width = w + 6
        height = h + 6
        
        return (x, y, width, height)
    
    def read_power(self, viewport, region):
        """Read power from region"""
        if region is None:
            return None, None
        
        x, y, w, h = region
        power_img = viewport[y:y+h, x:x+w]
        
        if self.debug:
            filename = 'debug_power_region.png'
            cv2.imwrite(filename, cv2.cvtColor(power_img, cv2.COLOR_RGB2BGR))
            self.debug_files.append(filename)
        
        green = power_img[:, :, 1]
        _, thresh = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        if self.debug:
            filename = 'debug_power_thresh.png'
            cv2.imwrite(filename, thresh)
            self.debug_files.append(filename)
        
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
            filename = 'debug_credits_region.png'
            cv2.imwrite(filename, cv2.cvtColor(credits_img, cv2.COLOR_RGB2BGR))
            self.debug_files.append(filename)
        
        green = credits_img[:, :, 1]
        _, thresh = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        if self.debug:
            filename = 'debug_credits_thresh.png'
            cv2.imwrite(filename, thresh)
            self.debug_files.append(filename)
        
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
            filename = 'debug_alarm_region.png'
            cv2.imwrite(filename, cv2.cvtColor(alarm_img, cv2.COLOR_RGB2BGR))
            self.debug_files.append(filename)
        
        gray = cv2.cvtColor(alarm_img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        if self.debug:
            filename = 'debug_alarm_thresh.png'
            cv2.imwrite(filename, thresh)
            self.debug_files.append(filename)
        
        text = pytesseract.image_to_string(thresh, config='--psm 10')  # Single character mode
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
        """Get game state with refined detection"""
        self.debug_files = []  # Reset file list
        
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
            filename = 'debug_visualization.png'
            cv2.imwrite(filename, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            self.debug_files.append(filename)
        
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
    print("REFINED SEQUENTIAL DETECTOR")
    print("="*60)
    
    detector = RefinedSequentialDetector()
    
    print("\nExpected: Power=10/20, Credits=72314, Alarm=2")
    input("\nPress ENTER to test...")
    
    state = detector.get_game_state()
    
    if state:
        print(f"\n✓ Results:")
        print(f"  Power: {state['power']}/{state['power_max']}")
        print(f"  Credits: {state['credits']}")
        print(f"  Alarm: {state['alarm_level']}")
        
        if detector.debug_files:
            print("\nDebug images:")
            print("open " + " ".join(detector.debug_files))
    else:
        print("✗ Failed")
        