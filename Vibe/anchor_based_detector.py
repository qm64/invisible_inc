from window_detector import WindowDetector
import cv2
import numpy as np
import pytesseract

class AnchorBasedDetector:
    """
    Anchor-based detection using Incognita profile as landmark
    Strategy: Find Incognita → Find PWR → Power left of PWR, Credits right of PWR
    """
    
    def __init__(self):
        self.detector = WindowDetector()
        self.debug = True
        self.debug_files = []
    
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
    
    def find_incognita_profile(self, viewport):
        """
        Find Incognita profile icon in top-left
        Returns: (x, y, width, height) or None
        """
        vh, vw = viewport.shape[:2]
        
        # Search top-left area (first 15% width, first 12% height)
        search_w = int(vw * 0.15)
        search_h = int(vh * 0.12)
        search_area = viewport[0:search_h, 0:search_w]
        
        # Incognita profile has distinctive cyan/teal border with white face
        # Look for high cyan (high green + high blue, low red)
        red = search_area[:, :, 0]
        green = search_area[:, :, 1]
        blue = search_area[:, :, 2]
        
        # Cyan mask: high green AND high blue, low red
        cyan_mask = ((green > 100) & (blue > 100) & (red < 80)).astype(np.uint8) * 255
        
        # White mask: all channels high
        white_mask = ((red > 180) & (green > 180) & (blue > 180)).astype(np.uint8) * 255
        
        # Combine - profile has both cyan border and white face
        combined_mask = cv2.bitwise_or(cyan_mask, white_mask)
        
        if self.debug:
            filename = 'debug_incognita_mask.png'
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
        
        # Add small padding
        x = max(0, left - 3)
        y = max(0, top - 3)
        width = right - left + 6
        height = bottom - top + 6
        
        return (x, y, width, height)
    
    def find_pwr_text(self, viewport, incognita_region):
        """
        Find "PWR" text above Incognita profile
        Returns: (x, y, width, height) or None
        """
        if incognita_region is None:
            # Fallback: search top-left
            search_left = 0
            search_right = int(viewport.shape[1] * 0.25)
            search_top = 0
            search_bottom = int(viewport.shape[0] * 0.06)
        else:
            ix, iy, iw, ih = incognita_region
            # Search above Incognita, same horizontal region
            search_left = max(0, ix - 20)
            search_right = ix + iw + 20
            search_top = 0
            search_bottom = iy
        
        search_area = viewport[search_top:search_bottom, search_left:search_right]
        
        # Find green text (PWR is cyan/green)
        green = search_area[:, :, 1]
        _, mask = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        
        if self.debug:
            filename = 'debug_pwr_search_area.png'
            cv2.imwrite(filename, cv2.cvtColor(search_area, cv2.COLOR_RGB2BGR))
            self.debug_files.append(filename)
            
            filename = 'debug_pwr_mask.png'
            cv2.imwrite(filename, mask)
            self.debug_files.append(filename)
        
        # Use OCR to find "PWR" text
        mask_large = cv2.resize(mask, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(mask_large, config='--psm 6')
        
        # Look for "PWR" in the text
        if 'PWR' not in text.upper():
            # Try to find it by bounding box
            rows = np.any(mask > 0, axis=1)
            cols = np.any(mask > 0, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                return None
            
            # Find rightmost group of text (PWR is after the numbers)
            # Scan from right to left for text
            h_projection = np.sum(mask, axis=0)
            
            # Find last significant text region
            pwr_right = len(h_projection) - 1
            pwr_left = 0
            
            # Scan from right
            for x in range(len(h_projection) - 1, -1, -1):
                if h_projection[x] > 0:
                    pwr_right = x
                    break
            
            # Scan left from pwr_right to find start of "PWR"
            gap_count = 0
            for x in range(pwr_right, -1, -1):
                if h_projection[x] == 0:
                    gap_count += 1
                    if gap_count > 5:  # Found gap before PWR
                        pwr_left = x + 5
                        break
                else:
                    gap_count = 0
            
            top = np.argmax(rows)
            bottom = len(rows) - np.argmax(rows[::-1])
            
            # Convert to viewport coordinates
            x = search_left + pwr_left
            y = search_top + top
            width = pwr_right - pwr_left + 5
            height = bottom - top + 4
            
            return (x, y, width, height)
        
        # If we found "PWR" via OCR, locate it in the mask
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        top = np.argmax(rows)
        bottom = len(rows) - np.argmax(rows[::-1])
        
        # Find rightmost text (PWR)
        h_projection = np.sum(mask[top:bottom, :], axis=0)
        pwr_right = len(h_projection) - 1
        
        for x in range(len(h_projection) - 1, -1, -1):
            if h_projection[x] > 0:
                pwr_right = x
                break
        
        # Scan left to find start of PWR (after gap)
        gap_count = 0
        pwr_left = 0
        for x in range(pwr_right, -1, -1):
            if h_projection[x] == 0:
                gap_count += 1
                if gap_count > 5:
                    pwr_left = x + 5
                    break
            else:
                gap_count = 0
        
        # Convert to viewport coordinates
        x = search_left + pwr_left
        y = search_top + top
        width = pwr_right - pwr_left + 5
        height = bottom - top + 4
        
        return (x, y, width, height)
    
    def find_power_value(self, viewport, pwr_region):
        """
        Find power value left of "PWR" text
        Returns: (x, y, width, height) or None
        """
        if pwr_region is None:
            return None
        
        px, py, pw, ph = pwr_region
        
        # Search left of PWR, same vertical position
        search_left = 0
        search_right = px - 5  # Small gap before PWR
        search_top = py - 5
        search_bottom = py + ph + 5
        
        if search_right <= search_left:
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
        width = right - left + 4
        height = bottom - top + 4
        
        return (x, y, width, height)
    
    def find_credits_value(self, viewport, pwr_region):
        """
        Find credits value right of "PWR" text
        Uses symmetry: credits distance from PWR ≈ power distance
        Returns: (x, y, width, height) or None
        """
        if pwr_region is None:
            return None
        
        px, py, pw, ph = pwr_region
        
        # Search right of PWR
        # Use similar width as power value (symmetry)
        search_left = px + pw + 5  # Small gap after PWR
        search_right = min(viewport.shape[1], search_left + 150)  # Reasonable width for credits
        search_top = py - 5
        search_bottom = py + ph + 5
        
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
        
        # Find end of digit sequence (before "CR" text if present)
        h_projection = np.sum(mask[top:bottom, :], axis=0)
        
        # Look for gap indicating end of numbers
        gap_count = 0
        credits_right = right
        for x in range(left, right):
            if h_projection[x] == 0:
                gap_count += 1
                if gap_count > 5:  # Found gap
                    credits_right = x - 5
                    break
            else:
                gap_count = 0
        
        # Convert to viewport coordinates
        x = search_left + left - 2
        y = search_top + top - 2
        width = credits_right - left + 4
        height = bottom - top + 4
        
        return (x, y, width, height)
    
    def find_alarm(self, viewport):
        """
        Find alarm digit - reuse working method from previous detector
        """
        vh, vw = viewport.shape[:2]
        
        # Search top-right for "SECURITY LEVEL" text
        search_left = int(vw * 0.85)
        search_right = vw
        search_top = 0
        search_bottom = int(vh * 0.10)
        
        search_area = viewport[search_top:search_bottom, search_left:search_right]
        
        # Look for yellow/bright text
        hsv = cv2.cvtColor(search_area, cv2.COLOR_RGB2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        gray = cv2.cvtColor(search_area, cv2.COLOR_RGB2GRAY)
        _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        combined_mask = cv2.bitwise_or(yellow_mask, bright_mask)
        
        rows = np.any(combined_mask > 0, axis=1)
        cols = np.any(combined_mask > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # Fallback
            ring_search_top = int(vh * 0.05)
            ring_search_bottom = int(vh * 0.20)
        else:
            sec_top = np.argmax(rows)
            sec_bottom = len(rows) - np.argmax(rows[::-1])
            ring_search_top = search_top + sec_bottom + 5
            ring_search_bottom = min(vh, ring_search_top + int(vh * 0.15))
        
        ring_search_area = viewport[ring_search_top:ring_search_bottom, search_left:search_right]
        
        # Find bright digit
        gray = cv2.cvtColor(ring_search_area, cv2.COLOR_RGB2GRAY)
        _, digit_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(digit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
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
        
        text = pytesseract.image_to_string(thresh, config='--psm 10')
        digits = ''.join(filter(str.isdigit, text))
        
        try:
            return int(digits) if digits else None
        except:
            return None
    
    def visualize_detections(self, viewport, incognita_region, pwr_region, power_region, credits_region, alarm_region):
        """Create annotated image"""
        vis = viewport.copy()
        
        if incognita_region:
            x, y, w, h = incognita_region
            cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 255), 2)
            cv2.putText(vis, "INCOGNITA", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        if pwr_region:
            x, y, w, h = pwr_region
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(vis, "PWR", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
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
        """Get game state using anchor-based detection"""
        self.debug_files = []
        
        window_img = self.detector.capture_game_window(auto_focus=True)
        if window_img is None:
            return None
        
        viewport, offset = self.find_viewport(window_img)
        if viewport is None:
            return None
        
        vh, vw = viewport.shape[:2]
        print(f"Viewport: {vw}x{vh}")
        
        # Step 1: Find Incognita profile (anchor)
        print("Finding Incognita profile...")
        incognita_region = self.find_incognita_profile(viewport)
        print(f"  Incognita region: {incognita_region}")
        
        # Step 2: Find "PWR" text
        print("Finding PWR text...")
        pwr_region = self.find_pwr_text(viewport, incognita_region)
        print(f"  PWR region: {pwr_region}")
        
        # Step 3: Find power value (left of PWR)
        print("Finding power value...")
        power_region = self.find_power_value(viewport, pwr_region)
        print(f"  Power region: {power_region}")
        
        # Step 4: Find credits value (right of PWR)
        print("Finding credits value...")
        credits_region = self.find_credits_value(viewport, pwr_region)
        print(f"  Credits region: {credits_region}")
        
        # Step 5: Find alarm
        print("Finding alarm...")
        alarm_region = self.find_alarm(viewport)
        print(f"  Alarm region: {alarm_region}")
        
        # Read values
        power_current, power_max = self.read_power(viewport, power_region)
        credits = self.read_credits(viewport, credits_region)
        alarm = self.read_alarm(viewport, alarm_region)
        
        # Visualize
        if self.debug:
            vis = self.visualize_detections(viewport, incognita_region, pwr_region, 
                                           power_region, credits_region, alarm_region)
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
    print("ANCHOR-BASED DETECTOR")
    print("="*60)
    
    detector = AnchorBasedDetector()
    
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
        