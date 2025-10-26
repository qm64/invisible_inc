from window_detector import WindowDetector
import cv2
import numpy as np
import pytesseract

class ImprovedAdaptiveDetector:
    """
    Improved adaptive detector with precise region finding
    Focus: Power, Credits, Alarm (non-overlapping core UI)
    """
    
    def __init__(self):
        self.detector = WindowDetector()
        self.debug = True  # Save debug images
    
    def find_viewport(self, window_img):
        """Find actual game content area (excluding black bars)"""
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
    
    def find_text_region(self, viewport, search_area_pct, color_channel='green', 
                         threshold=100, padding=(5, 5, 5, 5)):
        """
        Find tight bounding box around text in search area
        
        Args:
            viewport: The viewport image
            search_area_pct: (x%, y%, w%, h%) - where to search
            color_channel: 'green' or 'gray'
            threshold: Brightness threshold
            padding: (left, top, right, bottom) pixels to add around found text
            
        Returns:
            (x, y, width, height) in viewport coords, or None
        """
        vh, vw = viewport.shape[:2]
        
        # Convert percentage to pixels
        sx = int(vw * search_area_pct[0])
        sy = int(vh * search_area_pct[1])
        sw = int(vw * search_area_pct[2])
        sh = int(vh * search_area_pct[3])
        
        # Bounds check
        sx = max(0, min(sx, vw-1))
        sy = max(0, min(sy, vh-1))
        sw = min(sw, vw - sx)
        sh = min(sh, vh - sy)
        
        if sw <= 0 or sh <= 0:
            return None
        
        search_region = viewport[sy:sy+sh, sx:sx+sw]
        
        # Extract channel
        if color_channel == 'green':
            channel = search_region[:, :, 1]
        elif color_channel == 'gray':
            channel = cv2.cvtColor(search_region, cv2.COLOR_RGB2GRAY)
        else:
            channel = search_region[:, :, 1]
        
        # Threshold
        _, mask = cv2.threshold(channel, threshold, 255, cv2.THRESH_BINARY)
        
        # Find tight bounding box
        rows_with_content = np.any(mask > 0, axis=1)
        cols_with_content = np.any(mask > 0, axis=0)
        
        if not np.any(rows_with_content) or not np.any(cols_with_content):
            return None
        
        # Get bounds within search region
        top = np.argmax(rows_with_content)
        bottom = len(rows_with_content) - np.argmax(rows_with_content[::-1])
        left = np.argmax(cols_with_content)
        right = len(cols_with_content) - np.argmax(cols_with_content[::-1])
        
        # Add padding
        pad_left, pad_top, pad_right, pad_bottom = padding
        left = max(0, left - pad_left)
        top = max(0, top - pad_top)
        right = min(sw, right + pad_right)
        bottom = min(sh, bottom + pad_bottom)
        
        # Convert to viewport coordinates
        x = sx + left
        y = sy + top
        width = right - left
        height = bottom - top
        
        # Sanity check dimensions
        if width < 10 or height < 5:  # Too small to be real text
            return None
        if width > sw * 0.9:  # Probably found whole bar, not text
            return None
        
        return (x, y, width, height)
    
    def read_power(self, viewport, region):
        """Read power from region"""
        if region is None:
            return None, None
        
        x, y, w, h = region
        if x < 0 or y < 0 or x+w > viewport.shape[1] or y+h > viewport.shape[0]:
            return None, None
        
        power_img = viewport[y:y+h, x:x+w]
        
        # Save debug
        if self.debug:
            cv2.imwrite('debug_power_region.png', cv2.cvtColor(power_img, cv2.COLOR_RGB2BGR))
        
        green = power_img[:, :, 1]
        _, thresh = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        
        if thresh.size == 0 or thresh.shape[0] == 0 or thresh.shape[1] == 0:
            return None, None
        
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
        if x < 0 or y < 0 or x+w > viewport.shape[1] or y+h > viewport.shape[0]:
            return None
        
        credits_img = viewport[y:y+h, x:x+w]
        
        if self.debug:
            cv2.imwrite('debug_credits_region.png', cv2.cvtColor(credits_img, cv2.COLOR_RGB2BGR))
        
        green = credits_img[:, :, 1]
        _, thresh = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        
        if thresh.size == 0 or thresh.shape[0] == 0 or thresh.shape[1] == 0:
            return None
        
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
        if x < 0 or y < 0 or x+w > viewport.shape[1] or y+h > viewport.shape[0]:
            return None
        
        alarm_img = viewport[y:y+h, x:x+w]
        
        if self.debug:
            cv2.imwrite('debug_alarm_region.png', cv2.cvtColor(alarm_img, cv2.COLOR_RGB2BGR))
        
        gray = cv2.cvtColor(alarm_img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        if thresh.size == 0 or thresh.shape[0] == 0 or thresh.shape[1] == 0:
            return None
        
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
        """Create annotated image showing found regions"""
        vis = viewport.copy()
        
        # Draw search areas (yellow boxes)
        vh, vw = vis.shape[:2]
        
        # Power search area
        cv2.rectangle(vis, 
                     (0, 0), 
                     (int(vw*0.12), int(vh*0.06)), 
                     (0, 255, 255), 2)
        cv2.putText(vis, "Power Search", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Credits search area
        cv2.rectangle(vis,
                     (int(vw*0.10), 0),
                     (int(vw*0.25), int(vh*0.06)),
                     (0, 255, 255), 2)
        cv2.putText(vis, "Credits Search", (int(vw*0.11), 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Alarm search area
        cv2.rectangle(vis,
                     (int(vw*0.85), int(vh*0.05)),
                     (int(vw*1.0), int(vh*0.20)),
                     (0, 255, 255), 2)
        cv2.putText(vis, "Alarm Search", (int(vw*0.86), int(vh*0.07)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw found regions (green boxes)
        if power_region:
            x, y, w, h = power_region
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis, "POWER", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        if credits_region:
            x, y, w, h = credits_region
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis, "CREDITS", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        if alarm_region:
            x, y, w, h = alarm_region
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis, "ALARM", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return vis
    
    def get_game_state(self):
        """Get game state with improved detection"""
        window_img = self.detector.capture_game_window(auto_focus=True)
        if window_img is None:
            return None
        
        # Find viewport
        viewport, offset = self.find_viewport(window_img)
        if viewport is None:
            return None
        
        vh, vw = viewport.shape[:2]
        print(f"Viewport: {vw}x{vh}")
        
        # Find UI elements with tighter, non-overlapping search areas
        # Power: Top-left, first 12% of width
        power_region = self.find_text_region(
            viewport, 
            search_area_pct=(0.0, 0.0, 0.12, 0.06),
            color_channel='green',
            threshold=100,
            padding=(3, 2, 10, 2)  # Extra right padding for "/20 PWR"
        )
        
        # Credits: To the right of power, 10-25% of width
        credits_region = self.find_text_region(
            viewport,
            search_area_pct=(0.10, 0.0, 0.15, 0.06),
            color_channel='green',
            threshold=100,
            padding=(3, 2, 10, 2)  # Extra right padding for "CR"
        )
        
        # Alarm: Top-right, last 15% of width, 5-20% of height
        alarm_region = self.find_text_region(
            viewport,
            search_area_pct=(0.85, 0.05, 0.15, 0.15),
            color_channel='gray',
            threshold=150,
            padding=(5, 5, 5, 5)
        )
        
        print(f"Power region: {power_region}")
        print(f"Credits region: {credits_region}")
        print(f"Alarm region: {alarm_region}")
        
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
    print("IMPROVED ADAPTIVE DETECTOR")
    print("="*60)
    
    detector = ImprovedAdaptiveDetector()
    
    print("\nTesting with current window size...")
    print("Expected: Power=14/20, Credits=70306, Alarm=1")
    input("\nPress ENTER to test...")
    
    state = detector.get_game_state()
    
    if state:
        print(f"\n✓ Results:")
        print(f"  Power: {state['power']}/{state['power_max']}")
        print(f"  Credits: {state['credits']}")
        print(f"  Alarm: {state['alarm_level']}")
        print(f"  Viewport: {state['viewport_size']}")
        
        print("\nDebug images saved:")
        print("open debug_visualization.png debug_power_region.png debug_credits_region.png debug_alarm_region.png")
    else:
        print("✗ Failed to detect state")
        