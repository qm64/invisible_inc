from window_detector import WindowDetector
import cv2
import numpy as np
import pytesseract

class AdaptiveGameStateDetector:
    """
    Adaptive detector that finds UI elements by searching
    Works at any window size/aspect ratio
    """
    
    def __init__(self):
        self.detector = WindowDetector()
    
    def find_viewport(self, window_img):
        """
        Find actual game content area (excluding black bars)
        Returns viewport image and its offset in window
        """
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
    
    def find_ui_element(self, viewport, search_region, color_channel='green', threshold=100):
        """
        Find UI element by searching for colored text/graphics
        search_region: (x, y, width, height) in percentages of viewport
        Returns: (x, y, width, height) in pixels, or None
        """
        vh, vw = viewport.shape[:2]
        
        # Convert percentage region to pixels
        sx = int(vw * search_region[0])
        sy = int(vh * search_region[1])
        sw = int(vw * search_region[2])
        sh = int(vh * search_region[3])
        
        # Ensure valid bounds
        sx = max(0, min(sx, vw-1))
        sy = max(0, min(sy, vh-1))
        sw = min(sw, vw - sx)
        sh = min(sh, vh - sy)
        
        if sw <= 0 or sh <= 0:
            return None
        
        search_area = viewport[sy:sy+sh, sx:sx+sw]
        
        # Extract color channel
        if color_channel == 'green':
            channel = search_area[:, :, 1]
        elif color_channel == 'gray':
            channel = cv2.cvtColor(search_area, cv2.COLOR_RGB2GRAY)
        else:
            channel = search_area[:, :, 1]
        
        # Threshold
        _, mask = cv2.threshold(channel, threshold, 255, cv2.THRESH_BINARY)
        
        # Find bounding box of content
        rows_with_content = np.any(mask > 0, axis=1)
        cols_with_content = np.any(mask > 0, axis=0)
        
        if not np.any(rows_with_content) or not np.any(cols_with_content):
            return None
        
        top = np.argmax(rows_with_content)
        bottom = len(rows_with_content) - np.argmax(rows_with_content[::-1])
        left = np.argmax(cols_with_content)
        right = len(cols_with_content) - np.argmax(cols_with_content[::-1])
        
        # Convert back to viewport coordinates
        x = sx + left
        y = sy + top
        width = right - left
        height = bottom - top
        
        return (x, y, width, height)
    
    def read_power(self, viewport, region):
        """Read power from region"""
        if region is None:
            return None, None
        
        x, y, w, h = region
        power_img = viewport[y:y+h, x:x+w]
        
        green = power_img[:, :, 1]
        _, thresh = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        
        if thresh.size == 0:
            return None, None
        
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        text = pytesseract.image_to_string(thresh, config='--psm 7')
        
        try:
            parts = text.split('/')
            current = int(''.join(filter(str.isdigit, parts[0])))
            maximum = int(''.join(filter(str.isdigit, parts[1])))
            return current, maximum
        except:
            return None, None
    
    def read_credits(self, viewport, region):
        """Read credits from region"""
        if region is None:
            return None
        
        x, y, w, h = region
        credits_img = viewport[y:y+h, x:x+w]
        
        green = credits_img[:, :, 1]
        _, thresh = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        
        if thresh.size == 0:
            return None
        
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
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
        
        gray = cv2.cvtColor(alarm_img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        if thresh.size == 0:
            return None
        
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        text = pytesseract.image_to_string(thresh, config='--psm 7 digits')
        digits = ''.join(filter(str.isdigit, text))
        
        try:
            return int(digits) if digits else None
        except:
            return None
    
    def get_game_state(self):
        """Get game state adaptively"""
        window_img = self.detector.capture_game_window(auto_focus=True)
        if window_img is None:
            return None
        
        # Find viewport
        viewport, offset = self.find_viewport(window_img)
        if viewport is None:
            return None
        
        print(f"DEBUG: Viewport size: {viewport.shape[1]}x{viewport.shape[0]}")
        
        # Search for UI elements in approximate regions
        # These are rough areas where we expect to find each element
        power_region = self.find_ui_element(viewport, (0.0, 0.0, 0.15, 0.05))
        credits_region = self.find_ui_element(viewport, (0.08, 0.0, 0.12, 0.05))
        alarm_region = self.find_ui_element(viewport, (0.88, 0.05, 0.12, 0.15), color_channel='gray', threshold=150)
        
        print(f"DEBUG: Found power at {power_region}")
        print(f"DEBUG: Found credits at {credits_region}")
        print(f"DEBUG: Found alarm at {alarm_region}")
        
        # Read values
        power_current, power_max = self.read_power(viewport, power_region)
        credits = self.read_credits(viewport, credits_region)
        alarm = self.read_alarm(viewport, alarm_region)
        
        state = {
            'power': power_current,
            'power_max': power_max,
            'credits': credits,
            'alarm_level': alarm,
            'viewport_size': viewport.shape[:2],
            'window_size': window_img.shape[:2],
        }
        
        return state

# Test
if __name__ == "__main__":
    print("="*60)
    print("ADAPTIVE GAME STATE DETECTOR")
    print("="*60)
    
    detector = AdaptiveGameStateDetector()
    
    print("\nThis detector should work at ANY window size!")
    input("Press ENTER to test...")
    
    state = detector.get_game_state()
    
    if state:
        print(f"\n✓ Game State:")
        print(f"  Power: {state['power']}/{state['power_max']}")
        print(f"  Credits: {state['credits']}")
        print(f"  Alarm: {state['alarm_level']}")
        print(f"  Viewport: {state['viewport_size']}")
        print(f"  Window: {state['window_size']}")
    else:
        print("✗ Failed")