from window_detector import WindowDetector
import cv2
import numpy as np
import pytesseract

class GameStateDetector:
    """
    Detects all game state using measured layout proportions
    """
    
    def __init__(self):
        self.detector = WindowDetector()
        
        # Corrected layout from measurements
        self.layout = {
            'Power': {'x': 0.0, 'y': 0.0, 'width': 0.06, 'height': 0.035},
            'Credits': {'x': 0.0559, 'y': 0.0, 'width': 0.045, 'height': 0.035},
            'Alarm_integer': {'x': 0.96, 'y': 0.0902, 'width': 0.035, 'height': 0.0491},
            'Agent_icons': {'x': 0.0, 'y': 0.6658, 'width': 0.0459, 'height': 0.1716},
        }
    
    def extract_viewport(self, window_img):
        """Extract game viewport - same method that works in test_ocr_regions.py"""
        green = window_img[:, :, 1]
        green_threshold = cv2.threshold(green, 80, 255, cv2.THRESH_BINARY)[1]
        rows_with_green = np.any(green_threshold > 0, axis=1)
        cols_with_green = np.any(green_threshold > 0, axis=0)
        
        if not np.any(rows_with_green) or not np.any(cols_with_green):
            return None
        
        top = np.argmax(rows_with_green)
        bottom = len(rows_with_green) - np.argmax(rows_with_green[::-1])
        left = np.argmax(cols_with_green)
        right = len(cols_with_green) - np.argmax(cols_with_green[::-1])
        
        viewport = window_img[top:bottom, left:right]
        
        # Trim black borders - EXACTLY as in test_ocr_regions.py
        gray = cv2.cvtColor(viewport, cv2.COLOR_RGB2GRAY)
        for x in range(min(400, viewport.shape[1])):
            if np.mean(gray[:, x]) > 10:
                viewport = viewport[:, x:]
                break
        
        gray = cv2.cvtColor(viewport, cv2.COLOR_RGB2GRAY)
        for x in range(viewport.shape[1]-1, max(0, viewport.shape[1]-400), -1):
            if np.mean(gray[:, x]) > 10:
                viewport = viewport[:, :x+1]
                break
        
        for y in range(viewport.shape[0]-1, max(0, viewport.shape[0]-100), -1):
            if np.mean(gray[y, :]) > 10:
                viewport = viewport[:y+1, :]
                break
        
        return viewport

    def get_region(self, viewport, region_def):
        """Extract a region from viewport using proportions"""
        h, w = viewport.shape[:2]
        
        x = int(w * region_def['x'])
        y = int(h * region_def['y'])
        width = int(w * region_def['width'])
        height = int(h * region_def['height'])
        
        x = max(0, min(x, w-1))
        y = max(0, min(y, h-1))
        width = min(width, w - x)
        height = min(height, h - y)
        
        return viewport[y:y+height, x:x+width]
    
    def read_power(self, viewport):
        """Read power value (e.g., '10/20 PWR')"""
        power_img = self.get_region(viewport, self.layout['Power'])
        
        green = power_img[:, :, 1]
        _, thresh = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        text = pytesseract.image_to_string(thresh, config='--psm 7')
        
        try:
            parts = text.split('/')
            if len(parts) >= 2:
                current = int(''.join(filter(str.isdigit, parts[0])))
                maximum = int(''.join(filter(str.isdigit, parts[1])))
                return current, maximum
        except:
            pass
        
        return 0, 0
    
    def read_credits(self, viewport):
        """Read credits value"""
        credits_img = self.get_region(viewport, self.layout['Credits'])
        
        green = credits_img[:, :, 1]
        _, thresh = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        text = pytesseract.image_to_string(thresh, config='--psm 7 digits')
        digits = ''.join(filter(str.isdigit, text))
        
        try:
            return int(digits) if digits else 0
        except:
            return 0
    
    def read_alarm_level(self, viewport):
        """Read alarm level integer"""
        alarm_img = self.get_region(viewport, self.layout['Alarm_integer'])
        
        gray = cv2.cvtColor(alarm_img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        text = pytesseract.image_to_string(thresh, config='--psm 7 digits')
        digits = ''.join(filter(str.isdigit, text))
        
        try:
            return int(digits) if digits else 0
        except:
            return 0
    
    def count_agents(self, viewport):
        """Count agent icons"""
        icon_region = self.get_region(viewport, self.layout['Agent_icons'])
        
        green = icon_region[:, :, 1]
        _, mask = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        
        vertical_profile = np.sum(mask, axis=1)
        threshold = np.max(vertical_profile) * 0.3
        
        in_peak = False
        peak_count = 0
        
        for value in vertical_profile:
            if value > threshold and not in_peak:
                peak_count += 1
                in_peak = True
            elif value <= threshold:
                in_peak = False
        
        return peak_count
    
    def get_game_state(self):
        """Get complete game state"""
        window_img = self.detector.capture_game_window(auto_focus=False)
        if window_img is None:
            return None
        
        viewport = self.extract_viewport(window_img)
        if viewport is None:
            return None

        print(f"DEBUG: Viewport size: {viewport.shape}")  # ADD THIS LINE

        power_current, power_max = self.read_power(viewport)
        credits = self.read_credits(viewport)
        alarm = self.read_alarm_level(viewport)
        num_agents = self.count_agents(viewport)
        
        state = {
            'power': power_current,
            'power_max': power_max,
            'credits': credits,
            'alarm_level': alarm,
            'num_agents': num_agents,
            'viewport_size': viewport.shape[:2],
        }
        
        return state

# Test
if __name__ == "__main__":
    print("="*60)
    print("GAME STATE DETECTOR TEST")
    print("="*60)
    
    detector = GameStateDetector()
    
    print("\n1. Launch Invisible Inc")
    print("2. Start a mission")
    input("\nPress ENTER to detect state...")
    
    state = detector.get_game_state()
    
    if state:
        print(f"\n✓ Game State Detected:")
        print(f"  Power: {state['power']}/{state['power_max']}")
        print(f"  Credits: {state['credits']}")
        print(f"  Alarm Level: {state['alarm_level']}")
        print(f"  Agents: {state['num_agents']}")
        print(f"  Viewport: {state['viewport_size']}")
    else:
        print("✗ Could not detect game state")
