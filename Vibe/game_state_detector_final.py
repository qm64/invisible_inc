from window_detector import WindowDetector
import cv2
import numpy as np
import pytesseract

class GameStateDetector:
    """
    Game state detector using measured layout from 2560x1301 window
    """
    
    def __init__(self):
        self.detector = WindowDetector()
        
        # Layout from measurements (viewport-relative percentages)
        self.layout = {
            'power': {'x': 0.0, 'y': 0.013, 'width': 0.044, 'height': 0.016},
            'credits': {'x': 0.064, 'y': 0.014, 'width': 0.028, 'height': 0.015},
            'alarm': {'x': 0.959, 'y': 0.099, 'width': 0.016, 'height': 0.032},
            'agent_icons': {'x': 0.002, 'y': 0.682, 'width': 0.019, 'height': 0.032},
            'agent_profile': {'x': 0.002, 'y': 0.809, 'width': 0.083, 'height': 0.174},
        }
    
    def extract_viewport(self, window_img):
        """Extract viewport by detecting green UI elements"""
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
        
        return window_img[top:bottom, left:right]
    
    def get_region(self, viewport, region_def):
        """Extract region using percentages"""
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
        """Read power value"""
        power_img = self.get_region(viewport, self.layout['power'])
        
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
        
        return None, None
    
    def read_credits(self, viewport):
        """Read credits value"""
        credits_img = self.get_region(viewport, self.layout['credits'])
        
        green = credits_img[:, :, 1]
        _, thresh = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        text = pytesseract.image_to_string(thresh, config='--psm 7 digits')
        digits = ''.join(filter(str.isdigit, text))
        
        try:
            return int(digits) if digits else None
        except:
            return None
    
    def read_alarm_level(self, viewport):
        """Read alarm level"""
        alarm_img = self.get_region(viewport, self.layout['alarm'])
        
        gray = cv2.cvtColor(alarm_img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        text = pytesseract.image_to_string(thresh, config='--psm 7 digits')
        digits = ''.join(filter(str.isdigit, text))
        
        try:
            return int(digits) if digits else None
        except:
            return None
    
    def get_game_state(self):
        """Get complete game state"""
        window_img = self.detector.capture_game_window(auto_focus=False)
        if window_img is None:
            return None
        
        viewport = self.extract_viewport(window_img)
        if viewport is None:
            return None
        
        power_current, power_max = self.read_power(viewport)
        credits = self.read_credits(viewport)
        alarm = self.read_alarm_level(viewport)
        
        state = {
            'power': power_current,
            'power_max': power_max,
            'credits': credits,
            'alarm_level': alarm,
            'viewport_size': viewport.shape[:2],
        }
        
        return state

# Test
if __name__ == "__main__":
    print("="*60)
    print("GAME STATE DETECTOR - FINAL")
    print("="*60)
    
    detector = GameStateDetector()
    
    print("\n1. Keep window at maximized size (2560x1301)")
    print("2. Make sure it's your turn")
    input("\nPress ENTER to detect...")
    
    state = detector.get_game_state()
    
    if state:
        print(f"\n✓ Game State:")
        print(f"  Power: {state['power']}/{state['power_max']}")
        print(f"  Credits: {state['credits']}")
        print(f"  Alarm: {state['alarm_level']}")
        print(f"  Viewport: {state['viewport_size']}")
    else:
        print("✗ Failed to detect state")
        