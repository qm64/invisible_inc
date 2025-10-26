from window_detector import WindowDetector
import cv2
import numpy as np
import pytesseract

class GameStateDetector:
    """
    Game state detector using absolute pixel positions
    Measures from the detected power/credits positions
    """
    
    def __init__(self):
        self.detector = WindowDetector()
    
    def find_power_position(self, window_img):
        """
        Find power text to anchor our measurements
        Returns (x, y) of top-left corner of power display
        """
        # Search top-left area for green text
        search_region = window_img[0:100, 0:400]
        
        green = search_region[:, :, 1]
        _, mask = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        
        # Find leftmost green pixels
        cols_with_green = np.any(mask > 0, axis=0)
        if not np.any(cols_with_green):
            return None
        
        left = np.argmax(cols_with_green)
        return (left, 10)  # Approximate y position
    
    def get_game_state(self):
        """Get game state using absolute measurements from your window"""
        window_img = self.detector.capture_game_window(auto_focus=True)
        if window_img is None:
            return None
        
        # Use the exact measurements you provided (for 2560x1301 window)
        # These are absolute pixel positions
        regions = {
            'power': (141, 17, 100, 21),
            'credits': (286, 18, 64, 19),
            'alarm': (2328, 129, 36, 42),
        }
        
        # Extract and OCR each region
        power_img = window_img[17:38, 141:241]
        credits_img = window_img[18:37, 286:350]
        alarm_img = window_img[129:171, 2328:2364]
        
        # Save debug
        cv2.imwrite('debug_power_v2.png', cv2.cvtColor(power_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite('debug_credits_v2.png', cv2.cvtColor(credits_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite('debug_alarm_v2.png', cv2.cvtColor(alarm_img, cv2.COLOR_RGB2BGR))
        
        # OCR Power
        green = power_img[:, :, 1]
        _, thresh = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        power_text = pytesseract.image_to_string(thresh, config='--psm 7')
        
        try:
            parts = power_text.split('/')
            power_current = int(''.join(filter(str.isdigit, parts[0])))
            power_max = int(''.join(filter(str.isdigit, parts[1])))
        except:
            power_current, power_max = None, None
        
        # OCR Credits
        green = credits_img[:, :, 1]
        _, thresh = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        credits_text = pytesseract.image_to_string(thresh, config='--psm 7 digits')
        digits = ''.join(filter(str.isdigit, credits_text))
        
        try:
            credits = int(digits) if digits else None
        except:
            credits = None
        
        # OCR Alarm
        gray = cv2.cvtColor(alarm_img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        alarm_text = pytesseract.image_to_string(thresh, config='--psm 7 digits')
        digits = ''.join(filter(str.isdigit, alarm_text))
        
        try:
            alarm = int(digits) if digits else None
        except:
            alarm = None
        
        state = {
            'power': power_current,
            'power_max': power_max,
            'credits': credits,
            'alarm_level': alarm,
            'window_size': window_img.shape[:2],
        }
        
        return state

# Test
if __name__ == "__main__":
    print("="*60)
    print("GAME STATE DETECTOR V2 - ABSOLUTE POSITIONS")
    print("="*60)
    
    detector = GameStateDetector()
    
    print("\n1. Keep window at same size as when you measured")
    print("2. Make sure it's your turn")
    input("\nPress ENTER...")
    
    state = detector.get_game_state()
    
    if state:
        print(f"\n✓ Game State:")
        print(f"  Power: {state['power']}/{state['power_max']}")
        print(f"  Credits: {state['credits']}")
        print(f"  Alarm: {state['alarm_level']}")
        print(f"  Window: {state['window_size']}")
        
        print("\nDebug images:")
        print("open debug_power_v2.png debug_credits_v2.png debug_alarm_v2.png")
    else:
        print("✗ Failed")
