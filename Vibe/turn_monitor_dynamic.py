from window_detector import WindowDetector
import cv2
import numpy as np
import time
import pytesseract

class DynamicTurnMonitor:
    """Turn state monitor using dynamic window detection"""
    
    def __init__(self):
        self.detector = WindowDetector()
        
        # UI regions as percentages (work at any resolution)
        # Based on your 2560x1440 fullscreen coordinates converted to percentages:
        # Power was at (0, 0, 150, 50) in 2560x1440
        # = (0%, 0%, 5.9%, 3.5%)
        self.power_region = (0, 0, 0.08, 0.04)
        self.credits_region = (0.06, 0, 0.10, 0.04)
        self.agent_panel_region = (0, 0.70, 0.08, 0.25)  # Lower-left area
        
    def is_player_turn(self, img):
        """Detect if it's player's turn by checking agent panel visibility"""
        panel = self.detector.get_region(img, self.agent_panel_region)
        if panel is None:
            return False, 0.0
        
        # Extract green channel
        green = panel[:, :, 1]
        bright_green = np.sum(green > 100)
        total_pixels = green.size
        green_percentage = bright_green / total_pixels
        
        is_player = green_percentage > 0.05
        return is_player, green_percentage
    
    def get_power(self, img):
        """Read power from image"""
        power_img = self.detector.get_region(img, self.power_region)
        if power_img is None:
            return 0
        
        # Extract green channel
        green = power_img[:, :, 1]
        _, thresh = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        try:
            text = pytesseract.image_to_string(thresh, config='--psm 7')
            parts = text.split('/')
            if len(parts) >= 1:
                power = int(''.join(filter(str.isdigit, parts[0])))
                return power
        except:
            pass
        return 0
    
    def get_credits(self, img):
        """Read credits from image"""
        credits_img = self.detector.get_region(img, self.credits_region)
        if credits_img is None:
            return 0
        
        green = credits_img[:, :, 1]
        _, thresh = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        try:
            text = pytesseract.image_to_string(thresh, config='--psm 7 digits')
            digits = ''.join(filter(str.isdigit, text))
            if digits:
                return int(digits)
        except:
            pass
        return 0
    
    def get_full_state(self):
        """Get all game state at once"""
        img = self.detector.capture_game_window(auto_focus=False)
        
        if img is None:
            return None
        
        is_player, confidence = self.is_player_turn(img)
        
        state = {
            'is_player_turn': is_player,
            'turn_confidence': confidence,
            'power': self.get_power(img) if is_player else None,
            'credits': self.get_credits(img) if is_player else None,
        }
        
        return state

# Test
if __name__ == "__main__":
    print("="*60)
    print("DYNAMIC TURN MONITOR - TEST")
    print("="*60)
    
    monitor = DynamicTurnMonitor()
    
    print("\n1. Launch Invisible Inc in WINDOWED mode")
    print("2. Start a mission")
    print("3. Keep game window visible")
    input("\nPress ENTER to start monitoring...")
    
    print("\nMonitoring... (Press Ctrl+\\ to stop)\n")
    
    try:
        while True:
            state = monitor.get_full_state()
            
            if state is None:
                print("\r[ERROR] Could not capture window", end="", flush=True)
                time.sleep(1)
                continue
            
            # Clear line and print status
            print("\r" + " "*100, end="")
            
            turn_status = "YOUR TURN" if state['is_player_turn'] else "ENEMY TURN"
            confidence = f"{state['turn_confidence']*100:.1f}%"
            
            if state['is_player_turn']:
                status = f"[{turn_status}] Power: {state['power']:2d} | Credits: {state['credits']:4d} | Conf: {confidence}"
            else:
                status = f"[{turn_status}] Conf: {confidence}"
            
            print(f"\r{status}", end="", flush=True)
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped")
    