import mss
import cv2
import numpy as np
import time

class TurnStateMonitor:
    """
    Detects if it's the player's turn or enemy's turn
    Strategy: Agent panel visible = player turn, hidden = enemy turn
    """
    
    def __init__(self):
        self.sct = mss.mss()
        self.game_region = {
            "top": 0,
            "left": 0,
            "width": 2560,
            "height": 1440
        }
        
        # Agent panel region (lower-left where icons appear)
        # Based on your analysis: icons at x=~45, y=1014-1330
        self.agent_panel_region = (20, 1000, 150, 360)  # (x, y, width, height)
        
        # Regions we already have working
        self.power_region = (0, 0, 150, 50)
        self.credits_region = (160, 0, 200, 50)
        self.alarm_region = (2350, 100, 200, 120)
    
    def _capture_full_screen(self):
        """Capture full game screen"""
        screenshot = self.sct.grab(self.game_region)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    
    def _extract_region(self, img, region):
        """Extract a specific region from image"""
        x, y, w, h = region
        return img[y:y+h, x:x+w]
    
    def is_player_turn(self):
        """
        Detect if it's the player's turn
        Returns: True if player turn, False if enemy turn
        """
        img = self._capture_full_screen()
        
        # Extract agent panel region
        panel = self._extract_region(img, self.agent_panel_region)
        
        # Convert to grayscale
        gray = cv2.cvtColor(panel, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Check if there's significant green content (UI is green)
        # Extract green channel
        green = panel[:, :, 1]
        
        # Count bright green pixels (UI elements)
        bright_green = np.sum(green > 100)
        total_pixels = green.size
        green_percentage = bright_green / total_pixels
        
        # If more than 5% of pixels are bright green, panel is visible
        is_player = green_percentage > 0.05
        
        return is_player, green_percentage
    
    def get_power(self):
        """Read current power (reusing working code)"""
        img = self._capture_full_screen()
        power_img = self._extract_region(img, self.power_region)
        
        # Convert to RGB and extract green channel
        green = power_img[:, :, 1]
        _, thresh = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        import pytesseract
        text = pytesseract.image_to_string(thresh, config='--psm 7')
        
        try:
            parts = text.split('/')
            if len(parts) >= 1:
                power = int(''.join(filter(str.isdigit, parts[0])))
                return power
        except:
            pass
        return 0
    
    def get_credits(self):
        """Read current credits (reusing working code)"""
        img = self._capture_full_screen()
        credits_img = self._extract_region(img, self.credits_region)
        
        green = credits_img[:, :, 1]
        _, thresh = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        import pytesseract
        text = pytesseract.image_to_string(thresh, config='--psm 7 digits')
        
        try:
            digits = ''.join(filter(str.isdigit, text))
            if digits:
                return int(digits)
        except:
            pass
        return 0
    
    def get_alarm_level(self):
        """
        Read alarm level from upper-right
        TODO: Implement OCR for alarm number
        For now, return placeholder
        """
        # TODO: Extract alarm region and read number
        return -1  # -1 means "not implemented yet"
    
    def get_full_state(self):
        """Get all game state info at once"""
        is_player, green_pct = self.is_player_turn()
        
        state = {
            'is_player_turn': is_player,
            'turn_confidence': green_pct,
            'power': self.get_power() if is_player else None,
            'credits': self.get_credits() if is_player else None,
            'alarm': self.get_alarm_level()
        }
        
        return state
    
    def close(self):
        """Cleanup"""
        pass

# Test the monitor
if __name__ == "__main__":
    print("="*60)
    print("TURN STATE MONITOR - TEST")
    print("="*60)
    
    monitor = TurnStateMonitor()
    
    print("\n1. Launch Invisible Inc")
    print("2. Start a mission")
    print("3. This will monitor your game state")
    input("\nPress ENTER to start monitoring...")
    
    print("\nMonitoring... (Press Ctrl+\\ to stop)\n")
    
    try:
        while True:
            state = monitor.get_full_state()
            
            # Clear line and print status
            print("\r" + " "*80, end="")  # Clear line
            
            turn_status = "YOUR TURN" if state['is_player_turn'] else "ENEMY TURN"
            confidence = f"{state['turn_confidence']*100:.1f}%"
            
            if state['is_player_turn']:
                status = f"[{turn_status}] Power: {state['power']} | Credits: {state['credits']} | Confidence: {confidence}"
            else:
                status = f"[{turn_status}] Confidence: {confidence}"
            
            print(f"\r{status}", end="", flush=True)
            
            time.sleep(0.5)  # Update twice per second
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped")
    
    monitor.close()