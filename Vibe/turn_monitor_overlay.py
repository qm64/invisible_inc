import mss
import cv2
import numpy as np
import time
import pytesseract

class TurnStateMonitor:
    """Detects game state and shows overlay"""
    
    def __init__(self):
        self.sct = mss.mss()
        self.game_region = {
            "top": 0,
            "left": 0,
            "width": 2560,
            "height": 1440
        }
        
        self.agent_panel_region = (20, 1000, 150, 360)
        self.power_region = (0, 0, 150, 50)
        self.credits_region = (160, 0, 200, 50)
    
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
        """Detect if it's the player's turn"""
        img = self._capture_full_screen()
        panel = self._extract_region(img, self.agent_panel_region)
        green = panel[:, :, 1]
        bright_green = np.sum(green > 100)
        total_pixels = green.size
        green_percentage = bright_green / total_pixels
        is_player = green_percentage > 0.05
        return is_player, green_percentage
    
    def get_power(self):
        """Read current power"""
        img = self._capture_full_screen()
        power_img = self._extract_region(img, self.power_region)
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
    
    def get_credits(self):
        """Read current credits"""
        img = self._capture_full_screen()
        credits_img = self._extract_region(img, self.credits_region)
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
        """Get all game state"""
        is_player, green_pct = self.is_player_turn()
        
        state = {
            'is_player_turn': is_player,
            'turn_confidence': green_pct,
            'power': self.get_power() if is_player else None,
            'credits': self.get_credits() if is_player else None,
        }
        
        return state

# Test with overlay window
if __name__ == "__main__":
    print("="*60)
    print("TURN STATE MONITOR - WITH OVERLAY")
    print("="*60)
    
    monitor = TurnStateMonitor()
    
    print("\n1. Launch Invisible Inc and play")
    print("2. A small overlay window will show game state")
    print("3. Position it where you can see it while playing")
    print("\nPress Ctrl+\\ to stop\n")
    
    input("Press ENTER to start...")
    
    # Create overlay window
    cv2.namedWindow('Game State', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Game State', 400, 200)
    
    try:
        while True:
            state = monitor.get_full_state()
            
            # Create overlay image
            overlay = np.zeros((200, 400, 3), dtype=np.uint8)
            
            # Background color based on turn
            if state['is_player_turn']:
                overlay[:] = (0, 50, 0)  # Dark green for player turn
                turn_text = "YOUR TURN"
                color = (0, 255, 0)  # Bright green text
            else:
                overlay[:] = (50, 0, 0)  # Dark red for enemy turn
                turn_text = "ENEMY TURN"
                color = (0, 0, 255)  # Red text
            
            # Draw text
            cv2.putText(overlay, turn_text, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            if state['is_player_turn']:
                cv2.putText(overlay, f"Power: {state['power']}", (20, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(overlay, f"Credits: {state['credits']}", (20, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show confidence
            conf_text = f"Conf: {state['turn_confidence']*100:.1f}%"
            cv2.putText(overlay, conf_text, (20, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow('Game State', overlay)
            cv2.waitKey(100)  # Update 10 times per second
            
    except KeyboardInterrupt:
        print("\nStopped")
    
    cv2.destroyAllWindows()
    monitor.close()