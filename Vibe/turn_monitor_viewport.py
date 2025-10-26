from window_detector import WindowDetector
import cv2
import numpy as np
import time
import pytesseract

class ViewportTurnMonitor:
    """Turn monitor that handles letterboxed game viewport"""
    
    def __init__(self):
        self.detector = WindowDetector()
        self.viewport_cache = None
        self.viewport_cache_time = 0
        self.viewport_cache_duration = 5.0  # Re-detect viewport every 5 seconds
        
    def detect_viewport(self, img):
        """
        Detect the actual game viewport (excluding black bars)
        Returns: (left, top, right, bottom) or None
        """
        height, width = img.shape[:2]
        
        # Extract green channel (game UI is green)
        green = img[:, :, 1]
        
        # Find where there's significant green content
        green_threshold = cv2.threshold(green, 80, 255, cv2.THRESH_BINARY)[1]
        
        # Find first and last rows/columns with green
        rows_with_green = np.any(green_threshold > 0, axis=1)
        cols_with_green = np.any(green_threshold > 0, axis=0)
        
        if not np.any(rows_with_green) or not np.any(cols_with_green):
            return None
        
        # Find bounding box
        top = np.argmax(rows_with_green)
        bottom = len(rows_with_green) - np.argmax(rows_with_green[::-1])
        left = np.argmax(cols_with_green)
        right = len(cols_with_green) - np.argmax(cols_with_green[::-1])
        
        # Add small margin
        margin = 5
        top = max(0, top - margin)
        left = max(0, left - margin)
        bottom = min(height, bottom + margin)
        right = min(width, right + margin)
        
        return (left, top, right, bottom)
    
    def get_viewport(self, img, force_update=False):
        """Get viewport with caching"""
        current_time = time.time()
        
        if force_update or (current_time - self.viewport_cache_time) > self.viewport_cache_duration:
            viewport = self.detect_viewport(img)
            if viewport:
                self.viewport_cache = viewport
                self.viewport_cache_time = current_time
        
        return self.viewport_cache
    
    def extract_viewport(self, img):
        """Extract just the game viewport from window capture"""
        viewport = self.get_viewport(img)
        if viewport is None:
            return None
        
        left, top, right, bottom = viewport
        return img[top:bottom, left:right]
    
    def get_viewport_region(self, viewport_img, region_percent):
        """
        Extract region from viewport using percentages
        region_percent: (x_pct, y_pct, width_pct, height_pct)
        """
        if viewport_img is None:
            return None
        
        h, w = viewport_img.shape[:2]
        x_pct, y_pct, w_pct, h_pct = region_percent
        
        x = int(w * x_pct)
        y = int(h * y_pct)
        width = int(w * w_pct)
        height = int(h * h_pct)
        
        return viewport_img[y:y+height, x:x+width]
    
    def is_player_turn(self, viewport_img):
        """Detect player turn from agent panel"""
        # Agent panel is in lower-left of viewport
        panel = self.get_viewport_region(viewport_img, (0, 0.70, 0.08, 0.25))
        if panel is None:
            return False, 0.0
        
        green = panel[:, :, 1]
        bright_green = np.sum(green > 100)
        total_pixels = green.size
        green_percentage = bright_green / total_pixels
        
        is_player = green_percentage > 0.05
        return is_player, green_percentage
    
    def get_power(self, viewport_img):
        """Read power from viewport"""
        power_img = self.get_viewport_region(viewport_img, (0, 0, 0.08, 0.05))
        if power_img is None:
            return 0
        
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
    
    def get_credits(self, viewport_img):
        """Read credits from viewport"""
        credits_img = self.get_viewport_region(viewport_img, (0.08, 0, 0.10, 0.05))
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
        """Get complete game state"""
        # Capture window
        window_img = self.detector.capture_game_window(auto_focus=False)
        if window_img is None:
            return None
        
        # Extract viewport
        viewport_img = self.extract_viewport(window_img)
        if viewport_img is None:
            return None
        
        # Get state from viewport
        is_player, confidence = self.is_player_turn(viewport_img)
        
        state = {
            'is_player_turn': is_player,
            'turn_confidence': confidence,
            'power': self.get_power(viewport_img) if is_player else None,
            'credits': self.get_credits(viewport_img) if is_player else None,
        }
        
        return state

# Test
if __name__ == "__main__":
    print("="*60)
    print("VIEWPORT TURN MONITOR - TEST")
    print("="*60)
    
    monitor = ViewportTurnMonitor()
    
    print("\n1. Launch Invisible Inc in WINDOWED mode")
    print("2. Resize window however you like")
    print("3. Start a mission")
    input("\nPress ENTER to start monitoring...")
    
    print("\nMonitoring... (Press Ctrl+\\ to stop)\n")
    
    try:
        while True:
            state = monitor.get_full_state()
            
            if state is None:
                print("\r[ERROR] Could not capture window/viewport", end="", flush=True)
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
        