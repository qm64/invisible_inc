import Quartz
import mss
import cv2
import numpy as np
from AppKit import NSWorkspace, NSRunningApplication

class WindowDetector:
    """Detects and tracks the Invisible Inc game window"""
    
    def __init__(self, window_owner_name="InvisibleInc"):
        """
        Args:
            window_owner_name: The exact app owner name (e.g., "InvisibleInc")
        """
        self.window_owner_name = window_owner_name
        self.game_bounds = None
        self.last_check_time = 0
        self.check_interval = 2.0  # Re-check window position every 2 seconds
        
        # macOS window chrome offsets (title bar + borders)
        self.title_bar_height = 28  # Height of title bar with window controls
        self.border_left = 0         # Left border (usually 0)
        self.border_right = 0        # Right border (usually 0)
        self.border_bottom = 0       # Bottom border (usually 0)
        
    def focus_game_window(self):
        """
        Bring the game window to front
        Returns: True if successful, False otherwise
        """
        workspace = NSWorkspace.sharedWorkspace()
        running_apps = workspace.runningApplications()
        
        for app in running_apps:
            if app.localizedName() == self.window_owner_name:
                # Use the numeric constant instead of the named constant
                # NSApplicationActivateIgnoringOtherApps = 1 << 1
                success = app.activateWithOptions_(1 << 1)
                if success:
                    print(f"[Window] Focused '{self.window_owner_name}'")
                    import time
                    time.sleep(0.2)  # Give window time to come to front
                    return True
                else:
                    print(f"[Window] Failed to focus '{self.window_owner_name}'")
                    return False
        
        print(f"[Window] Could not find app '{self.window_owner_name}' to focus")
        return False
        
    def find_game_window(self):
        """
        Find Invisible Inc window using macOS APIs
        Returns: (x, y, width, height) or None
        """
        # Get list of all windows
        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID
        )
        
        # First pass: look for EXACT owner match (most reliable)
        for window in window_list:
            window_owner = window.get('kCGWindowOwnerName', '')
            
            if window_owner == self.window_owner_name:
                bounds = window.get('kCGWindowBounds', {})
                x = int(bounds.get('X', 0))
                y = int(bounds.get('Y', 0))
                width = int(bounds.get('Width', 0))
                height = int(bounds.get('Height', 0))
                
                if width > 0 and height > 0:
                    print(f"[Window] Found by exact owner: '{window_owner}'")
                    return (x, y, width, height)
        
        # Second pass: look for partial match (fallback)
        for window in window_list:
            window_owner = window.get('kCGWindowOwnerName', '')
            window_name = window.get('kCGWindowName', '')
            
            if self.window_owner_name.lower() in window_owner.lower() or \
               self.window_owner_name.lower() in window_name.lower():
                
                bounds = window.get('kCGWindowBounds', {})
                x = int(bounds.get('X', 0))
                y = int(bounds.get('Y', 0))
                width = int(bounds.get('Width', 0))
                height = int(bounds.get('Height', 0))
                
                if width > 0 and height > 0:
                    print(f"[Window] Found by partial match: '{window_owner}' / '{window_name}'")
                    return (x, y, width, height)
        
        return None
    
    def get_game_bounds(self, force_update=False):
        """
        Get current game window bounds, with caching
        Returns: (x, y, width, height)
        """
        import time
        current_time = time.time()
        
        # Re-check if interval passed or forced
        if force_update or (current_time - self.last_check_time) > self.check_interval:
            bounds = self.find_game_window()
            if bounds:
                self.game_bounds = bounds
                self.last_check_time = current_time
                print(f"[Window] Updated: x={bounds[0]}, y={bounds[1]}, size={bounds[2]}x{bounds[3]}")
        
        return self.game_bounds
    
    def capture_game_window(self, auto_focus=False):
        """
        Capture the game window (excluding title bar and borders)
        
        Args:
            auto_focus: If True, bring game to front before capturing
            
        Returns: numpy array of the image, or None if window not found
        """
        if auto_focus:
            self.focus_game_window()
        
        bounds = self.get_game_bounds()
        
        if not bounds:
            return None
        
        x, y, width, height = bounds
        
        # Adjust for window chrome (title bar and borders)
        x += self.border_left
        y += self.title_bar_height
        width -= (self.border_left + self.border_right)
        height -= (self.title_bar_height + self.border_bottom)
        
        with mss.mss() as sct:
            monitor = {
                "top": y,
                "left": x,
                "width": width,
                "height": height
            }
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    
    def get_region(self, img, region_percent):
        """
        Extract region using percentage-based coordinates
        region_percent: (x_pct, y_pct, width_pct, height_pct) where values are 0.0-1.0
        Example: (0, 0, 0.1, 0.05) = top-left 10% width, 5% height
        """
        if img is None:
            return None
        
        img_height, img_width = img.shape[:2]
        
        x_pct, y_pct, w_pct, h_pct = region_percent
        
        x = int(img_width * x_pct)
        y = int(img_height * y_pct)
        w = int(img_width * w_pct)
        h = int(img_height * h_pct)
        
        return img[y:y+h, x:x+w]

# Test the detector
if __name__ == "__main__":
    print("="*60)
    print("WINDOW DETECTOR TEST (with auto-focus)")
    print("="*60)
    
    detector = WindowDetector()  # Defaults to "InvisibleInc"
    
    print("\n1. Launch Invisible Inc in WINDOWED mode")
    print("2. You can switch to Terminal - auto-focus will handle it")
    input("\nPress ENTER to detect and capture window...")
    
    # Find window first
    bounds = detector.find_game_window()
    
    if bounds:
        x, y, w, h = bounds
        print(f"\n✓ Found game window!")
        print(f"  Position: ({x}, {y})")
        print(f"  Size: {w}x{h}")
        
        print("\nCapturing game window with auto-focus...")
        img = detector.capture_game_window(auto_focus=True)
        
        if img is not None:
            print(f"✓ Captured image: {img.shape}")
            
            # Save test capture
            cv2.imwrite('window_test.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print("✓ Saved as window_test.png")
            
            # Test region extraction (top-left 10%)
            print("\nTesting region extraction...")
            power_region = detector.get_region(img, (0, 0, 0.1, 0.05))
            if power_region is not None:
                print(f"✓ Power region: {power_region.shape}")
                cv2.imwrite('power_region_test.png', cv2.cvtColor(power_region, cv2.COLOR_RGB2BGR))
                print("✓ Saved as power_region_test.png")
                
            print("\n" + "="*60)
            print("Check the images:")
            print("  open window_test.png")
            print("  open power_region_test.png")
            print("="*60)
        else:
            print("✗ Failed to capture window")
    else:
        print("\n✗ Could not find Invisible Inc window")
        print("\nMake sure:")
        print("  - Game is running")
        print("  - Game is in windowed mode")
