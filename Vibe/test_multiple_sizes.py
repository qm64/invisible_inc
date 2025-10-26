from window_detector import WindowDetector
import cv2
import numpy as np
import pytesseract
import mss

class MultiSizeTest:
    """Test detector at multiple window sizes"""
    
    def __init__(self):
        self.detector = WindowDetector()
    
    def capture_full_screen(self):
        """Capture entire screen as fallback for fullscreen/maximized"""
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    
    def read_power(self, img, x, y, w, h):
        """Try to read power at given position"""
        if y+h > img.shape[0] or x+w > img.shape[1] or w <= 0 or h <= 0:
            return None, None
        
        power_img = img[y:y+h, x:x+w]
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
    
    def find_power_by_search(self, img):
        """Search for power text in top-left area"""
        # Search larger area
        search_h = min(150, img.shape[0])
        search_w = min(500, img.shape[1])
        search_area = img[0:search_h, 0:search_w]
        
        # Find green text
        green = search_area[:, :, 1]
        _, mask = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        
        # Find leftmost and topmost green content
        rows_with_green = np.any(mask > 0, axis=1)
        cols_with_green = np.any(mask > 0, axis=0)
        
        if not np.any(rows_with_green) or not np.any(cols_with_green):
            return None
        
        top = np.argmax(rows_with_green)
        left = np.argmax(cols_with_green)
        
        # Power should be near this position
        # Try to extract a reasonable box
        power_x = max(0, left - 5)
        power_y = max(0, top - 5)
        power_w = 100
        power_h = 30
        
        return (power_x, power_y, power_w, power_h)
    
    def test_window_size(self, use_screen_capture=False):
        """Test current window size"""
        if use_screen_capture:
            print("Using full screen capture...")
            img = self.capture_full_screen()
        else:
            img = self.detector.capture_game_window(auto_focus=True)
        
        if img is None:
            return None
        
        h, w = img.shape[:2]
        print(f"\nWindow size: {w}x{h}")
        
        # Method 1: Try absolute positions from 2560x1301 measurements
        # Scale proportionally
        scale_x = w / 2560
        scale_y = h / 1301
        
        power_x = int(141 * scale_x)
        power_y = int(17 * scale_y)
        power_w = int(100 * scale_x)
        power_h = int(21 * scale_y)
        
        print(f"Method 1 (scaled): Trying power at ({power_x},{power_y}) size {power_w}x{power_h}")
        p1_curr, p1_max = self.read_power(img, power_x, power_y, power_w, power_h)
        
        if p1_curr is not None:
            print(f"  ✓ Read: {p1_curr}/{p1_max}")
        else:
            print(f"  ✗ Failed")
        
        # Method 2: Search for power text
        print(f"Method 2 (search): Searching for power text...")
        found = self.find_power_by_search(img)
        
        p2_curr, p2_max = None, None
        
        if found:
            px, py, pw, ph = found
            print(f"  Found at ({px},{py}) size {pw}x{ph}")
            p2_curr, p2_max = self.read_power(img, px, py, pw, ph)
            
            if p2_curr is not None:
                print(f"  ✓ Read: {p2_curr}/{p2_max}")
                
                # Calculate where this is as a percentage
                x_pct = px / w
                y_pct = py / h
                print(f"  Position: {x_pct:.3f}x, {y_pct:.3f}y from top-left")
            else:
                print(f"  ✗ Could not read text")
        else:
            print(f"  ✗ Could not find power text")
        
        # Save debug
        cv2.imwrite('debug_current_window.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        return {
            'window_size': (w, h),
            'method1_result': (p1_curr, p1_max),
            'method2_result': (p2_curr, p2_max),
        }

# Test
if __name__ == "__main__":
    print("="*60)
    print("MULTI-SIZE TEST")
    print("="*60)
    
    tester = MultiSizeTest()
    
    sizes_to_test = [
        "Current 'full' size (tray + dock visible)",
        "Narrower/taller (more vertical)",
        "Wider/shorter (more horizontal)",
        "Maximized (Option+Click, will use screen capture)",
    ]
    
    results = []
    
    for i, size_desc in enumerate(sizes_to_test):
        print(f"\n{'='*60}")
        print(f"Test {i+1}/{len(sizes_to_test)}: {size_desc}")
        print('='*60)
        
        input("Adjust window size, then press ENTER...")
        
        # Use screen capture for test #4 (maximized)
        use_screen = (i == 3)  # Last test
        if use_screen:
            print("(Using full screen capture for this test)")
        
        result = tester.test_window_size(use_screen_capture=use_screen)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        w, h = result['window_size']
        m1 = result['method1_result']
        m2 = result['method2_result']
        
        print(f"\nTest {i}: {w}x{h} (aspect: {w/h:.2f})")
        print(f"  Method 1 (scaled): {m1[0]}/{m1[1] if m1[0] else 'FAILED'}")
        print(f"  Method 2 (search): {m2[0]}/{m2[1] if m2[0] else 'FAILED'}")
        