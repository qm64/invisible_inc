    def extract_alarm_level(self, image: np.ndarray) -> Optional[int]:
        """
        Extract alarm level number (0-6) from security clock.
        
        Uses image-relative positioning rather than requiring security_clock detection.
        The alarm level is always at a fixed position in the upper-right corner:
        - Approximately 2.3-3.5% from the right edge of the image
        - Approximately 9.7% from the top edge of the image
        
        Works with color-based extraction (yellow/orange/red) and grayscale fallback.
        
        Returns:
            int 0-6 if detected, None if extraction fails
        """
        try:
            import pytesseract
            import re
        except ImportError:
            if self.debug:
                print("⚠ pytesseract not installed, cannot extract alarm level")
            return None
        
        h, w = image.shape[:2]
        
        # Use image-relative positioning (works across resolutions)
        # Based on analysis: ~46-71px from right, ~109px from top (in 2000x1125 images)
        # As percentages: 97.3% across, 9.7% down
        clock_center_x = int(w * 0.973)
        clock_center_y = int(h * 0.097)
        
        # Extract a region around the alarm level number
        region_size = 50
        
        x1 = max(0, clock_center_x - region_size // 2)
        y1 = max(0, clock_center_y - region_size // 2)
        x2 = min(w, clock_center_x + region_size // 2)
        y2 = min(h, clock_center_y + region_size // 2)
        
        region = image[y1:y2, x1:x2]
        
        if self.debug:
            print(f"  Alarm level search region: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Convert to HSV for color-based extraction
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Prepare multiple color masks for different alarm levels
        methods = []
        
        # Yellow (low alarm levels 0-2)
        mask_yellow = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([45, 255, 255]))
        methods.append(("yellow", mask_yellow))
        
        # Orange (medium alarm levels 3-4)  
        mask_orange = cv2.inRange(hsv, np.array([8, 100, 100]), np.array([25, 255, 255]))
        methods.append(("orange", mask_orange))
        
        # Red (high alarm levels 5-6)
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])),
            cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        )
        methods.append(("red", mask_red))
        
        # Combined mask
        mask_combined = cv2.bitwise_or(mask_yellow, cv2.bitwise_or(mask_orange, mask_red))
        methods.append(("combined", mask_combined))
        
        # Grayscale threshold (fallback - works when color masks fail)
        region_scaled_gray = cv2.resize(region, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(region_scaled_gray, cv2.COLOR_BGR2GRAY)
        _, gray_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        gray_thresh_small = cv2.resize(gray_thresh, (region.shape[1], region.shape[0]), 
                                       interpolation=cv2.INTER_AREA)
        methods.append(("grayscale", gray_thresh_small))
        
        # Try OCR with each method, using both PSM 6 and PSM 8
        # PSM 6 (uniform block) works better for digits like 4
        # PSM 8 (single word) works better for simple digits like 0, 6
        best_result = None
        best_confidence = 0
        
        for method_name, mask in methods:
            # Scale up for better OCR
            scale = 8
            mask_scaled = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # Try both PSM modes
            for psm in [6, 8]:
                config = f'--psm {psm} -c tessedit_char_whitelist=0123456'
                text = pytesseract.image_to_string(mask_scaled, config=config).strip()
                
                # Look for single digit 0-6
                match = re.search(r'[0-6]', text)
                if match:
                    digit = int(match.group(0))
                    # Confidence based on exact match and number of pixels found
                    pixels = np.count_nonzero(mask)
                    confidence = 1.0 if text == str(digit) else 0.7
                    confidence += min(pixels / 1000, 0.3)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = (digit, f"{method_name}_PSM{psm}")
        
        if best_result:
            alarm_level, method = best_result
            if self.debug:
                print(f"  ✓ Alarm level: {alarm_level} (method: {method})")
            return alarm_level
        
        if self.debug:
            print(f"  ✗ Could not extract alarm level")
        
        return None
