# Test screen capture

import mss
import numpy as np
import cv2
from PIL import Image

# Capture entire screen
with mss.mss() as sct:
    # Get monitor info
    monitor = sct.monitors[1]  # Primary monitor
    print(f"Monitor resolution: {monitor['width']}x{monitor['height']}")
    
    # Capture screenshot
    screenshot = sct.grab(monitor)
    
    # Convert to numpy array
    img = np.array(screenshot)
    
    # Display with OpenCV
    cv2.imshow('Screen Capture', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Or save to file
    Image.frombytes('RGB', screenshot.size, screenshot.rgb).save('screenshot.png')
    print("Screenshot saved!")
