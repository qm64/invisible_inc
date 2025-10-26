# Capture the game screenshot

from invisible_inc_env import InvisibleIncEnv
import mss
import cv2
import numpy as np
import time

print("="*60)
print("CAPTURE INVISIBLE INC SCREENSHOT (FULL RES)")
print("="*60)

print("\n1. Launch Invisible Inc and get to an in-game screen")
print("   (during a mission, with UI visible)")
print("2. Switch back here and press ENTER")
input("\nPress ENTER to capture screenshot...")

print("\nCapturing in 3 seconds...")
time.sleep(3)

# Capture at FULL resolution (not resized)
with mss.mss() as sct:
    game_region = {
        "top": 0,
        "left": 0,
        "width": 2560,
        "height": 1440
    }
    
    screenshot = sct.grab(game_region)
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

# Save full resolution
cv2.imwrite('invisible_inc_full.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

print("\n✓ Full resolution screenshot saved as 'invisible_inc_full.png'")
print(f"  Size: {img.shape[1]}x{img.shape[0]} pixels")
print("\nOpen this file and look for:")
print("  - Power meter (top-left, format: 'x/max PWR')")
print("  - Credits (top-left, format: 'xxxx CR')")
print("  - Alarm tracker")
print("\nWe'll use this to build OCR detection.")
