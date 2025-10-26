import mss
import cv2
import numpy as np
import time

print("="*60)
print("RECAPTURE SINGLE SCREENSHOT")
print("="*60)

print("\nRecapturing: ui_elements/full_interface.png")
print("\nMake sure NO popups are visible!")
print("Show the clean game interface during your turn.")
input("\nPress ENTER when ready...")

print("Capturing in 3 seconds...")
time.sleep(3)

with mss.mss() as sct:
    screenshot = sct.grab(sct.monitors[1])
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

filepath = "game_docs/ui_elements/full_interface.png"
cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print(f"✓ Saved: {filepath}")
print("\nNow run analyze_screenshots.py again!")