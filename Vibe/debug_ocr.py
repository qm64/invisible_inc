from invisible_inc_env import InvisibleIncEnv
import mss
import cv2
import numpy as np
import pytesseract
import time

print("="*60)
print("DEBUG OCR READING")
print("="*60)

env = InvisibleIncEnv()

print("\n1. Launch Invisible Inc and start a mission")
print("2. Make sure power meter shows '5/20 PWR'")
print("3. Switch back and press ENTER")
input("\nPress ENTER to continue...")

print("\nCapturing in 3 seconds...")
time.sleep(3)

# Capture full resolution screenshot
with mss.mss() as sct:
    full_screenshot = sct.grab(env.game_region)
    img = np.array(full_screenshot)

# Extract power region
x, y, w, h = env.power_region
power_img = img[y:y+h, x:x+w]

# Save the raw power region
cv2.imwrite('power_raw.png', cv2.cvtColor(power_img, cv2.COLOR_BGRA2BGR))
print("✓ Saved power_raw.png")

# Convert to grayscale
gray = cv2.cvtColor(power_img, cv2.COLOR_BGRA2GRAY)
cv2.imwrite('power_gray.png', gray)
print("✓ Saved power_gray.png")

# Threshold
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
cv2.imwrite('power_thresh.png', thresh)
print("✓ Saved power_thresh.png")

# Try OCR with different configs
text1 = pytesseract.image_to_string(thresh, config='--psm 7')
text2 = pytesseract.image_to_string(thresh, config='--psm 6')
text3 = pytesseract.image_to_string(power_img, config='--psm 7')

print("\n" + "="*60)
print("OCR RESULTS:")
print("="*60)
print(f"From thresholded (psm 7): '{text1.strip()}'")
print(f"From thresholded (psm 6): '{text2.strip()}'")
print(f"From original (psm 7):    '{text3.strip()}'")

print("\n" + "="*60)
print("IMAGES SAVED - OPEN THEM:")
print("="*60)
print("open power_raw.png power_gray.png power_thresh.png")
print("\nLook at power_thresh.png - is the text clear and readable?")

env.close()