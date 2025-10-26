from invisible_inc_env import InvisibleIncEnv
import mss
import cv2
import numpy as np
import pytesseract
import time

print("="*60)
print("DEBUG CREDITS OCR READING")
print("="*60)

env = InvisibleIncEnv()

print("\n1. Launch Invisible Inc and start a mission")
print("2. Make sure you can see credits (should be 500)")
print("3. Switch back and press ENTER")
input("\nPress ENTER to continue...")

print("\nCapturing in 3 seconds...")
time.sleep(3)

# Capture full resolution screenshot
with mss.mss() as sct:
    full_screenshot = sct.grab(env.game_region)
    img = np.array(full_screenshot)

# Extract credits region
x, y, w, h = env.credits_region
credits_img = img[y:y+h, x:x+w]

# Save the raw credits region
cv2.imwrite('credits_raw.png', cv2.cvtColor(credits_img, cv2.COLOR_BGRA2BGR))
print("✓ Saved credits_raw.png")

# Convert BGRA to RGB
rgb = cv2.cvtColor(credits_img, cv2.COLOR_BGRA2RGB)
cv2.imwrite('credits_rgb.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
print("✓ Saved credits_rgb.png")

# Extract green channel
green_channel = rgb[:, :, 1]
cv2.imwrite('credits_green.png', green_channel)
print("✓ Saved credits_green.png")

# Threshold
_, thresh = cv2.threshold(green_channel, 100, 255, cv2.THRESH_BINARY)
cv2.imwrite('credits_thresh.png', thresh)
print("✓ Saved credits_thresh.png")

# Scale up
thresh_large = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
cv2.imwrite('credits_large.png', thresh_large)
print("✓ Saved credits_large.png")

# Try OCR
text1 = pytesseract.image_to_string(thresh_large, config='--psm 7 digits')
text2 = pytesseract.image_to_string(thresh_large, config='--psm 6 digits')

print("\n" + "="*60)
print("OCR RESULTS:")
print("="*60)
print(f"From large (psm 7): '{text1.strip()}'")
print(f"From large (psm 6): '{text2.strip()}'")

# Try to parse
try:
    digits = ''.join(filter(str.isdigit, text1))
    if digits:
        credits = int(digits)
        print(f"\nParsed credits: {credits}")
    else:
        print("\nNo digits found")
except Exception as e:
    print(f"\nFailed to parse credits: {e}")

print("\n" + "="*60)
print("IMAGES SAVED:")
print("="*60)
print("open credits_raw.png credits_green.png credits_thresh.png credits_large.png")

env.close()