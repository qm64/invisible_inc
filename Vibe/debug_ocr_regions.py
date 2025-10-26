from window_detector import WindowDetector
import cv2
import pytesseract

detector = WindowDetector()

print("="*60)
print("DEBUG OCR REGIONS")
print("="*60)

print("\n1. Launch Invisible Inc in WINDOWED mode")
print("2. Start a mission (so power/credits are visible)")
input("\nPress ENTER to capture...")

# Capture window
img = detector.capture_game_window(auto_focus=True)

if img is None:
    print("ERROR: Could not capture window")
    exit()

print(f"\nCaptured window: {img.shape}")
height, width = img.shape[:2]

# Save full capture
cv2.imwrite('debug_full.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print("✓ Saved debug_full.png")

# Test power region (percentage-based)
power_region_pct = (0, 0, 0.08, 0.04)
x_pct, y_pct, w_pct, h_pct = power_region_pct
x = int(width * x_pct)
y = int(height * y_pct)
w = int(width * w_pct)
h = int(height * h_pct)

print(f"\nPower region (percentage): {power_region_pct}")
print(f"Power region (pixels): x={x}, y={y}, w={w}, h={h}")

power_img = img[y:y+h, x:x+w]
cv2.imwrite('debug_power.png', cv2.cvtColor(power_img, cv2.COLOR_RGB2BGR))
print("✓ Saved debug_power.png")

# Try OCR on power
green = power_img[:, :, 1]
_, thresh = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
thresh_large = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
cv2.imwrite('debug_power_thresh.png', thresh_large)

text = pytesseract.image_to_string(thresh_large, config='--psm 7')
print(f"Power OCR result: '{text.strip()}'")

# Test credits region
credits_region_pct = (0.06, 0, 0.10, 0.04)
x_pct, y_pct, w_pct, h_pct = credits_region_pct
x = int(width * x_pct)
y = int(height * y_pct)
w = int(width * w_pct)
h = int(height * h_pct)

print(f"\nCredits region (percentage): {credits_region_pct}")
print(f"Credits region (pixels): x={x}, y={y}, w={w}, h={h}")

credits_img = img[y:y+h, x:x+w]
cv2.imwrite('debug_credits.png', cv2.cvtColor(credits_img, cv2.COLOR_RGB2BGR))
print("✓ Saved debug_credits.png")

# Try OCR on credits
green = credits_img[:, :, 1]
_, thresh = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
thresh_large = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
cv2.imwrite('debug_credits_thresh.png', thresh_large)

text = pytesseract.image_to_string(thresh_large, config='--psm 7 digits')
print(f"Credits OCR result: '{text.strip()}'")

print("\n" + "="*60)
print("CHECK THE IMAGES:")
print("="*60)
print("open debug_full.png")
print("open debug_power.png")
print("open debug_credits.png")
print("\nDo they show the power and credits text?")