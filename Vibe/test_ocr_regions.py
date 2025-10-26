from window_detector import WindowDetector
import cv2
import numpy as np
import pytesseract

detector = WindowDetector()

print("="*60)
print("TEST OCR REGIONS")
print("="*60)

input("\nPress ENTER to capture...")

window_img = detector.capture_game_window(auto_focus=True)

# Extract viewport with better black border trimming
green = window_img[:, :, 1]
green_threshold = cv2.threshold(green, 80, 255, cv2.THRESH_BINARY)[1]
rows_with_green = np.any(green_threshold > 0, axis=1)
cols_with_green = np.any(green_threshold > 0, axis=0)

top = np.argmax(rows_with_green)
bottom = len(rows_with_green) - np.argmax(rows_with_green[::-1])
left = np.argmax(cols_with_green)
right = len(cols_with_green) - np.argmax(cols_with_green[::-1])

viewport = window_img[top:bottom, left:right]

# Trim left black border more aggressively
gray = cv2.cvtColor(viewport, cv2.COLOR_RGB2GRAY)
trim_left = 0
for x in range(min(400, viewport.shape[1])):
    col_mean = np.mean(gray[:, x])
    if col_mean > 10:
        trim_left = x
        break

# Add small margin back for power display
trim_left = max(0, trim_left - 10)  # 10px margin on left
viewport = viewport[:, trim_left:]

# Trim right black border (less aggressive)
gray = cv2.cvtColor(viewport, cv2.COLOR_RGB2GRAY)
trim_right = viewport.shape[1]
for x in range(viewport.shape[1]-1, max(0, viewport.shape[1]-400), -1):
    col_mean = np.mean(gray[:, x])
    if col_mean > 10:
        trim_right = x + 1
        break

# Add margin for objectives text
trim_right = min(viewport.shape[1], trim_right + 50)  # 50px margin on right
viewport = viewport[:, :trim_right]

# Trim bottom black border
for y in range(viewport.shape[0]-1, max(0, viewport.shape[0]-100), -1):
    row_mean = np.mean(gray[y, :])
    if row_mean > 10:
        viewport = viewport[:y+1, :]
        break

h, w = viewport.shape[:2]
print(f"Viewport after trimming: {w}x{h}")

# Save trimmed viewport
cv2.imwrite('viewport_trimmed.png', cv2.cvtColor(viewport, cv2.COLOR_RGB2BGR))

# Test regions
layout = {
    'Power': {'x': 0.0, 'y': 0.0, 'width': 0.06, 'height': 0.035},  # Slightly wider
    'Credits': {'x': 0.0559, 'y': 0.0, 'width': 0.045, 'height': 0.035},  # Much wider (was 0.0315)
    'Alarm_integer': {'x': 0.96, 'y': 0.0902, 'width': 0.035, 'height': 0.0491},  # Wider and more centered (was x=0.9667, w=0.0191)
}

for name, region_def in layout.items():
    x = int(w * region_def['x'])
    y = int(h * region_def['y'])
    width = int(w * region_def['width'])
    height = int(h * region_def['height'])
    
    x = max(0, min(x, w-1))
    y = max(0, min(y, h-1))
    width = min(width, w - x)
    height = min(height, h - y)
    
    print(f"\n{name}:")
    print(f"  Pixels: x={x}, y={y}, w={width}, h={height}")
    
    if width <= 0 or height <= 0:
        print(f"  ✗ Invalid size!")
        continue
    
    region = viewport[y:y+height, x:x+width]
    
    cv2.imwrite(f'ocr_test_{name.lower()}.png', cv2.cvtColor(region, cv2.COLOR_RGB2BGR))
    
    if name in ['Power', 'Credits']:
        green_ch = region[:, :, 1]
        _, thresh = cv2.threshold(green_ch, 100, 255, cv2.THRESH_BINARY)
        thresh_big = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f'ocr_test_{name.lower()}_thresh.png', thresh_big)
        
        text = pytesseract.image_to_string(thresh_big, config='--psm 7')
        print(f"  OCR result: '{text.strip()}'")
    elif name == 'Alarm_integer':
        gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray_region, 150, 255, cv2.THRESH_BINARY)
        thresh_big = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f'ocr_test_{name.lower()}_thresh.png', thresh_big)
        
        text = pytesseract.image_to_string(thresh_big, config='--psm 7 digits')
        print(f"  OCR result: '{text.strip()}'")

print("\n" + "="*60)
print("CHECK IMAGES:")
print("="*60)
print("open viewport_trimmed.png ocr_test_power.png ocr_test_power_thresh.png ocr_test_credits.png ocr_test_credits_thresh.png ocr_test_alarm_integer.png ocr_test_alarm_integer_thresh.png")
