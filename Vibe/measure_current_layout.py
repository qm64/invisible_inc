from window_detector import WindowDetector
import cv2
import numpy as np

detector = WindowDetector()

print("="*60)
print("MEASURE CURRENT LAYOUT")
print("="*60)
print("\nThis will capture and let you click to measure regions")

input("\nPress ENTER to capture...")

window_img = detector.capture_game_window(auto_focus=True)

# Trim viewport
green = window_img[:, :, 1]
green_threshold = cv2.threshold(green, 80, 255, cv2.THRESH_BINARY)[1]
rows_with_green = np.any(green_threshold > 0, axis=1)
cols_with_green = np.any(green_threshold > 0, axis=0)

top = np.argmax(rows_with_green)
bottom = len(rows_with_green) - np.argmax(rows_with_green[::-1])
left = np.argmax(cols_with_green)
right = len(cols_with_green) - np.argmax(cols_with_green[::-1])

viewport = window_img[top:bottom, left:right]

# Trim black borders
gray = cv2.cvtColor(viewport, cv2.COLOR_RGB2GRAY)
for x in range(min(400, viewport.shape[1])):
    if np.mean(gray[:, x]) > 10:
        viewport = viewport[:, x:]
        break

gray = cv2.cvtColor(viewport, cv2.COLOR_RGB2GRAY)
for x in range(viewport.shape[1]-1, max(0, viewport.shape[1]-400), -1):
    if np.mean(gray[:, x]) > 10:
        viewport = viewport[:, :x+1]
        break

for y in range(viewport.shape[0]-1, max(0, viewport.shape[0]-100), -1):
    if np.mean(gray[y, :]) > 10:
        viewport = viewport[:y+1, :]
        break

h, w = viewport.shape[:2]
print(f"\nViewport: {w}x{h}")

# Draw measurement grid
viewport_display = cv2.cvtColor(viewport, cv2.COLOR_RGB2BGR)

# Draw existing regions for reference
regions = [
    ("Power (current)", (0, 0, 116, 41), (0, 255, 0)),
    ("Credits (WRONG)", (96, 0, 77, 41), (0, 0, 255)),
    ("Alarm (WRONG)", (2168, 106, 45, 54), (0, 0, 255)),
]

for name, (x, y, width, height), color in regions:
    cv2.rectangle(viewport_display, (x, y), (x+width, y+height), color, 2)
    cv2.putText(viewport_display, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

cv2.imwrite('layout_to_measure.png', viewport_display)

print("\n" + "="*60)
print("INSTRUCTIONS:")
print("="*60)
print("1. Open: open layout_to_measure.png")
print("2. Look at the green box (Power) - it's correct")
print("3. Look at the red boxes (Credits, Alarm) - they're wrong")
print("\n4. Using Preview's selection tool, measure:")
print("   - Credits text area (should be '65172 CR')")
print("   - Alarm number (just the '0' digit)")
print("\n5. Give me the coordinates as: (x, y, width, height)")
print("\nExample: Credits: x=200, y=5, w=120, h=35")
