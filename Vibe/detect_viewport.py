from window_detector import WindowDetector
import cv2
import numpy as np

detector = WindowDetector()

print("="*60)
print("DETECT GAME VIEWPORT")
print("="*60)

print("\n1. Launch Invisible Inc in WINDOWED mode")
print("2. Start a mission")
input("\nPress ENTER to detect viewport...")

# Capture window
img = detector.capture_game_window(auto_focus=True)

if img is None:
    print("ERROR: Could not capture window")
    exit()

height, width = img.shape[:2]
print(f"\nWindow size: {width}x{height}")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Find non-black regions (game content)
# Black bars are very dark (< 10), game content is brighter
_, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

# Find contours of non-black regions
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    print("ERROR: Could not find game content")
    exit()

# Get bounding box of largest contour (the game viewport)
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)

print(f"\nGame viewport detected:")
print(f"  Position: ({x}, {y})")
print(f"  Size: {w}x{h}")
print(f"  Aspect ratio: {w/h:.3f}")

# Calculate black bar sizes
top_bar = y
bottom_bar = height - (y + h)
left_bar = x
right_bar = width - (x + w)

print(f"\nBlack bars:")
print(f"  Top: {top_bar}px")
print(f"  Bottom: {bottom_bar}px")
print(f"  Left: {left_bar}px")
print(f"  Right: {right_bar}px")

# Draw the detected viewport on the image
img_annotated = img.copy()
cv2.rectangle(img_annotated, (x, y), (x+w, y+h), (0, 255, 0), 3)
cv2.imwrite('viewport_detected.png', cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))
print("\n✓ Saved viewport_detected.png (green rectangle shows viewport)")

# Extract just the game viewport
game_viewport = img[y:y+h, x:x+w]
cv2.imwrite('viewport_only.png', cv2.cvtColor(game_viewport, cv2.COLOR_RGB2BGR))
print("✓ Saved viewport_only.png (game content only)")

print("\n" + "="*60)
print("CHECK:")
print("="*60)
print("open viewport_detected.png")
print("\nDoes the green rectangle match the game content exactly?")