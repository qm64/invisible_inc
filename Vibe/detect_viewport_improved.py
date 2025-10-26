from window_detector import WindowDetector
import cv2
import numpy as np

detector = WindowDetector()

print("="*60)
print("DETECT GAME VIEWPORT (IMPROVED)")
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

# Method: Look for green UI elements (the game always has green UI)
# Extract green channel
green = img[:, :, 1]

# Find where there's significant green (UI elements)
green_threshold = cv2.threshold(green, 80, 255, cv2.THRESH_BINARY)[1]

# Find first and last rows/columns with green content
rows_with_green = np.any(green_threshold > 0, axis=1)
cols_with_green = np.any(green_threshold > 0, axis=0)

if not np.any(rows_with_green) or not np.any(cols_with_green):
    print("ERROR: Could not find green UI elements")
    exit()

# Find bounding box of green content
top = np.argmax(rows_with_green)
bottom = len(rows_with_green) - np.argmax(rows_with_green[::-1])
left = np.argmax(cols_with_green)
right = len(cols_with_green) - np.argmax(cols_with_green[::-1])

# Add small margin (UI might not extend to exact edges)
margin = 5
top = max(0, top - margin)
left = max(0, left - margin)
bottom = min(height, bottom + margin)
right = min(width, right + margin)

viewport_width = right - left
viewport_height = bottom - top

print(f"\nGame viewport detected (by green UI):")
print(f"  Position: ({left}, {top})")
print(f"  Size: {viewport_width}x{viewport_height}")
print(f"  Aspect ratio: {viewport_width/viewport_height:.3f}")

# Calculate black bar sizes
print(f"\nBlack bars:")
print(f"  Top: {top}px")
print(f"  Bottom: {height - bottom}px")
print(f"  Left: {left}px")
print(f"  Right: {width - right}px")

# Draw the detected viewport
img_annotated = img.copy()
cv2.rectangle(img_annotated, (left, top), (right, bottom), (0, 255, 0), 3)
cv2.imwrite('viewport_detected2.png', cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))
print("\n✓ Saved viewport_detected2.png")

# Extract viewport
game_viewport = img[top:bottom, left:right]
cv2.imwrite('viewport_only2.png', cv2.cvtColor(game_viewport, cv2.COLOR_RGB2BGR))
print("✓ Saved viewport_only2.png")

# Now test power region on the viewport
print("\n" + "="*60)
print("TESTING POWER/CREDITS REGIONS:")
print("="*60)

# Power should be at top-left of viewport
power_w = int(viewport_width * 0.08)
power_h = int(viewport_height * 0.05)
power_region = game_viewport[0:power_h, 0:power_w]
cv2.imwrite('viewport_power.png', cv2.cvtColor(power_region, cv2.COLOR_RGB2BGR))
print(f"\nPower region: {power_w}x{power_h}")
print("✓ Saved viewport_power.png")

# Credits should be to the right of power
credits_x = int(viewport_width * 0.08)
credits_w = int(viewport_width * 0.10)
credits_h = int(viewport_height * 0.05)
credits_region = game_viewport[0:credits_h, credits_x:credits_x+credits_w]
cv2.imwrite('viewport_credits.png', cv2.cvtColor(credits_region, cv2.COLOR_RGB2BGR))
print(f"Credits region: {credits_w}x{credits_h}")
print("✓ Saved viewport_credits.png")

print("\n" + "="*60)
print("CHECK:")
print("="*60)
print("open viewport_detected2.png")
print("open viewport_power.png")
print("open viewport_credits.png")
print("\nDoes the green rectangle match the game perfectly?")
print("Do power and credits regions show the text?")