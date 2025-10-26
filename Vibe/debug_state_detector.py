from window_detector import WindowDetector
import cv2
import numpy as np

detector = WindowDetector()

print("="*60)
print("DEBUG STATE DETECTOR")
print("="*60)

print("\n1. Launch Invisible Inc")
print("2. Start a mission")
input("\nPress ENTER to capture...")

# Capture window
window_img = detector.capture_game_window(auto_focus=True)

if window_img is None:
    print("✗ Could not capture window")
    exit()

print(f"\n✓ Captured window: {window_img.shape}")

# Save window capture
cv2.imwrite('debug_window.png', cv2.cvtColor(window_img, cv2.COLOR_RGB2BGR))
print("✓ Saved debug_window.png")

# Try to detect viewport
green = window_img[:, :, 1]
green_threshold = cv2.threshold(green, 80, 255, cv2.THRESH_BINARY)[1]

cv2.imwrite('debug_green_mask.png', green_threshold)
print("✓ Saved debug_green_mask.png")

rows_with_green = np.any(green_threshold > 0, axis=1)
cols_with_green = np.any(green_threshold > 0, axis=0)

print(f"\nRows with green: {np.sum(rows_with_green)}")
print(f"Cols with green: {np.sum(cols_with_green)}")

if not np.any(rows_with_green) or not np.any(cols_with_green):
    print("\n✗ No green detected - viewport detection failed")
    print("This might be:")
    print("  - Enemy turn (UI hidden)")
    print("  - Wrong color threshold")
    print("  - Game paused")
    exit()

# Find viewport bounds
top = np.argmax(rows_with_green)
bottom = len(rows_with_green) - np.argmax(rows_with_green[::-1])
left = np.argmax(cols_with_green)
right = len(cols_with_green) - np.argmax(cols_with_green[::-1])

print(f"\nViewport bounds:")
print(f"  top={top}, bottom={bottom}, left={left}, right={right}")
print(f"  Size: {right-left}x{bottom-top}")

# Extract viewport
viewport = window_img[top:bottom, left:right]
cv2.imwrite('debug_viewport.png', cv2.cvtColor(viewport, cv2.COLOR_RGB2BGR))
print(f"\n✓ Saved debug_viewport.png ({viewport.shape})")

print("\n" + "="*60)
print("CHECK:")
print("="*60)
print("open debug_window.png")
print("open debug_green_mask.png")
print("open debug_viewport.png")
