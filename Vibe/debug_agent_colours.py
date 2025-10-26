from window_detector import WindowDetector
import cv2
import numpy as np

detector = WindowDetector()

print("="*60)
print("DEBUG AGENT ICON COLORS")
print("="*60)

print("\n1. Launch Invisible Inc")
print("2. Make sure agent icons are visible on left side")
input("\nPress ENTER to analyze...")

# Capture viewport
window_img = detector.capture_game_window(auto_focus=True)

# Extract viewport (same logic as before)
green = window_img[:, :, 1]
green_threshold = cv2.threshold(green, 80, 255, cv2.THRESH_BINARY)[1]
rows_with_green = np.any(green_threshold > 0, axis=1)
cols_with_green = np.any(green_threshold > 0, axis=0)

top = np.argmax(rows_with_green)
bottom = len(rows_with_green) - np.argmax(rows_with_green[::-1])
left = np.argmax(cols_with_green)
right = len(cols_with_green) - np.argmax(cols_with_green[::-1])

margin = 5
top = max(0, top - margin)
left = max(0, left - margin)
bottom = min(window_img.shape[0], bottom + margin)
right = min(window_img.shape[1], right + margin)

viewport = window_img[top:bottom, left:right]
viewport_h, viewport_w = viewport.shape[:2]

# Extract agent panel
panel_y = int(viewport_h * 0.65)
panel_h = int(viewport_h * 0.35)
panel_w = int(viewport_w * 0.15)

agent_panel = viewport[panel_y:panel_y+panel_h, 0:panel_w]

# Save original
cv2.imwrite('debug_panel_original.png', cv2.cvtColor(agent_panel, cv2.COLOR_RGB2BGR))
print("✓ Saved debug_panel_original.png")

# Analyze colors in the panel
print("\nAnalyzing colors in agent panel...")

# Get average color
mean_color = agent_panel.mean(axis=(0,1))
print(f"Average RGB: R={mean_color[0]:.1f}, G={mean_color[1]:.1f}, B={mean_color[2]:.1f}")

# Show color channels separately
red_channel = agent_panel[:, :, 0]
green_channel = agent_panel[:, :, 1]
blue_channel = agent_panel[:, :, 2]

cv2.imwrite('debug_panel_red.png', red_channel)
cv2.imwrite('debug_panel_green.png', green_channel)
cv2.imwrite('debug_panel_blue.png', blue_channel)
print("✓ Saved color channel images")

# Try different thresholds on green channel
for threshold in [50, 80, 100, 120, 150]:
    _, mask = cv2.threshold(green_channel, threshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f'debug_panel_green_thresh_{threshold}.png', mask)
    pixel_count = np.sum(mask > 0)
    percent = (pixel_count / mask.size) * 100
    print(f"  Threshold {threshold}: {percent:.1f}% bright pixels")

# Convert to HSV and show components
hsv = cv2.cvtColor(agent_panel, cv2.COLOR_RGB2HSV)
hue = hsv[:, :, 0]
saturation = hsv[:, :, 1]
value = hsv[:, :, 2]

cv2.imwrite('debug_panel_hue.png', hue)
cv2.imwrite('debug_panel_saturation.png', saturation)
cv2.imwrite('debug_panel_value.png', value)
print("✓ Saved HSV channel images")

print("\n" + "="*60)
print("CHECK THESE IMAGES:")
print("="*60)
print("open debug_panel_original.png")
print("open debug_panel_green.png")
print("open debug_panel_green_thresh_100.png")
print("open debug_panel_hue.png")
print("\nWhich image shows the agent icons most clearly?")