from window_detector import WindowDetector
import cv2
import numpy as np

detector = WindowDetector()

print("="*60)
print("DETECT AGENT ICONS")
print("="*60)

print("\n1. Launch Invisible Inc in WINDOWED mode")
print("2. Start a mission with multiple agents visible")
input("\nPress ENTER to detect agents...")

# Capture and extract viewport
window_img = detector.capture_game_window(auto_focus=True)
if window_img is None:
    print("ERROR: Could not capture window")
    exit()

# Extract viewport (reusing logic from turn_monitor_viewport.py)
green = window_img[:, :, 1]
green_threshold = cv2.threshold(green, 80, 255, cv2.THRESH_BINARY)[1]
rows_with_green = np.any(green_threshold > 0, axis=1)
cols_with_green = np.any(green_threshold > 0, axis=0)

if not np.any(rows_with_green) or not np.any(cols_with_green):
    print("ERROR: Could not find viewport")
    exit()

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

print(f"\nViewport: {viewport_w}x{viewport_h}")

# Agent panel is in lower-left
# Based on your analysis: icons at x=~45-50, y=~1014-1330 in 2560x1440
# That's roughly x=2%, y=70-92% of viewport
panel_region = (0, 0.65, 0.15, 0.35)  # Left side, bottom 35%

x_pct, y_pct, w_pct, h_pct = panel_region
panel_x = int(viewport_w * x_pct)
panel_y = int(viewport_h * y_pct)
panel_w = int(viewport_w * w_pct)
panel_h = int(viewport_h * h_pct)

agent_panel = viewport[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w]

print(f"Agent panel region: {panel_w}x{panel_h}")

# Save for inspection
cv2.imwrite('agent_panel.png', cv2.cvtColor(agent_panel, cv2.COLOR_RGB2BGR))
print("✓ Saved agent_panel.png")

# Detect agent icons by looking for green circular regions
# Agent icons have distinct green borders

# Convert to HSV for better green detection
hsv = cv2.cvtColor(agent_panel, cv2.COLOR_RGB2HSV)

# Green color range in HSV
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# Create mask for green
mask = cv2.inRange(hsv, lower_green, upper_green)

cv2.imwrite('agent_panel_green_mask.png', mask)
print("✓ Saved agent_panel_green_mask.png")

# Find contours (potential agent icons)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by size (agent icons are roughly circular, certain size)
min_area = 100  # Adjust based on window size
max_area = 5000

agent_icons = []
for contour in contours:
    area = cv2.contourArea(contour)
    if min_area < area < max_area:
        x, y, w, h = cv2.boundingRect(contour)
        # Check if roughly circular (width ≈ height)
        aspect_ratio = w / float(h)
        if 0.7 < aspect_ratio < 1.3:  # Roughly square/circular
            agent_icons.append((x, y, w, h, area))

# Sort by Y coordinate (top to bottom)
agent_icons.sort(key=lambda icon: icon[1])

print(f"\nFound {len(agent_icons)} agent icons:")
for i, (x, y, w, h, area) in enumerate(agent_icons):
    print(f"  Agent {i+1}: pos=({x},{y}) size={w}x{h} area={area}")

# Draw rectangles around detected icons
agent_panel_annotated = agent_panel.copy()
for i, (x, y, w, h, _) in enumerate(agent_icons):
    cv2.rectangle(agent_panel_annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(agent_panel_annotated, str(i+1), (x+5, y+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite('agent_panel_detected.png', cv2.cvtColor(agent_panel_annotated, cv2.COLOR_RGB2BGR))
print("\n✓ Saved agent_panel_detected.png")

print("\n" + "="*60)
print("CHECK:")
print("="*60)
print("open agent_panel.png")
print("open agent_panel_green_mask.png")
print("open agent_panel_detected.png")
print("\nDo the green rectangles match the agent icons?")
