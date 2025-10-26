import cv2
import numpy as np

# Load screenshot
img = cv2.imread('agent_mode.png')

if img is None:
    print("ERROR: Could not load agent_mode.png")
    exit()

print(f"Full image size: {img.shape[1]}x{img.shape[0]}")

# Detect viewport by finding green UI elements (same method as before)
green = img[:, :, 1]
green_threshold = cv2.threshold(green, 80, 255, cv2.THRESH_BINARY)[1]

rows_with_green = np.any(green_threshold > 0, axis=1)
cols_with_green = np.any(green_threshold > 0, axis=0)

if not np.any(rows_with_green) or not np.any(cols_with_green):
    print("ERROR: Could not find viewport")
    exit()

# Find viewport bounds
top = np.argmax(rows_with_green)
bottom = len(rows_with_green) - np.argmax(rows_with_green[::-1])
left = np.argmax(cols_with_green)
right = len(cols_with_green) - np.argmax(cols_with_green[::-1])

# Add small margin
margin = 5
top = max(0, top - margin)
left = max(0, left - margin)
bottom = min(img.shape[0], bottom + margin)
right = min(img.shape[1], right + margin)

viewport_width = right - left
viewport_height = bottom - top

print(f"\nViewport detected:")
print(f"  Position: ({left}, {top})")
print(f"  Size: {viewport_width}x{viewport_height}")
print(f"  Black borders: top={top}px, left={left}px, right={img.shape[1]-right}px, bottom={img.shape[0]-bottom}px")

# Draw viewport on image
img_annotated = img.copy()
cv2.rectangle(img_annotated, (left, top), (right, bottom), (255, 0, 255), 8)
cv2.imwrite('viewport_detected_bounds.png', img_annotated)
print("\n✓ Saved viewport_detected_bounds.png")

# Now convert all measurements to viewport-relative percentages
print("\n" + "="*60)
print("CONVERTING MEASUREMENTS TO VIEWPORT PERCENTAGES:")
print("="*60)

agent_mode_regions = {
    'Power': (0, 0, 261, 92),
    'Credits': (261, 0, 174, 92),
    'Incognita_button': (0, 92, 423, 171),
    'Tactical_button': (2330, 0, 438, 180),
    'Menu_button': (4970, 0, 150, 92),
    'Info_bar': (2768, 0, 2202, 92),
    'Alarm_integer': (4883, 247, 101, 121),
    'Alarm_ring': (4770, 111, 323, 338),
    'Daemons': (4777, 452, 334, 1548),
    'Agent_profile': (0, 2378, 424, 488),
    'Agent_icons': (0, 1890, 231, 485),
    'Quick_actions': (425, 2688, 339, 141),
    'Augments_Inventory': (425, 2533, 1325, 161),
    'End_turn': (4585, 2734, 535, 143),
    'Rewind': (4378, 2735, 222, 142),
}

incognita_mode_regions = {
    'Programs': (0, 245, 900, 1298),
    'Incognita_profile': (0, 2378, 424, 488),
}

def to_percentages(x, y, w, h, viewport_left, viewport_top, viewport_w, viewport_h):
    """Convert pixel coords to viewport percentages"""
    # Adjust for viewport offset
    x_rel = x - viewport_left
    y_rel = y - viewport_top
    
    # Convert to percentages
    x_pct = x_rel / viewport_w
    y_pct = y_rel / viewport_h
    w_pct = w / viewport_w
    h_pct = h / viewport_h
    
    return (x_pct, y_pct, w_pct, h_pct)

print("\nAgent Mode Regions (as percentages):")
print("layout = {")
for name, (x, y, w, h) in agent_mode_regions.items():
    x_pct, y_pct, w_pct, h_pct = to_percentages(x, y, w, h, left, top, viewport_width, viewport_height)
    print(f"    '{name}': {{'x': {x_pct:.4f}, 'y': {y_pct:.4f}, 'width': {w_pct:.4f}, 'height': {h_pct:.4f}}},")
print("}")

print("\nIncognita Mode Additional Regions (as percentages):")
print("incognita_layout = {")
for name, (x, y, w, h) in incognita_mode_regions.items():
    x_pct, y_pct, w_pct, h_pct = to_percentages(x, y, w, h, left, top, viewport_width, viewport_height)
    print(f"    '{name}': {{'x': {x_pct:.4f}, 'y': {y_pct:.4f}, 'width': {w_pct:.4f}, 'height': {h_pct:.4f}}},")
print("}")

print("\n" + "="*60)
print("CHECK: open viewport_detected_bounds.png")
print("Does the magenta rectangle show the viewport correctly?")
print("="*60)
