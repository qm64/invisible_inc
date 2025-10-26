# Find the UI coordinates of an element

import cv2
import numpy as np

# Global variable to store clicks
clicks = []

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks"""
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        print(f"Click recorded: x={x}, y={y}")
        
        # Draw a circle where clicked
        cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Find UI Elements', img_display)

# Load the screenshot
img = cv2.imread('invisible_inc_full.png')
img_display = img.copy()

print("="*60)
print("FIND UI COORDINATES")
print("="*60)
print("\nInstructions:")
print("1. A window will open showing your screenshot")
print("2. Click on these locations IN ORDER:")
print("   - Top-left corner of power meter text")
print("   - Bottom-right corner of power meter text")
print("   - Top-left corner of credits text")
print("   - Bottom-right corner of credits text")
print("3. Press 'q' when done")
print("="*60)

# Create window and set mouse callback
cv2.namedWindow('Find UI Elements')
cv2.setMouseCallback('Find UI Elements', mouse_callback)

# Show image
cv2.imshow('Find UI Elements', img_display)

print("\nWindow opened! Start clicking...")

# Wait for user to finish
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

print("\n" + "="*60)
print("COORDINATES RECORDED")
print("="*60)

if len(clicks) >= 4:
    print("\nPower meter region:")
    print(f"  Top-left: x={clicks[0][0]}, y={clicks[0][1]}")
    print(f"  Bottom-right: x={clicks[1][0]}, y={clicks[1][1]}")
    print(f"  Region: ({clicks[0][0]}, {clicks[0][1]}, {clicks[1][0]-clicks[0][0]}, {clicks[1][1]-clicks[0][1]})")
    
    print("\nCredits region:")
    print(f"  Top-left: x={clicks[2][0]}, y={clicks[2][1]}")
    print(f"  Bottom-right: x={clicks[3][0]}, y={clicks[3][1]}")
    print(f"  Region: ({clicks[2][0]}, {clicks[2][1]}, {clicks[3][0]-clicks[2][0]}, {clicks[3][1]-clicks[2][1]})")
    
    print("\nCOPY THIS INTO YOUR CODE:")
    print("-"*60)
    print(f"power_region = ({clicks[0][0]}, {clicks[0][1]}, {clicks[1][0]-clicks[0][0]}, {clicks[1][1]-clicks[0][1]})")
    print(f"credits_region = ({clicks[2][0]}, {clicks[2][1]}, {clicks[3][0]-clicks[2][0]}, {clicks[3][1]-clicks[2][1]})")
else:
    print("\nNot enough clicks recorded. You need to click 4 times.")
