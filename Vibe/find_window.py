# Find the game window coordinates

import pyautogui
import time

print("="*60)
print("FIND GAME WINDOW COORDINATES")
print("="*60)

print("\nThis will help you find the coordinates of your game window.")
print("\nStep 1: Move your mouse to the TOP-LEFT corner of your game window...")
print("(You have 5 seconds)")
time.sleep(5)

top_left = pyautogui.position()
print(f"\n✓ Top-left recorded: x={top_left.x}, y={top_left.y}")

print("\nStep 2: Now move your mouse to the BOTTOM-RIGHT corner of your game window...")
print("(You have 5 seconds)")
time.sleep(5)

bottom_right = pyautogui.position()
print(f"\n✓ Bottom-right recorded: x={bottom_right.x}, y={bottom_right.y}")

# Calculate region
width = bottom_right.x - top_left.x
height = bottom_right.y - top_left.y

region = {
    "top": top_left.y,
    "left": top_left.x,
    "width": width,
    "height": height
}

print("\n" + "="*60)
print("YOUR GAME REGION:")
print("="*60)
print(f"\ngame_region = {region}")
print(f"\nSize: {width}x{height} pixels")
print("\n" + "="*60)
print("COPY THIS:")
print("="*60)
print(f"""
env = GameEnv(
    game_region={region},
    action_delay=0.2
)
""")
print("\nPaste this into your code to capture your game window!")
