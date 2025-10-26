# Find fullscreen mode coordinates

import pyautogui

print("="*60)
print("FULLSCREEN COORDINATES")
print("="*60)

# Get screen size
screen_width, screen_height = pyautogui.size()

print(f"\nYour screen resolution: {screen_width}x{screen_height}")

# Fullscreen region (entire screen)
region = {
    "top": 0,
    "left": 0,
    "width": screen_width,
    "height": screen_height
}

print("\n" + "="*60)
print("YOUR FULLSCREEN REGION:")
print("="*60)
print(f"\ngame_region = {region}")
print("\n" + "="*60)
print("COPY THIS:")
print("="*60)
print(f"""
env = GameEnv(
    game_region={region},
    action_delay=0.2
)
""")
