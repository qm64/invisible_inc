# Test keyboard and mouse control

import pyautogui
import time

# IMPORTANT: Set fail-safe (move mouse to corner to stop)
pyautogui.FAILSAFE = True

# Get screen size
screen_width, screen_height = pyautogui.size()
print(f"Screen size: {screen_width}x{screen_height}")

# Get current mouse position
x, y = pyautogui.position()
print(f"Mouse position: {x}, {y}")

print("Test starting in 3 seconds... Move mouse to corner to abort!")
time.sleep(3)

# Test keyboard
print("Pressing 'w' key...")
pyautogui.press('w')
time.sleep(0.5)

# Test multiple keys
print("Pressing 'w', 'a', 's', 'd'...")
for key in ['w', 'a', 's', 'd']:
    pyautogui.press(key)
    time.sleep(0.2)

# Test mouse movement
print("Moving mouse...")
pyautogui.moveTo(100, 100, duration=0.5)
pyautogui.click()

print("Test complete!")
