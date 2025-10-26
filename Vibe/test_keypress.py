# Test keypress

import pyautogui
import time

print("="*60)
print("DIRECT KEY PRESS TEST")
print("="*60)
print("\nSwitch to a text editor NOW!")
print("Test will start in 5 seconds...")
print("You should see 'wwwww' typed in your editor")
print("="*60)

time.sleep(5)

print("\nPressing 'w' key 5 times...")
for i in range(5):
    pyautogui.press('w')
    print(f"  Pressed {i+1}")
    time.sleep(0.5)

print("\n✓ Test complete")
print("Did you see 'wwwww' appear in your text editor?")

