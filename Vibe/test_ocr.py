from invisible_inc_env import InvisibleIncEnv
import time

print("="*60)
print("TEST OCR READING")
print("="*60)

env = InvisibleIncEnv()

print("\n1. Launch Invisible Inc and start a mission")
print("2. Make sure power and credits are visible")
print("3. Switch back and press ENTER")
input("\nPress ENTER to continue...")

print("\nReading UI in 3 seconds...")
time.sleep(3)

# Read power and credits
power = env._read_power()
credits = env._read_credits()

print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(f"Power:   {power}")
print(f"Credits: {credits}")
print("\nDoes this match what you see in the game?")

env.close()