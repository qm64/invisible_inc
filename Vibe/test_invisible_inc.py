# Test Invisible Inc game environment

from invisible_inc_env import InvisibleIncEnv
import time

print("="*60)
print("INVISIBLE INC ENVIRONMENT TEST")
print("="*60)

env = InvisibleIncEnv()

print("\n1. Launch Invisible Inc and start a mission")
print("2. Make sure the game is in fullscreen mode")
print("3. Switch back to terminal and press ENTER when ready")
input("\nPress ENTER to continue...")

print("\nStarting test in 3 seconds...")
print("The AI will take random actions!")
time.sleep(3)

print("\n[Starting episode]")
obs, info = env.reset()
print(f"✓ Observation captured: {obs.shape}")

print("\n[Taking 20 random actions - watch the game!]")
for step in range(20):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    
    action_name = {0:'W', 1:'A', 2:'S', 3:'D', 4:'SPACE', 5:'NONE'}[action]
    print(f"  Step {step+1}: {action_name:6s} | Reward: {reward:+.2f} | Done: {done}")
    
    if done or truncated:
        print("  Episode ended!")
        break

env.close()

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
print("\nDid you see the game respond to W/A/S/D/Space keys?")

