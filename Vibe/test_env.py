# Test game environment

from game_env import GameEnv
import time

print("="*60)
print("GAME ENVIRONMENT TEST")
print("="*60)

# Create environment
env = GameEnv(
    game_region={
        "top": 100,
        "left": 100,
        "width": 800,
        "height": 600
    },
    action_delay=0.2
)

print("\nStarting environment test...")
print("Switch to your text editor NOW!")
print("Test will start in 5 seconds...")
print("You should see keys being pressed in your editor!")
time.sleep(5)

print("\n[1/2] Resetting environment...")
obs, info = env.reset()
print(f"      ✓ Observation shape: {obs.shape}")

print("\n[2/2] Testing random actions for 10 steps...")
print("      Watch your editor - keys will be pressed!")
for step in range(10):
    # Random action
    action = env.action_space.sample()
    
    # Execute action
    obs, reward, done, truncated, info = env.step(action)
    
    action_name = {0:'W', 1:'A', 2:'S', 3:'D', 4:'SPACE', 5:'NONE'}[action]
    print(f"      Step {step+1}: Action={action_name}, Reward={reward:.2f}")
    
    if done:
        print("      Episode done!")
        obs, info = env.reset()

env.close()

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
print("\nDid you see W, A, S, D, and SPACE characters in your editor?")
