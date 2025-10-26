from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from invisible_inc_env import InvisibleIncEnv
import os
import signal
import sys
import time

print("="*60)
print("TRAIN INVISIBLE INC AI AGENT")
print("="*60)

# Global variables
model = None
env = None
steps_done = 0

def signal_handler(sig, frame):
    """Handle Ctrl+\ gracefully"""
    global model, env, steps_done
    print("\n\n[STOPPING] Received stop signal...")
    if model:
        model.save("invisible_inc_agent_partial")
        print(f"✓ Model saved after {steps_done} steps")
    if env:
        env.close()
    print("\n" + "="*60)
    print("TRAINING STOPPED")
    print("="*60)
    sys.exit(0)

# Set up signal handler for Ctrl+\
signal.signal(signal.SIGQUIT, signal_handler)

# Create environment
def make_env():
    return InvisibleIncEnv()

env = DummyVecEnv([make_env])

print("\nCreating PPO agent...")
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    tensorboard_log="./tensorboard_logs/"
)

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
print("\nInstructions:")
print("1. Launch Invisible Inc and start a mission")
print("2. Keep the game in fullscreen")
print("3. The AI will start playing in 10 seconds")
print("\nTO STOP TRAINING:")
print("  Switch to terminal and press: Ctrl+\\ (backslash)")
print("  Model will be saved immediately")
print("\nTraining for 10,000 steps...")
print("="*60)

time.sleep(10)

# Train
try:
    for i in range(5):  # 5 batches of 2048 steps = 10,240 steps
        print(f"\n[Batch {i+1}/5] Training...")
        model.learn(total_timesteps=2048, reset_num_timesteps=False)
        steps_done += 2048
        print(f"✓ Completed {steps_done}/10240 steps")
    
    # Save final model
    model.save("invisible_inc_agent")
    print(f"\n✓ Final model saved after {steps_done} steps")
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    model.save("invisible_inc_agent_error")
    print("✓ Model saved due to error")

env.close()

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)