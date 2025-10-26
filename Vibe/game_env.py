# Game environment

import gymnasium as gym
import numpy as np
import mss
import pyautogui
import cv2
from gymnasium import spaces
import time

class GameEnv(gym.Env):
    """
    Custom Gym environment for any game using screen capture
    """
    
    def __init__(self, game_region=None, action_delay=0.1):
        super(GameEnv, self).__init__()
        
        # Screen capture
        self.sct = mss.mss()
        
        # Define game window region (top, left, width, height)
        if game_region is None:
            # Default: top-left 800x600 area
            self.game_region = {
                "top": 100,
                "left": 100, 
                "width": 800,
                "height": 600
            }
        else:
            self.game_region = game_region
        
        # Delay between actions (give game time to respond)
        self.action_delay = action_delay
        
        # Define action space (0: W, 1: A, 2: S, 3: D, 4: Space, 5: No-op)
        self.action_space = spaces.Discrete(6)
        
        # Define observation space (84x84 RGB image)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(84, 84, 3),
            dtype=np.uint8
        )
        
        # Action mapping
        self.action_map = {
            0: 'w',      # Up/Forward
            1: 'a',      # Left
            2: 's',      # Down/Back
            3: 'd',      # Right
            4: 'space',  # Jump/Action
            5: None      # No action
        }
        
        # Enable fail-safe
        pyautogui.FAILSAFE = True
        
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Wait for game to be ready
        time.sleep(1)
        
        # Capture initial state
        obs = self._get_observation()
        
        return obs, {}
    
    def step(self, action):
        """Execute one action"""
        # Perform action
        key = self.action_map[action]
        if key is not None:
            pyautogui.press(key)
        
        # Wait for game to update
        time.sleep(self.action_delay)
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward (you'll customize this per game)
        reward = self._calculate_reward(obs)
        
        # Check if episode is done (you'll customize this)
        done = self._check_done(obs)
        truncated = False
        
        return obs, reward, done, truncated, {}
    
    def _get_observation(self):
        """Capture and process screen"""
        # Capture screen region
        screenshot = self.sct.grab(self.game_region)
        
        # Convert to numpy array
        img = np.array(screenshot)
        
        # Convert BGRA to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        
        # Resize to standard size (84x84 is common for RL)
        img = cv2.resize(img, (84, 84))
        
        return img
    
    def _calculate_reward(self, obs):
        """
        Calculate reward from observation
        YOU NEED TO CUSTOMIZE THIS FOR YOUR GAME
        """
        # Example: reward based on brightness (placeholder)
        # You'd typically use OCR, template matching, or color detection
        brightness = np.mean(obs)
        reward = 0.0
        
        # Add your game-specific reward logic here
        # Examples:
        # - OCR to read score
        # - Template matching to detect win/loss screens
        # - Color detection for health bars
        # - Pixel comparison with goal states
        
        return reward
    
    def _check_done(self, obs):
        """
        Check if episode is done
        YOU NEED TO CUSTOMIZE THIS FOR YOUR GAME
        """
        # Example: detect game over screen by color
        # You'd typically use template matching or OCR
        
        done = False
        
        # Add your game-specific done logic here
        # Examples:
        # - Detect "Game Over" text
        # - Detect death screen
        # - Timeout after N steps
        
        return done
    
    def render(self, mode='human'):
        """Display current observation"""
        obs = self._get_observation()
        cv2.imshow('Game View', obs)
        cv2.waitKey(1)
    
    def close(self):
        """Cleanup"""
        cv2.destroyAllWindows()
