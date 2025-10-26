from game_env import GameEnv
import mss
import cv2
import numpy as np
import pytesseract

class InvisibleIncEnv(GameEnv):
    """
    Customized environment for Invisible Inc
    """
    
    def __init__(self):
        # Verify screen
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            print(f"Detected monitor: {monitor}")
        
        game_region = {
            "top": 0,
            "left": 0,
            "width": 2560,
            "height": 1440
        }
        
        print(f"Using game_region: {game_region}")
        
        super().__init__(
            game_region=game_region,
            action_delay=0.3
        )
        
        # UI regions from find_ui_coordinates.py
        # FORMAT: (x, y, width, height)
        self.power_region = (0, 0, 150, 50)       # Full power text "XX/XX PWR"
        self.credits_region = (160, 0, 200, 50)   # Full credits "XXXXXX CR"
        
        self.steps_taken = 0
        self.max_steps_per_episode = 1000
        self.previous_power = 0
    
    def reset(self, seed=None, options=None):
        """Reset for new episode"""
        self.steps_taken = 0
        self.previous_power = 0
        return super().reset(seed=seed, options=options)
    
    def step(self, action):
        """Execute action and calculate rewards"""
        obs, _, done, truncated, info = super().step(action)
        
        self.steps_taken += 1
        
        # Calculate Invisible Inc specific rewards
        reward = self._calculate_invisible_inc_reward(obs)
        
        # Check if episode should end
        done = self._check_invisible_inc_done(obs)
        
        # Timeout after max steps
        if self.steps_taken >= self.max_steps_per_episode:
            truncated = True
        
        return obs, reward, done, truncated, info
    
    def _read_power(self):
        """Read power value from UI using OCR"""
        # Capture full resolution screenshot
        with mss.mss() as sct:
            full_screenshot = sct.grab(self.game_region)
            img = np.array(full_screenshot)
        
        # Extract power region
        x, y, w, h = self.power_region
        power_img = img[y:y+h, x:x+w]
        
        # Convert BGRA to RGB
        rgb = cv2.cvtColor(power_img, cv2.COLOR_BGRA2RGB)
        
        # Extract GREEN channel (text is green)
        green_channel = rgb[:, :, 1]
        
        # Threshold to get only bright green text
        _, thresh = cv2.threshold(green_channel, 100, 255, cv2.THRESH_BINARY)
        
        # Scale up 3x for better OCR
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # OCR
        text = pytesseract.image_to_string(thresh, config='--psm 7')
        
        # Parse "5/20 PWR" format
        try:
            # Extract just the numbers before the slash
            parts = text.split('/')
            if len(parts) >= 1:
                power = int(''.join(filter(str.isdigit, parts[0])))
                return power
        except:
            pass
        
        return 0  # Return 0 if we can't read it
    
    def _read_credits(self):
        """Read credits value from UI using OCR"""
        # Capture full resolution screenshot
        with mss.mss() as sct:
            full_screenshot = sct.grab(self.game_region)
            img = np.array(full_screenshot)
        
        # Extract credits region
        x, y, w, h = self.credits_region
        credits_img = img[y:y+h, x:x+w]
        
        # Convert BGRA to RGB
        rgb = cv2.cvtColor(credits_img, cv2.COLOR_BGRA2RGB)
        
        # Extract GREEN channel (text is green)
        green_channel = rgb[:, :, 1]
        
        # Threshold to get only bright green text
        _, thresh = cv2.threshold(green_channel, 100, 255, cv2.THRESH_BINARY)
        
        # Scale up 3x for better OCR
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # OCR - white text on black background
        text = pytesseract.image_to_string(
            thresh, 
            config='--psm 7 digits'
        )
        
        # Parse - extract all digits
        try:
            digits = ''.join(filter(str.isdigit, text))
            if digits:
                credits = int(digits)
                return credits
        except:
            pass
        
        return 0
    
    def _calculate_invisible_inc_reward(self, obs):
        """
        Reward function for Invisible Inc
        """
        reward = -0.01  # Small penalty per step
        
        # Read current power
        current_power = self._read_power()
        
        # Reward for gaining power
        if current_power > self.previous_power:
            power_gain = current_power - self.previous_power
            reward += power_gain * 0.5  # 0.5 reward per power gained
            print(f"  [POWER GAIN] +{power_gain} power! Reward: +{power_gain * 0.5:.2f}")
        
        self.previous_power = current_power
        
        return reward
    
    def _check_invisible_inc_done(self, obs):
        """Check if episode is over"""
        # TODO: Detect death/game over screen
        done = False
        return done