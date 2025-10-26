import mss
import cv2
import numpy as np
import time
import os

print("="*60)
print("INVISIBLE INC - GAME DOCUMENTATION SYSTEM")
print("="*60)

# Create directories for organizing screenshots
os.makedirs("game_docs/ui_elements", exist_ok=True)
os.makedirs("game_docs/agents", exist_ok=True)
os.makedirs("game_docs/items", exist_ok=True)
os.makedirs("game_docs/situations", exist_ok=True)

def capture_screenshot(name, description):
    """Capture and save a labeled screenshot"""
    print(f"\n{description}")
    print("Switch to game and set up the scene...")
    input("Press ENTER when ready to capture...")
    
    print("Capturing in 2 seconds...")
    time.sleep(2)
    
    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[1])
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    
    filepath = f"game_docs/{name}.png"
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"✓ Saved: {filepath}")

print("\nWe'll capture screenshots of key game elements.")
print("This will help us build the AI's understanding.\n")

# UI Elements
print("\n" + "="*60)
print("PART 1: UI ELEMENTS")
print("="*60)

capture_screenshot(
    "ui_elements/full_interface",
    "Show: Full game interface during your turn"
)

capture_screenshot(
    "ui_elements/agent_panel",
    "Show: Agent selection icons (lower left)"
)

capture_screenshot(
    "ui_elements/selected_agent",
    "Show: A selected agent with movement range visible"
)

capture_screenshot(
    "ui_elements/end_turn_button",
    "Show: Where is the 'End Turn' button/indicator?"
)

capture_screenshot(
    "ui_elements/pause_menu",
    "Show: ESC pause menu"
)

# Agent States
print("\n" + "="*60)
print("PART 2: AGENT STATES")
print("="*60)

capture_screenshot(
    "agents/agent_selected",
    "Show: Agent selected (ready to move)"
)

capture_screenshot(
    "agents/agent_moved",
    "Show: Agent after moving (turn used)"
)

capture_screenshot(
    "agents/movement_range",
    "Show: Blue squares showing where agent can move"
)

capture_screenshot(
    "agents/multiple_agents",
    "Show: Multiple agents visible, showing selection UI"
)

# Game States
print("\n" + "="*60)
print("PART 3: GAME STATES")
print("="*60)

capture_screenshot(
    "situations/player_turn",
    "Show: During your turn (normal play)"
)

capture_screenshot(
    "situations/enemy_turn",
    "Show: During enemy turn (if possible)"
)

capture_screenshot(
    "situations/alarm_level",
    "Show: Alarm tracker/security level indicator"
)

capture_screenshot(
    "situations/mission_objective",
    "Show: Mission objective display"
)

print("\n" + "="*60)
print("DOCUMENTATION COMPLETE")
print("="*60)
print("\nScreenshots saved in game_docs/ directory")
print("\nNext step: Review screenshots and identify:")
print("  - Clickable regions (agent icons, buttons)")
print("  - Detection regions (alarm level, turn state)")
print("  - Hotkey shortcuts")
