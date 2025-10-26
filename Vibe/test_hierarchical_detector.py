"""
Test script for Hierarchical Game State Detector
"""

from hierarchical_detector import HierarchicalGameStateDetector, DisplayMode
from window_detector import WindowDetector
import sys

def print_state(state):
    """Pretty print the game state"""
    if state is None:
        print("✗ Failed to detect game state")
        return
    
    print("\n" + "="*60)
    print("DETECTED GAME STATE")
    print("="*60)
    
    print(f"\n📊 Viewport: {state.viewport_size}")
    print(f"🎮 Mode: {state.mode.value}")
    print(f"👁️  Lower left visible: {state.lower_left_visible}")
    
    print("\n--- Upper Left ---")
    print(f"⚡ Power: {state.power}/{state.power_max}")
    print(f"💰 Credits: {state.credits}")
    
    print("\n--- Upper Right ---")
    print(f"🚨 Alarm Level: {state.alarm_level}")
    print(f"🔄 Turn: {state.turn}")
    print(f"📅 Day: {state.day}")
    print(f"🎯 Mission Type: {state.mission_type}")
    
    if state.lower_left_visible:
        print("\n--- Lower Left ---")
        print(f"👤 Agent: {state.agent_name}")
        print(f"⚡ AP: {state.action_points}")
    
    print("\n" + "="*60)

def main():
    print("="*60)
    print("HIERARCHICAL GAME STATE DETECTOR - TEST")
    print("="*60)
    
    # Initialize detector
    print("\nInitializing detector...")
    detector = HierarchicalGameStateDetector(WindowDetector())
    
    # Load templates
    template_dir = sys.argv[1] if len(sys.argv) > 1 else './templates'
    print(f"Loading templates from: {template_dir}")
    detector.load_templates(template_dir)
    
    # Test detection
    print("\nMake sure the game is running...")
    input("Press ENTER when ready to capture (auto-focus will activate)...")
    
    print("\nCapturing and detecting (auto-focusing game window)...")
    state = detector.get_game_state()
    
    # Print results
    print_state(state)
    
    # Show debug images
    if detector.debug and detector.debug_images:
        print(f"\n📁 Debug images saved:")
        for img in detector.debug_images:
            print(f"   {img}")
        print("\nTo view:")
        print(f"   open {' '.join(detector.debug_images)}")

if __name__ == "__main__":
    main()