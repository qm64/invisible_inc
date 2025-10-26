#!/usr/bin/env python3
"""
Intelligently select test frames for structural detector testing
Skips early frames before game window is detected
Only considers opponent turns after seeing valid player turns
"""

import json
import sys
from pathlib import Path

def select_test_frames(turn_phase_json_path, num_per_phase=3):
    """
    Select representative test frames from turn phase analysis
    
    Args:
        turn_phase_json_path: Path to turn_phase_analysis.json
        num_per_phase: Number of frames to select per phase type
    
    Returns:
        List of (frame_number, phase, confidence, description) tuples
    """
    
    with open(turn_phase_json_path, 'r') as f:
        data = json.load(f)
    
    frames = data['frame_results']
    
    # Find first valid player_normal frame (game window detected)
    first_valid_idx = None
    for i, frame_data in enumerate(frames):
        if frame_data['phase'] == 'player_normal' and frame_data['confidence'] >= 0.9:
            first_valid_idx = i
            break
    
    if first_valid_idx is None:
        print("ERROR: No valid player_normal frames found!")
        return []
    
    print(f"First valid game frame: {first_valid_idx}")
    print(f"Skipping frames 0-{first_valid_idx-1} (pre-game window detection)\n")
    
    # Filter to only valid frames (after first player_normal)
    valid_frames = frames[first_valid_idx:]
    
    # Separate by phase with confidence thresholds
    player_normal = [f for f in valid_frames 
                     if f['phase'] == 'player_normal' and f['confidence'] >= 0.85]
    player_action = [f for f in valid_frames 
                     if f['phase'] == 'player_action' and f['confidence'] >= 0.80]
    
    # Only consider opponent frames AFTER we've seen player frames
    # and only if they have reasonable confidence
    opponent = []
    seen_player = False
    for f in valid_frames:
        if f['phase'] in ['player_normal', 'player_action']:
            seen_player = True
        elif f['phase'] == 'opponent' and seen_player and f['confidence'] >= 0.70:
            opponent.append(f)
    
    print(f"Valid frames by phase:")
    print(f"  player_normal: {len(player_normal)}")
    print(f"  player_action: {len(player_action)}")
    print(f"  opponent: {len(opponent)}")
    print()
    
    # Select representative frames
    selected = []
    
    # Select player_normal frames from different parts of session
    if player_normal:
        indices = [
            0,  # Early
            len(player_normal) // 3,  # Early-mid
            len(player_normal) // 2,  # Mid
            len(player_normal) * 2 // 3,  # Late-mid
            len(player_normal) - 1  # End
        ]
        for idx in indices[:num_per_phase]:
            if idx < len(player_normal):
                f = player_normal[idx]
                desc = f"Planning phase (position {idx}/{len(player_normal)})"
                selected.append((f['frame'], 'player_normal', f['confidence'], desc))
    
    # Select player_action frames
    if player_action:
        # Spread across session
        step = max(1, len(player_action) // num_per_phase)
        for i in range(0, min(num_per_phase * step, len(player_action)), step):
            f = player_action[i]
            desc = "Agent action (profile hidden)"
            selected.append((f['frame'], 'player_action', f['confidence'], desc))
    
    # Select opponent frames if any valid ones exist
    if opponent:
        step = max(1, len(opponent) // num_per_phase)
        for i in range(0, min(num_per_phase * step, len(opponent)), step):
            f = opponent[i]
            desc = "Opponent turn (minimal UI)"
            selected.append((f['frame'], 'opponent', f['confidence'], desc))
    
    return sorted(selected, key=lambda x: x[0])  # Sort by frame number

def main():
    if len(sys.argv) < 2:
        print("Usage: python select_test_frames.py <turn_phase_analysis.json> [num_per_phase]")
        print("Example: python select_test_frames.py captures/20251022_201216/turn_phase_analysis.json 3")
        sys.exit(1)
    
    json_path = Path(sys.argv[1])
    num_per_phase = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        sys.exit(1)
    
    print(f"Analyzing: {json_path}")
    print(f"Selecting {num_per_phase} frames per phase type\n")
    print("="*70)
    
    selected = select_test_frames(json_path, num_per_phase)
    
    print("\n" + "="*70)
    print("SELECTED TEST FRAMES")
    print("="*70)
    
    for frame_num, phase, confidence, desc in selected:
        print(f"Frame {frame_num:04d} | {phase:15s} | conf={confidence:.2f} | {desc}")
    
    print("\n" + "="*70)
    print("BASH TEST COMMAND")
    print("="*70)
    
    # Generate bash array for test script
    captures_dir = json_path.parent
    print(f'\nTEST_FRAMES=(')
    for frame_num, phase, confidence, desc in selected:
        print(f'    "frame_{frame_num:06d}.png:{phase}:{desc}"')
    print(')')
    
    print(f"\n# To test these frames:")
    print(f"# ./test_structural_detector.sh {captures_dir} ./structural_detector.py")

if __name__ == "__main__":
    main()