"""
Merge multiple color signature files into one comprehensive database
Organized by game state and UI region
"""

import json
from pathlib import Path
from datetime import datetime

def merge_signatures():
    """Merge all color signature files into one comprehensive database"""
    
    # Load all signature files
    files = [
        'color_signatures_normal.json',
        'color_signatures_mainframe.json',
        'color_signatures_drone.json',
        'color_signatures_agent_moving.json'
    ]
    
    all_signatures = {}
    source_info = {}
    
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                mode = file.replace('color_signatures_', '').replace('.json', '')
                
                for name, sig in data['signatures'].items():
                    # Store signature with metadata
                    all_signatures[name] = sig
                    source_info[name] = {
                        'mode': mode,
                        'source_file': file,
                        'timestamp': data['timestamp'],
                        'source_image': data['source_image']
                    }
        except FileNotFoundError:
            print(f"Warning: {file} not found, skipping...")
    
    # Organize signatures by category
    organized = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'source_files': files,
            'total_signatures': len(all_signatures)
        },
        
        # Always present elements (all game states)
        'always_present': {
            'incognita_upper': all_signatures.get('incognita_upper'),
            'power': all_signatures.get('power'),
            'credits': all_signatures.get('credits'),
            'top_center': all_signatures.get('top_center'),
            'tactical_view': all_signatures.get('tactical_view'),
            'menu_hamburger': all_signatures.get('menu_hamburger'),
            'security_level': all_signatures.get('security_level'),
            'security_clock': all_signatures.get('security_clock'),
            'alarm_level': all_signatures.get('alarm_level'),
        },
        
        # Right side elements (usually present)
        'right_side': {
            'top_right': all_signatures.get('top_right'),
            'top_right_info': all_signatures.get('top_right_info'),
            'daemon_list': all_signatures.get('daemon_list'),
            'lower_right': all_signatures.get('lower_right'),
            'end_turn': all_signatures.get('end_turn'),
            'objectives': all_signatures.get('objectives'),
        },
        
        # Agent mode - full UI
        'agent_mode_full': {
            'upper_left_normal': all_signatures.get('upper_left_normal'),
            'lower_left_normal': all_signatures.get('lower_left_normal'),
            'agent_profile': all_signatures.get('agent_profile'),
            'agent_icons': all_signatures.get('agent_icons'),
            'actions': all_signatures.get('actions'),
            'augments': all_signatures.get('augments'),
            'inventory': all_signatures.get('inventory'),
        },
        
        # Agent mode - moving (partial UI)
        'agent_mode_moving': {
            'lower_left_agent_moving': all_signatures.get('lower_left_agent_moving'),
            'upper_right_agent_moving': all_signatures.get('upper_right_agent_moving'),
        },
        
        # Drone mode
        'drone_mode': {
            'lower_left_drone': all_signatures.get('lower_left_drone'),
        },
        
        # Mainframe mode
        'mainframe_mode': {
            'incognita_and_programs_region': all_signatures.get('incognita_and_programs_region'),
            'incognita_programs': all_signatures.get('incognita_programs'),
            'lower_left_mainframe': all_signatures.get('lower_left_mainframe'),
            'incognita_profile': all_signatures.get('incognita_profile'),
        },
        
        # Source information for debugging
        '_source_info': source_info
    }
    
    # Remove None entries
    def clean_none(d):
        if isinstance(d, dict):
            return {k: clean_none(v) for k, v in d.items() if v is not None}
        return d
    
    organized = clean_none(organized)
    
    # Save merged file
    output_file = 'color_signatures_merged.json'
    with open(output_file, 'w') as f:
        json.dump(organized, f, indent=2)
    
    print(f"\n✓ Merged {len(all_signatures)} signatures into {output_file}")
    print("\nSignatures by category:")
    for category, sigs in organized.items():
        if category.startswith('_') or category == 'metadata':
            continue
        if isinstance(sigs, dict):
            print(f"  {category}: {len(sigs)} signatures")
    
    # Also create a flattened Python version for easy import
    py_output = 'color_signatures_merged.py'
    with open(py_output, 'w') as f:
        f.write('"""Merged color signatures from all game modes"""\n')
        f.write(f'# Generated: {datetime.now().isoformat()}\n\n')
        f.write('COLOR_SIGNATURES = ')
        f.write(json.dumps(organized, indent=2))
    
    print(f"✓ Created Python version: {py_output}")
    
    # Create a quick reference guide
    guide_file = 'color_signatures_guide.txt'
    with open(guide_file, 'w') as f:
        f.write("COLOR SIGNATURES QUICK REFERENCE\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("VIEWPORT DETECTION (use these anchors):\n")
        f.write("  - tactical_view (top center)\n")
        f.write("  - menu_hamburger (top right)\n")
        f.write("  - security_clock (right side)\n")
        f.write("  - end_turn (bottom right)\n")
        f.write("  - incognita_upper (top left - always present)\n\n")
        
        f.write("MODE DETECTION:\n")
        f.write("  Agent Mode Full: Look for 'agent_profile' or 'agent_icons'\n")
        f.write("  Agent Mode Moving: Look for 'lower_left_agent_moving'\n")
        f.write("  Drone Mode: Look for 'lower_left_drone'\n")
        f.write("  Mainframe Mode: Look for 'incognita_programs'\n")
        f.write("  Blank Lower Left: None of the above found\n\n")
        
        f.write("DETECTION PRIORITY:\n")
        f.write("  1. Detect viewport using always_present elements\n")
        f.write("  2. Check for mainframe mode (incognita_programs)\n")
        f.write("  3. Check for agent mode elements\n")
        f.write("  4. If lower left empty, proceed with top/right only\n\n")
        
        f.write("VALUE READING LOCATIONS:\n")
        f.write("  Power: Above incognita_upper\n")
        f.write("  Credits: Above incognita_upper (below power)\n")
        f.write("  Security Level: Inside security_clock region\n")
        f.write("  Agent AP: Next to each agent icon\n")
        f.write("  Selected Agent: Read profile name text\n\n")
    
    print(f"✓ Created reference guide: {guide_file}")
    
    return organized

if __name__ == '__main__':
    merged = merge_signatures()
    print("\n" + "=" * 60)
    print("Merge complete! Files created:")
    print("  - color_signatures_merged.json")
    print("  - color_signatures_merged.py")
    print("  - color_signatures_guide.txt")
    print("=" * 60)
    