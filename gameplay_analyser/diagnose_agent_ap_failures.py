#!/usr/bin/env python3
"""
Agent AP Extraction Failure Diagnostics
Categorizes different failure modes and saves visual examples
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict
import subprocess
import re

def run_agent_extractor(frame_path):
    """Run agent_ap_extractor.py and parse output"""
    try:
        result = subprocess.run(
            ['python', 'agent_ap_extractor.py', frame_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout
    except Exception as e:
        return f"Error running extractor: {e}"

def parse_extractor_output(output):
    """Parse the extractor output into structured data"""
    data = {
        'profile_detected': False,
        'agents': [],
        'raw_output': output
    }
    
    # Check for profile detection
    if 'Profile detected: Yes' in output:
        data['profile_detected'] = True
    
    # Extract agent data
    agent_pattern = r'Agent (\d+):\s+(?:(\d+) AP|None AP)\s+\(text: \'([^\']*)\'\)'
    for match in re.finditer(agent_pattern, output):
        agent_idx = int(match.group(1))
        ap_value = int(match.group(2)) if match.group(2) else None
        ocr_text = match.group(3)
        
        data['agents'].append({
            'index': agent_idx,
            'ap': ap_value,
            'ocr_text': ocr_text
        })
    
    return data

def extract_roi_for_agent(frame_path, agent_idx):
    """Extract the ROI image for a specific agent"""
    img = cv2.imread(frame_path)
    if img is None:
        return None
    
    height, width = img.shape[:2]
    
    # Calculate profile box (same logic as extractor)
    profile_x = int(width * 0.0078125)  # 20/2560
    profile_y = int(height * 0.826389)  # 1190/1440
    profile_w = int(width * 0.076172)   # 195/2560
    profile_h = int(height * 0.159722)  # 230/1440
    
    # Calculate AP box dimensions
    ap_width = int(profile_w * 0.333)   # 65/195
    ap_height = int(profile_h * 0.109)  # 25/230
    ap_x = profile_x + int(profile_w * 0.220)  # 43/195 offset
    
    # Starting Y and spacing
    start_y = profile_y - int(profile_h * 0.304)  # 70/230 above profile
    spacing = int(profile_h * 0.161)  # 37/230 between agents
    
    # Calculate this agent's position
    agent_y = start_y + (agent_idx * spacing)
    
    # Extract ROI
    roi = img[agent_y:agent_y+ap_height, ap_x:ap_x+ap_width]
    
    return roi

def categorize_failure(data):
    """Categorize the type of failure"""
    if not data['profile_detected']:
        return 'no_profile'
    
    if len(data['agents']) == 0:
        return 'profile_no_agents'
    
    # Check for partial failures
    has_success = any(agent['ap'] is not None for agent in data['agents'])
    has_failure = any(agent['ap'] is None for agent in data['agents'])
    
    if has_success and has_failure:
        return 'partial_extraction'
    
    if not has_success:
        # All agents failed - check OCR patterns
        ap_only_count = sum(1 for agent in data['agents'] if agent['ocr_text'] == 'AP')
        if ap_only_count > 0:
            return 'ap_text_only'  # Sees "AP" but no number
        return 'complete_ocr_failure'
    
    return 'success'

def save_failure_examples(session_dir, diagnostic_dir):
    """Analyze all frames and save examples of each failure type"""
    frames_dir = os.path.join(session_dir, 'frames')
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.png')])
    
    # Statistics
    stats = defaultdict(int)
    failure_examples = defaultdict(list)  # failure_type -> list of (frame_num, agent_idx, ocr_text)
    
    print(f"Analyzing {len(frame_files)} frames...")
    print()
    
    for i, frame_file in enumerate(frame_files):
        frame_num = int(frame_file.replace('frame_', '').replace('.png', ''))
        frame_path = os.path.join(frames_dir, frame_file)
        
        # Run extractor
        output = run_agent_extractor(frame_path)
        data = parse_extractor_output(output)
        
        # Categorize
        category = categorize_failure(data)
        stats[category] += 1
        
        # Save examples of failures (up to 10 of each type)
        if category != 'success' and len(failure_examples[category]) < 10:
            # Save frame-level info
            failure_examples[category].append({
                'frame_num': frame_num,
                'frame_file': frame_file,
                'data': data
            })
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(frame_files)}...")
    
    print()
    
    # Create diagnostic output directories
    for category in failure_examples.keys():
        category_dir = os.path.join(diagnostic_dir, category)
        os.makedirs(category_dir, exist_ok=True)
    
    # Save ROI images for each failure
    print("Saving failure examples...")
    for category, examples in failure_examples.items():
        category_dir = os.path.join(diagnostic_dir, category)
        
        for example in examples:
            frame_num = example['frame_num']
            frame_file = example['frame_file']
            frame_path = os.path.join(frames_dir, frame_file)
            data = example['data']
            
            # For each failed agent in this frame
            for agent in data['agents']:
                if agent['ap'] is None:  # Failed extraction
                    roi = extract_roi_for_agent(frame_path, agent['index'])
                    if roi is not None and roi.size > 0:
                        # Save ROI
                        roi_filename = f"frame_{frame_num:06d}_agent{agent['index']}_ocr-{agent['ocr_text']}.png"
                        roi_path = os.path.join(category_dir, roi_filename)
                        cv2.imwrite(roi_path, roi)
    
    return stats, failure_examples

def print_report(stats, failure_examples, session_dir):
    """Print and save diagnostic report"""
    print()
    print("=" * 70)
    print("AGENT AP EXTRACTION FAILURE ANALYSIS")
    print("=" * 70)
    print()
    
    total = sum(stats.values())
    
    print(f"Total frames analyzed: {total}")
    print()
    print("Breakdown:")
    print("-" * 70)
    
    categories = [
        ('success', 'Successful extraction'),
        ('no_profile', 'No profile detected'),
        ('profile_no_agents', 'Profile detected, zero agents found'),
        ('partial_extraction', 'Partial extraction (some agents failed)'),
        ('ap_text_only', 'OCR sees "AP" but no number'),
        ('complete_ocr_failure', 'Complete OCR failure (all agents)')
    ]
    
    for cat_key, cat_name in categories:
        count = stats[cat_key]
        pct = (count / total * 100) if total > 0 else 0
        print(f"{cat_name:45} {count:4d} ({pct:5.1f}%)")
    
    print()
    print("=" * 70)
    print("FAILURE EXAMPLES SAVED")
    print("=" * 70)
    print()
    
    for cat_key, cat_name in categories:
        if cat_key == 'success':
            continue
        
        examples = failure_examples.get(cat_key, [])
        if examples:
            print(f"{cat_name}:")
            print(f"  Examples saved: {len(examples)}")
            print(f"  Frame numbers: {', '.join(str(ex['frame_num']) for ex in examples[:5])}" + 
                  (" ..." if len(examples) > 5 else ""))
            print()
    
    # Save detailed report
    report_path = os.path.join(session_dir, 'agent_ap_diagnostic_report.txt')
    with open(report_path, 'w') as f:
        f.write("AGENT AP EXTRACTION FAILURE ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total frames: {total}\n\n")
        
        for cat_key, cat_name in categories:
            count = stats[cat_key]
            pct = (count / total * 100) if total > 0 else 0
            f.write(f"{cat_name:45} {count:4d} ({pct:5.1f}%)\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("DETAILED FAILURE EXAMPLES\n")
        f.write("=" * 70 + "\n\n")
        
        for cat_key, cat_name in categories:
            if cat_key == 'success':
                continue
            
            examples = failure_examples.get(cat_key, [])
            if examples:
                f.write(f"\n{cat_name}:\n")
                f.write(f"Count: {stats[cat_key]} frames\n\n")
                
                for example in examples:
                    f.write(f"  Frame {example['frame_num']:06d}:\n")
                    for agent in example['data']['agents']:
                        if agent['ap'] is None:
                            f.write(f"    Agent {agent['index']}: OCR='{agent['ocr_text']}'\n")
                    f.write("\n")
    
    print(f"✓ Detailed report saved to {report_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_agent_ap_failures.py <session_directory>")
        print("Example: python diagnose_agent_ap_failures.py captures/20251022_201216")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    
    # Create diagnostic directory
    diagnostic_dir = os.path.join(session_dir, 'agent_ap_diagnostics')
    os.makedirs(diagnostic_dir, exist_ok=True)
    
    print("=" * 70)
    print("AGENT AP EXTRACTION FAILURE DIAGNOSTICS")
    print("=" * 70)
    print()
    
    # Analyze and save examples
    stats, failure_examples = save_failure_examples(session_dir, diagnostic_dir)
    
    # Print report
    print_report(stats, failure_examples, session_dir)
    
    print()
    print(f"✓ Visual examples saved to {diagnostic_dir}/")

if __name__ == "__main__":
    main()
    