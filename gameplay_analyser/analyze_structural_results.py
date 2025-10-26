#!/usr/bin/env python3
"""
Analyze structural_detector.py results across multiple frames
Generates a summary report showing what works and what doesn't
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def load_detection_json(json_path):
    """Load detection results from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def analyze_detections(results_dir):
    """Analyze all detection JSON files in results directory"""
    
    results_path = Path(results_dir)
    json_files = list(results_path.glob("*_detected.json"))
    
    if not json_files:
        print(f"No detection JSON files found in {results_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"Structural Detector Analysis")
    print(f"{'='*70}")
    print(f"Found {len(json_files)} detection results\n")
    
    # Track what elements were detected
    element_counts = defaultdict(int)
    element_confidence = defaultdict(list)
    extracted_values = defaultdict(list)
    
    frame_results = []
    
    for json_file in sorted(json_files):
        frame_name = json_file.stem.replace('_detected', '')
        data = load_detection_json(json_file)
        
        frame_info = {
            'frame': frame_name,
            'elements': [],
            'viewport': data.get('viewport', {}).get('bbox'),
            'values': {}
        }
        
        # Count detected elements
        for element_name, element_data in data.items():
            if element_name == 'viewport':
                continue
                
            element_counts[element_name] += 1
            
            if 'confidence' in element_data:
                element_confidence[element_name].append(element_data['confidence'])
            
            frame_info['elements'].append(element_name)
            
            # Check for extracted values (special elements ending in _value)
            if element_name.endswith('_value') or 'value' in element_data:
                value = element_data.get('value', 'N/A')
                extracted_values[element_name].append(value)
                frame_info['values'][element_name] = value
        
        frame_results.append(frame_info)
    
    # Print per-frame results
    print(f"{'='*70}")
    print("Per-Frame Detection Results")
    print(f"{'='*70}\n")
    
    for result in frame_results:
        print(f"Frame: {result['frame']}")
        print(f"  Elements detected: {len(result['elements'])}")
        print(f"  Viewport: {result['viewport']}")
        
        if result['values']:
            print(f"  Extracted values:")
            for val_name, val in result['values'].items():
                print(f"    {val_name}: {val}")
        
        print(f"  Elements: {', '.join(sorted(result['elements']))}")
        print()
    
    # Print summary statistics
    print(f"{'='*70}")
    print("Detection Summary Statistics")
    print(f"{'='*70}\n")
    
    print(f"Total frames analyzed: {len(json_files)}")
    print(f"\nElement Detection Rate:")
    
    for element_name in sorted(element_counts.keys()):
        count = element_counts[element_name]
        rate = count / len(json_files) * 100
        avg_conf = sum(element_confidence[element_name]) / len(element_confidence[element_name]) if element_confidence[element_name] else 0
        
        status = "✓" if rate >= 80 else "⚠" if rate >= 50 else "✗"
        conf_str = f"(avg conf: {avg_conf:.2f})" if element_confidence[element_name] else ""
        
        print(f"  {status} {element_name:30s}: {count}/{len(json_files)} ({rate:5.1f}%) {conf_str}")
    
    # Print extracted values summary
    if extracted_values:
        print(f"\n{'='*70}")
        print("Extracted Values Summary")
        print(f"{'='*70}\n")
        
        for value_name in sorted(extracted_values.keys()):
            values = extracted_values[value_name]
            success_count = sum(1 for v in values if v != 'N/A' and v is not None)
            success_rate = success_count / len(values) * 100
            
            status = "✓" if success_rate >= 80 else "⚠" if success_rate >= 50 else "✗"
            print(f"  {status} {value_name:30s}: {success_count}/{len(values)} ({success_rate:5.1f}%)")
            
            if success_count > 0:
                unique_values = set(v for v in values if v != 'N/A' and v is not None)
                print(f"      Unique values: {sorted(unique_values)}")
    
    # Analysis of what's working
    print(f"\n{'='*70}")
    print("Analysis")
    print(f"{'='*70}\n")
    
    reliable = [name for name, count in element_counts.items() 
                if count / len(json_files) >= 0.8]
    unreliable = [name for name, count in element_counts.items() 
                  if count / len(json_files) < 0.5]
    
    print("✓ RELIABLE DETECTIONS (≥80% success rate):")
    for name in sorted(reliable):
        print(f"  - {name}")
    
    if unreliable:
        print("\n✗ UNRELIABLE DETECTIONS (<50% success rate):")
        for name in sorted(unreliable):
            count = element_counts[name]
            rate = count / len(json_files) * 100
            print(f"  - {name} ({rate:.1f}%)")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_structural_results.py <results_directory>")
        print("Example: python analyze_structural_results.py captures/20251022_201216/structural_test_results")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    analyze_detections(results_dir)
