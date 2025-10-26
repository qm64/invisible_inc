"""
Extract templates from the detected viewport (not full window)
This ensures templates match what we'll be searching
"""

from hierarchical_detector import HierarchicalGameStateDetector
from window_detector import WindowDetector
import cv2
import sys

def main():
    print("="*60)
    print("VIEWPORT TEMPLATE EXTRACTOR")
    print("="*60)
    
    detector = HierarchicalGameStateDetector(WindowDetector())
    
    # Load existing templates for anchor detection
    print("\nLoading anchor templates...")
    detector.load_templates('./templates')
    
    print("\nCapturing and detecting viewport...")
    input("Press ENTER to capture...")
    
    window_img = detector.window_detector.capture_game_window(auto_focus=True)
    if window_img is None:
        print("Failed to capture window")
        return
    
    # Detect viewport
    viewport_result = detector.viewport_detector.find_viewport_with_all_anchors(window_img)
    if viewport_result is None:
        print("Viewport detection failed")
        return
    
    viewport_bbox, anchors = viewport_result
    viewport = window_img[viewport_bbox.y:viewport_bbox.y2, viewport_bbox.x:viewport_bbox.x2]
    
    print(f"✓ Viewport: {viewport_bbox.width}x{viewport_bbox.height} at offset ({viewport_bbox.x}, {viewport_bbox.y})")
    
    # Load measurements (adjusted for viewport offset)
    measurements_file = sys.argv[1] if len(sys.argv) > 1 else 'measurements_complete.txt'
    
    print(f"\nLoading measurements from {measurements_file}...")
    measurements = []
    with open(measurements_file, 'r') as f:
        for line in f:
            if ':' in line and 'x=' in line and not line.startswith('Image:') and not line.startswith('Size:'):
                name, coords = line.split(':', 1)
                name = name.strip()
                parts = coords.split(',')
                x = int(parts[0].split('=')[1].strip())
                y = int(parts[1].split('=')[1].strip())
                w = int(parts[2].split('=')[1].strip())
                h = int(parts[3].split('=')[1].strip())
                measurements.append((name, x, y, w, h))
    
    print(f"Loaded {len(measurements)} measurements")
    
    # Adjust measurements for viewport offset and extract from viewport
    print(f"\nExtracting templates from viewport (offset: {viewport_bbox.x}, {viewport_bbox.y})...")
    
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'templates_viewport'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    extracted = 0
    for name, x, y, w, h in measurements:
        # Adjust coordinates to viewport space
        vp_x = x - viewport_bbox.x
        vp_y = y - viewport_bbox.y
        
        # Check if region is within viewport
        if vp_x < 0 or vp_y < 0 or vp_x + w > viewport_bbox.width or vp_y + h > viewport_bbox.height:
            print(f"  ⚠️  Skipping '{name}' - outside viewport bounds")
            print(f"      Original: ({x},{y}) {w}x{h}")
            print(f"      Viewport: ({vp_x},{vp_y}) {w}x{h}")
            print(f"      Bounds check: x={vp_x}<0? y={vp_y}<0? x+w={vp_x+w}>{viewport_bbox.width}? y+h={vp_y+h}>{viewport_bbox.height}?")
            continue
        
        # Extract from viewport
        template = viewport[vp_y:vp_y+h, vp_x:vp_x+w]
        
        # Save
        filename = f"{output_dir}/{name}_template.png"
        cv2.imwrite(filename, cv2.cvtColor(template, cv2.COLOR_RGB2BGR))
        
        print(f"  ✓ {name}: viewport coords ({vp_x},{vp_y}) {w}x{h} -> {filename}")
        extracted += 1
    
    print(f"\n✓ Extracted {extracted} templates to {output_dir}/")
    print(f"\nTo use these templates:")
    print(f"  detector.load_templates('{output_dir}')")

if __name__ == "__main__":
    main()