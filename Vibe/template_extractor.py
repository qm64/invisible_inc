"""
Template Extractor - Creates template images from measurements
Extracts UI elements for template matching in the game state detector
"""

import cv2
import sys
import os
from typing import List, Tuple, Dict

def load_measurements(measurements_file: str) -> Tuple[str, Tuple[int, int], List[Tuple[str, int, int, int, int]]]:
    """
    Load measurements from file
    Returns: (image_path, image_size, measurements)
    """
    measurements = []
    image_path = None
    image_size = None
    
    with open(measurements_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Image:'):
                image_path = line.split(':', 1)[1].strip()
            elif line.startswith('Size:'):
                size_str = line.split(':', 1)[1].strip()
                w, h = map(int, size_str.split('x'))
                image_size = (w, h)
            elif ':' in line and 'x=' in line:
                # Parse measurement line
                name, coords = line.split(':', 1)
                name = name.strip()
                
                # Parse x=, y=, w=, h=
                parts = coords.split(',')
                x = int(parts[0].split('=')[1].strip())
                y = int(parts[1].split('=')[1].strip())
                w = int(parts[2].split('=')[1].strip())
                h = int(parts[3].split('=')[1].strip())
                
                measurements.append((name, x, y, w, h))
    
    return image_path, image_size, measurements


def extract_templates(image, measurements: List[Tuple[str, int, int, int, int]], 
                     output_dir: str, padding: int = 5):
    """
    Extract template images from measurements
    Adds padding around each region for better matching
    """
    os.makedirs(output_dir, exist_ok=True)
    
    img_h, img_w = image.shape[:2]
    extracted = {}
    
    for name, x, y, w, h in measurements:
        # Add padding (but stay within image bounds)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)
        
        # Extract region
        template = image[y1:y2, x1:x2]
        
        # Save template
        filename = f"{output_dir}/{name}_template.png"
        cv2.imwrite(filename, template)
        
        extracted[name] = {
            'filename': filename,
            'original_bbox': (x, y, w, h),
            'padded_bbox': (x1, y1, x2-x1, y2-y1),
            'size': template.shape[:2]
        }
        
        print(f"✓ Extracted '{name}': {template.shape[1]}x{template.shape[0]} -> {filename}")
    
    return extracted
    

def analyze_proportions(image_size: Tuple[int, int], measurements: List[Tuple[str, int, int, int, int]]):
    """
    Analyze proportional positions for better heuristics
    """
    img_w, img_h = image_size
    
    print("\n" + "="*60)
    print("PROPORTIONAL ANALYSIS")
    print("="*60)
    
    for name, x, y, w, h in measurements:
        x_pct = (x / img_w) * 100
        y_pct = (y / img_h) * 100
        x2_pct = ((x + w) / img_w) * 100
        y2_pct = ((y + h) / img_h) * 100
        
        print(f"\n{name}:")
        print(f"  Position: ({x}, {y}) = ({x_pct:.1f}%, {y_pct:.1f}%)")
        print(f"  Size: {w}x{h}")
        print(f"  Bounds: X [{x_pct:.1f}% -> {x2_pct:.1f}%], Y [{y_pct:.1f}% -> {y2_pct:.1f}%]")


def create_reference_guide(output_dir: str, image_size: Tuple[int, int], 
                          measurements: List[Tuple[str, int, int, int, int]]):
    """
    Create a reference guide with all measurements and proportions
    """
    img_w, img_h = image_size
    
    guide_path = f"{output_dir}/template_guide.txt"
    
    with open(guide_path, 'w') as f:
        f.write("TEMPLATE EXTRACTION GUIDE\n")
        f.write("="*60 + "\n\n")
        f.write(f"Source Image Size: {img_w}x{img_h}\n\n")
        
        f.write("EXTRACTED TEMPLATES:\n")
        f.write("-"*60 + "\n")
        
        for name, x, y, w, h in measurements:
            x_pct = (x / img_w) * 100
            y_pct = (y / img_h) * 100
            
            f.write(f"\n{name}:\n")
            f.write(f"  Absolute: x={x}, y={y}, w={w}, h={h}\n")
            f.write(f"  Relative: x={x_pct:.1f}%, y={y_pct:.1f}%\n")
            f.write(f"  Template: {name}_template.png\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("USAGE IN DETECTOR:\n")
        f.write("-"*60 + "\n")
        f.write("""
# Load templates
detector.templates.load_template('power', 'templates/power_template.png')
detector.templates.load_template('credits', 'templates/credits_template.png')
detector.templates.load_template('alarm', 'templates/alarm_template.png')
# ... etc

# Use for matching
power_bbox = detector.templates.match_template(viewport, 'power', threshold=0.8)
if power_bbox:
    # Found power display at power_bbox.x, power_bbox.y
    pass
        """)
    
    print(f"\n✓ Created reference guide: {guide_path}")


def visualize_templates(image, measurements: List[Tuple[str, int, int, int, int]], 
                       output_path: str):
    """
    Create visualization showing all measured regions
    """
    vis = image.copy()
    
    # Draw each measurement
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
    ]
    
    for i, (name, x, y, w, h) in enumerate(measurements):
        color = colors[i % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        
        # Draw label with background
        label = name
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Background rectangle for text
        cv2.rectangle(vis, (x, y-label_h-5), (x+label_w+4, y), color, -1)
        cv2.putText(vis, label, (x+2, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.imwrite(output_path, vis)
    print(f"\n✓ Created visualization: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python template_extractor.py <measurements.txt>")
        print("\nExample: python template_extractor.py measurements.txt")
        sys.exit(1)
    
    measurements_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "templates"
    
    print("="*60)
    print("TEMPLATE EXTRACTOR")
    print("="*60)
    
    # Load measurements
    print(f"\nLoading measurements from {measurements_file}...")
    image_path, image_size, measurements = load_measurements(measurements_file)
    
    if not image_path or not image_size:
        print("Error: Could not parse measurements file")
        sys.exit(1)
    
    print(f"  Image: {image_path}")
    print(f"  Size: {image_size[0]}x{image_size[1]}")
    print(f"  Measurements: {len(measurements)}")
    
    # Load image
    print(f"\nLoading image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        sys.exit(1)
    
    # Extract templates from measurements
    print(f"\nExtracting templates to '{output_dir}/'...")
    extracted = extract_templates(image, measurements, output_dir)
    
    # Analyze proportions
    analyze_proportions(image_size, measurements)
    
    # Create reference guide
    create_reference_guide(output_dir, image_size, measurements)
    
    # Create visualization
    vis_path = f"{output_dir}/measurements_visualization.png"
    visualize_templates(image, measurements, vis_path)
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"\nTemplates saved to: {output_dir}/")
    print(f"Total templates: {len(extracted) + 3}")  # +3 for anchor templates
    print(f"\nNext steps:")
    print(f"  1. Review templates in '{output_dir}/'")
    print(f"  2. Adjust anchor template extraction if needed")
    print(f"  3. Use detector.load_templates('{output_dir}') in your code")


if __name__ == "__main__":
    main()