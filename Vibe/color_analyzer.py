"""
Color Signature Analyzer for Game UI Elements
Interactively select regions to analyze their color characteristics
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class ColorSignatureAnalyzer:
    def __init__(self, screenshot_path):
        self.screenshot_path = Path(screenshot_path)
        self.image = cv2.imread(str(screenshot_path))
        if self.image is None:
            raise ValueError(f"Could not load image: {screenshot_path}")
        
        self.display_image = self.image.copy()
        self.selecting = False
        self.start_point = None
        self.current_rect = None
        self.signatures = {}
        self.element_name = None
        
        print("\nColor Signature Analyzer")
        print("=" * 50)
        print("Instructions:")
        print("  1. Click and drag to select a UI element")
        print("  2. Release to analyze the color signature")
        print("  3. Type a name for the element and press Enter")
        print("  4. Press 'r' to reset current selection")
        print("  5. Press 's' to save all signatures")
        print("  6. Press 'q' to quit")
        print("=" * 50)
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start_point = (x, y)
            self.current_rect = None
            
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
            self.display_image = self.image.copy()
            cv2.rectangle(self.display_image, self.start_point, (x, y), (0, 255, 0), 2)
            
        elif event == cv2.EVENT_LBUTTONUP and self.selecting:
            self.selecting = False
            self.current_rect = (
                min(self.start_point[0], x),
                min(self.start_point[1], y),
                max(self.start_point[0], x),
                max(self.start_point[1], y)
            )
            self.analyze_region()
    
    def analyze_region(self):
        if not self.current_rect:
            return
        
        x1, y1, x2, y2 = self.current_rect
        region = self.image[y1:y2, x1:x2]
        
        if region.size == 0:
            print("Selected region is too small")
            return
        
        # Convert to different color spaces
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        rgb_region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        
        # Calculate statistics
        signature = self.calculate_signature(region, hsv_region, rgb_region)
        
        # Display visualization
        self.visualize_signature(region, signature)
        
        # Prompt for name
        print("\n" + "=" * 50)
        print("Enter a name for this element (or press Enter to skip):")
        self.element_name = input("> ").strip()
        
        if self.element_name:
            self.signatures[self.element_name] = signature
            print(f"✓ Saved signature for '{self.element_name}'")
        
        # Draw permanent rectangle on display
        self.display_image = self.image.copy()
        cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if self.element_name:
            cv2.putText(self.display_image, self.element_name, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def calculate_signature(self, bgr_region, hsv_region, rgb_region):
        """Calculate comprehensive color signature"""
        
        # HSV statistics (best for color detection)
        hsv_mean = np.mean(hsv_region, axis=(0, 1))
        hsv_std = np.std(hsv_region, axis=(0, 1))
        hsv_min = np.min(hsv_region, axis=(0, 1))
        hsv_max = np.max(hsv_region, axis=(0, 1))
        
        # RGB statistics (for reference)
        rgb_mean = np.mean(rgb_region, axis=(0, 1))
        
        # Dominant colors (top 3)
        pixels = rgb_region.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        top_indices = np.argsort(counts)[-3:][::-1]
        dominant_colors = unique_colors[top_indices].tolist()
        dominant_counts = counts[top_indices].tolist()
        
        # Calculate color ranges for detection (HSV)
        # Use mean ± 2*std, clamped to actual min/max
        h_range = [
            max(0, int(hsv_mean[0] - 2*hsv_std[0])),
            min(179, int(hsv_mean[0] + 2*hsv_std[0]))
        ]
        s_range = [
            max(0, int(hsv_mean[1] - 2*hsv_std[1])),
            min(255, int(hsv_mean[1] + 2*hsv_std[1]))
        ]
        v_range = [
            max(0, int(hsv_mean[2] - 2*hsv_std[2])),
            min(255, int(hsv_mean[2] + 2*hsv_std[2]))
        ]
        
        signature = {
            'hsv': {
                'mean': hsv_mean.tolist(),
                'std': hsv_std.tolist(),
                'min': hsv_min.tolist(),
                'max': hsv_max.tolist(),
                'detection_range': {
                    'h': h_range,
                    's': s_range,
                    'v': v_range
                }
            },
            'rgb': {
                'mean': rgb_mean.tolist(),
                'dominant_colors': dominant_colors,
                'dominant_counts': dominant_counts
            },
            'region_size': {
                'width': bgr_region.shape[1],
                'height': bgr_region.shape[0]
            }
        }
        
        # Print summary
        print(f"\nColor Analysis:")
        print(f"  Region size: {signature['region_size']['width']}x{signature['region_size']['height']}")
        print(f"  HSV mean: H={hsv_mean[0]:.1f}, S={hsv_mean[1]:.1f}, V={hsv_mean[2]:.1f}")
        print(f"  Detection ranges:")
        print(f"    Hue: {h_range[0]}-{h_range[1]}")
        print(f"    Saturation: {s_range[0]}-{s_range[1]}")
        print(f"    Value: {v_range[0]}-{v_range[1]}")
        print(f"  Dominant RGB colors: {dominant_colors}")
        
        return signature
    
    def visualize_signature(self, region, signature):
        """Create visualization of color signature"""
        h = 200
        w = 600
        viz = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Show dominant colors
        color_width = w // 3
        for i, (color, count) in enumerate(zip(signature['rgb']['dominant_colors'],
                                               signature['rgb']['dominant_counts'])):
            x1 = i * color_width
            x2 = (i + 1) * color_width
            # Convert RGB to BGR for OpenCV
            bgr_color = (int(color[2]), int(color[1]), int(color[0]))
            cv2.rectangle(viz, (x1, 0), (x2, 100), bgr_color, -1)
            
            # Add text
            text = f"#{i+1}: {count} px"
            cv2.putText(viz, text, (x1 + 5, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Show HSV ranges as bars
        hsv_ranges = signature['hsv']['detection_range']
        y_base = 120
        
        # Hue bar (0-179)
        h_start = int((hsv_ranges['h'][0] / 179) * w)
        h_end = int((hsv_ranges['h'][1] / 179) * w)
        cv2.rectangle(viz, (h_start, y_base), (h_end, y_base + 20), (0, 255, 0), -1)
        cv2.putText(viz, f"H: {hsv_ranges['h'][0]}-{hsv_ranges['h'][1]}", 
                   (5, y_base + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Saturation bar (0-255)
        s_start = int((hsv_ranges['s'][0] / 255) * w)
        s_end = int((hsv_ranges['s'][1] / 255) * w)
        cv2.rectangle(viz, (s_start, y_base + 30), (s_end, y_base + 50), (0, 255, 0), -1)
        cv2.putText(viz, f"S: {hsv_ranges['s'][0]}-{hsv_ranges['s'][1]}", 
                   (5, y_base + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Value bar (0-255)
        v_start = int((hsv_ranges['v'][0] / 255) * w)
        v_end = int((hsv_ranges['v'][1] / 255) * w)
        cv2.rectangle(viz, (v_start, y_base + 60), (v_end, y_base + 80), (0, 255, 0), -1)
        cv2.putText(viz, f"V: {hsv_ranges['v'][0]}-{hsv_ranges['v'][1]}", 
                   (5, y_base + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('Color Signature', viz)
    
    def save_signatures(self):
        """Save all signatures to JSON file"""
        if not self.signatures:
            print("No signatures to save")
            return
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'source_image': str(self.screenshot_path),
            'signatures': self.signatures
        }
        
        output_path = Path('color_signatures.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Saved {len(self.signatures)} signatures to {output_path}")
        
        # Also create a Python dict file for easy import
        py_path = Path('color_signatures.py')
        with open(py_path, 'w') as f:
            f.write("# Auto-generated color signatures\n")
            f.write("# Generated: " + datetime.now().isoformat() + "\n\n")
            f.write("COLOR_SIGNATURES = ")
            f.write(json.dumps(self.signatures, indent=2))
        
        print(f"✓ Saved Python version to {py_path}")
    
    def run(self):
        """Main interaction loop"""
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.mouse_callback)
        
        while True:
            cv2.imshow('Image', self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.display_image = self.image.copy()
                self.current_rect = None
                print("Selection reset")
            elif key == ord('s'):
                self.save_signatures()
        
        cv2.destroyAllWindows()


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python color_analyzer.py <screenshot_path>")
        print("\nOr use the WindowDetector to capture a screenshot first:")
        print("  python window_detector.py")
        sys.exit(1)
    
    screenshot_path = sys.argv[1]
    
    analyzer = ColorSignatureAnalyzer(screenshot_path)
    analyzer.run()


if __name__ == '__main__':
    main()
    