#!/usr/bin/env python3
"""
Interactive measurement tool for marking bounding boxes on images.
Draw boxes, label them, and export measurements.
"""

import cv2
import sys
import json

# Global variables
drawing = False
ix, iy = -1, -1
boxes = []
temp_box = None
img_display = None
img_original = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, temp_box, img_display, boxes
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = img_display.copy()
            cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 0), 2)
            # Show current coords
            cv2.putText(temp_img, f"({ix},{iy}) -> ({x},{y})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Measure Tool', temp_img)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Save box
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        width = x2 - x1
        height = y2 - y1
        
        if width > 0 and height > 0:
            print(f"\nBox drawn: ({x1},{y1}) to ({x2},{y2}), size {width}x{height}")
            box_name = input(f"Enter name for this box (or press Enter to skip): ").strip()
            
            if box_name:
                boxes.append({
                    'name': box_name,
                    'x': x1,
                    'y': y1,
                    'width': width,
                    'height': height,
                    'x2': x2,
                    'y2': y2
                })
                print(f"✓ Saved as '{box_name}'")
                
                # Redraw all boxes
                redraw_boxes()
            else:
                print("Skipped (no name entered)")

def redraw_boxes():
    """Redraw all saved boxes on the display image"""
    global img_display, img_original, boxes
    
    img_display = img_original.copy()
    
    for i, box in enumerate(boxes):
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        name = box['name']
        
        # Draw rectangle
        cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw label with background
        label = f"{i+1}. {name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_display, (x, y-th-8), (x+tw+4, y), (0, 255, 0), -1)
        cv2.putText(img_display, label, (x+2, y-4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.imshow('Measure Tool', img_display)

def print_help():
    """Print help instructions"""
    print("\n" + "="*60)
    print("KEYBOARD COMMANDS:")
    print("="*60)
    print("  SPACE  - List all measurements")
    print("  r      - Reset (delete all boxes)")
    print("  u      - Undo last box")
    print("  h      - Show this help")
    print("  q      - Quit and save")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python measure_tool.py <image_file>")
        print("Example: python measure_tool.py frame_000079.png")
        sys.exit(1)
    
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"❌ Could not load image: {img_path}")
        sys.exit(1)
    
    img_original = img.copy()
    img_display = img.copy()
    img_h, img_w = img.shape[:2]
    
    print("="*60)
    print("MEASUREMENT TOOL")
    print("="*60)
    print(f"\nImage: {img_path}")
    print(f"Size: {img_w}x{img_h}")
    print("\nInstructions:")
    print("  - Click and drag to draw a bounding box")
    print("  - Enter a name in the terminal when prompted")
    print("  - Press SPACE to list all measurements")
    print("  - Press 'h' for help, 'q' to quit and save")
    
    cv2.namedWindow('Measure Tool', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Measure Tool', draw_rectangle)
    cv2.imshow('Measure Tool', img_display)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Quit
            break
        elif key == ord('r'):
            # Reset
            if boxes:
                confirm = input("\nReset all boxes? (y/n): ")
                if confirm.lower() == 'y':
                    boxes = []
                    img_display = img_original.copy()
                    cv2.imshow('Measure Tool', img_display)
                    print("✓ All boxes cleared")
        elif key == ord('u'):
            # Undo last
            if boxes:
                removed = boxes.pop()
                print(f"✓ Removed '{removed['name']}'")
                redraw_boxes()
            else:
                print("No boxes to undo")
        elif key == ord('h'):
            # Help
            print_help()
        elif key == ord(' '):
            # List measurements
            if boxes:
                print("\n" + "="*60)
                print("CURRENT MEASUREMENTS:")
                print("="*60)
                for i, box in enumerate(boxes):
                    print(f"{i+1}. {box['name']}: "
                          f"({box['x']},{box['y']}) to ({box['x2']},{box['y2']}) "
                          f"size {box['width']}x{box['height']}")
            else:
                print("\nNo measurements yet")
    
    cv2.destroyAllWindows()
    
    # Save measurements
    if boxes:
        print("\n" + "="*60)
        print("FINAL MEASUREMENTS:")
        print("="*60)
        print(f"\nImage: {img_path}")
        print(f"Size: {img_w}x{img_h}\n")
        
        for i, box in enumerate(boxes):
            print(f"{i+1}. {box['name']}:")
            print(f"   Upper-left: ({box['x']}, {box['y']})")
            print(f"   Lower-right: ({box['x2']}, {box['y2']})")
            print(f"   Size: {box['width']}x{box['height']}")
            print()
        
        # Save to text file
        txt_file = 'measurements.txt'
        with open(txt_file, 'w') as f:
            f.write(f"Image: {img_path}\n")
            f.write(f"Size: {img_w}x{img_h}\n\n")
            for i, box in enumerate(boxes):
                f.write(f"{i+1}. {box['name']}:\n")
                f.write(f"   Upper-left: ({box['x']}, {box['y']})\n")
                f.write(f"   Lower-right: ({box['x2']}, {box['y2']})\n")
                f.write(f"   Size: {box['width']}x{box['height']}\n\n")
        
        # Save to JSON file
        json_file = 'measurements.json'
        with open(json_file, 'w') as f:
            json.dump({
                'image': img_path,
                'width': img_w,
                'height': img_h,
                'boxes': boxes
            }, f, indent=2)
        
        print(f"✓ Saved to {txt_file} and {json_file}")
    else:
        print("\nNo measurements to save")