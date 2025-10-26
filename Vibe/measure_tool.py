import cv2
import sys

# Global variables
drawing = False
ix, iy = -1, -1
boxes = []
temp_box = None
img_display = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, temp_box, img_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = img_display.copy()
            cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Measure Tool', temp_img)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Save box
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        width = x2 - x1
        height = y2 - y1
        
        box_name = input(f"\nEnter name for box at ({x1},{y1}) size {width}x{height}: ")
        boxes.append((box_name, x1, y1, width, height))
        
        # Redraw all boxes
        temp_img = img_display.copy()
        for name, bx, by, bw, bh in boxes:
            cv2.rectangle(temp_img, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
            cv2.putText(temp_img, name, (bx, by-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('Measure Tool', temp_img)
        img_display[:] = temp_img

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python measure_tool.py <image_file>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Could not load image: {img_path}")
        sys.exit(1)
    
    img_display = img.copy()
    
    print("="*60)
    print("MEASUREMENT TOOL")
    print("="*60)
    print("\nInstructions:")
    print("  - Click and drag to draw a box")
    print("  - Enter a name in the terminal")
    print("  - Press 'q' to quit and save measurements")
    print("  - Press 'r' to reset all boxes")
    
    cv2.namedWindow('Measure Tool')
    cv2.setMouseCallback('Measure Tool', draw_rectangle)
    cv2.imshow('Measure Tool', img_display)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            boxes = []
            img_display = img.copy()
            cv2.imshow('Measure Tool', img_display)
    
    cv2.destroyAllWindows()
    
    # Save measurements
    print("\n" + "="*60)
    print("MEASUREMENTS:")
    print("="*60)
    
    img_h, img_w = img.shape[:2]
    print(f"\nImage size: {img_w}x{img_h}\n")
    
    for name, x, y, w, h in boxes:
        print(f"{name}: x={x}, y={y}, w={w}, h={h}")
    
    # Save to file
    with open('measurements.txt', 'w') as f:
        f.write(f"Image: {img_path}\n")
        f.write(f"Size: {img_w}x{img_h}\n\n")
        for name, x, y, w, h in boxes:
            f.write(f"{name}: x={x}, y={y}, w={w}, h={h}\n")
    
    print("\n✓ Saved to measurements.txt")