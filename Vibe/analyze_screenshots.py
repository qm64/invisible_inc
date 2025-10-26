import cv2
import os

print("="*60)
print("SCREENSHOT ANALYSIS TOOL")
print("="*60)
print("\nThis tool will help you identify key regions in each screenshot.")
print("Click on important UI elements to record their positions.\n")

screenshots = [
    ("ui_elements/full_interface.png", "Identify: Power, Credits, Alarm level positions"),
    ("ui_elements/agent_panel.png", "Identify: Each agent icon position"),
    ("ui_elements/selected_agent.png", "Identify: Selected agent indicator"),
    ("ui_elements/end_turn_button.png", "Identify: End turn button location"),
    ("agents/movement_range.png", "Note: Blue movement squares appearance"),
    ("situations/alarm_level.png", "Identify: Alarm tracker exact location"),
]

clicks = {}
current_file = None

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks"""
    global current_file
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_file not in clicks:
            clicks[current_file] = []
        clicks[current_file].append((x, y))
        print(f"  Click {len(clicks[current_file])}: x={x}, y={y}")
        
        # Draw circle
        cv2.circle(img_display, (x, y), 8, (0, 255, 0), 2)
        cv2.putText(img_display, str(len(clicks[current_file])), 
                   (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 255, 0), 2)
        cv2.imshow('Analyze Screenshot', img_display)

print("Instructions:")
print("- Click on important UI elements")
print("- Press 'n' for next screenshot")
print("- Press 'q' to quit and save\n")

for filepath, description in screenshots:
    full_path = f"game_docs/{filepath}"
    
    if not os.path.exists(full_path):
        print(f"⚠ Skipping {filepath} (not found)")
        continue
    
    img = cv2.imread(full_path)
    img_display = img.copy()
    current_file = filepath
    
    print("\n" + "="*60)
    print(f"FILE: {filepath}")
    print(f"TASK: {description}")
    print("="*60)
    print("Click on important areas, then press 'n' for next...")
    
    cv2.namedWindow('Analyze Screenshot')
    cv2.setMouseCallback('Analyze Screenshot', mouse_callback)
    cv2.imshow('Analyze Screenshot', img_display)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):  # Next
            break
        elif key == ord('q'):  # Quit
            cv2.destroyAllWindows()
            break
    
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# Save analysis
print("\n" + "="*60)
print("ANALYSIS RESULTS")
print("="*60)

with open("game_docs/analysis.txt", "w") as f:
    for filepath, points in clicks.items():
        print(f"\n{filepath}:")
        f.write(f"{filepath}:\n")
        for i, (x, y) in enumerate(points, 1):
            print(f"  Point {i}: ({x}, {y})")
            f.write(f"  Point {i}: ({x}, {y})\n")
        f.write("\n")

print("\n✓ Analysis saved to game_docs/analysis.txt")
print("\nNext: Let's discuss what you found!")
