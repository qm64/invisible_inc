"""
Diagnostic tool to capture and save both window and viewport
"""

from hierarchical_detector import HierarchicalGameStateDetector
from window_detector import WindowDetector
import cv2

def main():
    detector = HierarchicalGameStateDetector(WindowDetector())
    
    # Load templates
    print("Loading templates...")
    detector.load_templates('./templates')
    
    print("\nCapturing window...")
    input("Press ENTER to capture...")
    
    window_img = detector.window_detector.capture_game_window(auto_focus=True)
    
    if window_img is None:
        print("Failed to capture window")
        return
    
    # Save full window
    cv2.imwrite('diagnostic_full_window.png', cv2.cvtColor(window_img, cv2.COLOR_RGB2BGR))
    print(f"✓ Saved full window: {window_img.shape[1]}x{window_img.shape[0]} -> diagnostic_full_window.png")
    
    # Try to detect viewport
    viewport_result = detector.viewport_detector.find_viewport_with_all_anchors(window_img)
    
    if viewport_result:
        viewport_bbox, anchors = viewport_result
        print(f"\n✓ Viewport detected:")
        print(f"  Position: x={viewport_bbox.x}, y={viewport_bbox.y}")
        print(f"  Size: {viewport_bbox.width}x{viewport_bbox.height}")
        print(f"  Anchors: {list(anchors.keys())}")
        
        # Extract and save viewport
        viewport = window_img[viewport_bbox.y:viewport_bbox.y2, viewport_bbox.x:viewport_bbox.x2]
        cv2.imwrite('diagnostic_viewport.png', cv2.cvtColor(viewport, cv2.COLOR_RGB2BGR))
        print(f"✓ Saved viewport -> diagnostic_viewport.png")
        
        # Draw viewport boundary on full window
        window_with_viewport = window_img.copy()
        cv2.rectangle(window_with_viewport, 
                     (viewport_bbox.x, viewport_bbox.y),
                     (viewport_bbox.x2, viewport_bbox.y2),
                     (255, 0, 255), 3)
        cv2.imwrite('diagnostic_window_with_viewport.png', 
                   cv2.cvtColor(window_with_viewport, cv2.COLOR_RGB2BGR))
        print(f"✓ Saved window with viewport overlay -> diagnostic_window_with_viewport.png")
        
        # Save upper-left search region
        vh, vw = viewport.shape[:2]
        search_region = viewport[0:int(vh*0.1), 0:int(vw*0.2)]
        cv2.imwrite('diagnostic_upper_left_search.png', 
                   cv2.cvtColor(search_region, cv2.COLOR_RGB2BGR))
        print(f"✓ Saved upper-left search region: {search_region.shape[1]}x{search_region.shape[0]}")
        
    else:
        print("✗ Viewport detection failed")
    
    print("\n" + "="*60)
    print("COMPARISON:")
    print("="*60)
    print(f"Original measurement image: 2560x1301")
    print(f"Current window capture: {window_img.shape[1]}x{window_img.shape[0]}")
    if viewport_result:
        print(f"Current viewport: {viewport_bbox.width}x{viewport_bbox.height}")
        print(f"Viewport offset: ({viewport_bbox.x}, {viewport_bbox.y})")
    
    print("\nNext steps:")
    print("1. Compare diagnostic_full_window.png with measure_me.png")
    print("2. Check if viewport is cutting off power/credits")
    print("3. Use diagnostic_upper_left_search.png to see search region")
    print("4. Run measurement_tool.py on diagnostic_viewport.png to extract new templates")

if __name__ == "__main__":
    main()