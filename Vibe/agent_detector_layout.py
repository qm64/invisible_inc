from window_detector import WindowDetector
import cv2
import numpy as np
import pytesseract

class LayoutBasedDetector:
    """
    Detects game elements using fixed layout proportions
    Everything is relative to viewport dimensions
    """
    
    def __init__(self):
        self.detector = WindowDetector()
        
        # Layout proportions (percentage of viewport)
        # These are approximate and can be tuned
        self.layout = {
            # Large selected agent portrait (lower-left anchor)
            'selected_agent': {
                'x': 0.005,     # Further left
                'y': 0.58,      # Lower down
                'width': 0.11,  # Narrower
                'height': 0.40  # Height
            },
            
            # Small agent icons (above selected agent)
            # In Incognita mode, this shows agent portraits in top-left
            'agent_icons': {
                'x': 0.005,          # Far left
                'y': 0.04,           # Below power/credits
                'width': 0.07,       # Icon width
                'height': 0.50,      # Vertical region for all icons
                'icon_height': 0.10, # Each icon/portrait height
                'spacing': 0.005     # Space between
            },
            
            # AP text (right of agent icons)
            'ap_text': {
                'x': 0.08,       # Right of icons
                'y': 0.04,       # Same as icons
                'width': 0.05,   # Narrow text region
                'height': 0.50   # Same height as icons
            },
            
            # Quick actions (bottom-right corner)
            'quick_actions': {
                'x': 0.60,       # Right side
                'y': 0.75,       # Bottom
                'width': 0.38,   # Wide
                'height': 0.23   # Bottom area
            },
            
            # Augments (right side, middle)
            'augments': {
                'x': 0.60,       # Right side
                'y': 0.08,       # Upper area
                'width': 0.18,   # Column width
                'height': 0.30   # Height
            },
            
            # Inventory (right side, right of augments label)
            'inventory': {
                'x': 0.73,       # Right of augments
                'y': 0.08,       # Same y
                'width': 0.25,   # Rest of width
                'height': 0.30   # Same height
            }
        }
    
    def extract_viewport(self, window_img):
        """Extract game viewport from window"""
        green = window_img[:, :, 1]
        green_threshold = cv2.threshold(green, 80, 255, cv2.THRESH_BINARY)[1]
        rows_with_green = np.any(green_threshold > 0, axis=1)
        cols_with_green = np.any(green_threshold > 0, axis=0)
        
        if not np.any(rows_with_green) or not np.any(cols_with_green):
            return None, None
        
        top = np.argmax(rows_with_green)
        bottom = len(rows_with_green) - np.argmax(rows_with_green[::-1])
        left = np.argmax(cols_with_green)
        right = len(cols_with_green) - np.argmax(cols_with_green[::-1])
        
        margin = 5
        top = max(0, top - margin)
        left = max(0, left - margin)
        bottom = min(window_img.shape[0], bottom + margin)
        right = min(window_img.shape[1], right + margin)
        
        viewport = window_img[top:bottom, left:right]
        bounds = (left, top, right - left, bottom - top)
        
        return viewport, bounds
    
    def get_region(self, viewport, region_def):
        """Extract a region from viewport using proportions"""
        h, w = viewport.shape[:2]
        
        x = int(w * region_def['x'])
        y = int(h * region_def['y'])
        width = int(w * region_def['width'])
        height = int(h * region_def['height'])
        
        return viewport[y:y+height, x:x+width], (x, y, width, height)
    
    def count_agents(self, viewport):
        """
        Count number of small agent icons by looking at the agent icon column
        """
        icon_region, coords = self.get_region(viewport, self.layout['agent_icons'])
        
        # Threshold green channel
        green = icon_region[:, :, 1]
        _, mask = cv2.threshold(green, 100, 255, cv2.THRESH_BINARY)
        
        # Count vertical segments (each icon creates a bright segment)
        # Sum horizontally to get vertical profile
        vertical_profile = np.sum(mask, axis=1)
        
        # Find peaks (agent icons)
        # A peak is a region with high values
        threshold = np.max(vertical_profile) * 0.3
        in_peak = False
        peak_count = 0
        
        for value in vertical_profile:
            if value > threshold and not in_peak:
                peak_count += 1
                in_peak = True
            elif value <= threshold:
                in_peak = False
        
        return peak_count
    
    def get_ap_values(self, viewport, num_agents):
        """
        Read AP values for each agent
        Returns list of AP values (one per agent)
        """
        ap_region, coords = self.get_region(viewport, self.layout['ap_text'])
        
        if ap_region is None or ap_region.size == 0:
            return [0] * num_agents
        
        # Divide vertically by number of agents
        region_h = coords[3]
        icon_h = int(viewport.shape[0] * self.layout['agent_icons']['icon_height'])
        spacing = int(viewport.shape[0] * self.layout['agent_icons']['spacing'])
        
        ap_values = []
        start_y = 0
        
        for i in range(num_agents):
            # Extract AP region for this agent
            end_y = min(start_y + icon_h, region_h)
            
            if start_y >= region_h or end_y <= start_y:
                ap_values.append(0)
                start_y = end_y + spacing
                continue
            
            ap_img = ap_region[start_y:end_y, :]
            
            if ap_img.size == 0:
                ap_values.append(0)
                start_y = end_y + spacing
                continue
            
            # OCR on cyan/green text
            # Extract cyan channel (high in both green and blue)
            cyan = cv2.min(ap_img[:, :, 1], ap_img[:, :, 2])
            _, thresh = cv2.threshold(cyan, 100, 255, cv2.THRESH_BINARY)
            
            if thresh.size == 0:
                ap_values.append(0)
                start_y = end_y + spacing
                continue
            
            thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            
            # Save for debugging
            cv2.imwrite(f'debug_ap_{i+1}.png', thresh)
            
            # OCR
            text = pytesseract.image_to_string(thresh, config='--psm 7 digits')
            digits = ''.join(filter(str.isdigit, text))
            
            try:
                ap = int(digits) if digits else 0
            except:
                ap = 0
            
            ap_values.append(ap)
            start_y = end_y + spacing
        
        return ap_values
    
    def get_game_state(self):
        """Get complete game state using layout-based detection"""
        window_img = self.detector.capture_game_window(auto_focus=True)
        if window_img is None:
            return None, None
        
        viewport, bounds = self.extract_viewport(window_img)
        if viewport is None:
            return None, None
        
        # Count agents
        num_agents = self.count_agents(viewport)
        
        # Get AP for each agent
        ap_values = self.get_ap_values(viewport, num_agents)
        
        # Extract regions (for visualization/future use)
        selected_region, selected_coords = self.get_region(viewport, self.layout['selected_agent'])
        
        state = {
            'num_agents': num_agents,
            'ap_values': ap_values,
            'viewport_size': viewport.shape[:2],
            'selected_agent_region': selected_coords,
        }
        
        return state, viewport

# Test
if __name__ == "__main__":
    print("="*60)
    print("LAYOUT-BASED DETECTOR TEST")
    print("="*60)
    
    detector = LayoutBasedDetector()
    
    print("\n1. Launch Invisible Inc")
    print("2. Start a mission with agents visible")
    input("\nPress ENTER to detect...")
    
    state, viewport = detector.get_game_state()
    
    if state:
        print(f"\n✓ Detected {state['num_agents']} agents")
        print(f"  AP values: {state['ap_values']}")
        print(f"  Viewport size: {state['viewport_size']}")
        
        # Visualize layout
        viewport_vis = viewport.copy()
        h, w = viewport.shape[:2]
        
        # Draw all regions
        for name, region_def in detector.layout.items():
            if name == 'agent_icons':
                continue  # Skip for clarity
            
            x = int(w * region_def['x'])
            y = int(h * region_def['y'])
            width = int(w * region_def['width'])
            height = int(h * region_def['height'])
            
            color = (0, 255, 0) if name == 'selected_agent' else (255, 255, 0)
            cv2.rectangle(viewport_vis, (x, y), (x+width, y+height), color, 2)
            cv2.putText(viewport_vis, name, (x+5, y+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw agent icon locations
        icon_def = detector.layout['agent_icons']
        icon_x = int(w * icon_def['x'])
        icon_y_start = int(h * icon_def['y'])
        icon_w = int(w * icon_def['width'])
        icon_h = int(h * icon_def['icon_height'])
        spacing = int(h * icon_def['spacing'])
        
        for i in range(state['num_agents']):
            y = icon_y_start + i * (icon_h + spacing)
            cv2.rectangle(viewport_vis, (icon_x, y), (icon_x+icon_w, y+icon_h), (0, 255, 255), 2)
            cv2.putText(viewport_vis, f"Agent {i+1}: {state['ap_values'][i]} AP",
                       (icon_x+icon_w+5, y+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        cv2.imwrite('layout_visualization.png', cv2.cvtColor(viewport_vis, cv2.COLOR_RGB2BGR))
        print("\n✓ Saved layout_visualization.png")
        
        print("\n" + "="*60)
        print("CHECK: open layout_visualization.png")
        print("="*60)
    else:
        print("✗ Could not detect game state")
