from window_detector import WindowDetector
import cv2
import numpy as np

class AgentDetector:
    """Detects agent icons and selected agent"""
    
    def __init__(self):
        self.detector = WindowDetector()
    
    def extract_viewport(self, window_img):
        """Extract game viewport from window"""
        green = window_img[:, :, 1]
        green_threshold = cv2.threshold(green, 80, 255, cv2.THRESH_BINARY)[1]
        rows_with_green = np.any(green_threshold > 0, axis=1)
        cols_with_green = np.any(green_threshold > 0, axis=0)
        
        if not np.any(rows_with_green) or not np.any(cols_with_green):
            return None
        
        top = np.argmax(rows_with_green)
        bottom = len(rows_with_green) - np.argmax(rows_with_green[::-1])
        left = np.argmax(cols_with_green)
        right = len(cols_with_green) - np.argmax(cols_with_green[::-1])
        
        margin = 5
        top = max(0, top - margin)
        left = max(0, left - margin)
        bottom = min(window_img.shape[0], bottom + margin)
        right = min(window_img.shape[1], right + margin)
        
        return window_img[top:bottom, left:right]
    
    def extract_agent_panel(self, viewport):
        """Extract the agent panel region (lower-left)"""
        viewport_h, viewport_w = viewport.shape[:2]
        
        # Agent panel: left 15%, bottom 35%
        panel_y = int(viewport_h * 0.65)
        panel_h = int(viewport_h * 0.35)
        panel_w = int(viewport_w * 0.15)
        
        return viewport[panel_y:panel_y+panel_h, 0:panel_w]
    
    def merge_overlapping_agents(self, agents):
        """
        Merge agents that overlap or are very close together
        Uses multiple passes to catch all mergeable groups
        """
        if len(agents) <= 1:
            return agents
        
        # Keep merging until no more merges happen
        changed = True
        passes = 0
        max_passes = 5
        
        while changed and passes < max_passes:
            changed = False
            passes += 1
            
            merged = []
            used = set()
            
            for i, agent1 in enumerate(agents):
                if i in used:
                    continue
                
                # Start with this agent
                merged_box = {
                    'x': agent1['x'],
                    'y': agent1['y'],
                    'x2': agent1['x'] + agent1['width'],
                    'y2': agent1['y'] + agent1['height'],
                    'area': agent1['area']
                }
                used.add(i)
                
                # Check if any other agents overlap or are nearby
                for j, agent2 in enumerate(agents):
                    if j in used:
                        continue
                    
                    a2_x2 = agent2['x'] + agent2['width']
                    a2_y2 = agent2['y'] + agent2['height']
                    
                    # Check if boxes overlap or are very close (within 30px)
                    margin = 30
                    overlaps = not (merged_box['x2'] + margin < agent2['x'] or
                                a2_x2 + margin < merged_box['x'] or
                                merged_box['y2'] + margin < agent2['y'] or
                                a2_y2 + margin < merged_box['y'])
                    
                    if overlaps:
                        # Merge this agent into the merged box
                        merged_box['x'] = min(merged_box['x'], agent2['x'])
                        merged_box['y'] = min(merged_box['y'], agent2['y'])
                        merged_box['x2'] = max(merged_box['x2'], a2_x2)
                        merged_box['y2'] = max(merged_box['y2'], a2_y2)
                        merged_box['area'] += agent2['area']
                        used.add(j)
                        changed = True  # We merged something, need another pass
                
                # Convert back to agent format
                width = merged_box['x2'] - merged_box['x']
                height = merged_box['y2'] - merged_box['y']
                is_selected = merged_box['area'] > 5000
                
                merged.append({
                    'x': merged_box['x'],
                    'y': merged_box['y'],
                    'width': width,
                    'height': height,
                    'area': merged_box['area'],
                    'is_selected': is_selected
                })
            
            agents = merged
        
        return agents

    def detect_agents(self, agent_panel):
        """
        Detect all agent icons in the panel
        Returns: list of (x, y, width, height, is_selected)
        """
        panel_h, panel_w = agent_panel.shape[:2]
        
        # Threshold green channel to find bright areas (agent icons)
        green_channel = agent_panel[:, :, 1]
        _, mask = cv2.threshold(green_channel, 100, 255, cv2.THRESH_BINARY)
        
        # Morphological closing to connect nearby regions (merge portrait parts)
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Save debug mask
        cv2.imwrite('debug_mask_closed.png', mask)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        agents = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size
            if area < 150:  # Too small (noise)
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # CRITICAL: Agent icons should be on the LEFT side of the panel
            # Ignore anything on the right half (that's augments/actions)
            if x > panel_w * 0.5:  # Right half of panel
                continue
            
            # Check if roughly square/circular (agent icons)
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.4 or aspect_ratio > 2.5:  # Allow slightly more variation
                continue
            
            # Classify as selected (large) or regular (small)
            # Selected agent is MUCH larger
            is_selected = area > 5000
            
            agents.append({
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'area': area,
                'is_selected': is_selected
            })
        
        # Sort by Y coordinate (top to bottom)
        agents.sort(key=lambda a: a['y'])
        
        return agents

    def get_agents(self):
        """Get all agent information"""
        window_img = self.detector.capture_game_window(auto_focus=False)
        if window_img is None:
            return None
        
        viewport = self.extract_viewport(window_img)
        if viewport is None:
            return None
        
        agent_panel = self.extract_agent_panel(viewport)
        agents = self.detect_agents(agent_panel)
        
        return {
            'agents': agents,
            'agent_count': len(agents),
            'selected_agent': next((i for i, a in enumerate(agents) if a['is_selected']), None)
        }

# Test
if __name__ == "__main__":
    print("="*60)
    print("AGENT DETECTOR TEST")
    print("="*60)
    
    detector = AgentDetector()
    
    print("\n1. Launch Invisible Inc")
    print("2. Start a mission with agents visible")
    input("\nPress ENTER to detect agents...")
    
    # Get window and viewport
    window_img = detector.detector.capture_game_window(auto_focus=True)
    viewport = detector.extract_viewport(window_img)
    agent_panel = detector.extract_agent_panel(viewport)
    
    # Detect agents
    agents = detector.detect_agents(agent_panel)
    
    print(f"\nFound {len(agents)} agents:")
    for i, agent in enumerate(agents):
        agent_type = "SELECTED" if agent['is_selected'] else "regular"
        print(f"  Agent {i+1}: ({agent['x']},{agent['y']}) size={agent['width']}x{agent['height']} area={agent['area']} [{agent_type}]")
    
    # Visualize
    agent_panel_annotated = agent_panel.copy()
    for i, agent in enumerate(agents):
        color = (255, 0, 0) if agent['is_selected'] else (0, 255, 0)  # Red for selected, green for regular
        cv2.rectangle(agent_panel_annotated, 
                     (agent['x'], agent['y']), 
                     (agent['x'] + agent['width'], agent['y'] + agent['height']), 
                     color, 2)
        label = f"{i+1}" + (" SEL" if agent['is_selected'] else "")
        cv2.putText(agent_panel_annotated, label, 
                   (agent['x']+5, agent['y']+15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite('agents_detected_final.png', cv2.cvtColor(agent_panel_annotated, cv2.COLOR_RGB2BGR))
    print("\n✓ Saved agents_detected_final.png")
    
    print("\n" + "="*60)
    print("CHECK: open agents_detected_final.png")
    print("="*60)
    print("\nGreen boxes = regular agents")
    print("Red box = selected agent")
