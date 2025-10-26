import cv2

# Load your screenshots
agent_mode_img = cv2.imread('agent_mode.png')
incognita_mode_img = cv2.imread('incognita_mode.png')

if agent_mode_img is None or incognita_mode_img is None:
    print("ERROR: Could not load screenshots")
    print("Make sure 'agent_mode.png' and 'incognita_mode.png' are in the current directory")
    exit()

print(f"Agent mode image: {agent_mode_img.shape}")
print(f"Incognita mode image: {incognita_mode_img.shape}")

# Your measurements converted to (x, y, width, height)
# Format: (x, y, x2, y2) -> (x, y, width, height)

agent_mode_regions = {
    'Power': (0, 0, 261, 92),
    'Credits': (261, 0, 174, 92),  # width = 435-261
    'Incognita_button': (0, 92, 423, 171),  # height = 263-92
    'Tactical_button': (2330, 0, 438, 180),  # width = 2768-2330
    'Menu_button': (4970, 0, 150, 92),  # width = 5120-4970
    'Info_bar': (2768, 0, 2202, 92),  # width = 4970-2768
    'Alarm_integer': (4883, 247, 101, 121),  # width = 4984-4883, height = 368-247
    'Alarm_ring': (4770, 111, 323, 338),  # width = 5093-4770, height = 449-111
    'Daemons': (4777, 452, 334, 1548),  # width = 5111-4777, height = 2000-452
    'Agent_profile': (0, 2378, 424, 488),  # CORRECTED: height = 2866-2378
    'Agent_icons': (0, 1890, 231, 485),  # height = 2375-1890
    'Quick_actions': (425, 2688, 339, 141),  # width = 764-425, height = 2829-2688
    'Augments_Inventory': (425, 2533, 1325, 161),  # width = 1750-425, height = 2694-2533
    'End_turn': (4585, 2734, 535, 143),  # width = 5120-4585, height = 2877-2734
    'Rewind': (4378, 2735, 222, 142),  # CORRECTED: width = 4600-4378, height = 2877-2735
}

incognita_mode_regions = {
    'Programs': (0, 245, 900, 1298),  # CORRECTED: height = 1543-245
    'Incognita_profile': (0, 2378, 424, 488),  # CORRECTED: Same as agent profile
}

def draw_boxes(img, regions, title):
    """Draw labeled boxes on image"""
    img_annotated = img.copy()
    
    for name, (x, y, w, h) in regions.items():
        # Draw rectangle
        color = (0, 255, 0)  # Green
        cv2.rectangle(img_annotated, (x, y), (x+w, y+h), color, 4)
        
        # Draw label background
        label = name.replace('_', ' ')
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw black background for text
        cv2.rectangle(img_annotated, 
                     (x, y-text_h-10), 
                     (x+text_w+10, y), 
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(img_annotated, label, 
                   (x+5, y-5), 
                   font, font_scale, (0, 255, 0), thickness)
    
    return img_annotated

# Annotate images
print("\nAnnotating agent mode...")
agent_annotated = draw_boxes(agent_mode_img, agent_mode_regions, "Agent Mode")
cv2.imwrite('agent_mode_annotated.png', agent_annotated)
print("✓ Saved agent_mode_annotated.png")

print("\nAnnotating incognita mode...")
# Combine both common and incognita-specific regions
all_incognita_regions = {**agent_mode_regions, **incognita_mode_regions}
incognita_annotated = draw_boxes(incognita_mode_img, all_incognita_regions, "Incognita Mode")
cv2.imwrite('incognita_mode_annotated.png', incognita_annotated)
print("✓ Saved incognita_mode_annotated.png")

print("\n" + "="*60)
print("CHECK THE ANNOTATED IMAGES:")
print("="*60)
print("open agent_mode_annotated.png")
print("open incognita_mode_annotated.png")
print("\nVerify all green boxes are in the correct locations!")
