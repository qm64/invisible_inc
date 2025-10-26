import Quartz

search_term = "Invisible"

window_list = Quartz.CGWindowListCopyWindowInfo(
    Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
    Quartz.kCGNullWindowID
)

print("="*60)
print(f"DEBUG: Searching for '{search_term}'")
print("="*60)

for window in window_list:
    owner = window.get('kCGWindowOwnerName', '')
    name = window.get('kCGWindowName', '')
    bounds = window.get('kCGWindowBounds', {})
    width = bounds.get('Width', 0)
    height = bounds.get('Height', 0)
    
    # Only check substantial windows
    if width > 100 and height > 100:
        owner_lower = owner.lower()
        name_lower = name.lower()
        search_lower = search_term.lower()
        
        owner_match = search_lower in owner_lower
        name_match = search_lower in name_lower
        
        if owner_match or name_match:
            print(f"\n✓ MATCH FOUND!")
            print(f"  Owner: '{owner}'")
            print(f"  Name: '{name}'")
            print(f"  Owner contains '{search_term}': {owner_match}")
            print(f"  Name contains '{search_term}': {name_match}")
            print(f"  Size: {width}x{height}")