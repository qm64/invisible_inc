import Quartz

print("="*60)
print("LIST ALL WINDOWS")
print("="*60)

window_list = Quartz.CGWindowListCopyWindowInfo(
    Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
    Quartz.kCGNullWindowID
)

print("\nFound windows:\n")

for i, window in enumerate(window_list):
    owner = window.get('kCGWindowOwnerName', 'Unknown')
    name = window.get('kCGWindowName', '')
    bounds = window.get('kCGWindowBounds', {})
    width = bounds.get('Width', 0)
    height = bounds.get('Height', 0)
    
    # Only show substantial windows
    if width > 100 and height > 100:
        print(f"{i+1}. Owner: '{owner}'")
        if name:
            print(f"   Name: '{name}'")
        print(f"   Size: {width}x{height}")
        print()

print("="*60)
print("\nLook for Invisible Inc in the list above!")
print("Note the EXACT owner name and window name.")
