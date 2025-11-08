# Alarm Clock Detection Fix - Wrong Circle Selected

## Critical Bug Identified

From `frame_000330_detection.png`, the green circle (detected clock) is at the **LEFT edge** of the search rectangle, not at the actual alarm clock in the far top-right corner.

**Root cause:** The detector was selecting the circle with the most colored pixels, which could be ANY circle in the search region. The actual alarm clock was being ignored in favor of some other circular shape.

## Fundamental Issue

**Before:** 14.5% clock detection rate (108 of 743 frames)
**Problem:** Detector finds wrong circles, not the actual alarm clock

The alarm clock is **always at the far right edge of the screen**, but the detector was selecting whichever circle happened to have the most yellow/orange/red pixels, regardless of position.

## Three Fixes Applied

### 1. Rightmost Circle Priority

**Before:** Selected circle with highest color ratio
```python
if color_ratio > best_color_ratio and color_ratio > 0.1:
    best_color_ratio = color_ratio
    best_circle = ((int(cx), int(cy)), int(r))
```

**After:** Select rightmost circle (with minimum color threshold)
```python
if color_ratio > 0.1 and cx > best_x:
    best_x = cx
    best_circle = ((int(cx), int(cy)), int(r))
```

**Why:** The alarm clock is always at the rightmost position in the UI. Other circular shapes may have more colored pixels, but the clock's position is deterministic.

### 2. Tighter Search Region

**Before:** x > 75% width, y < 15% height
**After:** x > 85% width, y < 12% height

**Why:** 
- Clock is always in far top-right corner
- Tighter region reduces false positives from other UI elements
- Focuses search on where clock actually appears

### 3. More Sensitive Hough Parameters

**Before:** 
- `minDist=50` (circles must be 50px apart)
- `param2=30` (accumulator threshold)

**After:**
- `minDist=30` (circles can be closer - clock is small)
- `param2=20` (lower threshold = more sensitive detection)

**Why:** The alarm clock is a small circle that may be close to other UI elements. Lower thresholds increase chance of detecting it.

## Expected Improvement

**Before:** 14.5% detection rate (108/743 frames)
**Target:** 70-85% detection rate

The clock should be visible in most gameplay frames except:
- Menu screens
- Dialog overlays
- Transitions
- Camera movements away from tactical view

By selecting the rightmost circle instead of the one with most colored pixels, we should correctly identify the actual alarm clock in frames where it's visible.

## Debug Verification

The debug script now shows:
- All detected circles with positions
- Color ratio for each circle
- Which circle is selected (rightmost with color)
- Visual markers on the image

Example output:
```
Found 3 circles
  Circle 0: center=(1200,50), radius=35, in_region=True, color_ratio=0.45 <- SELECTED (rightmost with color)
  Circle 1: center=(1100,60), radius=30, in_region=True, color_ratio=0.52
  Circle 2: center=(900,45), radius=28, in_region=False, color_ratio=0.31
```

Circle 0 is selected because it's the rightmost (x=1200), even though Circle 1 has higher color_ratio (0.52 vs 0.45).

## Test Instructions

```bash
# Debug frame 330 to verify correct clock selection
python debug_alarm.py captures/20251021_224508/frames/frame_000330.png
```

Check that:
1. Green circle is now on the actual alarm clock (far right)
2. Debug output shows correct circle selected as "rightmost with color"
3. Blue search rectangle contains the clock

Then run full test:
```bash
python test_alarm_level.py captures/20251021_224508/frames
```

Clock detection should jump from 14.5% to 70-85%.

## Technical Notes

### Why Not Position-Based Only?

We still require 10% colored pixels because:
- Ensures detected circle actually has colored segments
- Prevents selecting random circles with no alarm color
- Validates it's clock-like (has yellow/orange/red content)

### Why Rightmost Works

The UI layout is consistent:
- Alarm clock: Far top-right corner
- Hamburger menu: Below clock, but still right side
- Other UI: Left and bottom areas

By selecting rightmost circle, we naturally get the clock since nothing else is positioned further right.

### Alternative Approaches Considered

1. **Template matching**: Overkill for simple position-based selection
2. **Structural features**: Clock's ring pattern is visible, but position is simpler
3. **Multiple candidates**: Could track clock position over time, but rightmost is sufficient

The rightmost heuristic is simple, fast, and leverages the deterministic UI layout.
