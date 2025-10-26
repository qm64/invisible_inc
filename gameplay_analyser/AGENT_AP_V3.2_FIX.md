# Agent AP Extractor v3.2 - Position Correction

## The Bug in v3.0/v3.1

**Problem:** I incorrectly "adjusted" the measured starting position when adding padding.

**v2 (working):**
```python
start_y = profile_y - int(profile_h * 0.304)  # 70/230 = y=1120 ✓ (measured)
ap_height = 25px
```

**v3.0/v3.1 (broken):**
```python
start_y = profile_y - int(profile_h * 0.348)  # 80/230 = y=1110 ✗ (moved above text!)
ap_height = 35px
```

By moving the starting position from y=1120 to y=1110, I placed the ROI **10 pixels above** where the actual text is located.

## The Fix in v3.2

**v3.2 (corrected):**
```python
start_y = profile_y - int(profile_h * 0.304)  # 70/230 = y=1120 ✓ (measured)
ap_height = 35px  # Provides padding around text
```

## Why This Works

The taller ROI (35px instead of 25px) provides padding **within the ROI** without needing to move the starting position.

**Text layout in 35px ROI starting at y=1120:**
- Pixels 1120-1125: Padding above text (5-6 pixels)
- Pixels 1125-1142: Actual text (17 pixels)
- Pixels 1142-1155: Padding below text (13 pixels)

This gives OCR the breathing room it needs while keeping the text in the correct measured position.

## Lesson Learned

**"Don't adjust measurements - trust them!"**

When adding padding for OCR:
1. **Measure where the text actually is** ✓
2. **Make the ROI taller/wider** ✓
3. **DON'T move the starting position** ✗ (my mistake)

The padding comes from the increased ROI size, not from repositioning.

## Test Results Expected

With v3.2, testing on frames 49, 122, 124 should now:
- Start ROI at correct position (y=1120 for Agent 0)
- Capture numbers that were previously clipped
- Improve success rate from 38% to ~50-55%
