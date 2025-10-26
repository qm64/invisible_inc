# Alarm Level Extractor v2 - Multi-Resolution Support

## Problem Identified

The original extractor was calibrated for 2000x1125 resolution (what I see after upload resizing), but your actual captures are 2560x1440 (and some 2540x1310). This caused the 67.8% failure rate.

## Solution

Created v2 with **resolution detection** and appropriate positioning for each resolution:

### Resolution-Specific Positioning

**2560x1440 / 2540x1310:**
- Search center: 98.3% across, 6% down
- Region size: 100x100 pixels
- Covers alarm positions from 1.7-10% vertical

**2000x1125:**
- Search center: 97.3% across, 9.7% down  
- Region size: 50x50 pixels
- Original calibration

## Files Updated

1. **[alarm_level_extractor_v2.py](computer:///mnt/user-data/outputs/alarm_level_extractor_v2.py)** - Standalone extractor with multi-resolution support
2. **[batch_test_alarm_v2.py](computer:///mnt/user-data/outputs/batch_test_alarm_v2.py)** - Updated batch test script

## Testing Instructions

From your `~/git/qm/invisible_inc/gameplay_analyser` directory:

```bash
# Test on the 711-frame session (2560x1440)
python batch_test_alarm_v2.py captures/20251022_201216
```

## Expected Results

**Old version:** 67.8% success rate  
**New version:** Should be 85-95% (closer to viewport detection rate)

The remaining failures will likely be:
- Opponent turn frames (no UI)
- Dialog/menu overlays
- Pre/post mission screens
- Special UI states

## Resolution Distribution

From your 711-frame session (20251022_201216):
- 655 frames at 2560x1440 (92%)
- 56 frames at 2540x1310 (8%)

Both are now supported.

## Next Steps

1. **Run the v2 batch test** on your 711-frame session
2. **Report the new success rate** - should be much better than 67.8%
3. **Check failed frames** - if specific patterns emerge, we can refine further
4. **Consider adding parallelization** - as you suggested, this would speed up batch testing 5-8x

## Integration

Once we validate the success rate, you can integrate v2 into your structural_detector.py by:
1. Replacing the `extract_alarm_level()` method
2. Updating version to 1.3.2
3. Adding resolution detection logic

## Why Upload Resizing Mattered

When you upload images to this chat, they get resized to ~2000x1125 for display. This meant:
- I was testing on **resized versions** (2000x1125)
- You were testing on **originals** (2560x1440)
- The positioning was correct for resized, wrong for originals

Now v2 handles both!
