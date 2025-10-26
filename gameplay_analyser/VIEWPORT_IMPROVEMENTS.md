# Structural Detector v1.3.0 - Viewport Inference Improvements

## Problem Solved

**Before v1.3.0:**
- Opponent turn frames had **0% viewport detection** success
- Relied solely on `power_text` as primary anchor
- Power/credits text sometimes missing during opponent turns
- Fallback method required 3+ edges, too strict
- No temporal consistency between frames

**Results on 12 test frames:**
- Player frames (5): ✅ 100% viewport detection
- Opponent frames (7): ❌ 0% viewport detection
- Overall: 41.7% success rate

## Solution: Multi-Strategy Viewport Inference

### Strategy 1: Power/Credits Text (Primary)
**When present** (player planning frames):
- Most accurate method
- Uses PWR/CREDITS text position to infer all edges
- Same as before, just prioritized

### Strategy 2: Always-Present Anchors (New!)
**For opponent turns** when power_text missing:
- Uses **hamburger_menu** (100% detected) in upper-right
- Uses **tactical_view** (100% detected) in top-center
- Optionally uses **security_clock** to refine
- These elements are present in ALL frames (player + opponent)

**Logic:**
```python
# Hamburger right edge → viewport right edge
viewport_right = hamburger.bbox[0] + hamburger.bbox[2]

# Tactical view top → viewport top edge  
viewport_top = tactical.bbox[1]

# Symmetry assumption: equal margins left/right
viewport_left = img_width - viewport_right

# Use last known height or estimate
viewport_height = last_known_viewport.height OR (img_height - top)
```

### Strategy 3: Temporal Consistency (New!)
**Frame-to-frame stability:**
- Viewport doesn't change between frames (unless user resizes)
- If detection fails, use **last known viewport**
- Stored as class variable `_last_known_viewport`
- Survives across frames in same session

**Benefits:**
- Handles brief detection failures gracefully
- Opponent turn frames can inherit from nearby player frames
- Forgives occasional missed detections

### Strategy 4: Relaxed Fallback
**Reduced requirements:**
- Old: needed 3+ edges
- New: only needs 2 edges if we have **top + right**
- Uses symmetry to estimate missing edges
- More forgiving on partial detections

## Usage

### Single Frame (as before)
```python
from structural_detector import StructuralDetector
import cv2

detector = StructuralDetector(debug=True)
image = cv2.imread('frame_000032.png')
elements = detector.detect_anchors(image)
viewport = detector.infer_viewport()
```

### Frame Sequence (NEW - Recommended)
```python
from structural_detector import StructuralDetector
import cv2
from pathlib import Path

# Reset cache at start of session
StructuralDetector.reset_temporal_cache()

detector = StructuralDetector(debug=False)

for frame_path in sorted(Path('frames').glob('frame_*.png')):
    image = cv2.imread(str(frame_path))
    elements = detector.detect_anchors(image)
    viewport = detector.infer_viewport()  # Uses temporal consistency!
    
    # Process frame...
```

### Batch Processing Script (NEW)
```bash
python batch_structural_detector.py captures/20251022_201216/frames

# With custom output directory
python batch_structural_detector.py captures/20251022_201216/frames output_detections

# With debug output
python batch_structural_detector.py captures/20251022_201216/frames --debug
```

**Features:**
- Automatically resets temporal cache at start
- Processes frames in order (maintains temporal consistency)
- Shows progress and statistics
- Saves visualizations and JSON for each frame

## Expected Improvements

**Predicted results on your 12 test frames:**

| Frame Type | Count | Old Viewport | New Viewport | Method |
|------------|-------|--------------|--------------|--------|
| Player planning | 5 | 100% | 100% | Power text |
| Opponent turn | 7 | 0% | **~85-100%** | Hamburger+tactical OR temporal |

**Overall expected:**
- Old: 41.7% viewport detection (5/12)
- New: **~90-100%** viewport detection (11-12/12)

**Why not 100%?**
- First opponent frame may fail if no prior player frame
- Rare cases where hamburger/tactical also fail
- But temporal cache will recover quickly

## When Viewport Detection Still Fails

**Acceptable failure scenarios:**
1. **First frame is opponent turn** - no prior viewport to inherit
2. **User resizes window during opponent turn** - will recover next player frame
3. **Hamburger + tactical both fail** - extremely rare given 100% detection rate

**In these cases:**
- Detector gracefully returns `None`
- Will recover on next frame with better anchors
- User accepts brief detection gaps during window resizing

## Testing

To verify improvements, re-run analysis on your 12 frames:

```bash
# Re-test with new version
./test_structural_detector.sh captures/20251022_201216 ./structural_detector.py

# Analyze results
python analyze_structural_results.py captures/20251022_201216/structural_test_results
```

**Expected outcome:**
- Opponent frames (31, 32, 33, 34, 75, 129, 130) should now have viewport detected
- More UI elements detected once viewport is available
- Detection rates for end_turn, objectives, power_text should improve on frames where present

## Files Included

1. **structural_detector.py** - Updated v1.3.0 with improved viewport inference
2. **batch_structural_detector.py** - NEW batch processor with temporal consistency
3. **VIEWPORT_IMPROVEMENTS.md** - This documentation

## Changelog v1.3.0

- ✅ Added temporal viewport caching across frames
- ✅ New viewport inference from hamburger + tactical_view
- ✅ Relaxed fallback requirements (2 edges instead of 3)
- ✅ Added `reset_temporal_cache()` class method
- ✅ Better handling of opponent turn frames
- ✅ Batch processing script with proper session handling

**Expected impact:**
- Viewport detection: 41.7% → ~95%+ 
- Opponent turn element detection: massively improved
- More robust to missing primary anchors
