# Priority #1: Persistent UI Anchor Detectors

This package contains 5 detector classes for always-present UI elements that serve as spatial anchors for game state detection in Invisible Inc.

## 📦 What's Included

### Detector Classes (`anchor_detectors.py`)

1. **EndTurnDetector** - Cyan "END TURN" button (lower-right)
   - Success rate: ~97%+
   - Most reliable turn phase indicator
   - HSV cyan detection (H: 85-100°)

2. **HamburgerMenuDetector** - 3-line menu icon (upper-right)
   - Success rate: 78.2%
   - Horizontal line pattern matching
   - Always present in both game modes

3. **TacticalViewDetector** - Polygon at top center
   - Success rate: 80.9%
   - Edge detection + contour analysis
   - Defines top edge of viewport

4. **PowerCreditsAnchorDetector** - Cyan "PWR" text (upper-left)
   - Success rate: 77.2%
   - Primary anchor for power/credits region
   - Used by OCR extraction detectors

5. **SecurityClockDetector** - Red/orange circular clock (upper-right)
   - Success rate: 38.4% ⚠️ NEEDS IMPROVEMENT
   - Included for completeness
   - Alarm level extraction works better with viewport-relative positioning

## 🚀 Quick Start

### Single Image Test

```bash
python test_anchor_detectors.py path/to/frame.png
```

This will:
- Detect all 5 anchors in the image
- Print detection results with confidence scores
- Save visualization to `frame_anchors.png`

### Session Batch Test

```bash
python test_anchor_detectors.py --session captures/20251022_201216
```

This will:
- Process up to 50 frames from the session
- Calculate success rates for each anchor
- Print summary statistics

### Debug Mode

```bash
python test_anchor_detectors.py frame.png --debug
```

Prints detailed detection information for troubleshooting.

## 📋 Integration Example

```python
from detector_framework import DetectorRegistry
from anchor_detectors import (
    EndTurnDetector,
    PowerCreditsAnchorDetector,
    # ... import others as needed
)

# Create registry
registry = DetectorRegistry()

# Register anchor detectors
registry.register(EndTurnDetector())
registry.register(PowerCreditsAnchorDetector())
# ... register others

# Detect on an image
results = registry.detect_all(image)

# Access results
if results['end_turn'].success:
    bbox = results['end_turn'].data['bbox']
    confidence = results['end_turn'].confidence
    print(f"END TURN button at {bbox} (confidence: {confidence:.2f})")
```

## 🎯 Use Cases

### Viewport Detection

Combine multiple anchors to infer viewport boundaries:

```python
# Detect anchors
results = registry.detect_all(image)

# Use results to estimate viewport
left_edge = results['power_credits_anchor'].data['bbox'][0]
right_edge = results['hamburger_menu'].data['bbox'][0] + bbox[2]
top_edge = results['tactical_view'].data['bbox'][1]
bottom_edge = results['end_turn'].data['bbox'][1] + bbox[3]
```

### Turn Phase Detection

Use END TURN button presence to determine game phase:

```python
if results['end_turn'].success:
    print("Player has control")
else:
    print("Opponent turn or game transition")
```

### Spatial Reference

Use anchors as reference points for detecting other UI elements:

```python
# Find power value relative to power_credits_anchor
pwr_anchor = results['power_credits_anchor'].data['bbox']
# OCR region is relative to this anchor...
```

## 📊 Expected Performance

Based on testing across 711 frames:

| Detector | Success Rate | Notes |
|----------|--------------|-------|
| END TURN | ~97%+ | Excellent, most reliable |
| Power/Credits | 77.2% | Good baseline |
| Hamburger Menu | 78.2% | Good baseline |
| Tactical View | 80.9% | Good baseline |
| Security Clock | 38.4% | Needs improvement |

**Baseline ~75-80%** is expected due to:
- Opponent turns (UI changes)
- Game mode switches (mainframe view)
- Dialog overlays (dimmed UI)
- Transition frames

## 🔧 Configuration

Each detector can be configured via `DetectorConfig`:

```python
from detector_framework import DetectorConfig

config = DetectorConfig(
    name="end_turn",
    type=DetectorType.STRUCTURAL,
    params={
        'hue_min': 85,
        'hue_max': 100,
        'sat_min': 100,
        'val_min': 100,
        'min_area': 1000
    }
)

detector = EndTurnDetector(config)
```

## 🐛 Known Issues

### Security Clock (38.4% success rate)

The security clock detector has reliability issues:
- Red/orange color detection is sensitive to lighting
- Circular shape detection can be confused by game elements
- Arc segments may not be contiguous

**Workaround**: Use viewport-relative positioning for alarm level extraction instead of depending on security clock detection.

## 📁 Files

- `anchor_detectors.py` - Detector class implementations
- `test_anchor_detectors.py` - Test and demonstration script
- `ANCHOR_DETECTORS_README.md` - This file

## 🔄 Dependencies

Requires the modular detector framework:
- `detector_framework.py` - Core framework classes

Standard libraries:
- OpenCV (`cv2`)
- NumPy
- Pathlib (built-in)

## 📝 Notes

- All detectors return standardized `DetectionResult` objects
- Bounding boxes use (x, y, width, height) format
- Confidence scores range from 0.0 to 1.0
- Failed detections have `success=False` and may include error messages
- Detectors are stateless and can process frames independently

## ➡️ Next Steps

After anchors are working, proceed to:
1. **Priority #2**: Power and Credits OCR extraction
2. **Priority #3**: Alarm level extraction
3. **Priority #4**: Agent status detection

These higher-level detectors will depend on the anchor results for spatial reference.
