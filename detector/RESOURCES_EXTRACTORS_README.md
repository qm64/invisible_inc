## Resources Extractors - Power and Credits OCR Detection

This document describes the `PowerExtractor` and `CreditsExtractor` detectors that extract resource values from the Invisible Inc game UI using OCR.

### Version: 1.0.0

---

## Overview

The resources extractors read power and credits values from the upper-left corner of the game UI. They depend on the `PowerCreditsAnchorDetector` to locate the text region, then use OCR with multiple thresholding approaches to extract the numeric values.

**Format:** `XX/YY PWR ZZZZZ`
- **XX/YY** = Current/Max Power (extracted by PowerExtractor)
- **PWR** = Cyan anchor text (used for detection)
- **ZZZZZ** = Credits amount (extracted by CreditsExtractor)

**Expected Success Rate:** ~75.9% 
- Failures occur during opponent turns, dialog overlays, and transitions
- This baseline is normal for OCR-based extractors in dynamic game states

---

## Architecture

### Detector Hierarchy

```
PowerCreditsAnchorDetector (Priority 1)
├── Detects cyan "PWR" text in upper-left
├── Returns bounding box of entire region
└── Success rate: ~77.2%

PowerExtractor (Priority 2)
├── Depends on: power_credits_anchor
├── OCRs the region to extract "XX/YY"
└── Success rate: ~75.9%

CreditsExtractor (Priority 2)
├── Depends on: power_credits_anchor  
├── OCRs the region to extract credits number
└── Success rate: ~75.9%
```

### Detection Strategy

Both extractors use the same multi-stage OCR approach:

1. **Region Extraction**
   - Get bounding box from anchor detector
   - Add 5px vertical padding for better text capture
   - Extract region from image

2. **Preprocessing**
   - Upscale 5× using cubic interpolation (for small text)
   - Convert to grayscale
   - Apply 3 different thresholding methods:
     - Simple binary threshold (T=100)
     - Otsu's adaptive threshold
     - Gaussian adaptive threshold

3. **OCR Extraction**
   - Run tesseract on each threshold variant
   - Use PSM mode 7 (single text line)
   - Whitelist: `0123456789/PWR `

4. **Value Parsing**
   - **Power**: Extract `(\d+)/(\d+)` pattern (e.g. "10/20")
   - **Credits**: Extract number after "PWR", or find longest number

---

## Usage

### Single Frame Testing

```bash
# Basic test
python test_resources_extractors.py frame.png

# With debug output
python test_resources_extractors.py frame.png --debug

# With validation
python test_resources_extractors.py frame.png --power 10/20 --credits 72314
```

### Session Testing

```bash
# Test entire session
python test_resources_extractors.py --session captures/20251022_201216

# Limit number of frames
python test_resources_extractors.py --session captures/20251022_201216 --max-frames 100

# Save results to JSON
python test_resources_extractors.py --session captures/20251022_201216 --output results.json
```

### Programmatic Usage

```python
import cv2
from detector_framework import DetectorRegistry
from anchor_detectors import PowerCreditsAnchorDetector
from resources_extractors import PowerExtractor, CreditsExtractor

# Setup registry
registry = DetectorRegistry()
registry.register(PowerCreditsAnchorDetector())
registry.register(PowerExtractor())
registry.register(CreditsExtractor())

# Load image
image = cv2.imread('frame.png')

# Run detection
results = registry.detect_all(image)

# Extract values
if results['power_extractor'].success:
    power = results['power_extractor'].data['power']
    print(f"Power: {power}")

if results['credits_extractor'].success:
    credits = results['credits_extractor'].data['credits']
    credits_int = results['credits_extractor'].data['credits_int']
    print(f"Credits: {credits} ({credits_int:,})")
```

---

## Configuration

### PowerExtractor Parameters

```python
config = DetectorConfig(
    name="power_extractor",
    type=DetectorType.OCR,
    dependencies=["power_credits_anchor"],
    params={
        'vertical_padding': 5,     # Pixels to add above/below text
        'upscale_factor': 5,       # Upscaling for better OCR
        'psm_mode': 7,             # Tesseract PSM mode (7 = single line)
        'whitelist': '0123456789/PWR '  # Allowed characters
    }
)
detector = PowerExtractor(config)
```

### CreditsExtractor Parameters

Same parameters as PowerExtractor. Both share the same OCR preprocessing pipeline.

---

## Output Format

### PowerExtractor Result

```python
{
    'success': True,
    'confidence': 0.8,
    'data': {
        'power': '10/20',      # String format
        'raw_format': '10/20'  # Same as power
    },
    'error': None
}
```

### CreditsExtractor Result

```python
{
    'success': True,
    'confidence': 0.8,
    'data': {
        'credits': '72314',     # String format
        'credits_int': 72314    # Integer format
    },
    'error': None
}
```

### Error Result

```python
{
    'success': False,
    'confidence': 0.0,
    'data': {},
    'error': 'Power/credits anchor not detected'
}
```

---

## Common Failure Modes

### 1. Opponent Turn (Expected)
**Symptom:** Both extractors fail together  
**Cause:** Power/credits UI hidden during opponent turns  
**Frequency:** ~13% of frames (normal gameplay)  
**Action:** This is expected behavior, not a bug

### 2. Dialog Overlays (Expected)
**Symptom:** Anchor detected but OCR fails  
**Cause:** UI elements dimmed behind dialog  
**Frequency:** ~5-10% when dialogs present  
**Action:** Expected, values resume after dialog closes

### 3. Transition Frames (Expected)
**Symptom:** Intermittent failures at phase boundaries  
**Cause:** UI elements fading in/out  
**Frequency:** ~2-3% during transitions  
**Action:** Normal, values stabilize quickly

### 4. OCR Misread (Rare)
**Symptom:** Extracted value doesn't match visual  
**Cause:** Poor threshold selection or lighting  
**Frequency:** ~1-2% of successful detections  
**Action:** Review with `--debug`, adjust thresholds if needed

---

## Testing Guidelines

### Validation Approach

1. **Test on known frames** - Use frames where you know the correct values
2. **Session testing** - Run on full 711-frame sessions to get baseline rate
3. **Expected baseline** - ~75.9% success is normal and acceptable
4. **Debug mode** - Use `--debug` to see OCR attempts and thresholding results

### Success Criteria

- **Anchor detection:** >75% (depends on game state distribution)
- **Power extraction:** >75% when anchor present
- **Credits extraction:** >75% when anchor present  
- **Both values:** >75% when anchor present

If rates fall significantly below these baselines, investigate:
- Tesseract installation
- Image quality/resolution
- Threshold parameters
- Game UI changes (updates/mods)

---

## Troubleshooting

### "pytesseract not installed"

```bash
pip install pytesseract --break-system-packages

# macOS
brew install tesseract

# Ubuntu
sudo apt-get install tesseract-ocr

# Verify installation
tesseract --version
```

### "Missing dependency: power_credits_anchor"

Ensure PowerCreditsAnchorDetector is registered before the extractors:

```python
registry = DetectorRegistry()
registry.register(PowerCreditsAnchorDetector())  # MUST be first
registry.register(PowerExtractor())
registry.register(CreditsExtractor())
```

### Low Success Rate (<60%)

1. Check if anchor detection rate is also low
2. Verify image quality (not scaled/compressed)
3. Try adjusting `upscale_factor` (3-7 range)
4. Test with `--debug` to see OCR output
5. Check if game UI has changed (updates/mods)

### Values Don't Match Visual

1. Enable `--debug` mode to see OCR attempts
2. Check if multiple thresholds agree
3. Verify the region bounding box is correct
4. Consider adjusting threshold parameters

---

## Integration with Existing System

These extractors integrate seamlessly with Priority 1 anchor detectors:

```python
# Complete setup with all Priority 1 + 2 detectors
from detector_framework import DetectorRegistry
from anchor_detectors import (
    EndTurnDetector,
    HamburgerMenuDetector,
    TacticalViewDetector,
    PowerCreditsAnchorDetector,
    SecurityClockDetector
)
from resources_extractors import PowerExtractor, CreditsExtractor

registry = DetectorRegistry()

# Priority 1: Anchors
registry.register(EndTurnDetector())
registry.register(HamburgerMenuDetector())
registry.register(TacticalViewDetector())
registry.register(PowerCreditsAnchorDetector())
registry.register(SecurityClockDetector())

# Priority 2: Resource extractors
registry.register(PowerExtractor())
registry.register(CreditsExtractor())

# Run all detectors
results = registry.detect_all(image)
```

---

## Next Steps

After validating the resources extractors, proceed to:

**Priority 3:** Alarm Level Detector (~78.8% baseline)
- OCR extraction of alarm level from security clock
- Depends on SecurityClockDetector

**Priority 4:** Agent AP Status (~49.9% baseline)  
- OCR extraction of agent action points
- Requires agent icon detection

---

## File Locations

```
resources_extractors.py          - PowerExtractor and CreditsExtractor classes
test_resources_extractors.py     - Testing script with validation
RESOURCES_EXTRACTORS_README.md   - This documentation file
```

---

## Version History

### 1.0.0 (2025-11-01)
- Initial implementation
- PowerExtractor with multi-threshold OCR
- CreditsExtractor with fallback parsing
- Comprehensive test script
- Full documentation

---

## References

- **Previous work:** structural_detector.py v1.2.6 (extract_power_value, extract_credits_value methods)
- **Success rate:** Based on 711-frame session testing
- **Framework:** Integrates with detector_framework.py v1.0.0
- **Dependencies:** PowerCreditsAnchorDetector (Priority 1)
