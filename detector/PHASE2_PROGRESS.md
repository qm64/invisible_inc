# Phase 2 Progress: Parallel Processing and Alarm Level Detection

## What's New

### 1. Parallel Processing for Resources Extraction

**File:** `test_resources_extractors.py`

Added parallel processing to the resources extraction testing with the following features:

- **Default workers:** N-2 (where N = available CPUs)
- **Command line option:** `--cpus k` to specify number of workers
- **Automatic validation:** Ensures worker count is between 1 and max CPUs

**Usage:**
```bash
# Use default (N-2) workers
python test_resources_extractors.py captures/20251021_224508

# Use specific number of workers
python test_resources_extractors.py captures/20251021_224508 --cpus 4

# Use all CPUs
python test_resources_extractors.py captures/20251021_224508 --cpus 10
```

**Output:**
- Total frames processed
- Anchor detection rate
- Power extraction rate
- Credits extraction rate
- Both extracted rate
- Sample of failed frames

### 2. Alarm Level Detection

**File:** `alarm_level_extractor.py`

New extractor that captures security alarm data:

**Major Alarm Level (Priority):**
- Single digit 0-6 from center of clock
- Uses OCR with multiple preprocessing strategies
- Tesseract with digit whitelist (0123456)

**Minor Alarm Level:**
- Counts filled segments (0-4 out of 5 total)
- Samples at 5 positions around clock circumference
- Detects yellow/orange/red colored segments

**Detection Strategy:**
1. Find security clock in top-right region (x > 75% width, y < 15% height)
2. Use color detection (yellow/orange/red HSV ranges)
3. Confirm with circle detection (Hough transform)
4. Extract center digit (OCR with upscaling)
5. Count filled segments (color sampling)

**Standalone Testing:**
```bash
# Test on single frame with visualization
python alarm_level_extractor.py captures/20251021_224508/frame_000103.png
```

### 3. Parallel Alarm Level Testing

**File:** `test_alarm_level.py`

Comprehensive testing script with parallel processing:

**Usage:**
```bash
# Use default workers
python test_alarm_level.py captures/20251021_224508

# Specify workers
python test_alarm_level.py captures/20251021_224508 --cpus 6
```

**Output:**
- Clock detection rate
- Major alarm extraction rate
- Minor alarm extraction rate
- Distribution of alarm levels (0-6)
- Distribution of segment fills (0-4)
- Average confidence scores
- Failed extraction samples

## Implementation Details

### HSV Color Ranges (Alarm Clock)

```python
# Yellow (low alarm): H 20-60°
yellow_lower = [20, 100, 100]
yellow_upper = [60, 255, 255]

# Orange (medium alarm): H 0-20°
orange_lower = [0, 100, 100]
orange_upper = [20, 255, 255]

# Red (high alarm): H 160-180°
red_lower = [160, 100, 100]
red_upper = [180, 255, 255]
```

### Clock Segment Positions

5 segments arranged in circle, starting from top:
- Top: -90°
- Top-right: -18°
- Bottom-right: 54°
- Bottom-left: 126°
- Top-left: 198°

Sampled at 70% of clock radius.

### OCR Configuration

```python
tesseract_config = '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456'
# PSM 10: Single character mode
# OEM 3: Default OCR engine
# Whitelist: Only digits 0-6
```

## Testing Strategy

1. **Start with single frame:**
   ```bash
   python alarm_level_extractor.py path/to/frame.png
   ```
   This shows visualization and detailed output.

2. **Run full session test:**
   ```bash
   python test_alarm_level.py captures/20251021_224508
   ```
   This provides statistics across all frames.

3. **Investigate failures:**
   - Check frames listed in "Clock found but major extraction failed"
   - Examine alarm level distribution to verify expected values
   - Review confidence scores to identify threshold issues

## Expected Challenges

Based on resources extraction experience:

1. **OCR Reliability:** 
   - Center digit may be small/dark
   - May need multiple preprocessing approaches
   - Could require brightness/contrast adjustment

2. **Segment Detection:**
   - Segments may not be uniformly colored
   - Edge cases at segment boundaries
   - Overlapping UI elements could interfere

3. **Clock Location:**
   - Position may vary slightly across frames
   - May be obscured by dialog boxes
   - Color intensity varies with alarm level

## Next Steps

1. Run `test_alarm_level.py` on your existing session
2. Check success rates for major and minor extraction
3. Review failed frames to identify patterns
4. Adjust parameters based on results:
   - OCR preprocessing if digit extraction fails
   - Segment sampling positions if count is incorrect
   - Clock detection thresholds if location fails

## Performance Notes

With parallel processing (N-2 workers on 10-core system):
- Expected throughput: ~30-40 fps
- 743-frame session: ~18-25 seconds
- Memory usage: ~200-300MB per worker

The major alarm level (0-6 digit) is prioritized as most important - success is based on extracting this value, with minor alarm as supplementary data.
