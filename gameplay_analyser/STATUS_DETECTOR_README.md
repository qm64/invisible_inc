# Game Status Detector for Invisible Inc

Extracts detailed game state information from captured gameplay frames.

## What It Detects

### 1. Resources
- **Power**: Current/Max (e.g., "12/15 PWR")
- **Credits**: Current amount (e.g., "250")

### 2. Turn Information
- **Turn number**: Current turn in mission
- **Alarm level**: 0-6 scale based on red indicator intensity
- **Tracker count**: Number of active trackers (TODO)

### 3. Agent Status (TODO)
- Agent name
- Action Points (AP) - current/max
- Inventory items with cooldowns
- Augment items
- Selection state

### 4. Incognita Programs (TODO)
- Active programs and their icons
- Cooldown states
- Variable PWR costs (affected by conditions/daemons)

## Architecture

### Detection Strategy

The detector uses a multi-stage approach:

1. **Viewport Detection**: Find the game window boundaries within the screenshot
2. **Region Extraction**: Extract specific UI regions using relative coordinates
3. **OCR Processing**: Preprocess regions and extract text using Tesseract
4. **Color Analysis**: Detect visual indicators (alarm level via red intensity)
5. **Status Compilation**: Aggregate all detected information into structured format

### UI Region Mapping

Regions are defined as relative coordinates (0.0-1.0) within the detected viewport:

```python
REGIONS = {
    'power_credits': (0.0, 0.0, 0.15, 0.08),    # Top-left
    'turn_number': (0.42, 0.0, 0.58, 0.06),     # Top-center  
    'alarm': (0.85, 0.0, 1.0, 0.08),            # Top-right
    'agent_panel': (0.0, 0.85, 0.25, 1.0),      # Bottom-left
    'incognita_switch': (0.0, 0.3, 0.08, 0.5),  # Left side
}
```

### OCR Pipeline

For text extraction:
1. Convert to grayscale
2. Increase contrast (alpha=2.0)
3. Apply binary threshold
4. Denoise with fastNlMeans
5. Upscale 3x for better OCR accuracy
6. Run Tesseract with digit-only whitelist

### Alarm Detection

Uses HSV color space to detect red pixels:
- Red ranges: H=[0-10, 160-180], S=[100-255], V=[100-255]
- Alarm level estimated from red pixel ratio
- Thresholds: <5% = level 0, 5-15% = level 1, ... >55% = level 6

## Usage

### Basic Analysis

```bash
# Analyze a captured session
python game_status_detector.py captures/session_20241021_194632

# With debug output
python game_status_detector.py captures/session_20241021_194632 --debug

# Custom worker count
python game_status_detector.py captures/session_20241021_194632 --workers 8

# Custom output location
python game_status_detector.py captures/session_20241021_194632 --output results.json
```

### Debug Utilities

Use the debug script to calibrate detection on sample frames:

```bash
# Show all detection regions overlaid on frame
python debug_status_detector.py captures/session_X/frames/frame_0050.png --show-regions

# Test OCR on text regions
python debug_status_detector.py captures/session_X/frames/frame_0050.png --test-ocr

# Analyze color detection (alarm)
python debug_status_detector.py captures/session_X/frames/frame_0050.png --analyze-colors

# Run all debug tests
python debug_status_detector.py captures/session_X/frames/frame_0050.png --all
```

Debug images are saved to `/tmp/status_debug/`:
- `regions_annotated.png` - Frame with all regions marked
- `power_credits_original.png` - Raw power/credits region
- `power_credits_processed.png` - Preprocessed for OCR
- `turn_number_*.png` - Turn number regions
- `alarm_region.png` - Alarm indicator region
- `alarm_red_mask.png` - Red pixel detection mask

## Output Format

Results are saved as JSON with this structure:

```json
{
  "session": "session_20241021_194632",
  "total_frames": 150,
  "frame_statuses": [
    {
      "frame_number": 0,
      "turn_number": 1,
      "resources": {
        "power_current": 12,
        "power_max": 15,
        "credits": 250
      },
      "alarm": {
        "level": 0,
        "ticks": null,
        "tracker_count": null
      },
      "agents": [],
      "incognita": {
        "active_programs": [],
        "available_pwr": null
      }
    }
  ],
  "summary": {
    "turn_range": {"min": 1, "max": 8},
    "resource_stats": {
      "power": {"min": 5, "max": 15, "avg": 10.2},
      "credits": {"min": 100, "max": 450, "avg": 275.5}
    },
    "alarm_stats": {"min": 0, "max": 3, "avg": 1.2},
    "detection_rates": {
      "turn_number": 0.95,
      "power": 0.92,
      "credits": 0.88,
      "alarm": 0.98
    }
  }
}
```

## Performance

- **Parallel processing**: Uses multiple workers (default: CPU count - 1)
- **Speed**: ~50 frames/second with 5 workers
- **Memory**: Processes frames independently, low memory footprint

## Installation

### System Requirements

Install system dependencies:

**macOS:**
```bash
# Tesseract for OCR
brew install tesseract

# Python Tkinter (required by pynput for input capture)
brew install python-tk@3.13
```

**Ubuntu/Debian:**
```bash
# Tesseract for OCR
sudo apt-get install tesseract-ocr

# Python Tkinter (required by pynput)
sudo apt-get install python3-tk
```

**Windows:**
- Tesseract: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Tkinter: Usually included with Python installation

### Python Dependencies

```bash
pip install -r requirements_status.txt
```

## Known Limitations

### Current Implementation

1. **Agent Panel**: Not yet implemented
   - Need to detect agent selection state
   - Parse AP, inventory, augments
   - Handle mainframe mode (Incognita panel)

2. **Incognita Programs**: Not yet implemented
   - Icon recognition needed
   - Cooldown parsing
   - Variable cost detection

3. **Viewport Detection**: Basic implementation
   - May fail on unusual resolutions
   - Assumes dark borders around game area
   - Falls back to full frame if detection uncertain

4. **OCR Accuracy**: Depends on frame quality
   - Small text may be difficult
   - Motion blur during captures affects accuracy
   - Different resolutions may need region calibration

### Region Calibration

The current region coordinates are estimates and may need adjustment based on:
- Screen resolution
- Window size (windowed vs fullscreen)
- UI scale settings in game

Use the debug script to visualize regions and adjust coordinates in `GameStatusDetector.REGIONS`.

## Integration with Turn Phase Detector

Combine with `turn_phase_detector.py` for complete analysis:

```bash
# 1. Detect turn phases
python turn_phase_detector.py captures/session_X

# 2. Extract game status
python game_status_detector.py captures/session_X

# Results:
# - captures/session_X/turn_phases.json
# - captures/session_X/game_status.json
```

Together, these provide:
- **When**: Turn phase timeline (player planning, player action, opponent)
- **What**: Detailed game state at each frame

## Future Enhancements

### High Priority
1. Implement agent panel detection
2. Implement Incognita program detection
3. Calibrate regions for different resolutions
4. Improve OCR accuracy for small numbers

### Medium Priority
1. Detect tracker count
2. Parse alarm "ticks" (progress to next level)
3. Detect daemon effects on program costs
4. Track inventory cooldowns

### Low Priority
1. Detect mission objectives state
2. Parse enemy visible state
3. Detect AP expenditure per action
4. Track KO'd agents

## Troubleshooting

### OCR returning empty strings
- Check debug images in `/tmp/status_debug/`
- Verify Tesseract is installed: `tesseract --version`
- Try adjusting preprocessing parameters
- Ensure frame quality is sufficient (not blurry)

### Viewport detection failing
- Use `--debug` flag to see detection details
- Check if game has unusual black bars or borders
- Manually adjust region coordinates if needed

### Low detection rates
- Verify regions are correctly positioned (use debug script)
- Check if UI scale in game is non-standard
- Ensure frames are from actual gameplay (not menus/loading)

### Slow performance
- Adjust `--workers` count
- Process subset of frames for testing
- Use SSD for frame storage

## Contributing

When adding new detection capabilities:

1. Add detection method to `GameStatusDetector` class
2. Update `GameStatus` dataclass with new fields
3. Add debug visualization to `debug_status_detector.py`
4. Update documentation with examples
5. Test on multiple resolutions and game states

## Credits

Part of the Invisible Inc gameplay capture and analysis system:
- `invisible_capture.py` - Gameplay capture
- `turn_phase_detector.py` - Turn phase classification
- `game_status_detector.py` - Game state extraction (this file)
- `web_viewer.py` - Session viewing and playback
