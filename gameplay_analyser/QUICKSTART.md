# Game Status Detector - Quick Start Guide

## What This Does

Extracts detailed game state from your Invisible Inc gameplay captures:
- **Resources**: Power (12/15), Credits (250)
- **Turn Info**: Turn number, Alarm level (0-6)
- **Agent Status**: AP, inventory, augments (TODO)
- **Incognita**: Programs, cooldowns (TODO)

## Installation

### 1. Install System Dependencies

**macOS:**
```bash
# Tesseract for OCR
brew install tesseract

# Python Tkinter (required by pynput for input capture)
brew install python-tk@3.13
```

**Ubuntu:**
```bash
# Tesseract for OCR
sudo apt-get install tesseract-ocr

# Python Tkinter (required by pynput)
sudo apt-get install python3-tk
```

**Windows:**
- Tesseract: Download from https://github.com/UB-Mannheim/tesseract/wiki
- Tkinter: Usually included with Python installation

### 2. Install Python Dependencies

```bash
pip install -r requirements_status.txt
```

## Basic Usage

### Analyze a Session

```bash
python game_status_detector.py captures/session_20241021_194632
```

This will:
- Process all frames in parallel
- Extract resources, turn number, alarm level
- Save results to `captures/session_20241021_194632/game_status.json`
- Print summary statistics

### Example Output

```
Analyzing 150 frames from session_20241021_194632...
Using 5 parallel workers
  Processed 50/150 frames...
  Processed 100/150 frames...
  Processed 150/150 frames...
Completed analysis of 150 frames

Results saved to: captures/session_20241021_194632/game_status.json

=== Summary ===
Turn range: 1 - 8

Detection success rates:
  turn_number: 95.3%
  power: 92.0%
  credits: 88.7%
  alarm: 98.0%

Resource stats:
  power: min=5, max=15, avg=10.2
  credits: min=100, max=450, avg=275.5

Alarm: min=0, max=3, avg=1.2
```

## Debug and Calibration

Before running full analysis, test on a sample frame:

```bash
# Pick a frame from your capture
python debug_status_detector.py captures/session_X/frames/frame_0050.png --all
```

This shows:
- Detected viewport boundaries
- All UI regions overlaid
- OCR results for text regions
- Color analysis for alarm detection
- Debug images saved to `/tmp/status_debug/`

### Review Debug Images

Check these files in `/tmp/status_debug/`:
- `regions_annotated.png` - Are the boxes in the right places?
- `power_credits_processed.png` - Can you read the numbers?
- `alarm_red_mask.png` - Does it highlight the alarm indicators?

If regions are misaligned, you may need to adjust coordinates in `game_status_detector.py`.

## Tips for Best Results

1. **Frame Quality**: Capture at 2-3 fps for clear frames (not blurry)
2. **Resolution**: Higher resolution = better OCR accuracy
3. **UI Visibility**: Avoid captures during heavy screen effects/animations
4. **Testing**: Always run debug script first on representative frames

## Common Issues

**"No text detected"**
- Run debug script to see preprocessed images
- Check if Tesseract is installed: `tesseract --version`
- Try a different frame (current one might be blurry)

**"Viewport detection failed"**
- Game window might have unusual borders
- Use `--debug` flag to see detection details
- Check debug images

**Low detection rates (<80%)**
- Regions might need calibration for your resolution
- Use debug script to visualize region placement
- Some frames might be from menus/transitions

## Next Steps

1. **Combine with Turn Phase Detector**:
   ```bash
   python turn_phase_detector.py captures/session_X
   python game_status_detector.py captures/session_X
   ```
   
2. **Analyze Multiple Sessions**:
   ```bash
   for session in captures/session_*; do
       python game_status_detector.py "$session"
   done
   ```

3. **Build Custom Analysis**:
   - Load `game_status.json` in your own scripts
   - Correlate with `turn_phases.json` for timeline analysis
   - Track resource usage patterns
   - Analyze alarm progression

## Files Included

- `game_status_detector.py` - Main detection script
- `debug_status_detector.py` - Debug and calibration tool
- `requirements_status.txt` - Python dependencies
- `STATUS_DETECTOR_README.md` - Full documentation
- `QUICKSTART.md` - This file

## Current Limitations

✅ **Working**:
- Power/Credits detection
- Turn number detection
- Alarm level detection (via color analysis)

🚧 **TODO**:
- Agent panel parsing (AP, inventory, augments)
- Incognita programs and cooldowns
- Tracker count
- Alarm tick progress

See `STATUS_DETECTOR_README.md` for implementation details and future enhancements.
