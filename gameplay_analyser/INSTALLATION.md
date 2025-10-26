# Complete Installation Guide
## Invisible Inc Game Analysis Toolkit

This guide covers installing everything you need for the complete analysis toolkit:
- Gameplay capture (`invisible_capture.py`)
- Turn phase detection (`turn_phase_detector.py`)
- Game status detection (`game_status_detector.py`)
- Web viewer (`web_viewer.py`)

## Step-by-Step Installation

### 1. Install System Dependencies

#### macOS

```bash
# Tesseract OCR (for game status detector)
brew install tesseract

# Python Tkinter (required by pynput for input capture)
brew install python-tk@3.13

# Optional: BlackHole (for system audio capture)
# Only needed if you want to capture game audio
brew install blackhole-2ch
```

#### Ubuntu/Debian

```bash
# Tesseract OCR (for game status detector)
sudo apt-get update
sudo apt-get install tesseract-ocr

# Python Tkinter (required by pynput)
sudo apt-get install python3-tk

# Optional: PulseAudio (for audio capture)
sudo apt-get install pulseaudio
```

#### Windows

1. **Tesseract OCR**: Download and install from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - Add Tesseract to your PATH
2. **Python Tkinter**: Usually included with Python installation
3. **Audio Capture**: May need additional configuration

### 2. Verify System Installations

```bash
# Check Tesseract
tesseract --version
# Should show: tesseract 5.x.x

# Check Python Tkinter (should not error)
python -c "import tkinter"

# Check if pynput can import (requires Tkinter)
python -c "from pynput import keyboard, mouse"
```

### 3. Install Python Dependencies

```bash
# Navigate to your project directory
cd /path/to/invisible-inc-analysis

# Install all Python packages
pip install -r requirements.txt
```

This installs:
- `mss` - Screen capture
- `pillow` - Image processing
- `pynput` - Keyboard/mouse input logging
- `sounddevice` - Audio capture
- `scipy` - Audio processing
- `numpy` - Numerical operations
- `opencv-python` - Computer vision
- `pytesseract` - OCR wrapper
- `pygetwindow` (Windows/Linux) - Window detection
- `pyobjc-framework-Quartz` (macOS) - Window detection

### 4. Verify Installation

```bash
# Test imports
python -c "import mss, cv2, pytesseract, pynput, numpy"
echo "All imports successful!"

# Quick version check
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

### 5. Test the Tools

#### Test Capture System

```bash
# Short 10-second test capture
python invisible_capture.py --fps 1 --duration 10

# Check if frames were captured
ls -lh captures/session_*/frames/
```

#### Test Turn Phase Detector

```bash
# Analyze the test capture
python turn_phase_detector.py captures/session_LATEST

# Check results
cat captures/session_LATEST/turn_phases.json
```

#### Test Game Status Detector

```bash
# Analyze the test capture
python game_status_detector.py captures/session_LATEST

# Check results
cat captures/session_LATEST/game_status.json
```

#### Test Debug Tools

```bash
# Pick any captured frame
python debug_status_detector.py captures/session_LATEST/frames/frame_0001.png --all

# Check debug output
ls -lh /tmp/status_debug/
```

## Troubleshooting

### "tesseract: command not found"

**Problem**: Tesseract not installed or not in PATH

**Solution**:
- macOS: `brew install tesseract`
- Ubuntu: `sudo apt-get install tesseract-ocr`
- Windows: Install from link above and add to PATH

### "No module named 'tkinter'" or "_tkinter.TclError"

**Problem**: Python Tkinter not installed

**Solution**:
- macOS: `brew install python-tk@3.13`
- Ubuntu: `sudo apt-get install python3-tk`
- Windows: Reinstall Python with Tkinter option checked

### "pynput" import fails on macOS

**Problem**: Permission issues or missing Tkinter

**Solution**:
1. Install Tkinter: `brew install python-tk@3.13`
2. Grant accessibility permissions to Terminal in System Preferences → Security & Privacy → Privacy → Accessibility

### "Could not find InvisibleInc window"

**Problem**: Game not running or window detection failing

**Solution**:
1. Make sure Invisible Inc is running
2. Try windowed mode instead of fullscreen
3. Check if window is actually named "InvisibleInc" (case-sensitive)

### Audio capture fails on macOS

**Problem**: Can't capture system audio without virtual device

**Solution**:
1. Install BlackHole: `brew install blackhole-2ch`
2. Set up Multi-Output Device in Audio MIDI Setup
3. Route game audio through BlackHole

### Low OCR accuracy

**Problem**: Game running at low resolution or frames are blurry

**Solution**:
1. Run game at higher resolution
2. Capture at lower FPS (2-3 fps instead of 5-10 fps) for clearer frames
3. Use debug script to check preprocessed images
4. Adjust region coordinates if UI elements are misaligned

## Directory Structure

After installation and first run, you should have:

```
invisible-inc-analysis/
├── invisible_capture.py          # Gameplay capture
├── turn_phase_detector.py        # Phase classification
├── game_status_detector.py       # Status extraction
├── debug_status_detector.py      # Debug utilities
├── web_viewer.py                 # Session viewer
├── requirements.txt              # Python dependencies
└── captures/                     # Captured sessions
    └── session_YYYYMMDD_HHMMSS/
        ├── frames/               # Screenshot frames
        ├── metadata.json         # Session info
        ├── inputs.jsonl          # Input events
        ├── turn_phases.json      # Phase analysis
        └── game_status.json      # Status analysis
```

## What Each Tool Does

### invisible_capture.py
- Captures screenshots at 2-3 FPS
- Logs keyboard/mouse input
- Records audio (optional)
- Creates organized session folders

### turn_phase_detector.py
- Detects "END TURN" button
- Classifies frames: player_normal, player_action, opponent
- Creates timeline of turn phases
- ~50 frames/second processing

### game_status_detector.py
- Extracts Power, Credits, Turn number, Alarm level
- Uses OCR and color analysis
- Parallel processing (~50 frames/second)
- Creates detailed status timeline

### debug_status_detector.py
- Visualizes detection regions
- Tests OCR accuracy
- Analyzes color detection
- Saves debug images

### web_viewer.py
- Browser-based session viewer
- Frame-by-frame playback
- Timeline scrubbing
- Speed controls

## Next Steps

Once everything is installed:

1. **Capture a mission**: `python invisible_capture.py --fps 2 --duration 1800`
2. **Analyze phases**: `python turn_phase_detector.py captures/session_LATEST`
3. **Extract status**: `python game_status_detector.py captures/session_LATEST`
4. **View results**: `python web_viewer.py captures`

## Getting Help

If you encounter issues:

1. Check debug output: `python game_status_detector.py --debug captures/session_X`
2. Verify regions: `python debug_status_detector.py frame.png --all`
3. Check system dependencies are installed correctly
4. Ensure game is running and window is detectable

## Platform-Specific Notes

### macOS
- Requires accessibility permissions for pynput
- BlackHole needed for system audio capture
- Use Homebrew for easy installation
- python-tk@3.13 specifically needed (not just python-tk)

### Linux
- May need to configure PulseAudio for audio capture
- Some distributions need additional opencv dependencies
- Window detection should work with X11 or Wayland

### Windows
- Tesseract must be manually added to PATH
- Window detection uses different library (pygetwindow)
- Audio capture may need WASAPI configuration

---

**Installation complete!** You're ready to capture and analyze Invisible Inc gameplay.
