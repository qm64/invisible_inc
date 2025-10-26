# Documentation Update - Tkinter Requirement Added

## What Changed

Added documentation for the **Python Tkinter** system dependency, which is required by `pynput` (used in `invisible_capture.py` for keyboard/mouse input logging).

## Why This Matters

On macOS and Linux, `pynput` requires Tkinter to be installed at the system level. Without it:
- `invisible_capture.py` will fail when trying to log input events
- You'll see errors like: `No module named 'tkinter'` or `_tkinter.TclError`

On Windows, Tkinter is usually included with Python, so this is primarily a macOS/Linux issue.

## Files Updated

### 1. **INSTALLATION.md** (NEW)
- Comprehensive installation guide for entire toolkit
- Step-by-step instructions for macOS, Ubuntu, Windows
- Troubleshooting section
- Verification steps

### 2. **requirements.txt** (NEW)
- Unified requirements file for entire project
- Includes all dependencies (capture + analysis)
- Clear comments about system dependencies
- Platform-specific packages with conditionals

### 3. **requirements_status.txt** (UPDATED)
- Added prominent system dependencies section
- Documents both Tesseract and Tkinter requirements
- Platform-specific installation commands

### 4. **README.md** (UPDATED)
- Added Tkinter to Quick Start section
- Links to comprehensive INSTALLATION.md
- Updated Requirements section

### 5. **QUICKSTART.md** (UPDATED)
- Installation section now includes Tkinter
- Separated by platform (macOS, Ubuntu, Windows)
- Clearer instructions

### 6. **STATUS_DETECTOR_README.md** (UPDATED)
- Installation section includes Tkinter
- Platform-specific commands

### 7. **SYSTEM_SUMMARY.md** (UPDATED)
- Deployment section mentions Tkinter
- Docker example updated
- Cross-platform notes enhanced

## Installation Commands

### macOS
```bash
brew install tesseract python-tk@3.13
pip install -r requirements.txt
```

### Ubuntu
```bash
sudo apt-get install tesseract-ocr python3-tk
pip install -r requirements.txt
```

### Windows
- Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
- Tkinter usually included with Python
- `pip install -r requirements.txt`

## Testing the Fix

After installing Tkinter, verify:

```bash
# Should not error
python -c "import tkinter"

# Should work now
python -c "from pynput import keyboard, mouse"

# Test capture
python invisible_capture.py --fps 1 --duration 10
```

## Why It Was Missing

The original documentation assumed users either:
1. Already had Tkinter installed (common on some systems)
2. Would encounter the error and search for the solution
3. Were only using the game status detector (which doesn't need pynput)

However, for a fresh install on macOS (especially with Homebrew Python), Tkinter is not included by default and must be explicitly installed.

## Impact

- **Low** - Most users who got this far already had Tkinter
- **Medium** - Critical for new users on macOS with Homebrew Python
- **High** - Prevents confusion and failed installations

## Related Issues

If users still have problems with pynput on macOS after installing Tkinter:
1. Check accessibility permissions: System Preferences → Security & Privacy → Privacy → Accessibility
2. Grant Terminal (or your IDE) permission to monitor input
3. Restart Terminal/IDE after granting permissions

## Summary

All documentation now clearly states that Python Tkinter is a required system dependency. Installation instructions are provided for all platforms, and a comprehensive installation guide (INSTALLATION.md) walks through the entire setup process with troubleshooting.

---

**Date**: October 23, 2025  
**Issue**: Missing Tkinter documentation  
**Resolution**: Added to all relevant documentation files  
**Files Changed**: 7 files (2 new, 5 updated)
