# Modular Detector Framework

A flexible framework for composing multiple computer vision detectors for Invisible Inc game state analysis.

## What It Does

Provides a common interface that allows different detection systems to work together:
- **Turn phase detection** - Player planning, player action, or opponent turn
- **Structural UI detection** - Game elements (power, credits, menus, agent icons)
- **Extensible architecture** - Easy to add new detectors

Key benefits:
- Detectors can use results from other detectors (dependencies)
- Automatic execution ordering (topological sort)
- Standardized result format
- Easy to extend without modifying existing code

## Installation

### Prerequisites

```bash
# Install Tesseract OCR (required for structural detector)
brew install tesseract

# Create/activate your virtual environment
python3 -m venv venv
source venv/bin/activate
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python test_framework.py
```

## Quick Start

### Single Frame Analysis

```python
import cv2
from detector_framework import DetectorRegistry
from detector_adapters import TurnPhaseDetectorAdapter, StructuralDetectorAdapter

# Load image
img = cv2.imread("test_frame.png")

# Setup registry
registry = DetectorRegistry()
registry.register(TurnPhaseDetectorAdapter())
registry.register(StructuralDetectorAdapter())

# Run detection
results = registry.detect_all(img)

# Access results
phase = results['turn_phase'].data['phase']
elements = results['structural'].data['elements']
print(f"Turn phase: {phase}")
print(f"Found {len(elements)} UI elements")
```

### Batch Processing

```python
from pathlib import Path
from detector_framework import DetectorPipeline

# Setup
registry = DetectorRegistry()
registry.register(TurnPhaseDetectorAdapter())
pipeline = DetectorPipeline(registry)

# Process all frames
frames = sorted(Path("captures/session/frames").glob("*.png"))
results = pipeline.process_session(frames, save_to="results.json")

# Get statistics
summary = pipeline.get_summary()
print(f"Success rate: {summary['detectors']['turn_phase']['success_rate']:.1%}")
```

### Run Examples

```bash
# Single frame
python example_framework_usage.py test_frame.png

# Full session
python example_framework_usage.py --session captures/20251021_200738
```

## Architecture

### Core Components

**BaseDetector** - Abstract base class for all detectors
- `detect(image, context, **kwargs)` - Main detection method
- `get_name()` - Unique identifier
- `get_type()` - Category (STRUCTURAL, PHASE, OCR, etc.)
- `get_dependencies()` - List of required detectors

**DetectorRegistry** - Manages multiple detectors
- Handles registration
- Resolves dependencies
- Computes execution order
- Runs detectors in sequence

**DetectorPipeline** - Batch processing
- Process frames in sequence
- Aggregate results
- Generate statistics

**DetectionResult** - Standardized output
```python
{
    'detector_name': 'turn_phase',
    'success': True,
    'confidence': 0.87,
    'data': {'phase': 'player_normal', ...},
    'error': None
}
```

### Dependency Flow

```
Image → [turn_phase] → phase, confidence
      → [structural] → elements, viewport
      → [viewport]   → viewport coords (depends on structural)
```

The registry automatically:
- Runs `structural` before `viewport` (dependency)
- Passes `structural` results to `viewport` in the `context` parameter
- Detects circular dependencies

## Available Detectors

### TurnPhaseDetectorAdapter
- **Type:** PHASE
- **Dependencies:** None
- **Detects:** Player planning, player action, or opponent turn
- **Output:** `phase`, `features`, confidence scores

### StructuralDetectorAdapter
- **Type:** STRUCTURAL
- **Dependencies:** None
- **Detects:** UI elements (power, credits, menus, agent icons)
- **Output:** `elements` dict, `viewport` info, element count

### ViewportDetectorAdapter
- **Type:** STRUCTURAL
- **Dependencies:** structural
- **Detects:** Game viewport boundaries
- **Output:** Viewport coordinates and dimensions

## Creating New Detectors

### Basic Template

See `detector_template.py` for complete examples. Basic structure:

```python
from detector_framework import BaseDetector, DetectionResult, DetectorType

class MyDetector(BaseDetector):
    def detect(self, image, context=None, **kwargs):
        # Your detection logic
        result_data = self._analyze(image)
        
        return DetectionResult(
            detector_name=self.get_name(),
            detector_type=self.get_type(),
            success=True,
            confidence=0.95,
            data=result_data
        )
    
    def get_name(self):
        return "my_detector"
    
    def get_type(self):
        return DetectorType.CUSTOM
```

### Using Dependencies

```python
class DependentDetector(BaseDetector):
    def __init__(self, config=None):
        if config is None:
            config = DetectorConfig(
                name="dependent",
                type=DetectorType.CUSTOM,
                dependencies=["structural"]  # Requires structural detector
            )
        super().__init__(config)
    
    def detect(self, image, context=None, **kwargs):
        # Access dependency results
        structural_result = context['structural']
        elements = structural_result.data['elements']
        
        # Your logic using those results...
```

### Register and Use

```python
registry.register(MyDetector())
results = registry.detect_all(img)
my_data = results['my_detector'].data
```

## Advanced Usage

### Selective Detection

Run only specific detectors:
```python
results = registry.detect_all(img, detectors=['turn_phase'])
```

### Dynamic Enable/Disable

```python
detector = registry.get_detector('structural')
detector.disable()  # Won't run
results = registry.detect_all(img)
detector.enable()   # Re-enable
```

### Reset State Between Sessions

```python
registry.reset_all()  # Clears temporal caches
# Process new session...
```

### Custom Configuration

```python
config = DetectorConfig(
    name="my_detector",
    type=DetectorType.CUSTOM,
    dependencies=['structural'],
    params={'threshold': 0.8, 'debug': True}
)
detector = MyDetector(config)
```

## Migration from Standalone Scripts

**Before:**
```python
from structural_detector import StructuralDetector
detector = StructuralDetector()
elements = detector.detect_anchors(img)
```

**After:**
```python
from detector_framework import DetectorRegistry
from detector_adapters import StructuralDetectorAdapter

registry = DetectorRegistry()
registry.register(StructuralDetectorAdapter())
results = registry.detect_all(img)
elements = results['structural'].data['elements']
```

The adapters make your existing `structural_detector.py` and `turn_phase_detector.py` compatible with the framework.

## File Structure

```
your_project/
├── detector_framework.py           # Core framework
├── detector_adapters.py            # Wraps existing detectors
├── detector_template.py            # Template for new detectors
├── example_framework_usage.py      # Usage examples
├── requirements.txt                # Python dependencies
├── test_framework.py               # Verification tests
├── structural_detector.py          # Your existing detector
└── turn_phase_detector.py          # Your existing detector
```

## Troubleshooting

### Import Error: "No module named 'structural_detector'"
Place `structural_detector.py` in the same directory as the framework files.

### Import Error: "No module named 'cv2'"
Install OpenCV: `pip install opencv-python`

### Tesseract Error
Install Tesseract: `brew install tesseract`

### Detector Failed
Check the error field:
```python
result = results['structural']
if not result.success:
    print(f"Error: {result.error}")
```

## Requirements

- Python 3.8+
- OpenCV (opencv-python >= 4.8.0)
- NumPy (numpy >= 1.24.0)
- Tesseract OCR (pytesseract >= 0.3.10)
- Tesseract system binary (brew install tesseract)

## Performance

- **Single frame:** ~100-200ms (depends on enabled detectors)
- **Batch processing:** Use pipeline for efficiency
- **Memory:** Results stored in memory - consider chunking for very long sessions

## License

MIT License
