#!/usr/bin/env python3
"""
Framework Verification Tests

Quick sanity check that the framework is installed and working correctly.

Usage:
    python test_framework.py
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import cv2
        print("  ✓ OpenCV (cv2)")
    except ImportError:
        print("  ✗ OpenCV not found. Install with: pip install opencv-python")
        return False
    
    try:
        import numpy as np
        print("  ✓ NumPy")
    except ImportError:
        print("  ✗ NumPy not found. Install with: pip install numpy")
        return False
    
    try:
        import pytesseract
        print("  ✓ pytesseract")
    except ImportError:
        print("  ✗ pytesseract not found. Install with: pip install pytesseract")
        return False
    
    try:
        from detector_framework import (
            BaseDetector, DetectorRegistry, DetectorPipeline,
            DetectionResult, DetectorType
        )
        print("  ✓ detector_framework")
    except ImportError as e:
        print(f"  ✗ detector_framework import failed: {e}")
        return False
    
    try:
        from detector_adapters import TurnPhaseDetectorAdapter
        print("  ✓ detector_adapters")
    except ImportError as e:
        print(f"  ✗ detector_adapters import failed: {e}")
        return False
    
    return True


def test_tesseract():
    """Check if Tesseract OCR is installed"""
    print("\nTesting Tesseract OCR...")
    
    import subprocess
    
    try:
        result = subprocess.run(
            ['tesseract', '--version'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"  ✓ {version}")
            return True
        else:
            print("  ✗ Tesseract found but version check failed")
            return False
    except FileNotFoundError:
        print("  ✗ Tesseract not found. Install with: brew install tesseract")
        return False
    except Exception as e:
        print(f"  ✗ Error checking Tesseract: {e}")
        return False


def test_basic_functionality():
    """Test basic framework functionality"""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from detector_framework import DetectorRegistry
        from detector_adapters import TurnPhaseDetectorAdapter
        
        # Create a test image (solid color)
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_img[:] = (50, 50, 50)  # Dark gray
        
        # Create registry and add detector
        registry = DetectorRegistry()
        registry.register(TurnPhaseDetectorAdapter())
        
        print("  ✓ Registry created and detector registered")
        
        # Run detection
        results = registry.detect_all(test_img)
        
        print("  ✓ Detection executed")
        
        # Check result structure
        assert 'turn_phase' in results, "Missing turn_phase results"
        result = results['turn_phase']
        assert hasattr(result, 'success'), "Result missing 'success' field"
        assert hasattr(result, 'confidence'), "Result missing 'confidence' field"
        assert hasattr(result, 'data'), "Result missing 'data' field"
        
        print("  ✓ Result structure valid")
        print(f"    - Phase detected: {result.data.get('phase', 'N/A')}")
        print(f"    - Confidence: {result.confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_existing_detectors():
    """Check if existing detector files are present"""
    print("\nChecking for existing detectors...")
    
    has_structural = Path('structural_detector.py').exists()
    has_turn_phase = Path('turn_phase_detector.py').exists()
    
    if has_structural:
        print("  ✓ structural_detector.py found")
    else:
        print("  ⚠ structural_detector.py not found (optional)")
    
    if has_turn_phase:
        print("  ✓ turn_phase_detector.py found")
    else:
        print("  ⚠ turn_phase_detector.py not found (optional)")
    
    if has_structural:
        try:
            from detector_adapters import StructuralDetectorAdapter
            print("  ✓ StructuralDetectorAdapter can be imported")
        except ImportError as e:
            print(f"  ⚠ StructuralDetectorAdapter import issue: {e}")
    
    return True  # Not critical for framework itself


def main():
    """Run all tests"""
    print("="*60)
    print("MODULAR DETECTOR FRAMEWORK - VERIFICATION TESTS")
    print("="*60)
    
    all_passed = True
    
    # Core imports
    if not test_imports():
        all_passed = False
    
    # Tesseract (warning only if missing)
    test_tesseract()
    
    # Basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    # Existing detectors (informational)
    test_existing_detectors()
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nFramework is ready to use!")
        print("\nNext steps:")
        print("  1. Place your structural_detector.py and turn_phase_detector.py here")
        print("  2. Try: python example_framework_usage.py <test_image.png>")
        print("  3. See README.md for full documentation")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above and run again.")
        sys.exit(1)
    
    print("="*60)


if __name__ == "__main__":
    main()
