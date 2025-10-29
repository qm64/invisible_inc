"""
Detector Template

Copy this file to create a new detector for the modular framework.

Instructions:
1. Copy this file and rename it (e.g., my_new_detector.py)
2. Replace all "Template" references with your detector name
3. Implement the detect() method with your logic
4. Update get_name() and get_type()
5. Add dependencies if needed
6. Import and register in your main script

Version: 1.0.0
"""

from typing import Dict, Optional, Any, List
import numpy as np
import cv2

from detector_framework import (
    BaseDetector,
    DetectionResult,
    DetectorType,
    DetectorConfig,
    create_simple_result,
    create_error_result
)


# =============================================================================
# SIMPLE DETECTOR (no dependencies)
# =============================================================================

class TemplateDetector(BaseDetector):
    """
    [DESCRIPTION]
    Brief description of what this detector does.
    
    Detects: [What UI elements or states]
    Output: [What goes in the data dict]
    Dependencies: [None or list of required detectors]
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None, **params):
        """
        Initialize the detector.
        
        Args:
            config: Optional configuration
            **params: Detector-specific parameters
        """
        # Set up default config
        if config is None:
            config = DetectorConfig(
                name="template",  # CHANGE THIS
                type=DetectorType.CUSTOM,  # CHANGE THIS if needed
                enabled=True,
                dependencies=[],  # Add dependencies here if needed
                params=params
            )
        
        super().__init__(config)
        
        # Initialize your detector's state
        self.threshold = config.params.get('threshold', 0.5)
        self.debug = config.params.get('debug', False)
        
        # Add any other initialization here
    
    def detect(self,
               image: np.ndarray,
               context: Optional[Dict[str, DetectionResult]] = None,
               **kwargs) -> DetectionResult:
        """
        Perform detection on the image.
        
        Args:
            image: BGR image from OpenCV
            context: Results from previously-run detectors
            **kwargs: Additional parameters
            
        Returns:
            DetectionResult with your findings
        """
        try:
            # YOUR DETECTION LOGIC GOES HERE
            
            # Example: Simple color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            height, width = image.shape[:2]
            
            # Detect specific color range
            lower_bound = np.array([100, 50, 50])
            upper_bound = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # Calculate how much of the color is present
            color_ratio = np.sum(mask > 0) / (height * width)
            
            # Determine if detection was successful
            detected = color_ratio > self.threshold
            confidence = min(color_ratio / self.threshold, 1.0) if detected else 0.0
            
            # Prepare result data
            data = {
                'detected': detected,
                'color_ratio': float(color_ratio),
                'threshold_used': self.threshold
            }
            
            # Add debug info if requested
            debug_info = None
            if self.debug:
                debug_info = {
                    'mask_pixels': int(np.sum(mask > 0)),
                    'total_pixels': height * width
                }
            
            # Return result
            return DetectionResult(
                detector_name=self.get_name(),
                detector_type=self.get_type(),
                success=detected,
                confidence=confidence,
                data=data,
                debug_info=debug_info
            )
            
        except Exception as e:
            # Return error result if something goes wrong
            return create_error_result(
                detector_name=self.get_name(),
                detector_type=self.get_type(),
                error=str(e)
            )
    
    def get_name(self) -> str:
        """Return unique detector identifier"""
        return "template"  # CHANGE THIS
    
    def get_type(self) -> DetectorType:
        """Return detector category"""
        return DetectorType.CUSTOM  # CHANGE THIS if appropriate
    
    def reset(self):
        """Reset any cached state (called between sessions)"""
        # Add any cleanup logic here if your detector maintains state
        pass


# =============================================================================
# DEPENDENT DETECTOR (uses results from other detectors)
# =============================================================================

class DependentTemplateDetector(BaseDetector):
    """
    Example detector that depends on another detector's results.
    
    This shows how to:
    - Declare dependencies
    - Access results from other detectors
    - Handle missing dependencies
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        if config is None:
            config = DetectorConfig(
                name="dependent_template",
                type=DetectorType.CUSTOM,
                enabled=True,
                dependencies=["structural"],  # REQUIRED: list detector names
                params={}
            )
        
        super().__init__(config)
    
    def detect(self,
               image: np.ndarray,
               context: Optional[Dict[str, DetectionResult]] = None,
               **kwargs) -> DetectionResult:
        """
        Detect using results from dependency detectors.
        """
        # Check if required detector ran
        if context is None or 'structural' not in context:
            return create_error_result(
                detector_name=self.get_name(),
                detector_type=self.get_type(),
                error="Missing required 'structural' detector results"
            )
        
        # Get results from dependency
        structural_result = context['structural']
        
        # Check if dependency succeeded
        if not structural_result.success:
            return create_error_result(
                detector_name=self.get_name(),
                detector_type=self.get_type(),
                error="Structural detection failed"
            )
        
        try:
            # Use results from structural detector
            elements = structural_result.data.get('elements', {})
            
            # YOUR LOGIC HERE using the dependency data
            # Example: Count specific element types
            panel_count = sum(1 for name in elements.keys() if 'panel' in name)
            
            return create_simple_result(
                detector_name=self.get_name(),
                detector_type=self.get_type(),
                data={
                    'panel_count': panel_count,
                    'total_elements': len(elements)
                },
                confidence=1.0
            )
            
        except Exception as e:
            return create_error_result(
                detector_name=self.get_name(),
                detector_type=self.get_type(),
                error=str(e)
            )
    
    def get_name(self) -> str:
        return "dependent_template"
    
    def get_type(self) -> DetectorType:
        return DetectorType.CUSTOM


# =============================================================================
# STATEFUL DETECTOR (maintains state across frames)
# =============================================================================

class StatefulTemplateDetector(BaseDetector):
    """
    Example detector that maintains state across frames.
    
    Useful for:
    - Tracking changes over time
    - Smoothing noisy detections
    - Detecting events/transitions
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        if config is None:
            config = DetectorConfig(
                name="stateful_template",
                type=DetectorType.CUSTOM,
                enabled=True,
                dependencies=[],
                params={}
            )
        
        super().__init__(config)
        
        # State variables
        self.previous_value = None
        self.frame_count = 0
        self.history = []
        self.max_history = 10
    
    def detect(self,
               image: np.ndarray,
               context: Optional[Dict[str, DetectionResult]] = None,
               **kwargs) -> DetectionResult:
        """
        Detect with temporal context.
        """
        try:
            # Detect current value (example)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            current_value = float(np.mean(gray))
            
            # Calculate change from previous frame
            changed = False
            change_amount = 0.0
            
            if self.previous_value is not None:
                change_amount = abs(current_value - self.previous_value)
                changed = change_amount > 10.0  # Threshold
            
            # Update history
            self.history.append(current_value)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            # Calculate smoothed value
            smoothed_value = float(np.mean(self.history))
            
            # Update state
            self.previous_value = current_value
            self.frame_count += 1
            
            return create_simple_result(
                detector_name=self.get_name(),
                detector_type=self.get_type(),
                data={
                    'current_value': current_value,
                    'smoothed_value': smoothed_value,
                    'changed': changed,
                    'change_amount': change_amount,
                    'frame_count': self.frame_count
                },
                confidence=1.0
            )
            
        except Exception as e:
            return create_error_result(
                detector_name=self.get_name(),
                detector_type=self.get_type(),
                error=str(e)
            )
    
    def get_name(self) -> str:
        return "stateful_template"
    
    def get_type(self) -> DetectorType:
        return DetectorType.CUSTOM
    
    def reset(self):
        """IMPORTANT: Reset state between sessions"""
        self.previous_value = None
        self.frame_count = 0
        self.history = []


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def example_usage():
    """Show how to use the template detectors"""
    import cv2
    from detector_framework import DetectorRegistry
    
    # Load image
    img = cv2.imread("test_frame.png")
    
    # Create registry
    registry = DetectorRegistry()
    
    # Register detectors
    registry.register(TemplateDetector(threshold=0.1, debug=True))
    registry.register(StatefulTemplateDetector())
    
    # Run detection
    results = registry.detect_all(img)
    
    # Access results
    template_result = results['template']
    print(f"Template detected: {template_result.data['detected']}")
    print(f"Confidence: {template_result.confidence:.2f}")
    
    stateful_result = results['stateful_template']
    print(f"Current value: {stateful_result.data['current_value']:.1f}")
    print(f"Smoothed value: {stateful_result.data['smoothed_value']:.1f}")


if __name__ == "__main__":
    example_usage()
