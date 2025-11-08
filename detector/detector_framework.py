"""
Minimal Detector Framework for Testing

Core classes needed for anchor_detectors.py and resources_extractors.py
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field


class DetectorType(Enum):
    """Types of detectors"""
    STRUCTURAL = "structural"  # UI element detection
    PHASE = "phase"           # Game phase classification
    OCR = "ocr"              # Text extraction
    CUSTOM = "custom"         # User-defined


@dataclass
class DetectorConfig:
    """Configuration for a detector"""
    name: str
    type: DetectorType
    dependencies: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Standardized result from a detector"""
    detector_name: str
    detector_type: DetectorType
    success: bool
    confidence: float
    data: Dict[str, Any]
    error: Optional[str] = None
    debug_info: Optional[Dict] = None


class BaseDetector(ABC):
    """
    Abstract base class for all detectors.
    All detectors must implement this interface.
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config
        self._enabled = True
    
    @abstractmethod
    def detect(self, image, context=None, **kwargs) -> DetectionResult:
        """
        Detect features in the image.
        
        Args:
            image: BGR image array from OpenCV
            context: Results from dependent detectors (keyed by detector name)
            **kwargs: Additional detector-specific parameters
            
        Returns:
            DetectionResult with findings
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return unique detector identifier"""
        pass
    
    @abstractmethod
    def get_type(self) -> DetectorType:
        """Return detector category"""
        pass
    
    def get_dependencies(self) -> List[str]:
        """Return list of detector names this detector depends on"""
        return self.config.dependencies if self.config else []
    
    def is_enabled(self) -> bool:
        """Check if detector is enabled"""
        return self._enabled
    
    def enable(self):
        """Enable this detector"""
        self._enabled = True
    
    def disable(self):
        """Disable this detector"""
        self._enabled = False
    
    def reset(self):
        """Reset any cached state (called between sessions)"""
        pass


class DetectorRegistry:
    """
    Central registry for managing multiple detectors.
    Handles dependency resolution and execution order.
    """
    
    def __init__(self):
        self._detectors: Dict[str, BaseDetector] = {}
        self._execution_order: List[str] = []
    
    def register(self, detector: BaseDetector) -> None:
        """Register a detector"""
        name = detector.get_name()
        if name in self._detectors:
            raise ValueError(f"Detector '{name}' already registered")
        
        self._detectors[name] = detector
        self._update_execution_order()
    
    def unregister(self, name: str) -> None:
        """Remove a detector"""
        if name in self._detectors:
            del self._detectors[name]
            self._update_execution_order()
    
    def get_detector(self, name: str) -> Optional[BaseDetector]:
        """Get detector by name"""
        return self._detectors.get(name)
    
    def list_detectors(self) -> List[str]:
        """List all registered detector names"""
        return list(self._detectors.keys())
    
    def _update_execution_order(self) -> None:
        """
        Compute execution order based on dependencies using topological sort.
        """
        # Simple topological sort
        visited = set()
        order = []
        
        def visit(name):
            if name in visited:
                return
            visited.add(name)
            
            if name in self._detectors:
                detector = self._detectors[name]
                for dep in detector.get_dependencies():
                    visit(dep)
                order.append(name)
        
        for name in self._detectors:
            visit(name)
        
        self._execution_order = order
    
    def detect_all(self, image, detectors=None, debug=False, **kwargs):
        """
        Run all (or specified) enabled detectors on an image.
        
        Args:
            image: Image to process
            detectors: Optional list of detector names to run
            debug: Enable debug output
            **kwargs: Additional parameters passed to detectors
        
        Returns:
            Dict mapping detector names to DetectionResults
        """
        results = {}
        
        # Determine which detectors to run
        if detectors is None:
            detectors_to_run = self._execution_order
        else:
            detectors_to_run = [d for d in self._execution_order if d in detectors]
        
        # Run detectors in dependency order
        for name in detectors_to_run:
            detector = self._detectors[name]
            
            if not detector.is_enabled():
                continue
            
            try:
                result = detector.detect(image, context=results, debug=debug, **kwargs)
                results[name] = result
            except Exception as e:
                # Create error result
                results[name] = DetectionResult(
                    detector_name=name,
                    detector_type=detector.get_type(),
                    success=False,
                    confidence=0.0,
                    data={},
                    error=f"Exception during detection: {str(e)}"
                )
        
        return results
    
    def reset_all(self):
        """Reset all detectors"""
        for detector in self._detectors.values():
            detector.reset()


# Helper functions
def create_simple_result(detector_name: str,
                        detector_type: DetectorType,
                        data: Dict[str, Any],
                        confidence: float = 1.0) -> DetectionResult:
    """Helper to create a simple successful result"""
    return DetectionResult(
        detector_name=detector_name,
        detector_type=detector_type,
        success=True,
        confidence=confidence,
        data=data
    )


def create_error_result(detector_name: str,
                       detector_type: DetectorType,
                       error: str) -> DetectionResult:
    """Helper to create an error result"""
    return DetectionResult(
        detector_name=detector_name,
        detector_type=detector_type,
        success=False,
        confidence=0.0,
        data={},
        error=error
    )
