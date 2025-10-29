"""
Modular Detector Framework for Invisible Inc Analysis

Provides a common interface for different detection systems:
- Structural UI detection
- Turn phase detection  
- OCR extraction
- Future detectors...

Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path
import numpy as np
import json
from enum import Enum


class DetectorType(Enum):
    """Categories of detectors"""
    STRUCTURAL = "structural"  # UI element detection
    PHASE = "phase"            # Turn phase classification
    OCR = "ocr"                # Text extraction
    STATE = "state"            # Game state extraction
    CUSTOM = "custom"          # User-defined


@dataclass
class DetectionResult:
    """Standardized result from any detector"""
    detector_name: str
    detector_type: DetectorType
    success: bool
    confidence: float  # 0.0 to 1.0
    data: Dict[str, Any]
    frame_index: Optional[int] = None
    timestamp: Optional[float] = None
    debug_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        result = asdict(self)
        result['detector_type'] = self.detector_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionResult':
        """Create from dict"""
        if 'detector_type' in data:
            data['detector_type'] = DetectorType(data['detector_type'])
        return cls(**data)


@dataclass
class DetectorConfig:
    """Configuration for a detector"""
    name: str
    type: DetectorType
    enabled: bool = True
    dependencies: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


class BaseDetector(ABC):
    """
    Abstract base class for all detectors.
    
    All detectors must implement:
    - detect(): Process an image and return results
    - get_name(): Return detector identifier
    - get_type(): Return detector category
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig(
            name=self.get_name(),
            type=self.get_type()
        )
        self._enabled = self.config.enabled
    
    @abstractmethod
    def detect(self, 
               image: np.ndarray,
               context: Optional[Dict[str, DetectionResult]] = None,
               **kwargs) -> DetectionResult:
        """
        Perform detection on an image.
        
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
        return self.config.dependencies
    
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
        Raises ValueError if circular dependencies detected.
        """
        # Build dependency graph: name -> set of dependencies
        graph = {name: set(det.get_dependencies()) 
                for name, det in self._detectors.items()}
        
        # Check all dependencies exist
        for name, deps in graph.items():
            for dep in deps:
                if dep not in graph:
                    raise ValueError(f"Unknown dependency: {dep}")
        
        # Topological sort (Kahn's algorithm)
        # In-degree = number of dependencies a node has
        in_degree = {name: len(deps) for name, deps in graph.items()}
        
        # Start with nodes that have no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            current = queue.pop(0)
            order.append(current)
            
            # Find all nodes that depend on current and decrease their in-degree
            for name, deps in graph.items():
                if current in deps:
                    deps.remove(current)
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)
        
        if len(order) != len(graph):
            raise ValueError("Circular dependency detected")
        
        self._execution_order = order
    
    def detect_all(self,
                   image: np.ndarray,
                   frame_index: Optional[int] = None,
                   detectors: Optional[List[str]] = None,
                   **kwargs) -> Dict[str, DetectionResult]:
        """
        Run multiple detectors in dependency order.
        
        Args:
            image: Input image
            frame_index: Optional frame number for tracking
            detectors: Optional list of specific detectors to run (runs all if None)
            **kwargs: Additional parameters passed to all detectors
            
        Returns:
            Dict mapping detector names to their results
        """
        results = {}
        
        # Determine which detectors to run
        if detectors is None:
            to_run = self._execution_order
        else:
            # Filter and maintain dependency order
            to_run = [d for d in self._execution_order if d in detectors]
        
        # Execute detectors in order
        for name in to_run:
            detector = self._detectors[name]
            
            if not detector.is_enabled():
                continue
            
            try:
                result = detector.detect(
                    image=image,
                    context=results,
                    frame_index=frame_index,
                    **kwargs
                )
                results[name] = result
                
            except Exception as e:
                # Create error result
                results[name] = DetectionResult(
                    detector_name=name,
                    detector_type=detector.get_type(),
                    success=False,
                    confidence=0.0,
                    data={},
                    frame_index=frame_index,
                    error=str(e)
                )
        
        return results
    
    def reset_all(self):
        """Reset all detectors (call between sessions)"""
        for detector in self._detectors.values():
            detector.reset()


class DetectorPipeline:
    """
    High-level pipeline for processing frames with multiple detectors.
    Handles batch processing, result aggregation, and output formatting.
    """
    
    def __init__(self, registry: DetectorRegistry):
        self.registry = registry
        self.session_results: List[Dict[str, DetectionResult]] = []
    
    def process_frame(self,
                      image: np.ndarray,
                      frame_index: Optional[int] = None,
                      **kwargs) -> Dict[str, DetectionResult]:
        """Process a single frame with all enabled detectors"""
        return self.registry.detect_all(image, frame_index, **kwargs)
    
    def process_session(self,
                       frame_paths: List[Path],
                       save_to: Optional[Path] = None,
                       **kwargs) -> List[Dict[str, DetectionResult]]:
        """
        Process multiple frames in sequence.
        
        Args:
            frame_paths: List of image file paths
            save_to: Optional path to save results JSON
            **kwargs: Additional parameters for detectors
            
        Returns:
            List of result dicts (one per frame)
        """
        import cv2
        
        self.session_results = []
        
        for idx, frame_path in enumerate(frame_paths):
            img = cv2.imread(str(frame_path))
            if img is None:
                print(f"Warning: Could not load {frame_path}")
                continue
            
            results = self.process_frame(img, frame_index=idx, **kwargs)
            self.session_results.append(results)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(frame_paths)} frames...")
        
        if save_to:
            self.save_results(save_to)
        
        return self.session_results
    
    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def save_results(self, output_path: Path):
        """Save session results to JSON"""
        serializable = []
        for frame_results in self.session_results:
            frame_dict = {
                name: result.to_dict() 
                for name, result in frame_results.items()
            }
            serializable.append(frame_dict)
        
        # Convert numpy types to native Python types
        serializable = self._convert_numpy_types(serializable)
        
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"✓ Results saved to: {output_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate summary statistics across all frames"""
        if not self.session_results:
            return {}
        
        summary = {
            'total_frames': len(self.session_results),
            'detectors': {}
        }
        
        # Aggregate per-detector stats
        for detector_name in self.session_results[0].keys():
            successes = sum(
                1 for frame in self.session_results 
                if frame[detector_name].success
            )
            avg_confidence = np.mean([
                frame[detector_name].confidence 
                for frame in self.session_results
            ])
            
            summary['detectors'][detector_name] = {
                'success_rate': successes / len(self.session_results),
                'avg_confidence': float(avg_confidence)
            }
        
        return summary


# Utility functions

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