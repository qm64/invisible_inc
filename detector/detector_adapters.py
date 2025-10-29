"""
Adapters for Existing Detectors

Wraps the existing StructuralDetector and TurnPhaseDetector
to work with the new modular framework.

Version: 1.0.0
"""

from typing import Dict, Optional, Any
import numpy as np
from pathlib import Path

from detector_framework import (
    BaseDetector, DetectionResult, DetectorType, DetectorConfig,
    create_simple_result, create_error_result
)

# Import existing detectors
# Note: These imports work when the files are in the same directory
try:
    from structural_detector import StructuralDetector as OriginalStructuralDetector, DetectedElement
except ImportError:
    print("Warning: structural_detector.py not found. StructuralDetectorAdapter will not work.")
    OriginalStructuralDetector = None
    DetectedElement = None


class StructuralDetectorAdapter(BaseDetector):
    """
    Adapter for the existing StructuralDetector.
    
    Wraps structural UI element detection in the common interface.
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None, debug: bool = False):
        # Default config
        if config is None:
            config = DetectorConfig(
                name="structural",
                type=DetectorType.STRUCTURAL,
                enabled=True,
                dependencies=[],  # No dependencies
                params={'debug': debug}
            )
        
        super().__init__(config)
        
        # Create the wrapped detector
        self.debug = config.params.get('debug', debug)
        if OriginalStructuralDetector is not None:
            self._detector = OriginalStructuralDetector(debug=self.debug)
        else:
            raise ImportError("StructuralDetector not available")
    
    def detect(self,
               image: np.ndarray,
               context: Optional[Dict[str, DetectionResult]] = None,
               **kwargs) -> DetectionResult:
        """
        Run structural detection on the image.
        
        Returns:
            DetectionResult with 'elements' dict in data field
        """
        try:
            # Run the original detector
            elements = self._detector.detect_anchors(image)
            
            # Calculate overall confidence based on how many key elements found
            key_elements = ['power_credits', 'hamburger_menu', 'security_clock']
            found_key = sum(1 for key in key_elements if key in elements)
            confidence = found_key / len(key_elements)
            
            # Convert DetectedElement objects to serializable dicts
            elements_dict = {}
            for name, element in elements.items():
                elements_dict[name] = {
                    'bbox': element.bbox,
                    'center': element.center,
                    'confidence': element.confidence,
                    'method': element.method
                }
            
            # Extract viewport info if available
            viewport_info = None
            if hasattr(self._detector, '_inferred_viewport'):
                viewport = self._detector._inferred_viewport
                if viewport:
                    viewport_info = {
                        'x': viewport[0],
                        'y': viewport[1],
                        'width': viewport[2],
                        'height': viewport[3]
                    }
            
            return create_simple_result(
                detector_name=self.get_name(),
                detector_type=self.get_type(),
                data={
                    'elements': elements_dict,
                    'element_count': len(elements),
                    'viewport': viewport_info
                },
                confidence=confidence
            )
            
        except Exception as e:
            return create_error_result(
                detector_name=self.get_name(),
                detector_type=self.get_type(),
                error=str(e)
            )
    
    def get_name(self) -> str:
        return "structural"
    
    def get_type(self) -> DetectorType:
        return DetectorType.STRUCTURAL
    
    def reset(self):
        """Reset temporal cache between sessions"""
        if hasattr(OriginalStructuralDetector, 'reset_temporal_cache'):
            OriginalStructuralDetector.reset_temporal_cache()


# For turn phase detection, we need a different approach since it uses
# a standalone function. Let's create a detector that wraps that logic.

class TurnPhaseDetectorAdapter(BaseDetector):
    """
    Adapter for turn phase detection.
    
    Detects whether the current frame is:
    - player_normal: Planning phase
    - player_action: Executing action
    - opponent: Enemy turn
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        if config is None:
            config = DetectorConfig(
                name="turn_phase",
                type=DetectorType.PHASE,
                enabled=True,
                dependencies=[],  # Can work independently
                params={}
            )
        
        super().__init__(config)
    
    def detect(self,
               image: np.ndarray,
               context: Optional[Dict[str, DetectionResult]] = None,
               **kwargs) -> DetectionResult:
        """
        Detect turn phase from image.
        
        Returns:
            DetectionResult with phase, features in data field
        """
        import cv2
        
        try:
            # Extract features (same logic as analyze_single_frame)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            height, width = image.shape[:2]
            
            features = {}
            
            # PRIMARY: "END TURN" button (lower-right)
            end_turn_h_start = int(height * 0.85)
            end_turn_w_start = int(width * 0.80)
            end_turn_region = image[end_turn_h_start:, end_turn_w_start:]
            end_turn_hsv = hsv[end_turn_h_start:, end_turn_w_start:]
            
            # Detect cyan button
            cyan_mask = cv2.inRange(end_turn_hsv,
                                   np.array([85, 80, 80]),
                                   np.array([100, 255, 255]))
            cyan_pixels = np.sum(cyan_mask > 0)
            end_turn_area = end_turn_region.shape[0] * end_turn_region.shape[1]
            features['end_turn_cyan_ratio'] = cyan_pixels / end_turn_area if end_turn_area > 0 else 0
            
            # Dark cyan background
            dark_cyan_mask = cv2.inRange(end_turn_hsv,
                                        np.array([85, 60, 40]),
                                        np.array([100, 255, 120]))
            dark_cyan_pixels = np.sum(dark_cyan_mask > 0)
            features['end_turn_dark_ratio'] = dark_cyan_pixels / end_turn_area if end_turn_area > 0 else 0
            
            end_turn_score = features['end_turn_cyan_ratio'] + features['end_turn_dark_ratio']
            features['end_turn_present'] = end_turn_score > 0.03
            
            # SECONDARY: Profile pic (lower-left)
            profile_h_start = int(height * 0.80)
            profile_w_end = int(width * 0.15)
            profile_region = image[profile_h_start:, :profile_w_end]
            profile_hsv = hsv[profile_h_start:, :profile_w_end]
            
            # Skin tones
            skin_mask = cv2.inRange(profile_hsv,
                                   np.array([0, 20, 50]),
                                   np.array([30, 255, 255]))
            skin_pixels = np.sum(skin_mask > 0)
            profile_area = profile_region.shape[0] * profile_region.shape[1]
            features['profile_skin_ratio'] = skin_pixels / profile_area if profile_area > 0 else 0
            
            # Colorful UI
            sat_mask = cv2.inRange(profile_hsv,
                                  np.array([0, 100, 50]),
                                  np.array([180, 255, 255]))
            sat_pixels = np.sum(sat_mask > 0)
            features['profile_color_ratio'] = sat_pixels / profile_area if profile_area > 0 else 0
            
            # Edges
            profile_gray = cv2.cvtColor(profile_region, cv2.COLOR_BGR2GRAY)
            profile_edges = cv2.Canny(profile_gray, 50, 150)
            features['profile_edge_ratio'] = np.sum(profile_edges > 0) / profile_area if profile_area > 0 else 0
            
            # Profile presence score
            profile_score = (
                features['profile_skin_ratio'] * 2.0 +
                features['profile_color_ratio'] * 1.5 +
                features['profile_edge_ratio'] * 1.0
            )
            features['profile_presence_score'] = profile_score
            
            # CLASSIFICATION
            phase = 'unknown'
            confidence = 0.0
            
            if features['end_turn_present']:
                if profile_score > 0.10:
                    phase = 'player_normal'
                    confidence = min(end_turn_score * 5 + profile_score * 2, 1.0)
                else:
                    phase = 'player_action'
                    confidence = min(end_turn_score * 5, 0.9)
            else:
                phase = 'opponent'
                # Check for enemy banner
                banner_h_start = int(height * 0.4)
                banner_h_end = int(height * 0.6)
                banner_region = image[banner_h_start:banner_h_end, :]
                banner_hsv = hsv[banner_h_start:banner_h_end, :]
                
                red_mask1 = cv2.inRange(banner_hsv,
                                       np.array([0, 100, 100]),
                                       np.array([10, 255, 255]))
                red_mask2 = cv2.inRange(banner_hsv,
                                       np.array([170, 100, 100]),
                                       np.array([180, 255, 255]))
                banner_red = cv2.bitwise_or(red_mask1, red_mask2)
                banner_area = banner_region.shape[0] * banner_region.shape[1]
                banner_red_ratio = np.sum(banner_red > 0) / banner_area if banner_area > 0 else 0
                
                if banner_red_ratio > 0.05:
                    confidence = min(banner_red_ratio * 10, 1.0)
                else:
                    confidence = 0.7
            
            return create_simple_result(
                detector_name=self.get_name(),
                detector_type=self.get_type(),
                data={
                    'phase': phase,
                    'features': features,
                    'end_turn_score': float(end_turn_score),
                    'profile_score': float(profile_score)
                },
                confidence=confidence
            )
            
        except Exception as e:
            return create_error_result(
                detector_name=self.get_name(),
                detector_type=self.get_type(),
                error=str(e)
            )
    
    def get_name(self) -> str:
        return "turn_phase"
    
    def get_type(self) -> DetectorType:
        return DetectorType.PHASE


# Example of a simple custom detector

class ViewportDetectorAdapter(BaseDetector):
    """
    Extracts viewport information from structural detection results.
    
    Demonstrates how to create a detector that depends on another.
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        if config is None:
            config = DetectorConfig(
                name="viewport",
                type=DetectorType.STRUCTURAL,
                enabled=True,
                dependencies=["structural"],  # Requires structural detector
                params={}
            )
        
        super().__init__(config)
    
    def detect(self,
               image: np.ndarray,
               context: Optional[Dict[str, DetectionResult]] = None,
               **kwargs) -> DetectionResult:
        """
        Extract viewport from structural detection.
        
        Requires 'structural' detector to have run first.
        """
        if context is None or 'structural' not in context:
            return create_error_result(
                detector_name=self.get_name(),
                detector_type=self.get_type(),
                error="Missing required 'structural' detector results"
            )
        
        structural_result = context['structural']
        if not structural_result.success:
            return create_error_result(
                detector_name=self.get_name(),
                detector_type=self.get_type(),
                error="Structural detection failed"
            )
        
        viewport = structural_result.data.get('viewport')
        
        if viewport is None:
            return DetectionResult(
                detector_name=self.get_name(),
                detector_type=self.get_type(),
                success=False,
                confidence=0.0,
                data={},
                error="No viewport detected"
            )
        
        return create_simple_result(
            detector_name=self.get_name(),
            detector_type=self.get_type(),
            data={
                'viewport': viewport,
                'x': viewport['x'],
                'y': viewport['y'],
                'width': viewport['width'],
                'height': viewport['height']
            },
            confidence=1.0
        )
    
    def get_name(self) -> str:
        return "viewport"
    
    def get_type(self) -> DetectorType:
        return DetectorType.STRUCTURAL
