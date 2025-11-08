"""
Resources Extractors - Power and Credits OCR Detection

OCR-based extractors that read power and credits values from the game UI.
These depend on the PowerCreditsAnchorDetector to locate the text region.

Version: 1.0.0
"""

from typing import Dict, Optional, Any
import numpy as np
import cv2
import re

from detector_framework import (
    BaseDetector, DetectionResult, DetectorType, DetectorConfig,
    create_simple_result, create_error_result
)


class PowerExtractor(BaseDetector):
    """
    Extract power value from power/credits region.
    
    Format: "XX/YY PWR CREDITS" - extracts the XX/YY part
    
    Dependencies:
    - power_credits_anchor: Provides the bounding box of the power/credits text region
    
    Success rate: ~75.9% (fails during opponent turns, dialogs, transitions)
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        if config is None:
            config = DetectorConfig(
                name="power_extractor",
                type=DetectorType.OCR,
                dependencies=["power_credits_anchor"],
                params={
                    'vertical_padding': 5,  # Pixels to add above/below for better OCR
                    'upscale_factor': 5,    # Upscale for better OCR of small text
                    'psm_mode': 7,          # Tesseract page segmentation mode (single line)
                    'whitelist': '0123456789/PWR '
                }
            )
        super().__init__(config)
    
    def detect(self, image: np.ndarray, context: Optional[Dict[str, DetectionResult]] = None,
               **kwargs) -> DetectionResult:
        """Extract power value using OCR"""
        
        debug = kwargs.get('debug', False)
        
        # Check dependency
        if not context or 'power_credits_anchor' not in context:
            return create_error_result(
                self.get_name(),
                self.get_type(),
                "Missing dependency: power_credits_anchor"
            )
        
        anchor_result = context['power_credits_anchor']
        if not anchor_result.success:
            return create_error_result(
                self.get_name(),
                self.get_type(),
                "Power/credits anchor not detected"
            )
        
        # Get bounding box from anchor
        bbox = anchor_result.data.get('bbox')
        if not bbox:
            return create_error_result(
                self.get_name(),
                self.get_type(),
                "No bounding box in anchor result"
            )
        
        # Extract power value
        try:
            power_value = self._extract_power_ocr(image, bbox, debug)
            
            if power_value:
                return create_simple_result(
                    self.get_name(),
                    self.get_type(),
                    {
                        'power': power_value,
                        'raw_format': power_value  # e.g. "10/20"
                    },
                    confidence=0.8
                )
            else:
                return create_error_result(
                    self.get_name(),
                    self.get_type(),
                    "Could not parse power value from OCR"
                )
                
        except Exception as e:
            return create_error_result(
                self.get_name(),
                self.get_type(),
                f"OCR extraction failed: {str(e)}"
            )
    
    def _extract_power_ocr(self, image: np.ndarray, bbox: tuple, debug: bool = False) -> Optional[str]:
        """
        Internal OCR extraction for power values.
        
        Args:
            image: Full game frame
            bbox: (x, y, w, h) tuple from power_credits_anchor
            debug: Whether to print debug info
        
        Returns:
            Power string in "XX/YY" format, or None if extraction fails
        """
        try:
            import pytesseract
        except ImportError:
            if debug:
                print("⚠ pytesseract not installed, cannot extract power value")
            return None
        
        x, y, w, h = bbox
        params = self.config.params
        
        # Add vertical padding to capture full text height
        pad = params['vertical_padding']
        region_y = max(0, y - pad)
        region_h = min(h + 2*pad, image.shape[0] - region_y)
        
        # Extract the power/credits region
        region = image[region_y:region_y+region_h, x:x+w]
        
        if debug:
            print(f"  Power OCR region: {w}×{region_h}px")
        
        # Upscale significantly for small text
        scale = params['upscale_factor']
        region_scaled = cv2.resize(region, None, fx=scale, fy=scale, 
                                   interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(region_scaled, cv2.COLOR_BGR2GRAY)
        
        # Try multiple thresholding approaches
        _, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        
        # OCR configuration
        config = f'--psm {params["psm_mode"]} -c tessedit_char_whitelist={params["whitelist"]}'
        
        # Try OCR with different thresholds
        for i, thresh in enumerate([thresh1, thresh2, thresh3], 1):
            text = pytesseract.image_to_string(thresh, config=config).strip()
            
            if debug and text:
                print(f"  Power OCR attempt {i}: '{text}'")
            
            # Parse pattern: XX/YY (power format)
            # First try full pattern with PWR
            match = re.search(r'(\d+)\s*/\s*(\d+)\s+PWR', text)
            if match:
                return f"{match.group(1)}/{match.group(2)}"
            
            # Fallback: just find XX/YY pattern
            match = re.search(r'(\d+)\s*/\s*(\d+)', text)
            if match:
                return f"{match.group(1)}/{match.group(2)}"
        
        return None
    
    def get_name(self) -> str:
        return "power_extractor"
    
    def get_type(self) -> DetectorType:
        return DetectorType.OCR


class CreditsExtractor(BaseDetector):
    """
    Extract credits value from power/credits region.
    
    Format: "XX/YY PWR ZZZZZ" - extracts the ZZZZZ part
    Credits can be any non-negative integer (0, 50, 72314, 1000000, etc.)
    
    Dependencies:
    - power_credits_anchor: Provides the bounding box of the power/credits text region
    
    Success rate: ~75.9% (fails during opponent turns, dialogs, transitions)
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        if config is None:
            config = DetectorConfig(
                name="credits_extractor",
                type=DetectorType.OCR,
                dependencies=["power_credits_anchor"],
                params={
                    'vertical_padding': 5,  # Pixels to add above/below for better OCR
                    'upscale_factor': 5,    # Upscale for better OCR of small text
                    'psm_mode': 7,          # Tesseract page segmentation mode (single line)
                    'whitelist': '0123456789/PWR '
                }
            )
        super().__init__(config)
    
    def detect(self, image: np.ndarray, context: Optional[Dict[str, DetectionResult]] = None,
               **kwargs) -> DetectionResult:
        """Extract credits value using OCR"""
        
        debug = kwargs.get('debug', False)
        
        # Check dependency
        if not context or 'power_credits_anchor' not in context:
            return create_error_result(
                self.get_name(),
                self.get_type(),
                "Missing dependency: power_credits_anchor"
            )
        
        anchor_result = context['power_credits_anchor']
        if not anchor_result.success:
            return create_error_result(
                self.get_name(),
                self.get_type(),
                "Power/credits anchor not detected"
            )
        
        # Get bounding box from anchor
        bbox = anchor_result.data.get('bbox')
        if not bbox:
            return create_error_result(
                self.get_name(),
                self.get_type(),
                "No bounding box in anchor result"
            )
        
        # Extract credits value
        try:
            credits_value = self._extract_credits_ocr(image, bbox, debug)
            
            if credits_value is not None:  # Allow 0 as valid value
                return create_simple_result(
                    self.get_name(),
                    self.get_type(),
                    {
                        'credits': credits_value,
                        'credits_int': int(credits_value)
                    },
                    confidence=0.8
                )
            else:
                return create_error_result(
                    self.get_name(),
                    self.get_type(),
                    "Could not parse credits value from OCR"
                )
                
        except Exception as e:
            return create_error_result(
                self.get_name(),
                self.get_type(),
                f"OCR extraction failed: {str(e)}"
            )
    
    def _extract_credits_ocr(self, image: np.ndarray, bbox: tuple, debug: bool = False) -> Optional[str]:
        """
        Internal OCR extraction for credits values.
        
        Args:
            image: Full game frame
            bbox: (x, y, w, h) tuple from power_credits_anchor
            debug: Whether to print debug info
        
        Returns:
            Credits string (e.g. "72314"), or None if extraction fails
        """
        try:
            import pytesseract
        except ImportError:
            if debug:
                print("⚠ pytesseract not installed, cannot extract credits value")
            return None
        
        x, y, w, h = bbox
        params = self.config.params
        
        # Add vertical padding to capture full text height
        pad = params['vertical_padding']
        region_y = max(0, y - pad)
        region_h = min(h + 2*pad, image.shape[0] - region_y)
        
        # Extract the power/credits region
        region = image[region_y:region_y+region_h, x:x+w]
        
        if debug:
            print(f"  Credits OCR region: {w}×{region_h}px")
        
        # Upscale significantly for small text
        scale = params['upscale_factor']
        region_scaled = cv2.resize(region, None, fx=scale, fy=scale, 
                                   interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(region_scaled, cv2.COLOR_BGR2GRAY)
        
        # Try multiple thresholding approaches
        _, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        
        # OCR configuration
        config = f'--psm {params["psm_mode"]} -c tessedit_char_whitelist={params["whitelist"]}'
        
        # Try OCR with different thresholds
        results = []
        for i, thresh in enumerate([thresh1, thresh2, thresh3], 1):
            text = pytesseract.image_to_string(thresh, config=config).strip()
            
            if debug and text:
                print(f"  Credits OCR attempt {i}: '{text}'")
            
            if text:
                results.append(text)
        
        # Parse the OCR results
        for text in results:
            # Try to match full pattern: "XX/YY PWR ZZZZZ"
            match = re.search(r'(\d+)\s*/\s*(\d+)\s+PWR\s+(\d+)', text)
            if match:
                credits = match.group(3)
                if debug:
                    print(f"  Parsed credits from full pattern: {credits}")
                return credits
        
        # Fallback: Find all numbers and return the longest one
        # (Power is XX/YY so individual parts are 1-2 digits, credits are typically 4-6 digits)
        for text in results:
            all_numbers = re.findall(r'\d+', text)
            if debug and all_numbers:
                print(f"  All numbers found: {all_numbers}")
            
            if all_numbers:
                # Return the longest number (likely credits)
                longest = max(all_numbers, key=len)
                if len(longest) >= 3:  # Credits are typically 3+ digits
                    return longest
        
        return None
    
    def get_name(self) -> str:
        return "credits_extractor"
    
    def get_type(self) -> DetectorType:
        return DetectorType.OCR
