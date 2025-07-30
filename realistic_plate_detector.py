import cv2
import numpy as np
import logging

class RealisticLicensePlateDetector:
    """
    A realistic license plate detector that focuses on accurate detection
    without making false claims about text extraction capabilities.
    """
    
    def __init__(self, min_area=800, max_area=8000, min_aspect=2.0, max_aspect=5.0, 
                 canny_low=80, canny_high=200):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.canny_low = canny_low
        self.canny_high = canny_high
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def detect_and_save(self, input_path, output_path):
        """Detect license plates and save annotated result."""
        try:
            # Load image
            image = cv2.imread(input_path)
            if image is None:
                self.logger.error(f"Could not load image: {input_path}")
                return 0, [], []
            
            self.logger.info(f"🔍 Processing image: {image.shape}")
            
            # Create a copy for annotation
            result_image = image.copy()
            
            # Detect license plates using multiple methods
            plates = self.detect_license_plates(image)
            
            self.logger.info(f"📊 Found {len(plates)} license plate candidates")
            
            # Annotate and extract details
            plate_details = []
            confidence_scores = []
            
            for i, (x, y, w, h, confidence, method) in enumerate(plates):
                # Draw bounding box
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add label
                label = f"Plate {i+1} ({confidence:.1%})"
                cv2.putText(result_image, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Extract plate region
                plate_roi = image[y:y+h, x:x+w]
                
                # Analyze the plate region honestly
                text_analysis = self.analyze_plate_text(plate_roi)
                
                plate_info = {
                    'plate_number': i + 1,
                    'text': text_analysis,
                    'confidence': confidence,
                    'method': method,
                    'position': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'roi': plate_roi
                }
                
                plate_details.append(plate_info)
                confidence_scores.append(confidence)
                
                self.logger.info(f"✅ Detected license plate {i+1}: text='{text_analysis}', "
                               f"pos=({x},{y}), size=({w}x{h}), confidence={confidence:.3f}, method={method}")
            
            # Save result image
            cv2.imwrite(output_path, result_image)
            self.logger.info(f"💾 Saved result to: {output_path}")
            
            return len(plates), confidence_scores, plate_details
            
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return 0, [], []
    
    def detect_license_plates(self, image):
        """Detect license plates using computer vision techniques."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing
            processed = self.preprocess_image(gray)
            
            # Use multiple detection methods
            plates = []
            
            # Method 1: Edge-based detection
            edge_plates = self.detect_by_edges(processed, gray)
            for plate in edge_plates:
                plates.append(plate + ('edge-based',))
            
            # Method 2: Morphological operations
            morph_plates = self.detect_by_morphology(processed, gray)
            for plate in morph_plates:
                plates.append(plate + ('morphological',))
            
            # Method 3: Contour analysis
            contour_plates = self.detect_by_contours(processed, gray)
            for plate in contour_plates:
                plates.append(plate + ('contour-based',))
            
            # Remove overlapping detections
            filtered_plates = self.remove_overlaps(plates)
            
            return filtered_plates
            
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []
    
    def preprocess_image(self, gray):
        """Preprocess image for better detection."""
        try:
            # Noise reduction
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Histogram equalization
            equalized = cv2.equalizeHist(denoised)
            
            return equalized
            
        except Exception as e:
            return gray
    
    def detect_by_edges(self, processed, original):
        """Detect license plates using edge detection."""
        try:
            plates = []
            
            # Apply Canny edge detection
            edges = cv2.Canny(processed, self.canny_low, self.canny_high)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                
                # Filter based on size and aspect ratio
                if (self.min_area <= area <= self.max_area and
                    self.min_aspect <= aspect_ratio <= self.max_aspect):
                    
                    # Calculate confidence based on rectangularity and edge density
                    confidence = self.calculate_confidence(contour, original[y:y+h, x:x+w])
                    
                    if confidence > 0.15:  # Minimum confidence threshold
                        plates.append((x, y, w, h, confidence))
            
            return plates
            
        except Exception as e:
            return []
    
    def detect_by_morphology(self, processed, original):
        """Detect license plates using morphological operations."""
        try:
            plates = []
            
            # Apply morphological operations to highlight text regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(processed, cv2.MORPH_GRADIENT, kernel)
            
            # Threshold
            _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                
                if (self.min_area <= area <= self.max_area and
                    self.min_aspect <= aspect_ratio <= self.max_aspect):
                    
                    confidence = self.calculate_confidence(contour, original[y:y+h, x:x+w])
                    
                    if confidence > 0.15:
                        plates.append((x, y, w, h, confidence))
            
            return plates
            
        except Exception as e:
            return []
    
    def detect_by_contours(self, processed, original):
        """Detect license plates using contour analysis."""
        try:
            plates = []
            
            # Apply adaptive threshold
            adaptive = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                
                if (self.min_area <= area <= self.max_area and
                    self.min_aspect <= aspect_ratio <= self.max_aspect):
                    
                    confidence = self.calculate_confidence(contour, original[y:y+h, x:x+w])
                    
                    if confidence > 0.15:
                        plates.append((x, y, w, h, confidence))
            
            return plates
            
        except Exception as e:
            return []
    
    def calculate_confidence(self, contour, roi):
        """Calculate confidence score for a potential license plate."""
        try:
            # Basic geometric analysis
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            
            # Rectangularity score
            rectangularity = area / rect_area if rect_area > 0 else 0
            
            # Aspect ratio score
            aspect_ratio = w / h if h > 0 else 0
            aspect_score = 1.0 if self.min_aspect <= aspect_ratio <= self.max_aspect else 0.5
            
            # Text-like features in the ROI
            text_score = self.analyze_text_features(roi)
            
            # Combine scores
            confidence = (rectangularity * 0.3 + aspect_score * 0.3 + text_score * 0.4)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            return 0.0
    
    def analyze_text_features(self, roi):
        """Analyze if the ROI contains text-like features."""
        try:
            if roi.size == 0:
                return 0.0
            
            # Convert to grayscale if needed
            if len(roi.shape) == 3:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = roi
            
            # Apply edge detection
            edges = cv2.Canny(gray_roi, 50, 150)
            
            # Count edges (text regions should have many edges)
            edge_density = np.count_nonzero(edges) / edges.size
            
            # Look for rectangular components (characters)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            char_like_contours = 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect = w / h if h > 0 else 0
                area = cv2.contourArea(contour)
                
                # Check if this looks like a character
                if 0.2 <= aspect <= 2.0 and area > 20:
                    char_like_contours += 1
            
            # Score based on edge density and character-like shapes
            text_score = min(edge_density * 2 + (char_like_contours / 10), 1.0)
            
            return text_score
            
        except Exception as e:
            return 0.0
    
    def analyze_plate_text(self, plate_roi):
        """Honestly analyze what can be determined about the plate text."""
        try:
            if plate_roi.size == 0:
                return "Empty region"
            
            # Convert to grayscale
            if len(plate_roi.shape) == 3:
                gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_roi
            
            # Analyze the characteristics we can detect
            height, width = gray.shape
            
            # Check for text-like patterns
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size
            
            # Find potential character regions
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            char_regions = 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect = w / h if h > 0 else 0
                area = cv2.contourArea(contour)
                
                if 0.2 <= aspect <= 2.0 and area > 15:
                    char_regions += 1
            
            # Provide honest assessment
            if char_regions >= 4 and edge_density > 0.1:
                return "Text detected (OCR not available)"
            elif char_regions >= 2:
                return "Possible text (unclear)"
            else:
                return "No clear text pattern"
                
        except Exception as e:
            return "Analysis failed"
    
    def remove_overlaps(self, plates):
        """Remove overlapping plate detections."""
        try:
            if len(plates) <= 1:
                return plates
            
            # Sort by confidence (descending)
            plates.sort(key=lambda x: x[4], reverse=True)
            
            keep = []
            for i, (x1, y1, w1, h1, conf1, method1) in enumerate(plates):
                overlaps = False
                
                for x2, y2, w2, h2, conf2, method2 in keep:
                    # Calculate overlap
                    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    overlap_area = overlap_x * overlap_y
                    
                    area1 = w1 * h1
                    area2 = w2 * h2
                    
                    # If overlap is significant, skip this detection
                    if overlap_area > 0.3 * min(area1, area2):
                        overlaps = True
                        break
                
                if not overlaps:
                    keep.append((x1, y1, w1, h1, conf1, method1))
            
            return keep
            
        except Exception as e:
            return plates