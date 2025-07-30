import cv2
import numpy as np
import os

class ImprovedLicensePlateDetector:
    """
    Improved license plate detector with better text detection and filtering.
    """
    
    def __init__(self, min_area=1000, max_area=8000, min_aspect_ratio=2.0, max_aspect_ratio=5.0,
                 canny_low=80, canny_high=200, min_rect_ratio=0.7):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_rect_ratio = min_rect_ratio
        
    def detect_and_save(self, input_path, output_path):
        """Main detection method that processes image and saves result."""
        try:
            # Load image
            image = cv2.imread(input_path)
            if image is None:
                print(f"❌ Could not load image: {input_path}")
                return 0, []
            
            print(f"🔍 Processing image: {image.shape}")
            
            # Create a copy for drawing
            result_image = image.copy()
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Preprocess image
            processed = self.preprocess_for_text_detection(gray)
            
            # Find license plate candidates using multiple methods
            candidates = self.find_license_plate_candidates(processed, gray)
            
            # Filter and validate candidates
            valid_plates = self.validate_license_plates(candidates, gray, image.shape)
            
            print(f"📊 Found {len(valid_plates)} valid license plate candidates")
            
            # Draw results
            confidence_scores = []
            for i, (contour, rect, confidence) in enumerate(valid_plates):
                x, y, w, h = rect
                
                # Draw rectangle around detected plate
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                # Add confidence score
                confidence_scores.append(confidence)
                cv2.putText(result_image, f'Plate {i+1}: {confidence:.2f}', 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                print(f"✅ Detected license plate {i+1}: pos=({x},{y}), size=({w}x{h}), confidence={confidence:.3f}")
            
            # Save result
            cv2.imwrite(output_path, result_image)
            print(f"💾 Saved result to: {output_path}")
            
            return len(valid_plates), confidence_scores
            
        except Exception as e:
            print(f"❌ Detection error: {str(e)}")
            return 0, []
    
    def preprocess_for_text_detection(self, gray):
        """Preprocess image specifically for license plate text detection."""
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply CLAHE for better local contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(filtered)
        
        return enhanced
    
    def find_license_plate_candidates(self, processed, original_gray):
        """Find potential license plate regions using multiple detection methods."""
        candidates = []
        
        # Method 1: Edge-based detection
        edge_candidates = self.detect_with_edges(processed)
        candidates.extend(edge_candidates)
        
        # Method 2: Text region detection
        text_candidates = self.detect_text_regions(processed)
        candidates.extend(text_candidates)
        
        # Method 3: Contour-based detection with morphological operations
        morph_candidates = self.detect_with_morphology(processed)
        candidates.extend(morph_candidates)
        
        print(f"📍 Found {len(candidates)} initial candidates")
        return candidates
    
    def detect_with_edges(self, gray):
        """Detect license plates using edge detection."""
        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        
        # Apply morphological operations to connect text
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_horizontal)
        
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel_vertical)
        
        # Find contours
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            if area > 500:  # Initial area filter
                candidates.append((contour, (x, y, w, h), 0.5))
        
        return candidates
    
    def detect_text_regions(self, gray):
        """Detect text-like regions that could be license plates."""
        # Create text detection kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Apply morphological gradient to highlight text
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Threshold
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Connect text horizontally
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Find contours
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            if area > 300:
                candidates.append((contour, (x, y, w, h), 0.6))
        
        return candidates
    
    def detect_with_morphology(self, gray):
        """Detect license plates using morphological operations."""
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Create rectangular kernel for license plate detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        
        # Apply morphological operations
        morph = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)
        
        # Threshold again
        _, morph = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Connect regions
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        connected = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel2)
        
        # Find contours
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            if area > 400:
                candidates.append((contour, (x, y, w, h), 0.4))
        
        return candidates
    
    def validate_license_plates(self, candidates, gray, image_shape):
        """Validate candidates to find actual license plates."""
        valid_plates = []
        
        for contour, rect, base_confidence in candidates:
            x, y, w, h = rect
            area = cv2.contourArea(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Apply strict filters
            if not self.passes_geometric_filters(area, aspect_ratio, w, h):
                continue
            
            # Extract ROI for text analysis
            roi = gray[y:y+h, x:x+w] if y+h <= gray.shape[0] and x+w <= gray.shape[1] else None
            if roi is None or roi.size == 0:
                continue
            
            # Analyze text characteristics
            text_score = self.analyze_license_plate_text(roi)
            if text_score < 0.1:  # Lower minimum text score to be less restrictive
                continue
            
            # Check rectangularity
            rect_area = w * h
            rect_ratio = area / rect_area if rect_area > 0 else 0
            if rect_ratio < self.min_rect_ratio:
                continue
            
            # Calculate final confidence
            confidence = self.calculate_final_confidence(
                contour, area, aspect_ratio, rect_ratio, text_score, y, image_shape[0]
            )
            
            if confidence > 0.2:  # Lower minimum confidence threshold
                valid_plates.append((contour, rect, confidence))
        
        # Sort by confidence and remove overlapping detections
        valid_plates = self.remove_overlapping_detections(valid_plates)
        valid_plates.sort(key=lambda x: x[2], reverse=True)
        
        return valid_plates
    
    def passes_geometric_filters(self, area, aspect_ratio, width, height):
        """Check if candidate passes geometric filters."""
        # Area filter
        if area < self.min_area or area > self.max_area:
            return False
        
        # Aspect ratio filter
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False
        
        # Size filter - more lenient
        if width < 40 or height < 12 or width > 400 or height > 120:
            return False
        
        return True
    
    def analyze_license_plate_text(self, roi):
        """Analyze if the region contains license plate-like text."""
        if roi.size == 0:
            return 0.0
        
        try:
            # Apply adaptive threshold
            binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Analyze character-like components
            num_labels, labels = cv2.connectedComponents(binary)
            
            char_count = 0
            valid_chars = 0
            
            for label in range(1, num_labels):
                component_mask = (labels == label).astype(np.uint8) * 255
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    area = cv2.contourArea(contours[0])
                    aspect = w / h if h > 0 else 0
                    
                    char_count += 1
                    
                    # Check if component looks like a character - more lenient
                    if (0.1 <= aspect <= 3.0 and  # More flexible character aspect ratio
                        10 <= area <= 2000 and     # Larger character area range
                        w >= 3 and h >= 5 and      # Smaller minimum size
                        w <= 80 and h <= 100):     # Larger maximum size
                        valid_chars += 1
            
            # Calculate text scores - more generous scoring
            char_density = valid_chars / max(char_count, 1) if char_count > 0 else 0.5
            char_count_score = min(valid_chars / 4.0, 1.0)  # License plates have ~3-8 chars
            
            # Edge density (license plates have strong edges)
            edges = cv2.Canny(roi, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size
            
            # Contrast (license plates have high contrast)
            contrast = np.std(roi) / 255.0
            
            # Basic shape score - license plates are rectangular
            aspect_score = 0.5 if roi.shape[1] > roi.shape[0] * 1.5 else 0.2
            
            # Combine scores with more balanced weighting
            text_score = (char_density * 0.3 + 
                         char_count_score * 0.2 + 
                         edge_density * 0.3 + 
                         contrast * 0.1 +
                         aspect_score * 0.1)
            
            return min(text_score, 1.0)
            
        except Exception as e:
            print(f"Text analysis error: {e}")
            return 0.0
    
    def calculate_final_confidence(self, contour, area, aspect_ratio, rect_ratio, text_score, y_pos, image_height):
        """Calculate final confidence score."""
        # Base confidence from text analysis (most important)
        confidence = text_score * 0.5
        
        # Rectangularity
        confidence += rect_ratio * 0.2
        
        # Aspect ratio scoring
        if 2.5 <= aspect_ratio <= 4.0:
            confidence += 0.15
        elif 2.0 <= aspect_ratio < 2.5 or 4.0 < aspect_ratio <= 5.0:
            confidence += 0.1
        else:
            confidence += 0.05
        
        # Size scoring
        if 1500 <= area <= 6000:
            confidence += 0.1
        elif 1000 <= area < 1500 or 6000 < area <= 8000:
            confidence += 0.05
        
        # Position scoring (license plates are typically in lower part of image)
        relative_pos = y_pos / image_height
        if 0.4 <= relative_pos <= 0.9:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def remove_overlapping_detections(self, detections):
        """Remove overlapping detections using Non-Maximum Suppression."""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda x: x[2], reverse=True)
        
        keep = []
        for i, (contour1, rect1, conf1) in enumerate(detections):
            x1, y1, w1, h1 = rect1
            
            # Check for significant overlap with higher confidence detections
            overlaps = False
            for j in range(i):
                if j < len(keep):
                    _, rect2, _ = keep[j]
                    x2, y2, w2, h2 = rect2
                    
                    # Calculate overlap
                    overlap_area = max(0, min(x1+w1, x2+w2) - max(x1, x2)) * max(0, min(y1+h1, y2+h2) - max(y1, y2))
                    union_area = w1*h1 + w2*h2 - overlap_area
                    
                    if overlap_area / union_area > 0.3:  # 30% overlap threshold
                        overlaps = True
                        break
            
            if not overlaps:
                keep.append((contour1, rect1, conf1))
        
        return keep