import cv2
import numpy as np
import os

class LicensePlateDetector:
    """
    Classical computer vision license plate detector using OpenCV.
    Detects rectangular regions with high-contrast edges and appropriate aspect ratios.
    """
    
    def __init__(self, min_area=800, max_area=50000, min_aspect_ratio=1.2, max_aspect_ratio=6.0,
                 canny_low=50, canny_high=150, min_rect_ratio=0.6):
        """
        Initialize the license plate detector with configurable parameters.
        
        Args:
            min_area: Minimum contour area for valid license plates
            max_area: Maximum contour area for valid license plates
            min_aspect_ratio: Minimum width/height ratio for license plates
            max_aspect_ratio: Maximum width/height ratio for license plates
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            min_rect_ratio: Minimum ratio of contour area to bounding rectangle area
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_rect_ratio = min_rect_ratio
        
    def preprocess_image(self, image):
        """
        Preprocess the input image for license plate detection with text-focused approach.
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed grayscale image ready for edge detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply adaptive histogram equalization for better local contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        equalized = clahe.apply(filtered)
        
        return equalized
    
    def detect_text_regions(self, gray_image):
        """
        Detect text-like regions using morphological operations.
        This helps identify areas with characters (like license plates).
        
        Args:
            gray_image: Preprocessed grayscale image
            
        Returns:
            Binary image with text regions highlighted
        """
        # Apply morphological gradient to highlight text boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grad = cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, kernel)
        
        # Threshold to get binary image
        _, text_thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Connect text characters horizontally
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        connected = cv2.morphologyEx(text_thresh, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Fill small gaps
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, vertical_kernel)
        
        return connected
    
    def detect_edges(self, gray_image):
        """
        Detect edges in the preprocessed image using Canny edge detector.
        
        Args:
            gray_image: Preprocessed grayscale image
            
        Returns:
            Binary edge image
        """
        # Apply Canny edge detection
        edges = cv2.Canny(gray_image, self.canny_low, self.canny_high)
        
        # Apply morphological operations to close gaps in edges
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_rect)
        
        # Use a horizontal kernel to connect horizontal edges (license plates are horizontal)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Use a small rectangular kernel for final cleanup
        final_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, final_kernel)
        
        return edges
    
    def find_contours(self, edge_image):
        """
        Find contours in the edge image using multiple approaches.
        
        Args:
            edge_image: Binary edge image
            
        Returns:
            List of contours
        """
        # Try multiple contour detection approaches
        contours1, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Combine and deduplicate contours
        all_contours = list(contours1) + list(contours2)
        
        # Remove very small contours early
        filtered_contours = [c for c in all_contours if cv2.contourArea(c) > 100]
        
        return filtered_contours
    
    def filter_license_plate_contours(self, contours, gray_image):
        """
        Advanced filtering specifically for license plate characteristics including text analysis.
        
        Args:
            contours: List of contours to filter
            gray_image: Original grayscale image for text analysis
            
        Returns:
            List of tuples (contour, bounding_rect, confidence_score)
        """
        valid_plates = []
        area_filtered = 0
        aspect_filtered = 0
        text_filtered = 0
        
        for i, contour in enumerate(contours):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate area and aspect ratio
            area = cv2.contourArea(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Debug logging for first few contours
            if i < 10:
                print(f"  Contour {i}: area={area:.0f}, aspect={aspect_ratio:.2f}, size=({w}x{h})")
            
            # More lenient area filtering for license plates
            if area < 500 or area > 12000:  # Expanded range for different plate sizes
                area_filtered += 1
                continue
                
            # License plate aspect ratio filtering - more lenient
            if aspect_ratio < 1.2 or aspect_ratio > 6.0:
                aspect_filtered += 1
                continue
            
            # Size filtering - more flexible dimensions
            if w < 40 or h < 12 or w > 400 or h > 120:
                aspect_filtered += 1
                continue
            
            # Text density analysis - more lenient threshold
            roi = gray_image[y:y+h, x:x+w]
            if roi.size == 0:
                continue
                
            # Check for text-like characteristics with lower threshold
            text_score = self.analyze_text_characteristics(roi)
            if text_score < 0.15:  # Much lower threshold for text detection
                text_filtered += 1
                continue
            
            # Check rectangularity
            rect_area = w * h
            rect_ratio = area / rect_area if rect_area > 0 else 0
            
            if rect_ratio < 0.6:  # License plates should be reasonably rectangular
                continue
            
            # Calculate enhanced confidence score
            confidence = self.calculate_license_plate_confidence(contour, area, aspect_ratio, rect_ratio, text_score)
            
            valid_plates.append((contour, (x, y, w, h), confidence))
            print(f"  ✅ License plate candidate: area={area:.0f}, aspect={aspect_ratio:.2f}, text_score={text_score:.2f}, confidence={confidence:.3f}")
        
        print(f"📋 Advanced filtering: {area_filtered} area-filtered, {aspect_filtered} size-filtered, {text_filtered} text-filtered")
        
        # Sort by confidence score (highest first)
        valid_plates.sort(key=lambda x: x[2], reverse=True)
        
        return valid_plates
    
    def analyze_text_characteristics(self, roi):
        """
        Simplified text analysis for license plate detection.
        
        Args:
            roi: Region of interest (grayscale image patch)
            
        Returns:
            Text-like score between 0 and 1
        """
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 30:
            return 0.0
        
        try:
            # Apply threshold to get binary image
            _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Simple edge density check
            edges = cv2.Canny(roi, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size
            
            # Count connected components (characters should create multiple components)
            num_labels, _ = cv2.connectedComponents(binary)
            component_score = min(num_labels / 10.0, 1.0)  # More components = more likely text
            
            # Aspect ratio check - license plates are wider than tall
            aspect_bonus = 0.2 if roi.shape[1] > roi.shape[0] * 1.5 else 0.0
            
            # Combine scores with more lenient weighting
            text_score = edge_density * 0.5 + component_score * 0.3 + aspect_bonus
            return min(text_score, 1.0)
            
        except Exception as e:
            print(f"Text analysis error: {e}")
            return 0.5  # Give neutral score if analysis fails
    
    def calculate_license_plate_confidence(self, contour, area, aspect_ratio, rect_ratio, text_score):
        """
        Calculate confidence score specifically for license plate detection.
        
        Args:
            contour: The contour
            area: Contour area
            aspect_ratio: Width/height ratio
            rect_ratio: Contour area / bounding rectangle area
            text_score: Text-like characteristics score
            
        Returns:
            Confidence score between 0 and 1
        """
        # Start with rectangularity
        confidence = rect_ratio * 0.3
        
        # Text score is very important for license plates
        confidence += text_score * 0.4
        
        # Aspect ratio scoring - typical license plates are 2:1 to 4:1
        if 2.0 <= aspect_ratio <= 4.0:
            confidence += 0.2  # Perfect aspect ratio
        elif 1.5 <= aspect_ratio <= 2.0:
            confidence += 0.1  # Acceptable square-ish plates
        else:
            confidence += 0.05  # Non-standard but possible
        
        # Size scoring - typical license plate areas
        if 1500 <= area <= 5000:
            confidence += 0.1  # Ideal size
        elif 800 <= area <= 1500 or 5000 <= area <= 8000:
            confidence += 0.05  # Acceptable size
        
        return min(confidence, 1.0)
    
    def filter_contours(self, contours):
        """
        Filter contours based on area, aspect ratio, and rectangularity.
        
        Args:
            contours: List of contours to filter
            
        Returns:
            List of tuples (contour, bounding_rect, confidence_score)
        """
        valid_plates = []
        area_filtered = 0
        aspect_filtered = 0
        rect_filtered = 0
        
        for i, contour in enumerate(contours):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate area and aspect ratio
            area = cv2.contourArea(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Debug logging for first few contours
            if i < 10:
                print(f"  Contour {i}: area={area:.0f}, aspect={aspect_ratio:.2f}, size=({w}x{h})")
            
            # Filter by area - be more restrictive for very large areas
            if area < self.min_area:
                area_filtered += 1
                continue
            
            # Much stricter filter for very large areas (likely car outline)
            if area > 15000:
                area_filtered += 1
                continue
                
            # Filter by aspect ratio
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                aspect_filtered += 1
                continue
            
            # Check rectangularity (how close the contour is to a rectangle)
            rect_area = w * h
            rect_ratio = area / rect_area if rect_area > 0 else 0
            
            if rect_ratio < self.min_rect_ratio:
                rect_filtered += 1
                continue
            
            # Calculate confidence score based on multiple factors
            confidence = self.calculate_confidence(contour, area, aspect_ratio, rect_ratio)
            
            valid_plates.append((contour, (x, y, w, h), confidence))
            print(f"  ✅ Valid plate candidate: area={area:.0f}, aspect={aspect_ratio:.2f}, rect_ratio={rect_ratio:.2f}, confidence={confidence:.3f}")
        
        print(f"📋 Filtering results: {area_filtered} area-filtered, {aspect_filtered} aspect-filtered, {rect_filtered} rectangularity-filtered")
        
        # Sort by confidence score (highest first)
        valid_plates.sort(key=lambda x: x[2], reverse=True)
        
        return valid_plates
    
    def calculate_confidence(self, contour, area, aspect_ratio, rect_ratio):
        """
        Calculate confidence score for a detected region.
        
        Args:
            contour: The contour
            area: Contour area
            aspect_ratio: Width/height ratio
            rect_ratio: Contour area / bounding rectangle area
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from rectangularity
        confidence = rect_ratio
        
        # Strong penalty for very large areas (likely entire car outline)
        if area > 20000:
            confidence *= 0.3  # Heavy penalty for very large regions
        
        # Bonus for optimal aspect ratio (around 2:1 to 4:1 for license plates)
        if 2.0 <= aspect_ratio <= 4.5:
            confidence *= 1.3  # Strong bonus for typical license plate ratios
        elif 1.5 <= aspect_ratio <= 2.0:
            confidence *= 1.1  # Moderate bonus for square-ish plates
        else:
            confidence *= 0.7  # Penalty for unusual ratios
        
        # Bonus for reasonable size (license plates are typically 1000-10000 pixels)
        if 1000 <= area <= 10000:
            confidence *= 1.2  # Bonus for typical size
        elif area < 1000:
            confidence *= 0.6  # Penalty for too small
        
        # Check contour approximation (rectangles should have 4 vertices)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            confidence *= 1.3  # Strong bonus for rectangular shape
        elif len(approx) <= 6:
            confidence *= 1.1  # Moderate bonus for nearly rectangular
        else:
            confidence *= 0.8  # Penalty for complex shapes
        
        return min(confidence, 1.0)
    
    def remove_overlapping_detections(self, valid_plates, overlap_threshold=0.1):
        """
        Remove overlapping detections using Non-Maximum Suppression with confidence filtering.
        
        Args:
            valid_plates: List of (contour, bounding_rect, confidence) tuples
            overlap_threshold: Minimum overlap ratio to consider as duplicate
            
        Returns:
            Filtered list of non-overlapping detections
        """
        if len(valid_plates) <= 1:
            return valid_plates
        
        # More lenient confidence filtering
        high_confidence_plates = [(c, r, conf) for c, r, conf in valid_plates if conf > 0.4]
        
        if not high_confidence_plates:
            # If no high-confidence detections, take the best candidates with lower threshold
            moderate_confidence_plates = [(c, r, conf) for c, r, conf in valid_plates if conf > 0.2]
            return moderate_confidence_plates[:1] if moderate_confidence_plates else []
        
        # Apply NMS to high-confidence detections
        final_plates = []
        
        for i, (contour1, rect1, conf1) in enumerate(high_confidence_plates):
            x1, y1, w1, h1 = rect1
            is_duplicate = False
            
            for j, (contour2, rect2, conf2) in enumerate(final_plates):
                x2, y2, w2, h2 = rect2
                
                # Calculate intersection over union (IoU)
                intersection_x = max(x1, x2)
                intersection_y = max(y1, y2)
                intersection_w = min(x1 + w1, x2 + w2) - intersection_x
                intersection_h = min(y1 + h1, y2 + h2) - intersection_y
                
                if intersection_w > 0 and intersection_h > 0:
                    intersection_area = intersection_w * intersection_h
                    union_area = w1 * h1 + w2 * h2 - intersection_area
                    iou = intersection_area / union_area if union_area > 0 else 0
                    
                    if iou > overlap_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                final_plates.append((contour1, rect1, conf1))
        
        # For typical single license plate images, return only the best detection
        return final_plates[:1]
    
    def detect_license_plates(self, image):
        """
        Focused license plate detection targeting vehicle rear areas.
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (number_of_plates, list_of_confidence_scores, list_of_bounding_rectangles)
        """
        print(f"🔍 Starting detection on image shape: {image.shape}")
        print(f"📊 Detection parameters: area({self.min_area}-8000), aspect({self.min_aspect_ratio}-{self.max_aspect_ratio}), canny({self.canny_low}-{self.canny_high})")
        
        # Preprocess image
        gray = self.preprocess_image(image)
        
        # Focus on lower 60% of image where license plates are typically located
        height = gray.shape[0]
        roi_start = int(height * 0.4)  # Start from 40% down from top
        gray_roi = gray[roi_start:, :]
        
        print(f"🎯 Focusing on vehicle area: {gray_roi.shape} (lower 60% of image)")
        
        # Apply more aggressive preprocessing for license plate detection
        processed = self.enhance_license_plate_regions(gray_roi)
        
        # Find contours in the processed ROI
        contours = self.find_license_plate_contours(processed)
        
        # Adjust contour coordinates back to full image
        adjusted_contours = []
        for contour in contours:
            adjusted_contour = contour.copy()
            adjusted_contour[:, :, 1] += roi_start  # Add ROI offset to y coordinates
            adjusted_contours.append(adjusted_contour)
        
        print(f"📐 Found {len(adjusted_contours)} contours in vehicle area")
        
        # Filter with strict license plate criteria
        valid_plates = self.filter_strict_license_plates(adjusted_contours, gray)
        print(f"✅ {len(valid_plates)} contours passed strict filtering")
        
        # If no valid plates found, try alternative edge-based detection
        if len(valid_plates) == 0:
            print("🔄 No plates found, trying edge-based detection...")
            edge_plates = self.fallback_edge_detection(gray, roi_start)
            valid_plates.extend(edge_plates)
            print(f"🔍 Edge detection found {len(edge_plates)} additional candidates")
        
        # Remove overlapping detections
        final_plates = self.remove_overlapping_detections(valid_plates)
        print(f"🎯 Final detection count: {len(final_plates)}")
        
        # Extract results
        num_plates = len(final_plates)
        confidence_scores = [conf for _, _, conf in final_plates]
        bounding_rects = [rect for _, rect, _ in final_plates]
        
        return num_plates, confidence_scores, bounding_rects
    
    def enhance_license_plate_regions(self, gray_roi):
        """
        Enhanced preprocessing specifically for license plate detection.
        
        Args:
            gray_roi: Grayscale ROI focusing on vehicle area
            
        Returns:
            Processed binary image highlighting license plate regions
        """
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(gray_roi, 11, 17, 17)
        
        # Adaptive threshold to handle varying lighting
        adaptive_thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to connect text characters
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        
        # Close gaps horizontally (connect characters)
        processed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel_horizontal)
        # Close gaps vertically (solidify text)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_vertical)
        
        return processed
    
    def find_license_plate_contours(self, processed_roi):
        """
        Find contours specifically targeting license plate shapes.
        
        Args:
            processed_roi: Binary processed ROI
            
        Returns:
            List of contours
        """
        # Find external contours
        contours, _ = cv2.findContours(processed_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter out very small contours immediately
        min_contour_area = 300
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        
        return filtered_contours
    
    def filter_strict_license_plates(self, contours, full_gray_image):
        """
        Very strict filtering specifically for license plates.
        
        Args:
            contours: List of contours
            full_gray_image: Full grayscale image for analysis
            
        Returns:
            List of valid license plate candidates
        """
        valid_plates = []
        
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            if i < 5:  # Debug first few
                print(f"  Candidate {i}: area={area:.0f}, aspect={aspect_ratio:.2f}, size=({w}x{h}), pos=({x},{y})")
            
            # Debug output for filtering
            if i < 5:
                print(f"    Filtering: area={area:.0f} (want 600-15000), aspect={aspect_ratio:.2f} (want 2.0-5.0), size=({w}x{h})")
            
            # More lenient license plate criteria
            # Area: expanded range for license plates in different image sizes
            if area < 600 or area > 15000:
                if i < 5: print(f"    ❌ Rejected by area")
                continue
                
            # Aspect ratio: license plates are typically 2:1 to 5:1
            if aspect_ratio < 2.0 or aspect_ratio > 5.0:
                if i < 5: print(f"    ❌ Rejected by aspect ratio")
                continue
                
            # Dimensions: more flexible for different image scales
            if w < 50 or w > 300 or h < 15 or h > 80:
                if i < 5: print(f"    ❌ Rejected by dimensions")
                continue
            
            # Check rectangularity
            rect_area = w * h
            rect_ratio = area / rect_area if rect_area > 0 else 0
            if rect_ratio < 0.7:  # Should be quite rectangular
                continue
            
            # Position check - license plates should be in reasonable vehicle positions
            # Not at very top or very edges of the ROI
            roi_height = full_gray_image.shape[0] * 0.6  # ROI is 60% of image
            relative_y = (y - int(full_gray_image.shape[0] * 0.4)) / roi_height
            if relative_y < 0.1 or relative_y > 0.9:  # Skip if too close to ROI edges
                continue
            
            # Text analysis on the region
            roi = full_gray_image[y:y+h, x:x+w]
            if roi.size == 0:
                continue
                
            text_score = self.analyze_license_plate_text(roi)
            
            # Calculate confidence
            confidence = self.calculate_strict_confidence(contour, area, aspect_ratio, rect_ratio, text_score, w, h)
            
            if confidence > 0.3:  # Only accept reasonably confident detections
                valid_plates.append((contour, (x, y, w, h), confidence))
                print(f"  ✅ License plate: area={area:.0f}, aspect={aspect_ratio:.2f}, text={text_score:.2f}, conf={confidence:.3f}")
        
        # Sort by confidence
        valid_plates.sort(key=lambda x: x[2], reverse=True)
        return valid_plates
    
    def analyze_license_plate_text(self, roi):
        """
        Analyze text characteristics specifically for license plates.
        
        Args:
            roi: License plate region of interest
            
        Returns:
            Text score between 0 and 1
        """
        if roi.size == 0 or roi.shape[0] < 15 or roi.shape[1] < 50:
            return 0.0
        
        try:
            # Threshold the ROI
            _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Count white pixels (text should have reasonable coverage)
            white_ratio = np.count_nonzero(binary) / binary.size
            white_score = 1.0 if 0.1 < white_ratio < 0.4 else 0.5
            
            # Edge density
            edges = cv2.Canny(roi, 30, 100)
            edge_density = np.count_nonzero(edges) / edges.size
            edge_score = min(edge_density * 10, 1.0)
            
            # Check for horizontal structure (license plates have horizontal layout)
            horizontal_projection = np.sum(binary, axis=1)
            has_horizontal_structure = np.var(horizontal_projection) > 1000
            structure_score = 0.3 if has_horizontal_structure else 0.0
            
            return white_score * 0.4 + edge_score * 0.4 + structure_score + 0.2
            
        except Exception:
            return 0.2
    
    def calculate_strict_confidence(self, contour, area, aspect_ratio, rect_ratio, text_score, width, height):
        """
        Calculate confidence score with strict license plate criteria.
        
        Args:
            contour: Contour
            area: Area
            aspect_ratio: Width/height ratio
            rect_ratio: Rectangularity ratio
            text_score: Text analysis score
            width: Width
            height: Height
            
        Returns:
            Confidence score
        """
        confidence = 0.0
        
        # Rectangularity (30%)
        confidence += rect_ratio * 0.3
        
        # Text score (40%)
        confidence += text_score * 0.4
        
        # Aspect ratio bonus (20%)
        if 2.5 <= aspect_ratio <= 4.0:  # Ideal license plate ratio
            confidence += 0.2
        elif 2.0 <= aspect_ratio <= 2.5 or 4.0 <= aspect_ratio <= 4.5:
            confidence += 0.1
        
        # Size bonus (10%)
        if 1500 <= area <= 4000 and 80 <= width <= 160 and 25 <= height <= 45:
            confidence += 0.1
        elif 1000 <= area <= 6000:
            confidence += 0.05
            
        return min(confidence, 1.0)
    
    def fallback_edge_detection(self, full_gray_image, roi_start):
        """
        Fallback edge-based detection method for license plates.
        
        Args:
            full_gray_image: Full grayscale image
            roi_start: Start position of ROI
            
        Returns:
            List of license plate candidates
        """
        height = full_gray_image.shape[0]
        roi_end = height
        
        # Work on the ROI
        roi = full_gray_image[roi_start:roi_end, :]
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        
        # Edge detection with multiple thresholds
        edges1 = cv2.Canny(blurred, 30, 100)
        edges2 = cv2.Canny(blurred, 50, 150)
        combined_edges = cv2.bitwise_or(edges1, edges2)
        
        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours for license plate characteristics
        candidates = []
        
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Adjust coordinates back to full image
            full_y = y + roi_start
            
            if i < 3:  # Debug first few
                print(f"  Edge candidate {i}: area={area:.0f}, aspect={aspect_ratio:.2f}, size=({w}x{h}), pos=({x},{full_y})")
            
            # License plate criteria (more lenient for fallback)
            if (800 <= area <= 8000 and 
                2.0 <= aspect_ratio <= 5.0 and 
                50 <= w <= 250 and 
                15 <= h <= 70):
                
                # Simple confidence based on area and aspect ratio
                confidence = 0.3  # Base confidence for fallback method
                
                # Bonus for good aspect ratio
                if 2.5 <= aspect_ratio <= 4.0:
                    confidence += 0.2
                
                # Bonus for reasonable size
                if 1000 <= area <= 4000:
                    confidence += 0.1
                
                candidates.append((contour, (x, full_y, w, h), confidence))
                print(f"  ✅ Edge candidate accepted: conf={confidence:.3f}")
        
        # Sort by confidence and return best candidates
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:2]  # Return up to 2 best candidates
    
    def draw_detections(self, image, bounding_rects, confidence_scores):
        """
        Create output showing detected license plates - either cropped plates or full image with boxes.
        
        Args:
            image: Input BGR image
            bounding_rects: List of bounding rectangles
            confidence_scores: List of confidence scores
            
        Returns:
            Image with license plate extractions or bounding boxes
        """
        if len(bounding_rects) == 0:
            return image
        
        # If we found license plates, create a composite image showing both original and cropped plates
        if len(bounding_rects) == 1:
            # For single detection, show the cropped license plate prominently
            x, y, w, h = bounding_rects[0]
            
            # Add some padding around the license plate
            padding = 10
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(image.shape[1], x + w + padding)
            y_end = min(image.shape[0], y + h + padding)
            
            # Extract license plate region
            plate_crop = image[y_start:y_end, x_start:x_end]
            
            # Scale up the license plate for better visibility
            scale_factor = 3
            if plate_crop.shape[0] > 0 and plate_crop.shape[1] > 0:
                scaled_plate = cv2.resize(plate_crop, 
                                        (plate_crop.shape[1] * scale_factor, 
                                         plate_crop.shape[0] * scale_factor), 
                                        interpolation=cv2.INTER_CUBIC)
            else:
                scaled_plate = plate_crop
            
            # Create result image with original and scaled plate
            result_height = max(image.shape[0], scaled_plate.shape[0] + 100)
            result_width = image.shape[1] + scaled_plate.shape[1] + 50
            result_image = np.zeros((result_height, result_width, 3), dtype=np.uint8)
            result_image[:] = (240, 240, 240)  # Light gray background
            
            # Place original image with bounding box
            original_with_box = image.copy()
            cv2.rectangle(original_with_box, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Add label
            label = f"Detected Plate: {confidence_scores[0]:.2f}"
            cv2.putText(original_with_box, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Place images in result
            result_image[:image.shape[0], :image.shape[1]] = original_with_box
            
            # Place scaled license plate
            if scaled_plate.shape[0] > 0 and scaled_plate.shape[1] > 0:
                plate_y = 50
                plate_x = image.shape[1] + 25
                
                # Add border around license plate
                cv2.rectangle(result_image, 
                            (plate_x - 5, plate_y - 5), 
                            (plate_x + scaled_plate.shape[1] + 5, plate_y + scaled_plate.shape[0] + 5), 
                            (0, 255, 0), 3)
                
                # Place the scaled license plate
                result_image[plate_y:plate_y + scaled_plate.shape[0], 
                           plate_x:plate_x + scaled_plate.shape[1]] = scaled_plate
                
                # Add title above the license plate
                cv2.putText(result_image, "Extracted License Plate", 
                           (plate_x, plate_y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            return result_image
        
        else:
            # Multiple detections - use traditional bounding box approach
            result_image = image.copy()
            
            for i, ((x, y, w, h), confidence) in enumerate(zip(bounding_rects, confidence_scores)):
                # Draw bounding rectangle
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add label with confidence score
                label = f"Plate {i+1}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(result_image, (x, y - label_size[1] - 10), 
                             (x + label_size[0], y), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(result_image, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            return result_image
    
    def detect_and_save(self, input_path, output_path):
        """
        Complete detection pipeline: load image, detect plates, save result.
        
        Args:
            input_path: Path to input image
            output_path: Path to save output image
            
        Returns:
            Tuple of (number_of_plates, list_of_confidence_scores)
        """
        # Load image
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Failed to load image: {input_path}")
        
        # Detect license plates
        num_plates, confidence_scores, bounding_rects = self.detect_license_plates(image)
        
        # Draw detections
        result_image = self.draw_detections(image, bounding_rects, confidence_scores)
        
        # Save result
        cv2.imwrite(output_path, result_image)
        
        return num_plates, confidence_scores

def main():
    """
    Command-line interface for license plate detection.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python plate_detector.py <input_image> [output_image]")
        print("Example: python plate_detector.py input.jpg output.jpg")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output.jpg"
    
    # Initialize detector with default parameters
    detector = LicensePlateDetector()
    
    try:
        # Detect and save
        num_plates, confidence_scores = detector.detect_and_save(input_path, output_path)
        
        # Print results
        print(f"✅ Detection completed successfully!")
        print(f"📸 Input: {input_path}")
        print(f"💾 Output: {output_path}")
        print(f"🎯 License plates detected: {num_plates}")
        
        if confidence_scores:
            print("🔍 Confidence scores:")
            for i, score in enumerate(confidence_scores):
                print(f"   Plate {i+1}: {score:.3f}")
        else:
            print("ℹ️  No license plates detected in the image.")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
