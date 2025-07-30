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
        Precise license plate detection using MSER (Maximally Stable Extremal Regions) approach.
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (number_of_plates, list_of_confidence_scores, list_of_bounding_rectangles)
        """
        print(f"🔍 Starting precise detection on image shape: {image.shape}")
        
        # Preprocess image
        gray = self.preprocess_image(image)
        
        # Focus on lower 70% of image where license plates are located
        height = gray.shape[0]
        roi_start = int(height * 0.3)  # Start from 30% down from top
        gray_roi = gray[roi_start:, :]
        
        print(f"🎯 Focusing on vehicle area: {gray_roi.shape} (lower 70% of image)")
        
        # Use multiple detection approaches
        candidates = []
        
        # Method 1: MSER-based text region detection
        mser_candidates = self.detect_with_mser(gray_roi, roi_start)
        candidates.extend(mser_candidates)
        print(f"📍 MSER detection found {len(mser_candidates)} candidates")
        
        # Method 2: Sobel edge-based detection
        sobel_candidates = self.detect_with_sobel(gray_roi, roi_start)
        candidates.extend(sobel_candidates)
        print(f"📊 Sobel detection found {len(sobel_candidates)} candidates")
        
        # Method 3: Adaptive threshold detection
        adaptive_candidates = self.detect_with_adaptive_threshold(gray_roi, roi_start)
        candidates.extend(adaptive_candidates)
        print(f"🔧 Adaptive detection found {len(adaptive_candidates)} candidates")
        
        # Method 4: Fallback contour detection if no good candidates found
        if len(candidates) == 0:
            print("🔄 No candidates found, trying fallback contour detection...")
            contour_candidates = self.detect_with_contours(gray_roi, roi_start)
            candidates.extend(contour_candidates)
            print(f"📐 Contour detection found {len(contour_candidates)} candidates")
        
        # Combine and filter all candidates
        if candidates:
            # Remove duplicates and overlaps
            filtered_candidates = self.filter_and_merge_candidates(candidates, gray)
            print(f"✅ {len(filtered_candidates)} candidates after filtering")
            
            # Extract results
            num_plates = len(filtered_candidates)
            confidence_scores = [conf for _, _, conf in filtered_candidates]
            bounding_rects = [rect for _, rect, _ in filtered_candidates]
            
            return num_plates, confidence_scores, bounding_rects
        else:
            print("❌ No license plates detected")
            return 0, [], []
    
    def enhance_license_plate_regions(self, gray_roi):
        """
        Targeted preprocessing to find rectangular license plate regions.
        
        Args:
            gray_roi: Grayscale ROI focusing on vehicle area
            
        Returns:
            Processed binary image highlighting license plate regions
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        
        # Use Canny edge detection to find edges
        edges = cv2.Canny(blurred, 50, 200)
        
        # Small morphological closing to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def find_license_plate_contours(self, processed_roi):
        """
        Find contours using hierarchical approach to avoid merged regions.
        
        Args:
            processed_roi: Binary processed ROI
            
        Returns:
            List of contours
        """
        # Find all contours with hierarchy
        contours, hierarchy = cv2.findContours(processed_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area and basic shape
        filtered_contours = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Skip very small and very large contours
            if area < 100 or area > 50000:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Basic license plate shape filtering
            if (1.5 <= aspect_ratio <= 6.0 and 
                w >= 30 and h >= 10 and 
                area >= 300):
                filtered_contours.append(contour)
        
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
    
    def simple_rectangle_detection(self, full_gray_image, roi_start):
        """
        Simple sliding window approach to find rectangular license plate regions.
        
        Args:
            full_gray_image: Full grayscale image
            roi_start: Start position of ROI
            
        Returns:
            List of license plate candidates
        """
        height = full_gray_image.shape[0]
        width = full_gray_image.shape[1]
        roi_end = height
        
        # Work on the ROI
        roi = full_gray_image[roi_start:roi_end, :]
        
        # Apply edge detection
        edges = cv2.Canny(roi, 50, 150)
        
        # Find rectangular regions using template matching approach
        candidates = []
        
        # Define more precise license plate size ranges (width, height)
        plate_sizes = [
            (80, 20), (90, 22), (100, 25), (110, 28),    # Smaller plates
            (120, 30), (130, 32), (140, 35), (150, 38)   # Standard plates
        ]
        
        for plate_w, plate_h in plate_sizes:
            if plate_w > roi.shape[1] or plate_h > roi.shape[0]:
                continue
                
            # Slide window across the ROI
            step_x, step_y = 5, 3  # Step size for sliding
            
            for y in range(0, roi.shape[0] - plate_h, step_y):
                for x in range(0, roi.shape[1] - plate_w, step_x):
                    # Extract window
                    window = roi[y:y+plate_h, x:x+plate_w]
                    edge_window = edges[y:y+plate_h, x:x+plate_w]
                    
                    # Analyze the window for license plate characteristics
                    confidence = self.analyze_window_for_license_plate(window, edge_window)
                    
                    if confidence > 0.6:  # Higher threshold for more precise detection
                        # Convert coordinates back to full image
                        full_y = y + roi_start
                        full_x = x
                        
                        # Create a simple rectangular contour
                        contour = np.array([
                            [[full_x, full_y]],
                            [[full_x + plate_w, full_y]],
                            [[full_x + plate_w, full_y + plate_h]],
                            [[full_x, full_y + plate_h]]
                        ], dtype=np.int32)
                        
                        candidates.append((contour, (full_x, full_y, plate_w, plate_h), confidence))
                        print(f"  ✅ Window candidate: size=({plate_w}x{plate_h}), pos=({full_x},{full_y}), conf={confidence:.3f}")
        
        # Remove overlapping candidates
        filtered_candidates = self.remove_overlapping_windows(candidates)
        
        # Sort by confidence and return best candidates
        filtered_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Refine the best candidates by checking for actual license plate content
        refined_candidates = []
        for i, (contour, rect, conf) in enumerate(filtered_candidates[:5]):  # Check top 5
            x, y, w, h = rect
            region = full_gray_image[y:y+h, x:x+w]
            is_valid = self.validate_license_plate_region(region)
            print(f"    Validation {i}: pos=({x},{y}), size=({w}x{h}), conf={conf:.3f}, valid={is_valid}")
            
            if is_valid:
                refined_candidates.append((contour, rect, conf))
            elif len(refined_candidates) == 0 and i >= 2:  # If no valid found after 3 attempts, lower standards
                print("    No valid plates found, accepting best candidate with lower validation")
                refined_candidates.append((contour, rect, conf * 0.8))  # Reduce confidence but accept
                break
        
        # If we have valid candidates, return the best one; otherwise return the highest confidence
        if len(refined_candidates) > 0:
            return refined_candidates[:1]
        elif len(filtered_candidates) > 0:
            print("    No validation passed, returning best unvalidated candidate")
            return filtered_candidates[:1]
        else:
            return []
    
    def analyze_window_for_license_plate(self, window, edge_window):
        """
        Analyze a window to determine if it contains a license plate.
        
        Args:
            window: Grayscale window
            edge_window: Edge-detected window
            
        Returns:
            Confidence score
        """
        if window.size == 0:
            return 0.0
            
        try:
            confidence = 0.0
            
            # Edge density check - license plates should have moderate edge density
            edge_ratio = np.count_nonzero(edge_window) / edge_window.size
            if 0.15 <= edge_ratio <= 0.35:  # Optimal range for license plate text
                confidence += 0.4
            elif 0.1 <= edge_ratio < 0.15 or 0.35 < edge_ratio <= 0.5:
                confidence += 0.2
            else:
                confidence -= 0.1  # Penalize poor edge density
            
            # Horizontal structure check
            horizontal_projection = np.sum(edge_window, axis=1)
            if len(horizontal_projection) > 0 and np.var(horizontal_projection) > 100:
                confidence += 0.2
            
            # Vertical structure check (characters should create vertical variations)
            vertical_projection = np.sum(edge_window, axis=0)
            if len(vertical_projection) > 0 and np.var(vertical_projection) > 50:
                confidence += 0.2
            
            # Contrast check
            mean_intensity = np.mean(window)
            std_intensity = np.std(window)
            if std_intensity > 20:  # Good contrast
                confidence += 0.2
            
            # Central region intensity (license plates often have text in center)
            h, w = window.shape
            center_region = window[h//4:3*h//4, w//4:3*w//4]
            if center_region.size > 0:
                center_edges = edge_window[h//4:3*h//4, w//4:3*w//4]
                center_edge_ratio = np.count_nonzero(center_edges) / center_edges.size
                if center_edge_ratio > 0.1:
                    confidence += 0.1
            
            return min(confidence, 1.0)
            
        except:
            return 0.0
    
    def remove_overlapping_windows(self, candidates):
        """
        Remove overlapping window candidates using simple overlap detection.
        
        Args:
            candidates: List of (contour, rect, confidence) tuples
            
        Returns:
            Filtered list of candidates
        """
        if len(candidates) <= 1:
            return candidates
        
        # Sort by confidence
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        filtered = []
        for i, (contour, rect, conf) in enumerate(candidates):
            x1, y1, w1, h1 = rect
            
            # Check overlap with already accepted candidates
            overlaps = False
            for _, (x2, y2, w2, h2), _ in filtered:
                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                area1 = w1 * h1
                area2 = w2 * h2
                
                # If significant overlap, skip this candidate
                if overlap_area > 0.3 * min(area1, area2):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append((contour, rect, conf))
        
        return filtered
    
    def validate_license_plate_region(self, region):
        """
        Final validation to ensure detected region contains license plate characteristics.
        
        Args:
            region: Grayscale image region
            
        Returns:
            Boolean indicating if this is likely a license plate
        """
        if region.size == 0 or region.shape[0] < 15 or region.shape[1] < 50:
            return False
            
        try:
            # Apply threshold
            _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Check for reasonable text coverage (more lenient)
            white_pixels = np.count_nonzero(binary)
            white_ratio = white_pixels / binary.size
            if not (0.05 <= white_ratio <= 0.7):  # More lenient range
                return False
            
            # Check for connected components (characters) - more lenient
            num_labels, _ = cv2.connectedComponents(binary)
            if num_labels < 2 or num_labels > 25:  # More lenient range
                return False
            
            # Check horizontal distribution of content - more lenient
            horizontal_profile = np.sum(binary, axis=0)
            non_zero_cols = np.count_nonzero(horizontal_profile)
            horizontal_coverage = non_zero_cols / len(horizontal_profile)
            if horizontal_coverage < 0.2:  # More lenient threshold
                return False
            
            # Check for reasonable vertical distribution - more lenient
            vertical_profile = np.sum(binary, axis=1)
            if len(vertical_profile) > 0 and np.max(vertical_profile) > 0:
                peak_rows = np.where(vertical_profile > np.max(vertical_profile) * 0.2)[0]
                if len(peak_rows) < region.shape[0] * 0.2:  # More lenient threshold
                    return False
                
            return True
            
        except:
            return False
    
    def detect_with_mser(self, gray_roi, roi_start):
        """
        Use MSER with strict license plate constraints.
        
        Args:
            gray_roi: Grayscale ROI
            roi_start: ROI start position
            
        Returns:
            List of candidates
        """
        try:
            # Create MSER detector with tighter parameters for license plates
            mser = cv2.MSER.create(
                min_area=300,    # Minimum area for license plate text
                max_area=1800,   # Maximum area to avoid large regions
                delta=8,         # Stability parameter
                max_variation=0.25,
                min_diversity=0.2
            )
            
            # Detect regions
            regions, _ = mser.detectRegions(gray_roi)
            
            candidates = []
            for region in regions:
                if len(region) < 15:  # Skip very small regions
                    continue
                    
                # Get bounding rectangle
                region_array = np.array(region, dtype=np.int32).reshape(-1, 1, 2)
                x, y, w, h = cv2.boundingRect(region_array)
                aspect_ratio = w / h if h > 0 else 0
                
                # Strict license plate constraints
                if (80 <= w <= 140 and 18 <= h <= 35 and 
                    2.8 <= aspect_ratio <= 4.2):
                    
                    # Additional validation - check if region looks like text
                    region_img = gray_roi[y:y+h, x:x+w]
                    if self.looks_like_license_plate_text(region_img):
                        
                        # Convert coordinates to full image
                        full_y = y + roi_start
                        
                        # Create contour
                        contour = np.array([
                            [[x, full_y]], [[x + w, full_y]], 
                            [[x + w, full_y + h]], [[x, full_y + h]]
                        ], dtype=np.int32)
                        
                        confidence = 0.8  # Higher confidence for validated regions
                        candidates.append((contour, (x, full_y, w, h), confidence))
            
            return candidates
            
        except Exception as e:
            print(f"MSER detection error: {e}")
            return []
    
    def looks_like_license_plate_text(self, region):
        """
        Check if a region looks like license plate text.
        
        Args:
            region: Grayscale image region
            
        Returns:
            Boolean indicating if region looks like license plate text
        """
        if region.size == 0 or region.shape[0] < 15 or region.shape[1] < 60:
            return False
            
        try:
            # Check for horizontal text patterns
            # License plates have strong horizontal edges
            sobelx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
            
            horizontal_edges = np.sum(np.abs(sobelx))
            vertical_edges = np.sum(np.abs(sobely))
            
            # License plates should have more vertical edges (from characters)
            if vertical_edges == 0:
                return False
                
            edge_ratio = vertical_edges / (horizontal_edges + 1)
            
            # Check intensity variation (text should have good contrast)
            intensity_std = np.std(region)
            
            # Check for character-like patterns
            # Apply threshold to find potential characters
            _, thresh = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours in thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count potential character regions
            char_count = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                char_aspect = w / h if h > 0 else 0
                char_area = w * h
                
                # Character-like dimensions
                if (8 <= w <= 25 and 12 <= h <= 30 and 
                    0.3 <= char_aspect <= 1.5 and char_area > 50):
                    char_count += 1
            
            # License plates typically have 6-10 characters
            return (edge_ratio > 0.8 and intensity_std > 25 and 4 <= char_count <= 12)
            
        except:
            return False
    
    def detect_with_sobel(self, gray_roi, roi_start):
        """
        Use Sobel edge detection to find license plate regions.
        
        Args:
            gray_roi: Grayscale ROI
            roi_start: ROI start position
            
        Returns:
            List of candidates
        """
        try:
            # Apply Sobel operators
            sobelx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            magnitude = np.uint8(magnitude / magnitude.max() * 255)
            
            # Threshold and morphological operations
            magnitude_uint8 = magnitude.astype(np.uint8)
            _, thresh = cv2.threshold(magnitude_uint8, 50, 255, cv2.THRESH_BINARY)
            
            # Horizontal morphological closing to connect text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candidates = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # License plate filtering
                if (800 <= area <= 4000 and 
                    2.0 <= aspect_ratio <= 5.0 and
                    60 <= w <= 180 and 15 <= h <= 45):
                    
                    full_y = y + roi_start
                    confidence = 0.6  # Base confidence for Sobel
                    candidates.append((contour, (x, full_y, w, h), confidence))
            
            return candidates
            
        except Exception as e:
            print(f"Sobel detection error: {e}")
            return []
    
    def detect_with_adaptive_threshold(self, gray_roi, roi_start):
        """
        Use adaptive thresholding to detect license plate regions.
        
        Args:
            gray_roi: Grayscale ROI
            roi_start: ROI start position
            
        Returns:
            List of candidates
        """
        try:
            # Apply adaptive threshold
            adaptive = cv2.adaptiveThreshold(
                gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
            
            processed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel_h)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_v)
            
            # Find contours
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candidates = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # License plate filtering
                if (500 <= area <= 3000 and 
                    2.0 <= aspect_ratio <= 5.0 and
                    50 <= w <= 160 and 12 <= h <= 40):
                    
                    full_y = y + roi_start
                    confidence = 0.5  # Base confidence for adaptive
                    candidates.append((contour, (x, full_y, w, h), confidence))
            
            return candidates
            
        except Exception as e:
            print(f"Adaptive threshold detection error: {e}")
            return []
    
    def detect_with_contours(self, gray_roi, roi_start):
        """
        Fallback contour-based detection for license plates.
        
        Args:
            gray_roi: Grayscale ROI
            roi_start: ROI start position
            
        Returns:
            List of candidates
        """
        try:
            # Enhanced preprocessing for better contour detection
            blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)
            
            # Canny edge detection with optimal parameters
            edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)
            
            # Morphological operations to connect text
            kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_rect)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candidates = []
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # License plate specific filtering
                if (1000 <= area <= 4000 and 
                    2.5 <= aspect_ratio <= 4.5 and
                    80 <= w <= 150 and 20 <= h <= 40):
                    
                    # Check rectangularity
                    rect_area = w * h
                    rectangularity = area / rect_area if rect_area > 0 else 0
                    
                    if rectangularity > 0.7:  # Must be reasonably rectangular
                        full_y = y + roi_start
                        confidence = 0.6 * rectangularity  # Base confidence modified by shape
                        candidates.append((contour, (x, full_y, w, h), confidence))
            
            return candidates
            
        except Exception as e:
            print(f"Contour detection error: {e}")
            return []
    
    def filter_and_merge_candidates(self, candidates, full_gray_image):
        """
        Filter and merge candidates to find the single best license plate.
        
        Args:
            candidates: List of all candidates
            full_gray_image: Full grayscale image
            
        Returns:
            Filtered list with single best candidate
        """
        if not candidates:
            return []
        
        print(f"    Processing {len(candidates)} candidates for filtering")
        
        # First, validate all candidates and score them
        scored_candidates = []
        for i, (contour, rect, conf) in enumerate(candidates):
            x, y, w, h = rect
            region = full_gray_image[y:y+h, x:x+w]
            
            # Enhanced scoring system
            validation_score = 1.0 if self.validate_license_plate_region(region) else 0.3
            size_score = self.score_license_plate_size(w, h)
            position_score = self.score_license_plate_position(y, full_gray_image.shape[0])
            text_quality_score = self.score_text_quality(region)
            
            # Combined score
            final_score = (conf * 0.3 + validation_score * 0.3 + 
                          size_score * 0.2 + position_score * 0.1 + text_quality_score * 0.1)
            
            scored_candidates.append((contour, rect, final_score))
            print(f"    Candidate {i}: pos=({x},{y}), size=({w}x{h}), score={final_score:.3f}")
        
        # Sort by final score
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Group nearby candidates (they might be parts of the same license plate)
        grouped_candidates = self.group_nearby_candidates(scored_candidates)
        
        # Select the best group/candidate
        if grouped_candidates:
            best_group = grouped_candidates[0]
            if len(best_group) > 1:
                # Merge the group into a single bounding box
                merged_candidate = self.merge_candidate_group(best_group)
                return [merged_candidate]
            else:
                return [best_group[0]]
        
        return []
    
    def score_license_plate_size(self, width, height):
        """Score based on typical license plate dimensions."""
        aspect_ratio = width / height if height > 0 else 0
        
        # Ideal license plate: width 120-160, height 30-40, aspect ratio 3-4
        if 120 <= width <= 160 and 30 <= height <= 40 and 3.0 <= aspect_ratio <= 4.0:
            return 1.0
        elif 100 <= width <= 180 and 25 <= height <= 45 and 2.5 <= aspect_ratio <= 4.5:
            return 0.8
        elif 80 <= width <= 200 and 20 <= height <= 50 and 2.0 <= aspect_ratio <= 5.0:
            return 0.6
        else:
            return 0.3
    
    def score_license_plate_position(self, y_pos, image_height):
        """Score based on typical license plate position on vehicle."""
        # License plates are typically in the lower 60% of the image
        relative_pos = y_pos / image_height
        if 0.6 <= relative_pos <= 0.9:  # Ideal position
            return 1.0
        elif 0.5 <= relative_pos <= 0.95:  # Good position
            return 0.8
        else:
            return 0.4
    
    def score_text_quality(self, region):
        """Score the quality of text in the region."""
        if region.size == 0:
            return 0.0
            
        try:
            # Check edge density
            edges = cv2.Canny(region, 50, 150)
            edge_ratio = np.count_nonzero(edges) / edges.size
            
            # Check contrast
            std_intensity = float(np.std(region))
            contrast_score = min(std_intensity / 40.0, 1.0)
            
            # Combine scores
            return (edge_ratio * 5.0 + contrast_score) / 2.0
        except:
            return 0.3
    
    def group_nearby_candidates(self, candidates):
        """Group candidates that are close to each other (likely same license plate)."""
        if len(candidates) <= 1:
            return [candidates]
        
        groups = []
        used = set()
        
        for i, (contour1, rect1, score1) in enumerate(candidates):
            if i in used:
                continue
                
            x1, y1, w1, h1 = rect1
            group = [(contour1, rect1, score1)]
            used.add(i)
            
            # Find nearby candidates
            for j, (contour2, rect2, score2) in enumerate(candidates[i+1:], i+1):
                if j in used:
                    continue
                    
                x2, y2, w2, h2 = rect2
                
                # Check if candidates are close (horizontally or vertically adjacent)
                horizontal_distance = abs(x1 - x2)
                vertical_distance = abs(y1 - y2)
                
                if (horizontal_distance < max(w1, w2) * 1.5 and 
                    vertical_distance < max(h1, h2) * 1.5):
                    group.append((contour2, rect2, score2))
                    used.add(j)
            
            groups.append(group)
        
        # Sort groups by best score in each group
        groups.sort(key=lambda g: max(score for _, _, score in g), reverse=True)
        return groups
    
    def merge_candidate_group(self, group):
        """Merge a group of candidates into a single bounding box."""
        if len(group) == 1:
            return group[0]
        
        # Find bounding box that encompasses all candidates
        min_x = min(x for _, (x, y, w, h), _ in group)
        min_y = min(y for _, (x, y, w, h), _ in group)
        max_x = max(x + w for _, (x, y, w, h), _ in group)
        max_y = max(y + h for _, (x, y, w, h), _ in group)
        
        merged_w = max_x - min_x
        merged_h = max_y - min_y
        
        # Use the highest confidence from the group
        best_confidence = max(score for _, _, score in group)
        
        # Create merged contour
        merged_contour = np.array([
            [[min_x, min_y]], [[max_x, min_y]], 
            [[max_x, max_y]], [[min_x, max_y]]
        ], dtype=np.int32)
        
        print(f"    Merged group into: pos=({min_x},{min_y}), size=({merged_w}x{merged_h}), conf={best_confidence:.3f}")
        
        return (merged_contour, (min_x, min_y, merged_w, merged_h), best_confidence)
    
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
