import cv2
import numpy as np
import os

class FinalLicensePlateDetector:
    """
    Final license plate detector focused on actual text detection.
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
            
            # Focus on lower part of image where license plates are typically located
            height = image.shape[0]
            roi_start = int(height * 0.3)  # Focus on lower 70% of image
            gray_roi = gray[roi_start:, :]
            
            print(f"🎯 Focusing on lower region: {gray_roi.shape}")
            
            # Find license plates using text-focused approach
            license_plates = self.find_license_plates_by_text(gray_roi, roi_start)
            
            print(f"📊 Found {len(license_plates)} license plate candidates")
            
            # Draw results
            confidence_scores = []
            for i, (x, y, w, h, confidence) in enumerate(license_plates):
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
            
            return len(license_plates), confidence_scores
            
        except Exception as e:
            print(f"❌ Detection error: {str(e)}")
            return 0, []
    
    def find_license_plates_by_text(self, gray_roi, roi_start):
        """Find license plates by detecting text patterns."""
        # Method 1: Morphological text detection
        text_regions = self.detect_text_morphology(gray_roi)
        
        # Method 2: MSER text detection  
        mser_regions = self.detect_text_mser(gray_roi)
        
        # Method 3: Edge-based text detection
        edge_regions = self.detect_text_edges(gray_roi)
        
        # Combine all regions
        all_regions = text_regions + mser_regions + edge_regions
        
        # Filter and validate
        license_plates = self.validate_and_filter_plates(all_regions, gray_roi, roi_start)
        
        return license_plates
    
    def detect_text_morphology(self, gray):
        """Detect text using morphological operations."""
        # Apply morphological gradient to highlight text boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Threshold to get binary image
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Connect text characters horizontally
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Fill small gaps vertically
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, vertical_kernel)
        
        # Find contours
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area > 200:  # Minimum area
                regions.append((x, y, w, h, 'morph'))
        
        return regions
    
    def detect_text_mser(self, gray):
        """Detect text using MSER (Maximally Stable Extremal Regions)."""
        try:
            # Create MSER detector
            mser = cv2.MSER.create(
                min_area=60,
                max_area=1400,
                delta=5
            )
            
            # Detect regions
            regions, _ = mser.detectRegions(gray)
            
            # Group nearby regions that could be characters in the same license plate
            bounding_boxes = []
            for region in regions:
                if len(region) > 10:
                    region_array = np.array(region, dtype=np.int32).reshape(-1, 1, 2)
                    x, y, w, h = cv2.boundingRect(region_array)
                    bounding_boxes.append([x, y, w, h])
            
            # Group nearby character regions
            if bounding_boxes:
                license_regions = self.group_character_regions(bounding_boxes)
                return [(x, y, w, h, 'mser') for x, y, w, h in license_regions]
        
        except Exception as e:
            print(f"MSER detection error: {e}")
        
        return []
    
    def detect_text_edges(self, gray):
        """Detect text using edge detection."""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area > 150:
                regions.append((x, y, w, h, 'edge'))
        
        return regions
    
    def group_character_regions(self, bounding_boxes):
        """Group character regions that belong to the same license plate."""
        if not bounding_boxes:
            return []
        
        # Sort by y coordinate first, then x
        bounding_boxes.sort(key=lambda box: (box[1], box[0]))
        
        groups = []
        current_group = [bounding_boxes[0]]
        
        for i in range(1, len(bounding_boxes)):
            x, y, w, h = bounding_boxes[i]
            
            # Check if this box should be grouped with current group
            can_group = False
            for gx, gy, gw, gh in current_group:
                # Check vertical alignment and horizontal proximity
                if (abs(y - gy) < max(h, gh) * 0.5 and  # Similar vertical position
                    abs(x - (gx + gw)) < gw * 2):  # Reasonable horizontal distance
                    can_group = True
                    break
            
            if can_group:
                current_group.append(bounding_boxes[i])
            else:
                if len(current_group) >= 3:  # At least 3 characters
                    groups.append(current_group)
                current_group = [bounding_boxes[i]]
        
        # Don't forget the last group
        if len(current_group) >= 3:
            groups.append(current_group)
        
        # Convert groups to bounding rectangles
        license_regions = []
        for group in groups:
            if len(group) >= 3:  # At least 3 characters for a license plate
                min_x = min(box[0] for box in group)
                min_y = min(box[1] for box in group)
                max_x = max(box[0] + box[2] for box in group)
                max_y = max(box[1] + box[3] for box in group)
                
                w = max_x - min_x
                h = max_y - min_y
                
                # Add some padding
                padding_x = int(w * 0.1)
                padding_y = int(h * 0.2)
                
                license_regions.append((
                    max(0, min_x - padding_x),
                    max(0, min_y - padding_y),
                    w + 2 * padding_x,
                    h + 2 * padding_y
                ))
        
        return license_regions
    
    def validate_and_filter_plates(self, regions, gray_roi, roi_start):
        """Validate and filter potential license plate regions."""
        valid_plates = []
        
        for x, y, w, h, method in regions:
            # Basic geometric filters
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Skip regions that don't meet basic criteria
            if (area < 800 or area > 15000 or
                aspect_ratio < 1.5 or aspect_ratio > 6.0 or
                w < 50 or h < 15 or w > 350 or h > 100):
                continue
            
            # Extract ROI for detailed analysis
            roi = gray_roi[y:y+h, x:x+w] if (y+h <= gray_roi.shape[0] and 
                                             x+w <= gray_roi.shape[1]) else None
            if roi is None or roi.size == 0:
                continue
            
            # Analyze text characteristics
            text_score = self.analyze_text_quality(roi)
            
            # Calculate confidence based on multiple factors
            confidence = self.calculate_confidence(area, aspect_ratio, text_score, method)
            
            if confidence > 0.3:  # Minimum confidence threshold
                # Convert coordinates back to full image
                full_y = y + roi_start
                valid_plates.append((x, full_y, w, h, confidence))
        
        # Remove overlapping detections and sort by confidence
        valid_plates = self.remove_overlapping_detections(valid_plates)
        valid_plates.sort(key=lambda x: x[4], reverse=True)
        
        return valid_plates[:3]  # Return top 3 candidates
    
    def analyze_text_quality(self, roi):
        """Analyze the quality of text in the region."""
        if roi.size == 0:
            return 0.0
        
        try:
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Find connected components (potential characters)
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
                    
                    # Check if this looks like a character
                    if (0.2 <= aspect <= 1.5 and  # Character aspect ratio
                        area > 30 and              # Minimum area
                        w >= 5 and h >= 8 and      # Minimum dimensions
                        w <= 30 and h <= 50):      # Maximum dimensions
                        valid_chars += 1
            
            # Calculate text quality scores
            char_ratio = valid_chars / max(char_count, 1) if char_count > 0 else 0
            char_count_score = min(valid_chars / 5.0, 1.0)  # Expect ~5-8 characters
            
            # Edge density (text has strong edges)
            edges = cv2.Canny(roi, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size
            
            # Contrast (license plates have high contrast)
            contrast = np.std(roi) / 255.0
            
            # Horizontal distribution (characters spread across width)
            horizontal_profile = np.sum(binary, axis=0)
            non_zero_cols = np.count_nonzero(horizontal_profile)
            horizontal_coverage = non_zero_cols / len(horizontal_profile) if len(horizontal_profile) > 0 else 0
            
            # Combine scores
            text_score = (char_ratio * 0.3 + 
                         char_count_score * 0.25 + 
                         edge_density * 0.2 + 
                         contrast * 0.15 + 
                         horizontal_coverage * 0.1)
            
            return min(text_score, 1.0)
            
        except Exception as e:
            print(f"Text analysis error: {e}")
            return 0.0
    
    def calculate_confidence(self, area, aspect_ratio, text_score, method):
        """Calculate overall confidence score."""
        confidence = 0.0
        
        # Text quality is most important
        confidence += text_score * 0.5
        
        # Aspect ratio scoring (license plates are typically 2.5:1 to 4:1)
        if 2.5 <= aspect_ratio <= 4.0:
            confidence += 0.2
        elif 2.0 <= aspect_ratio < 2.5 or 4.0 < aspect_ratio <= 5.0:
            confidence += 0.15
        else:
            confidence += 0.05
        
        # Size scoring
        if 1500 <= area <= 6000:
            confidence += 0.15
        elif 1000 <= area <= 1500 or 6000 <= area <= 8000:
            confidence += 0.1
        else:
            confidence += 0.05
        
        # Method bonus
        method_bonus = {
            'mser': 0.1,    # MSER is good for text
            'morph': 0.08,  # Morphology is decent
            'edge': 0.05    # Edges are less reliable
        }
        confidence += method_bonus.get(method, 0.05)
        
        return min(confidence, 1.0)
    
    def remove_overlapping_detections(self, detections):
        """Remove overlapping detections using Non-Maximum Suppression."""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda x: x[4], reverse=True)
        
        keep = []
        for i, (x1, y1, w1, h1, conf1) in enumerate(detections):
            overlaps = False
            
            for x2, y2, w2, h2, conf2 in keep:
                # Calculate overlap
                overlap_area = max(0, min(x1+w1, x2+w2) - max(x1, x2)) * max(0, min(y1+h1, y2+h2) - max(y1, y2))
                union_area = w1*h1 + w2*h2 - overlap_area
                
                if overlap_area / union_area > 0.3:  # 30% overlap threshold
                    overlaps = True
                    break
            
            if not overlaps:
                keep.append((x1, y1, w1, h1, conf1))
        
        return keep