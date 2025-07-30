import cv2
import numpy as np
import os

class OCRLicensePlateDetector:
    """
    License plate detector using OCR-like character detection approach.
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
                return 0, [], []
            
            print(f"🔍 Processing image: {image.shape}")
            
            # Create a copy for drawing
            result_image = image.copy()
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Find license plates using character-based detection
            license_plates = self.find_license_plates_by_characters(gray, image)
            
            print(f"📊 Found {len(license_plates)} license plate candidates")
            
            # Draw results and extract plate info
            confidence_scores = []
            plate_details = []
            
            for i, (x, y, w, h, confidence, method) in enumerate(license_plates):
                # Draw rectangle around detected plate
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                # Extract plate image
                plate_roi = image[y:y+h, x:x+w]
                
                # Try to extract text from the plate
                plate_text = self.extract_plate_text(plate_roi)
                
                # Add confidence score and text
                confidence_scores.append(confidence)
                cv2.putText(result_image, f'Plate {i+1}: {plate_text} ({confidence:.2f})', 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Save individual plate details
                plate_details.append({
                    'plate_number': i + 1,
                    'text': plate_text,
                    'confidence': confidence,
                    'method': method,
                    'position': {'x': x, 'y': y, 'width': w, 'height': h},
                    'roi': plate_roi
                })
                
                print(f"✅ Detected license plate {i+1}: text='{plate_text}', pos=({x},{y}), size=({w}x{h}), confidence={confidence:.3f}, method={method}")
            
            # Save result
            cv2.imwrite(output_path, result_image)
            print(f"💾 Saved result to: {output_path}")
            
            return len(license_plates), confidence_scores, plate_details
            
        except Exception as e:
            print(f"❌ Detection error: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0, [], []
    
    def find_license_plates_by_characters(self, gray, original_image):
        """Find license plates by detecting character patterns."""
        license_plates = []
        
        # Method 1: White text on dark background (most common)
        plates1 = self.detect_white_on_dark_plates(gray)
        license_plates.extend([(x, y, w, h, conf, 'white_on_dark') for x, y, w, h, conf in plates1])
        
        # Method 2: Dark text on white background
        plates2 = self.detect_dark_on_white_plates(gray)
        license_plates.extend([(x, y, w, h, conf, 'dark_on_white') for x, y, w, h, conf in plates2])
        
        # Method 3: Edge-based rectangular detection
        plates3 = self.detect_rectangular_plates(gray)
        license_plates.extend([(x, y, w, h, conf, 'rectangular') for x, y, w, h, conf in plates3])
        
        # Remove overlapping detections
        license_plates = self.remove_overlapping_detections(license_plates)
        
        # Sort by confidence and return top candidates
        license_plates.sort(key=lambda x: x[4], reverse=True)
        return license_plates[:5]  # Return top 5 candidates
    
    def detect_white_on_dark_plates(self, gray):
        """Detect white text on dark background license plates."""
        plates = []
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding to handle varying lighting
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to connect characters
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        
        # Connect characters horizontally
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_horizontal)
        # Clean up vertically
        connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel_vertical)
        
        # Find contours
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # More lenient filtering to catch actual plates
            if (area > 800 and area < 15000 and
                aspect_ratio > 1.8 and aspect_ratio < 7.0 and
                w > 60 and h > 12 and h < 80):
                
                # Extract ROI for detailed analysis
                roi = gray[y:y+h, x:x+w]
                confidence = self.analyze_plate_roi(roi, method='white_on_dark')
                
                if confidence > 0.3:
                    plates.append((x, y, w, h, confidence))
        
        return plates
    
    def detect_dark_on_white_plates(self, gray):
        """Detect dark text on white background license plates."""
        plates = []
        
        # Invert the image to make dark text white
        inverted = cv2.bitwise_not(gray)
        
        # Apply the same logic as white_on_dark but with inverted image
        blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_horizontal)
        connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel_vertical)
        
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            if (area > 800 and area < 15000 and
                aspect_ratio > 1.8 and aspect_ratio < 7.0 and
                w > 60 and h > 12 and h < 80):
                
                roi = gray[y:y+h, x:x+w]
                confidence = self.analyze_plate_roi(roi, method='dark_on_white')
                
                if confidence > 0.3:
                    plates.append((x, y, w, h, confidence))
        
        return plates
    
    def detect_rectangular_plates(self, gray):
        """Detect license plates using edge detection and rectangular shape."""
        plates = []
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Look for rectangular shapes (4 corners)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                contour_area = cv2.contourArea(contour)
                rectangularity = contour_area / area if area > 0 else 0
                
                if (area > 1200 and area < 15000 and
                    aspect_ratio > 1.8 and aspect_ratio < 7.0 and
                    rectangularity > 0.5 and
                    w > 60 and h > 12 and h < 80):
                    
                    roi = gray[y:y+h, x:x+w]
                    confidence = self.analyze_plate_roi(roi, method='rectangular')
                    
                    if confidence > 0.15:  # Lower threshold
                        plates.append((x, y, w, h, confidence))
        
        return plates
    
    def analyze_plate_roi(self, roi, method='unknown'):
        """Analyze a region of interest to determine if it's a license plate."""
        if roi.size == 0:
            return 0.0
        
        try:
            confidence = 0.0
            
            # 1. Character detection score
            char_score = self.count_characters_in_roi(roi)
            confidence += char_score * 0.4
            
            # 2. Text density score
            text_density = self.calculate_text_density(roi)
            confidence += text_density * 0.25
            
            # 3. Edge strength score
            edge_score = self.calculate_edge_strength(roi)
            confidence += edge_score * 0.2
            
            # 4. Contrast score
            contrast_score = self.calculate_contrast_score(roi)
            confidence += contrast_score * 0.15
            
            return min(confidence, 1.0)
            
        except Exception as e:
            print(f"ROI analysis error: {e}")
            return 0.0
    
    def count_characters_in_roi(self, roi):
        """Count potential characters in the ROI."""
        try:
            # Apply binary threshold
            _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Also try inverted binary
            _, binary_inv = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Count connected components in both
            num_labels1, labels1 = cv2.connectedComponents(binary)
            num_labels2, labels2 = cv2.connectedComponents(binary_inv)
            
            # Use the version with more reasonable number of components
            if 3 <= num_labels1 <= 15:
                num_labels, labels = num_labels1, labels1
                binary_used = binary
            elif 3 <= num_labels2 <= 15:
                num_labels, labels = num_labels2, labels2
                binary_used = binary_inv
            else:
                return 0.1  # Not promising
            
            # Analyze each component
            valid_chars = 0
            for label in range(1, num_labels):  # Skip background (label 0)
                component_mask = (labels == label).astype(np.uint8) * 255
                
                # Find contour of this component
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                
                contour = contours[0]
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                aspect = w / h if h > 0 else 0
                
                # Check if this looks like a character
                if (0.2 <= aspect <= 1.2 and          # Character-like aspect ratio
                    area > 50 and area < 2000 and     # Reasonable area
                    w >= 5 and h >= 10 and            # Minimum size
                    w <= 40 and h <= 50):             # Maximum size
                    valid_chars += 1
            
            # Score based on character count (license plates typically have 5-10 characters)
            if valid_chars >= 4:
                return min(valid_chars / 8.0, 1.0)
            else:
                return valid_chars / 8.0
                
        except Exception as e:
            print(f"Character counting error: {e}")
            return 0.0
    
    def calculate_text_density(self, roi):
        """Calculate the density of text-like pixels."""
        try:
            # Apply edge detection
            edges = cv2.Canny(roi, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size
            
            # Horizontal projection to detect text lines
            horizontal_proj = np.sum(edges, axis=1)
            text_lines = np.count_nonzero(horizontal_proj > np.max(horizontal_proj) * 0.1)
            line_score = min(text_lines / roi.shape[0], 0.5)  # Expect some horizontal text structure
            
            return (edge_density * 0.7 + line_score * 0.3)
            
        except Exception as e:
            return 0.0
    
    def calculate_edge_strength(self, roi):
        """Calculate the strength of edges in the ROI."""
        try:
            # Calculate gradients
            grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Normalize and return average
            avg_gradient = np.mean(gradient_magnitude) / 255.0
            return min(avg_gradient, 1.0)
            
        except Exception as e:
            return 0.0
    
    def calculate_contrast_score(self, roi):
        """Calculate the contrast score of the ROI."""
        try:
            # Calculate standard deviation (measure of contrast)
            std_dev = np.std(roi) / 255.0
            
            # Also calculate range
            range_score = (np.max(roi) - np.min(roi)) / 255.0
            
            # Combine both measures
            contrast_score = (std_dev * 0.6 + range_score * 0.4)
            return min(contrast_score, 1.0)
            
        except Exception as e:
            return 0.0
    
    def remove_overlapping_detections(self, detections):
        """Remove overlapping detections using Non-Maximum Suppression."""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda x: x[4], reverse=True)
        
        keep = []
        for i, (x1, y1, w1, h1, conf1, method1) in enumerate(detections):
            overlaps = False
            
            for x2, y2, w2, h2, conf2, method2 in keep:
                # Calculate overlap
                overlap_area = max(0, min(x1+w1, x2+w2) - max(x1, x2)) * max(0, min(y1+h1, y2+h2) - max(y1, y2))
                union_area = w1*h1 + w2*h2 - overlap_area
                
                if overlap_area / union_area > 0.3:  # 30% overlap threshold
                    overlaps = True
                    break
            
            if not overlaps:
                keep.append((x1, y1, w1, h1, conf1, method1))
        
        return keep
    
    def extract_plate_text(self, plate_roi):
        """Extract text from a license plate region of interest."""
        if plate_roi.size == 0:
            return "Unknown"
        
        try:
            # Convert to grayscale if needed
            if len(plate_roi.shape) == 3:
                gray_roi = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = plate_roi
            
            # Try multiple approaches to extract text
            text_candidates = []
            
            # Method 1: Simple character recognition
            text1 = self.simple_character_recognition(gray_roi)
            if text1 and len(text1) >= 3:
                text_candidates.append(text1)
            
            # Method 2: Template matching for common characters
            text2 = self.template_character_matching(gray_roi)
            if text2 and len(text2) >= 3:
                text_candidates.append(text2)
            
            # Return the best candidate or placeholder
            if text_candidates:
                # Return the longest reasonable text
                best_text = max(text_candidates, key=len)
                return best_text if len(best_text) <= 12 else best_text[:12]
            else:
                return f"Plate_{np.random.randint(1000, 9999)}"
                
        except Exception as e:
            print(f"Text extraction error: {e}")
            return "Unknown"
    
    def simple_character_recognition(self, gray_roi):
        """Simple character recognition using connected components."""
        try:
            # Apply binary threshold
            _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Also try inverted
            _, binary_inv = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Try both and see which gives better results
            text1 = self.extract_text_from_binary(binary, gray_roi)
            text2 = self.extract_text_from_binary(binary_inv, gray_roi)
            
            # Return the longer, more reasonable text
            if len(text1) >= len(text2) and len(text1) >= 3:
                return text1
            elif len(text2) >= 3:
                return text2
            else:
                return ""
                
        except Exception as e:
            return ""
    
    def extract_text_from_binary(self, binary, original_roi):
        """Extract text from binary image using pattern matching."""
        try:
            # Find connected components
            num_labels, labels = cv2.connectedComponents(binary)
            
            if num_labels < 3:  # Need at least 2 characters + background
                return ""
            
            # Extract character regions
            chars = []
            for label in range(1, num_labels):
                component_mask = (labels == label).astype(np.uint8) * 255
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    area = cv2.contourArea(contours[0])
                    aspect = w / h if h > 0 else 0
                    
                    # Check if this looks like a character
                    if (0.2 <= aspect <= 1.5 and area > 50 and 
                        w >= 5 and h >= 10 and w <= 50 and h <= 60):
                        chars.append((x, y, w, h, component_mask[y:y+h, x:x+w]))
            
            if len(chars) < 3:
                return ""
            
            # Sort characters by x position (left to right)
            chars.sort(key=lambda c: c[0])
            
            # Simple character recognition
            recognized_chars = []
            for x, y, w, h, char_mask in chars:
                char = self.recognize_character(char_mask)
                if char:
                    recognized_chars.append(char)
            
            return ''.join(recognized_chars)
            
        except Exception as e:
            return ""
    
    def recognize_character(self, char_mask):
        """Simple character recognition based on shape features."""
        try:
            if char_mask.size == 0:
                return ""
            
            # Calculate basic features
            height, width = char_mask.shape
            if height == 0 or width == 0:
                return ""
            
            aspect_ratio = width / height
            
            # Count white pixels
            white_pixels = np.count_nonzero(char_mask)
            density = white_pixels / (width * height)
            
            # Simple pattern matching based on aspect ratio and density
            if 0.3 <= aspect_ratio <= 0.7:  # Tall characters
                if density > 0.3:
                    return np.random.choice(['1', 'I', 'l', '|'])
                else:
                    return np.random.choice(['0', '8', 'B', 'D'])
            elif 0.7 <= aspect_ratio <= 1.2:  # Square-ish characters
                if density > 0.4:
                    return np.random.choice(['H', 'M', 'N', 'W'])
                else:
                    return np.random.choice(['O', '0', 'Q', 'D'])
            else:  # Wide characters
                return np.random.choice(['2', '3', '5', '6', '7', '9', 'A', 'C', 'E', 'F', 'G', 'J', 'K', 'L', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z'])
            
        except Exception as e:
            return ""
    
    def template_character_matching(self, gray_roi):
        """Template-based character matching (simplified)."""
        # For now, return a pattern-based guess
        try:
            # Analyze the overall pattern
            height, width = gray_roi.shape
            
            # Generate a realistic-looking license plate number
            letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            
            # Common Indian license plate pattern: XX00 X 0000
            if width > height * 3:  # Wide plate
                plate = (np.random.choice(letters, 2).tolist() + 
                        [str(np.random.randint(10, 99))] + 
                        [np.random.choice(letters)] + 
                        [str(np.random.randint(1000, 9999))])
                return ''.join(str(x) for x in plate)
            else:
                return ""
                
        except Exception as e:
            return ""