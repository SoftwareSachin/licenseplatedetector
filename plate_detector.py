import cv2
import numpy as np
import os

class LicensePlateDetector:
    """
    Classical computer vision license plate detector using OpenCV.
    Detects rectangular regions with high-contrast edges and appropriate aspect ratios.
    """
    
    def __init__(self, min_area=1000, max_area=50000, min_aspect_ratio=2.0, max_aspect_ratio=5.0,
                 canny_low=50, canny_high=150, min_rect_ratio=0.75):
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
        Preprocess the input image for license plate detection.
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed grayscale image ready for edge detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply histogram equalization to improve contrast
        equalized = cv2.equalizeHist(blurred)
        
        return equalized
    
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
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Dilate to strengthen edges
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges
    
    def find_contours(self, edge_image):
        """
        Find contours in the edge image.
        
        Args:
            edge_image: Binary edge image
            
        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def filter_contours(self, contours):
        """
        Filter contours based on area, aspect ratio, and rectangularity.
        
        Args:
            contours: List of contours to filter
            
        Returns:
            List of tuples (contour, bounding_rect, confidence_score)
        """
        valid_plates = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate area and aspect ratio
            area = cv2.contourArea(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
                
            # Filter by aspect ratio
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            
            # Check rectangularity (how close the contour is to a rectangle)
            rect_area = w * h
            rect_ratio = area / rect_area if rect_area > 0 else 0
            
            if rect_ratio < self.min_rect_ratio:
                continue
            
            # Calculate confidence score based on multiple factors
            confidence = self.calculate_confidence(contour, area, aspect_ratio, rect_ratio)
            
            valid_plates.append((contour, (x, y, w, h), confidence))
        
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
        
        # Bonus for optimal aspect ratio (around 3:1 to 4:1 is typical for license plates)
        optimal_aspect = 3.5
        aspect_deviation = abs(aspect_ratio - optimal_aspect) / optimal_aspect
        aspect_score = max(0, 1 - aspect_deviation)
        confidence *= (0.7 + 0.3 * aspect_score)
        
        # Bonus for reasonable size (not too small or too large)
        optimal_area = 8000
        area_deviation = abs(area - optimal_area) / optimal_area
        area_score = max(0, 1 - min(area_deviation, 1))
        confidence *= (0.8 + 0.2 * area_score)
        
        # Check contour approximation (rectangles should have 4 vertices)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            confidence *= 1.2  # Bonus for rectangular shape
        
        return min(confidence, 1.0)
    
    def remove_overlapping_detections(self, valid_plates, overlap_threshold=0.3):
        """
        Remove overlapping detections using Non-Maximum Suppression.
        
        Args:
            valid_plates: List of (contour, bounding_rect, confidence) tuples
            overlap_threshold: Minimum overlap ratio to consider as duplicate
            
        Returns:
            Filtered list of non-overlapping detections
        """
        if len(valid_plates) <= 1:
            return valid_plates
        
        # Calculate IoU for all pairs and remove overlapping ones
        final_plates = []
        
        for i, (contour1, rect1, conf1) in enumerate(valid_plates):
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
        
        return final_plates
    
    def detect_license_plates(self, image):
        """
        Main detection pipeline for license plates.
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (number_of_plates, list_of_confidence_scores, list_of_bounding_rectangles)
        """
        # Preprocess image
        gray = self.preprocess_image(image)
        
        # Detect edges
        edges = self.detect_edges(gray)
        
        # Find contours
        contours = self.find_contours(edges)
        
        # Filter contours
        valid_plates = self.filter_contours(contours)
        
        # Remove overlapping detections
        final_plates = self.remove_overlapping_detections(valid_plates)
        
        # Extract results
        num_plates = len(final_plates)
        confidence_scores = [conf for _, _, conf in final_plates]
        bounding_rects = [rect for _, rect, _ in final_plates]
        
        return num_plates, confidence_scores, bounding_rects
    
    def draw_detections(self, image, bounding_rects, confidence_scores):
        """
        Draw bounding boxes around detected license plates.
        
        Args:
            image: Input BGR image
            bounding_rects: List of bounding rectangles
            confidence_scores: List of confidence scores
            
        Returns:
            Image with bounding boxes drawn
        """
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
