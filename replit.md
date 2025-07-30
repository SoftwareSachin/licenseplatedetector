# License Plate Detector

## Overview

This is a classical computer vision application for detecting license plates in images without using deep learning or pre-trained models. The system uses OpenCV and traditional image processing techniques to identify rectangular regions with characteristics typical of license plates (high contrast edges, specific aspect ratios, appropriate size).

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Technology**: HTML5, CSS3, Bootstrap 5, Vanilla JavaScript
- **Structure**: Single-page application with responsive design
- **Components**: Image upload interface, parameter controls, real-time preview, results display
- **Styling**: Custom CSS with Bootstrap framework for responsive layout and modern UI

### Backend Architecture
- **Framework**: Flask (Python web framework)
- **Structure**: Simple REST API with file upload handling
- **Design Pattern**: Modular approach with separated concerns (detection logic in separate class)
- **File Management**: Temporary file storage for uploads and processed outputs

### Core Processing Module
- **Technology**: OpenCV for computer vision operations
- **Algorithm**: Classical edge detection and contour analysis
- **Parameters**: Configurable detection thresholds (area, aspect ratio, edge detection settings)

## Key Components

### 1. Flask Web Application (`app.py`)
- **Purpose**: Main web server and API endpoints
- **Key Features**:
  - File upload handling with security checks
  - Image processing pipeline coordination
  - RESTful API for detection requests
  - Static file serving for results

### 2. License Plate Detector (`plate_detector.py`)
- **Purpose**: Core computer vision processing
- **Algorithm Steps**:
  - Grayscale conversion and noise reduction
  - Histogram equalization for contrast enhancement
  - Canny edge detection
  - Contour detection and filtering
  - Geometric validation (area, aspect ratio, rectangularity)
- **Configurable Parameters**:
  - Area constraints (1000-50000 pixels)
  - Aspect ratio limits (2:1 to 5:1)
  - Edge detection thresholds
  - Shape validation criteria

### 3. Frontend Interface
- **Upload Component**: Drag-and-drop file selection with validation
- **Parameter Controls**: Collapsible accordion with detection settings
- **Results Display**: Side-by-side original and processed image comparison
- **Download Feature**: Direct download of processed results

## Data Flow

1. **Image Upload**: User selects image file through web interface
2. **Validation**: File type and size validation on both client and server
3. **Processing**: Image sent to Flask backend via AJAX
4. **Detection Pipeline**:
   - Image preprocessing (grayscale, blur, equalization)
   - Edge detection using Canny algorithm
   - Contour extraction and filtering
   - Geometric validation against license plate criteria
   - Bounding box drawing on detected regions
5. **Response**: Processed image returned with detection metadata
6. **Display**: Results shown in web interface with download option

## External Dependencies

### Python Libraries
- **Flask**: Web framework for API and file handling
- **OpenCV (cv2)**: Computer vision operations
- **NumPy**: Numerical operations and array handling
- **Werkzeug**: Secure filename handling

### Frontend Libraries
- **Bootstrap 5**: UI framework and responsive design
- **Font Awesome 6**: Icon library
- **Vanilla JavaScript**: No additional frameworks, pure DOM manipulation

### File System Requirements
- **Upload Directory**: Temporary storage for user-uploaded images
- **Output Directory**: Storage for processed results
- **Static Assets**: CSS, JavaScript, and other frontend resources

## Deployment Strategy

### Development Setup
- **File Structure**: Standard Flask application layout
- **Static Files**: Served directly by Flask development server
- **Temporary Storage**: Local filesystem directories for uploads/outputs

### Production Considerations
- **File Size Limits**: 16MB maximum upload size configured
- **Security**: Filename sanitization and file type validation
- **Storage**: Temporary file cleanup needed for production deployment
- **Scalability**: Single-threaded processing suitable for demonstration, would need optimization for high traffic

### Configuration
- **Environment Variables**: None currently used, all settings hardcoded
- **Directory Structure**: Auto-creation of required directories on startup
- **Error Handling**: Basic error responses with JSON formatting

## Key Design Decisions

### Classical Computer Vision Approach
- **Problem**: Detect license plates without deep learning
- **Solution**: Multi-stage filtering using geometric and visual characteristics
- **Rationale**: Demonstrates fundamental computer vision concepts and requires no training data

### Modular Architecture
- **Problem**: Separation of concerns between web interface and processing logic
- **Solution**: Separate detector class with configurable parameters
- **Benefits**: Reusable detection logic, easy parameter tuning, testable components

### Web-based Interface
- **Problem**: Accessible demonstration of computer vision capabilities
- **Solution**: Flask web application with modern responsive interface
- **Benefits**: No installation required, visual feedback, parameter experimentation

### Parameter Configurability
- **Problem**: Different images may require different detection settings
- **Solution**: Exposed detection parameters through web interface
- **Benefits**: Educational value, adaptability to different license plate styles