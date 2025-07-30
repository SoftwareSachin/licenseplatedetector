// License Plate Detector Frontend JavaScript

class LicensePlateDetector {
    constructor() {
        this.initializeElements();
        this.bindEvents();
        this.initializeTooltips();
    }

    initializeElements() {
        // Form elements
        this.uploadForm = document.getElementById('uploadForm');
        this.imageFile = document.getElementById('imageFile');
        this.detectBtn = document.getElementById('detectBtn');
        
        // Display elements
        this.uploadPlaceholder = document.getElementById('uploadPlaceholder');
        this.originalImageContainer = document.getElementById('originalImageContainer');
        this.originalImage = document.getElementById('originalImage');
        this.resultImageContainer = document.getElementById('resultImageContainer');
        this.resultImage = document.getElementById('resultImage');
        this.loadingSpinner = document.getElementById('loadingSpinner');
        this.downloadLink = document.getElementById('downloadLink');
        
        // Results elements
        this.resultsCard = document.getElementById('resultsCard');
        this.resultsContent = document.getElementById('resultsContent');
        
        // Toast elements
        this.errorToast = new bootstrap.Toast(document.getElementById('errorToast'));
        this.errorMessage = document.getElementById('errorMessage');
    }

    bindEvents() {
        // File input change event
        this.imageFile.addEventListener('change', (e) => {
            this.handleFileSelect(e);
        });

        // Form submit event
        this.uploadForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.detectLicensePlates();
        });

        // Parameter change events for real-time validation
        this.bindParameterValidation();
    }

    bindParameterValidation() {
        const parameters = ['minArea', 'maxArea', 'minAspect', 'maxAspect', 'cannyLow', 'cannyHigh'];
        
        parameters.forEach(param => {
            const element = document.getElementById(param);
            if (element) {
                element.addEventListener('input', () => {
                    this.validateParameters();
                });
            }
        });
    }

    initializeTooltips() {
        // Initialize Bootstrap tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Validate file type
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff', 'image/webp'];
        if (!allowedTypes.includes(file.type)) {
            this.showError('Please select a valid image file (JPG, PNG, GIF, BMP, TIFF, WebP)');
            this.imageFile.value = '';
            return;
        }

        // Validate file size (16MB max)
        const maxSize = 16 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showError('File size must be less than 16MB');
            this.imageFile.value = '';
            return;
        }

        // Display preview
        this.displayImagePreview(file);
    }

    displayImagePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.originalImage.src = e.target.result;
            this.uploadPlaceholder.style.display = 'none';
            this.originalImageContainer.style.display = 'block';
            this.resultImageContainer.style.display = 'none';
            this.resultsCard.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    validateParameters() {
        const minArea = parseInt(document.getElementById('minArea').value);
        const maxArea = parseInt(document.getElementById('maxArea').value);
        const minAspect = parseFloat(document.getElementById('minAspect').value);
        const maxAspect = parseFloat(document.getElementById('maxAspect').value);
        const cannyLow = parseInt(document.getElementById('cannyLow').value);
        const cannyHigh = parseInt(document.getElementById('cannyHigh').value);

        let isValid = true;
        let errors = [];

        // Validate area ranges
        if (minArea >= maxArea) {
            errors.push('Minimum area must be less than maximum area');
            isValid = false;
        }

        // Validate aspect ratio ranges
        if (minAspect >= maxAspect) {
            errors.push('Minimum aspect ratio must be less than maximum aspect ratio');
            isValid = false;
        }

        // Validate Canny thresholds
        if (cannyLow >= cannyHigh) {
            errors.push('Canny low threshold must be less than high threshold');
            isValid = false;
        }

        // Update UI based on validation
        this.detectBtn.disabled = !isValid;
        
        if (!isValid && errors.length > 0) {
            // Could show validation errors in real-time
            console.log('Validation errors:', errors);
        }

        return isValid;
    }

    async detectLicensePlates() {
        if (!this.validateParameters()) {
            this.showError('Please fix parameter validation errors');
            return;
        }

        const file = this.imageFile.files[0];
        if (!file) {
            this.showError('Please select an image file');
            return;
        }

        // Show loading state
        this.setLoadingState(true);

        try {
            // Prepare form data
            const formData = new FormData(this.uploadForm);

            // Make API request
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok && result.success) {
                this.displayResults(result);
            } else {
                this.showError(result.error || 'Detection failed');
            }
        } catch (error) {
            console.error('Detection error:', error);
            this.showError('Network error: Please check your connection and try again');
        } finally {
            this.setLoadingState(false);
        }
    }

    displayResults(result) {
        // Display result image
        this.resultImage.src = result.output_url;
        this.downloadLink.href = result.output_url;
        this.resultImageContainer.style.display = 'block';
        this.resultImageContainer.classList.add('fade-in-up');

        // Display detection statistics
        this.displayDetectionStats(result);
        this.resultsCard.style.display = 'block';
        this.resultsCard.classList.add('fade-in-up');

        // Scroll to results
        this.resultImageContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    displayDetectionStats(result) {
        const { num_plates, confidence_scores, processing_time, plate_details } = result;

        let html = `
            <div class="detection-summary mb-3">
                <div class="row text-center">
                    <div class="col-4">
                        <div class="status-badge status-info">
                            <i class="fas fa-car"></i>
                            <div class="fw-bold">${num_plates}</div>
                            <small>Plates Detected</small>
                        </div>
                    </div>
                    <div class="col-4">
                        <div class="status-badge status-success">
                            <i class="fas fa-clock"></i>
                            <div class="fw-bold">${processing_time}s</div>
                            <small>Processing Time</small>
                        </div>
                    </div>
                    <div class="col-4">
                        <div class="status-badge ${num_plates > 0 ? 'status-success' : 'status-warning'}">
                            <i class="fas ${num_plates > 0 ? 'fa-check' : 'fa-exclamation-triangle'}"></i>
                            <div class="fw-bold">${num_plates > 0 ? 'Success' : 'None Found'}</div>
                            <small>Status</small>
                        </div>
                    </div>
                </div>
            </div>
        `;

        if (num_plates > 0 && plate_details && plate_details.length > 0) {
            html += `
                <h6 class="mb-3"><i class="fas fa-car"></i> Detected License Plates</h6>
                <div class="results-scroll">
            `;

            plate_details.forEach((plate, index) => {
                const confidencePercent = (plate.confidence * 100).toFixed(1);
                const confidenceClass = plate.confidence >= 0.7 ? 'success' : plate.confidence >= 0.5 ? 'warning' : 'danger';
                
                html += `
                    <div class="detection-result mb-3 p-3 border rounded">
                        <div class="row">
                            <div class="col-md-4">
                                <img src="${plate.crop_url}" alt="License Plate ${plate.plate_number}" 
                                     class="img-fluid rounded border" style="max-height: 100px;">
                            </div>
                            <div class="col-md-8">
                                <div class="d-flex justify-content-between align-items-center mb-2">
                                    <h6 class="fw-bold mb-0">Plate ${plate.plate_number}</h6>
                                    <span class="badge bg-${confidenceClass}">${confidencePercent}%</span>
                                </div>
                                <div class="mb-2">
                                    <strong>Text:</strong> 
                                    <span class="badge bg-primary fs-6">${plate.text}</span>
                                </div>
                                <div class="confidence-bar mb-2">
                                    <div class="confidence-fill bg-${confidenceClass}" style="width: ${confidencePercent}%"></div>
                                </div>
                                <small class="text-muted">
                                    Method: ${plate.method} | 
                                    Position: ${plate.position.x}, ${plate.position.y} | 
                                    Size: ${plate.position.width}×${plate.position.height}
                                </small>
                            </div>
                        </div>
                    </div>
                `;
            });

            html += '</div>';
        } else if (num_plates > 0) {
            // Fallback to old format if plate_details not available
            html += `
                <h6 class="mb-3"><i class="fas fa-chart-bar"></i> Detection Details</h6>
                <div class="results-scroll">
            `;

            confidence_scores.forEach((confidence, index) => {
                const confidencePercent = (confidence * 100).toFixed(1);
                const confidenceClass = confidence >= 0.7 ? 'success' : confidence >= 0.5 ? 'warning' : 'danger';
                
                html += `
                    <div class="detection-result">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span class="fw-bold">License Plate ${index + 1}</span>
                            <span class="badge bg-${confidenceClass}">${confidencePercent}%</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                        </div>
                        <small class="text-muted">Confidence Score: ${confidence.toFixed(3)}</small>
                    </div>
                `;
            });

            html += '</div>';

            // Add summary statistics
            const confidenceValues = plate_details && plate_details.length > 0 
                ? plate_details.map(p => p.confidence) 
                : confidence_scores;
            const avgConfidence = confidenceValues.reduce((a, b) => a + b, 0) / confidenceValues.length;
            const maxConfidence = Math.max(...confidenceValues);
            const minConfidence = Math.min(...confidenceValues);

            html += `
                <div class="mt-3 p-3 bg-light rounded">
                    <h6 class="mb-2">Summary Statistics</h6>
                    <div class="row">
                        <div class="col-4 text-center">
                            <small class="text-muted">Average</small>
                            <div class="fw-bold">${(avgConfidence * 100).toFixed(1)}%</div>
                        </div>
                        <div class="col-4 text-center">
                            <small class="text-muted">Highest</small>
                            <div class="fw-bold">${(maxConfidence * 100).toFixed(1)}%</div>
                        </div>
                        <div class="col-4 text-center">
                            <small class="text-muted">Lowest</small>
                            <div class="fw-bold">${(minConfidence * 100).toFixed(1)}%</div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            html += `
                <div class="alert alert-warning">
                    <i class="fas fa-info-circle"></i>
                    <strong>No license plates detected.</strong>
                    <br>Try adjusting the detection parameters or use a different image with clearer license plates.
                </div>
                <div class="mt-3">
                    <h6>Suggestions:</h6>
                    <ul class="small">
                        <li>Ensure the image has good lighting and contrast</li>
                        <li>License plates should be clearly visible and not too blurry</li>
                        <li>Try lowering the minimum area if plates are small</li>
                        <li>Adjust Canny thresholds for different edge detection sensitivity</li>
                    </ul>
                </div>
            `;
        }

        this.resultsContent.innerHTML = html;
    }

    setLoadingState(isLoading) {
        if (isLoading) {
            this.detectBtn.disabled = true;
            this.detectBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            this.loadingSpinner.style.display = 'block';
            this.resultImageContainer.style.display = 'none';
            this.resultsCard.style.display = 'none';
        } else {
            this.detectBtn.disabled = false;
            this.detectBtn.innerHTML = '<i class="fas fa-search"></i> Detect License Plates';
            this.loadingSpinner.style.display = 'none';
        }
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorToast.show();
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new LicensePlateDetector();
});

// Additional utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function validateImageFile(file) {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff', 'image/webp'];
    const maxSize = 16 * 1024 * 1024; // 16MB

    if (!allowedTypes.includes(file.type)) {
        return { valid: false, error: 'Invalid file type. Please select an image file.' };
    }

    if (file.size > maxSize) {
        return { valid: false, error: 'File too large. Maximum size is 16MB.' };
    }

    return { valid: true };
}

// Export for potential testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { LicensePlateDetector, formatFileSize, validateImageFile };
}
