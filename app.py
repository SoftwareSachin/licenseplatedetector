import os
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import cv2
import numpy as np
from realistic_plate_detector import RealisticLicensePlateDetector
import uuid
import time
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  # needed for url_for to generate with https

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_plates():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        original_filename = file.filename or 'uploaded_image'
        filename = secure_filename(original_filename)
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'jpg'
        input_filename = f"{unique_id}_input.{file_ext}"
        output_filename = f"{unique_id}_output.jpg"
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Save uploaded file
        file.save(input_path)
        
        # Get detection parameters from form
        min_area = int(request.form.get('min_area', 1000))
        max_area = int(request.form.get('max_area', 50000))
        min_aspect = float(request.form.get('min_aspect', 2.0))
        max_aspect = float(request.form.get('max_aspect', 5.0))
        canny_low = int(request.form.get('canny_low', 50))
        canny_high = int(request.form.get('canny_high', 150))
        
        # Initialize realistic detector with custom parameters
        detector = RealisticLicensePlateDetector(
            min_area=min_area,
            max_area=max_area,
            min_aspect=min_aspect,
            max_aspect=max_aspect,
            canny_low=canny_low,
            canny_high=canny_high
        )
        
        # Detect license plates
        start_time = time.time()
        num_plates, confidence_scores, plate_details = detector.detect_and_save(input_path, output_path)
        processing_time = time.time() - start_time
        
        # Save individual plate crops for dashboard
        plate_crops = []
        for i, plate_info in enumerate(plate_details):
            crop_filename = f"{unique_id}_plate_{i+1}.jpg"
            crop_path = os.path.join(app.config['OUTPUT_FOLDER'], crop_filename)
            cv2.imwrite(crop_path, plate_info['roi'])
            
            plate_crops.append({
                'plate_number': plate_info['plate_number'],
                'text': plate_info['text'],
                'confidence': plate_info['confidence'],
                'method': plate_info['method'],
                'crop_url': url_for('get_output', filename=crop_filename),
                'position': plate_info['position']
            })
        
        # Clean up input file
        os.remove(input_path)
        
        return jsonify({
            'success': True,
            'num_plates': num_plates,
            'confidence_scores': confidence_scores,
            'processing_time': round(processing_time, 3),
            'output_url': url_for('get_output', filename=output_filename),
            'plate_details': plate_crops
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/output/<filename>')
def get_output(filename):
    try:
        return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))
    except FileNotFoundError:
        return jsonify({'error': 'Output file not found'}), 404

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'service': 'License Plate Detector'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
