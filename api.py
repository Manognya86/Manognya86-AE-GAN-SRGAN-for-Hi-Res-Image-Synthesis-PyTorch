# api.py
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import uuid
import json
from inference_api import AEGANInference  # Import your inference class
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ENHANCED_FOLDER'] = 'enhanced'
app.config['FEEDBACK_FILE'] = 'feedback.json'
app.config['MODEL_DIR'] = 'models/latest'  # Path to your model weights

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ENHANCED_FOLDER'], exist_ok=True)

# Initialize feedback file if it doesn't exist
if not os.path.exists(app.config['FEEDBACK_FILE']):
    with open(app.config['FEEDBACK_FILE'], 'w') as f:
        json.dump({"feedback": []}, f)

# Initialize the AEGAN model
try:
    enhancer = AEGANInference(app.config['MODEL_DIR'])
    logger.info("AEGAN model successfully loaded")
except Exception as e:
    logger.error(f"Failed to initialize AEGAN model: {str(e)}")
    enhancer = None

@app.route('/api/enhance', methods=['POST'])
def enhance_image():
    if not enhancer:
        return jsonify({"error": "Model not initialized"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Read the file bytes
        image_bytes = file.read()
        
        # Enhance the image using the AEGAN model
        enhanced_filename, message = enhancer.enhance_image(image_bytes)
        
        if not enhanced_filename:
            return jsonify({"error": message}), 500
        
        # Generate unique ID for this enhancement
        enhancement_id = str(uuid.uuid4())
        
        # Move the enhanced file to our enhanced folder
        new_path = os.path.join(app.config['ENHANCED_FOLDER'], f"{enhancement_id}_{secure_filename(enhanced_filename)}")
        os.rename(enhanced_filename, new_path)
        
        return jsonify({
            "message": message,
            "enhanced_url": f"/api/enhanced/{enhancement_id}/{secure_filename(enhanced_filename)}"
        })
        
    except Exception as e:
        logger.error(f"Enhancement error: {str(e)}")
        return jsonify({"error": f"Enhancement failed: {str(e)}"}), 500

@app.route('/api/enhanced/<enhancement_id>/<filename>', methods=['GET'])
def get_enhanced_image(enhancement_id, filename):
    enhanced_path = os.path.join(app.config['ENHANCED_FOLDER'], f"{enhancement_id}_{filename}")
    if os.path.exists(enhanced_path):
        return send_file(enhanced_path, mimetype='image/png')
    return jsonify({"error": "Enhanced image not found"}), 404

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    if not data or 'rating' not in data:
        return jsonify({"error": "Invalid feedback data"}), 400
    
    try:
        # Load existing feedback
        with open(app.config['FEEDBACK_FILE'], 'r') as f:
            feedback_data = json.load(f)
        
        # Add new feedback
        new_feedback = {
            "id": str(uuid.uuid4()),
            "username": "current_user",  # In real app, get from auth
            "rating": data['rating'],
            "comment": data.get('comment', ''),
            "created_at": datetime.now().isoformat()
        }
        feedback_data['feedback'].append(new_feedback)
        
        # Save updated feedback
        with open(app.config['FEEDBACK_FILE'], 'w') as f:
            json.dump(feedback_data, f)
        
        return jsonify({"message": "Feedback submitted successfully"}), 201
    except Exception as e:
        logger.error(f"Feedback submission error: {str(e)}")
        return jsonify({"error": f"Failed to save feedback: {str(e)}"}), 500

@app.route('/api/feedback', methods=['GET'])
def get_feedback():
    try:
        with open(app.config['FEEDBACK_FILE'], 'r') as f:
            feedback_data = json.load(f)
        return jsonify(feedback_data)
    except Exception as e:
        logger.error(f"Feedback retrieval error: {str(e)}")
        return jsonify({"error": f"Failed to load feedback: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "model_loaded": enhancer is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)