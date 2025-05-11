from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageDraw
import os
import base64
from io import BytesIO
import json
import torch
import torch.nn as nn
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables
model = None
model_load_status = {"status": "not_started", "progress": 0}

def load_model():
    global model, model_load_status
    if model is None:
        try:
            logger.debug("Starting model loading...")
            # Import here to reduce cold start time
            from ultralytics import YOLO
            
            model_load_status = {"status": "initializing", "progress": 10}
            logger.debug("Ultralytics imported successfully")
            
            # Determine model path based on environment (local vs Render)
            if os.environ.get("RENDER"):
                # For Render deployment
                model_path = os.path.join(os.getcwd(), "models", "PHCoinClassifier", "best.pt")
                logger.debug(f"Running on Render, using path: {model_path}")
            else:
                # For local development
                model_path = os.path.join("models", "PHCoinClassifier", "best.pt")
                logger.debug(f"Running locally, using path: {model_path}")
            
            if not os.path.exists(model_path):
                logger.error(f"Error: Model file not found at {model_path}")
                # Try a fallback path for Render
                fallback_path = os.path.join("/opt/render/project/src", "models", "PHCoinClassifier", "best.pt")
                logger.debug(f"Trying fallback path: {fallback_path}")
                
                if os.path.exists(fallback_path):
                    model_path = fallback_path
                    logger.debug(f"Using fallback path: {model_path}")
                else:
                    model_load_status = {"status": "error", "progress": 0, "message": "Model file not found in repository"}
                    logger.error(f"Error: Model file not found at fallback path either")
                    return False
                
            # Set torch options to handle safe loading on PyTorch versions
            logger.debug(f"PyTorch version: {torch.__version__}")
            
            # For PyTorch 2.0+, explicitly set device
            device = torch.device('cpu')
            logger.debug(f"Using device: {device}")
            
            # Load model with explicit arguments
            model_load_status = {"status": "loading", "progress": 50}
            logger.debug(f"Loading model from {model_path}...")
            model = YOLO(model_path, task='detect')
            
            model_load_status = {"status": "completed", "progress": 100}
            logger.debug("Model loaded successfully!")
            return True
        except Exception as e:
            model_load_status = {"status": "error", "progress": 0, "message": str(e)}
            logger.error(f"Error loading model: {e}")
            return False
    return True

# Mapping class names to denominations
denomination_map = {
    '5 Front': 5,
    '5 Back': 5,
    '10 Front': 10,
    '10 Back': 10,
    '1_Front': 1,
    '1_Back': 1,
    '20 Front': 20,
    '20 Back': 20
}

# Define a function for classifying the coin and counting them
def classify_coin_and_count(image):
    # Ensure model is loaded
    if not load_model():
        return {}, 0, [], False, False
        
    results = model(image)  # Run inference using YOLO model
    
    # Extract boxes, confidences, and labels from results
    boxes = results[0].boxes.xyxy  # Bounding box coordinates (x_min, y_min, x_max, y_max)
    confidences = results[0].boxes.conf  # Confidence scores for each detection
    labels = results[0].boxes.cls  # Class indices (coin types)
    
    # Initialize coin counts and total amount
    total_amount = 0
    coin_counts = {5: 0, 10: 0, 1: 0, 20: 0}  # Added 20 PHP coin
    
    # Store bounding box data to pass to frontend
    bounding_boxes = []
    
    # Check if there are any detections at all
    has_detections = len(boxes) > 0
    valid_detections = False
    
    for i, box in enumerate(boxes):
        class_name = results[0].names[int(labels[i])]
        confidence = float(confidences[i])
        
        # Only count valid coin types (5 Front, 10 Front, 1 Front, 20 Front)
        if class_name in denomination_map:
            valid_detections = True
            coin_value = denomination_map[class_name]
            coin_counts[coin_value] += 1
            total_amount += coin_value
            
            # Add bounding box info to list
            bounding_boxes.append({
                'box': box.tolist(),  # Convert tensor to list: [x_min, y_min, x_max, y_max]
                'class': class_name,
                'confidence': confidence,
                'value': coin_value
            })
    
    return coin_counts, total_amount, bounding_boxes, has_detections, valid_detections

# Function to convert PIL Image to base64 string for display
def image_to_data_uri(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

# Endpoint to check model loading status
@app.route("/model-status", methods=["GET"])
def model_status():
    return jsonify(model_load_status)

# Home route for uploading image
@app.route("/", methods=["GET", "POST"])
def index():
    # Don't load the model on initial page load
    if request.method == "GET":
        logger.debug("Rendering initial page without loading model")
        return render_template("index.html", first_load=True)
    
    if request.method == "POST":
        image_file = request.files.get("coin_image")
        
        # Check if an image file was uploaded
        if image_file and image_file.filename.endswith(('png', 'jpg', 'jpeg')):
            try:
                logger.debug("Processing uploaded image")
                # Open the uploaded image
                image = Image.open(image_file.stream)
                
                # Convert image to data URI for display in results
                image_data_uri = image_to_data_uri(image)
                
                # Classify the image, count the coins, and calculate the total
                # This is where model loading will happen if needed
                coin_counts, total_amount, bounding_boxes, has_detections, valid_detections = classify_coin_and_count(image)
                
                # Remove keys with zero counts
                coin_counts = {k: v for k, v in coin_counts.items() if v > 0}
                
                # Render the result on the webpage
                return render_template("index.html", 
                                      coin_counts=coin_counts, 
                                      total_amount=total_amount, 
                                      image_data_uri=image_data_uri,
                                      bounding_boxes=json.dumps(bounding_boxes),
                                      image_width=image.width,
                                      image_height=image.height,
                                      has_detections=has_detections,
                                      valid_detections=valid_detections)
            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                logger.error(error_message)
                return render_template("index.html", error_message=error_message)
        else:
            return render_template("index.html", error_message="Please upload a valid image file (PNG, JPG, JPEG).")

# Add a simple health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

# For local development
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
