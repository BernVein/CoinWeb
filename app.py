from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageDraw
import os
import base64
from io import BytesIO
import json
import urllib.request
import tempfile
import time
import gdown
import torch
import torch.serialization
import os

# Initialize Flask app
app = Flask(__name__)

# For Vercel deployment, we'll load the model dynamically when needed
# to avoid issues with model size limitations
model = None
model_download_status = {"status": "not_started", "progress": 0}

def download_model_from_gdrive():
    global model_download_status
    model_download_status = {"status": "downloading", "progress": 10}
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.join("models", "PHCoinClassifier"), exist_ok=True)
    
    # Google Drive file ID from the shared link
    file_id = "12Guiu6h4ZK562yucv0DfSVLbGwKL3rN7"
    model_path = os.path.join("models", "PHCoinClassifier", "best.pt")
    
    try:
        model_download_status = {"status": "downloading", "progress": 30}
        # Use gdown to download file from Google Drive
        output = model_path
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)
        
        model_download_status = {"status": "completed", "progress": 100}
        return model_path
    except Exception as e:
        model_download_status = {"status": "error", "progress": 0, "message": str(e)}
        print(f"Error downloading model: {e}")
        return None

def load_model():
    global model, model_download_status
    if model is None:
        try:
            # Import here to reduce cold start time
            from ultralytics import YOLO
            import ultralytics.nn.tasks
            
            # Fix for PyTorch 2.6+ weights_only=True default
            # Add safe globals to allow loading YOLO model
            torch.serialization.add_safe_globals([
                ultralytics.nn.tasks.DetectionModel,
                ultralytics.nn.modules.Conv,
                ultralytics.nn.modules.block.C2f,
                ultralytics.nn.modules.block.SPPF,
                ultralytics.nn.modules.Head
            ])
            
            # Check if model exists locally, if not download it
            model_path = os.path.join("models", "PHCoinClassifier", "best.pt")
            if not os.path.exists(model_path):
                model_download_status = {"status": "starting", "progress": 5}
                model_path = download_model_from_gdrive()
                if model_path is None:
                    return False
                
            # Load your trained YOLOv8 model
            model_download_status = {"status": "loading", "progress": 90}
            model = YOLO(model_path)
            model_download_status = {"status": "completed", "progress": 100}
            return True
        except Exception as e:
            model_download_status = {"status": "error", "progress": 0, "message": str(e)}
            print(f"Error loading model: {e}")
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

# Endpoint to check model download status
@app.route("/model-status", methods=["GET"])
def model_status():
    return jsonify(model_download_status)

# Home route for uploading image
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image_file = request.files.get("coin_image")
        
        # Check if an image file was uploaded
        if image_file and image_file.filename.endswith(('png', 'jpg', 'jpeg')):
            try:
                # Open the uploaded image
                image = Image.open(image_file.stream)
                
                # Convert image to data URI for display in results
                image_data_uri = image_to_data_uri(image)
                
                # Classify the image, count the coins, and calculate the total
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
                return render_template("index.html", error_message=error_message)
        else:
            return render_template("index.html", error_message="Please upload a valid image file (PNG, JPG, JPEG).")
    
    return render_template("index.html", first_load=True)

# Add a simple health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

# For Vercel serverless deployment
app.debug = False

# Get port from environment variable for Render
port = int(os.environ.get("PORT", 8080))

# Ensure the Flask app runs
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
