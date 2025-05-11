from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageDraw
import os
import base64
from io import BytesIO
import json
import torch
import torch.serialization

# Initialize Flask app
app = Flask(__name__)

# Global variables
model = None
model_load_status = {"status": "not_started", "progress": 0}

def load_model():
    global model, model_load_status
    if model is None:
        try:
            # Import here to reduce cold start time
            from ultralytics import YOLO
            import ultralytics.nn.tasks
            import torch.nn.modules.container
            
            # Fix for PyTorch 2.6+ weights_only=True default
            # Add safe globals to allow loading YOLO model
            model_load_status = {"status": "initializing", "progress": 10}
            
            # Get a list of safe globals to add
            safe_globals = [
                ultralytics.nn.tasks.DetectionModel,
                ultralytics.nn.modules.Conv,
                ultralytics.nn.modules.block.C2f,
                ultralytics.nn.modules.block.SPPF,
                torch.nn.modules.container.Sequential,  # Add PyTorch Sequential module
                torch.nn.modules.activation.SiLU,       # Add SiLU activation
                torch.nn.modules.conv.Conv2d,           # Add Conv2d
                torch.nn.modules.batchnorm.BatchNorm2d  # Add BatchNorm2d
            ]
            
            # Dynamically check if Head module exists
            try:
                head_module = getattr(ultralytics.nn.modules, 'Head')
                safe_globals.append(head_module)
                print("Added Head module to safe globals")
            except AttributeError:
                # Head module doesn't exist in this version
                print("Head module not found in ultralytics.nn.modules, skipping")
                
                # Try to find alternative modules for different Ultralytics versions
                try:
                    import ultralytics.nn.modules.head
                    head_module = ultralytics.nn.modules.head.Detect
                    safe_globals.append(head_module)
                    print("Added Detect module to safe globals")
                except (ImportError, AttributeError):
                    print("Could not find alternative head modules")
            
            # Add all safe globals to torch serialization
            for sg in safe_globals:
                torch.serialization.add_safe_globals([sg])
            
            # Use model from the repository
            model_path = os.path.join("models", "PHCoinClassifier", "best.pt")
            
            if not os.path.exists(model_path):
                model_load_status = {"status": "error", "progress": 0, "message": "Model file not found in repository"}
                print(f"Error: Model file not found at {model_path}")
                return False
                
            # Load your trained YOLOv8 model
            model_load_status = {"status": "loading", "progress": 50}
            print(f"Loading model from {model_path}...")
            
            try:
                # First attempt - standard loading
                model = YOLO(model_path)
                model_load_status = {"status": "completed", "progress": 100}
                print("Model loaded successfully!")
                return True
            except Exception as first_error:
                print(f"First attempt at loading model failed: {first_error}")
                
                try:
                    # Second attempt - use a direct torch.load with weights_only=False
                    # Note: Only use with trusted model files!
                    print("Attempting alternate loading method with weights_only=False...")
                    
                    # Override YOLO's loading mechanism by monkey patching torch.load
                    original_torch_load = torch.load
                    
                    def patched_torch_load(*args, **kwargs):
                        kwargs['weights_only'] = False
                        return original_torch_load(*args, **kwargs)
                    
                    # Apply the patch
                    torch.load = patched_torch_load
                    
                    # Try to load the model again
                    model = YOLO(model_path)
                    
                    # Restore original torch.load
                    torch.load = original_torch_load
                    
                    model_load_status = {"status": "completed", "progress": 100}
                    print("Model loaded successfully with fallback method!")
                    return True
                except Exception as second_error:
                    # Restore original torch.load if not already restored
                    torch.load = original_torch_load
                    
                    model_load_status = {"status": "error", "progress": 0, "message": f"Failed to load model: {second_error}"}
                    print(f"All attempts to load model failed. Final error: {second_error}")
                    return False
        except Exception as e:
            model_load_status = {"status": "error", "progress": 0, "message": str(e)}
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

# Endpoint to check model loading status
@app.route("/model-status", methods=["GET"])
def model_status():
    return jsonify(model_load_status)

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

# For deployment
app.debug = False

# Get port from environment variable for Render
port = int(os.environ.get("PORT", 10000))

# Ensure the Flask app runs
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
