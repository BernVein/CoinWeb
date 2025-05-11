from flask import Flask, request, render_template
from PIL import Image, ImageDraw
from ultralytics import YOLO
import os
import base64
from io import BytesIO
import json

# Initialize Flask app
app = Flask(__name__)

# Path to the model
model_path = os.path.join("models", "PHCoinClassifier", "best.pt")

# Load your trained YOLOv8 model
model = YOLO(model_path)

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
    
    return render_template("index.html")

# Ensure the Flask app runs
if __name__ == "__main__":
    app.run(debug=True)
