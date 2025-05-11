from flask import Flask, request, render_template
from PIL import Image
from ultralytics import YOLO
import os

# Initialize Flask app
app = Flask(__name__)

# Path to the model
model_path = os.path.join("models", "PHCoinClassifier", "best.pt")

# Load your trained YOLOv8 model
model = YOLO(model_path)

# Define a function for classifying the coin
def classify_coin(image):
    results = model(image)  # Run inference using YOLO model
    
    # Extract boxes, confidences, and labels from results
    boxes = results[0].boxes.xyxy  # Bounding box coordinates (x_min, y_min, x_max, y_max)
    confidences = results[0].boxes.conf  # Confidence scores for each detection
    labels = results[0].boxes.cls  # Class indices (coin types)
    
    detections = []
    for i, box in enumerate(boxes):
        class_name = results[0].names[int(labels[i])]
        confidence = confidences[i].item()  # Convert tensor to Python float
        box_coords = box.tolist()  # Convert tensor to list of coordinates
        
        detections.append({
            'class': class_name,
            'confidence': confidence,
            'box': box_coords
        })
    
    return detections

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
                
                # Classify the image and get detections
                detections = classify_coin(image)
                
                # Render the result on the webpage
                return render_template("index.html", detections=detections)
            except Exception as e:
                return f"Error processing image: {e}", 400
        else:
            return "Please upload a valid image file (PNG, JPG, JPEG).", 400
    
    return render_template("index.html")

# Ensure the Flask app runs
if __name__ == "__main__":
    app.run(debug=True)
