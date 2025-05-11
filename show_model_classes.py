from ultralytics import YOLO
import os

# Path to the model
model_path = os.path.join("models", "PHCoinClassifier", "best.pt")

# Load the model
try:
    model = YOLO(model_path)
    
    # Get the class names
    class_names = model.names
    
    print("=== Available Classes in the Model ===")
    for idx, name in class_names.items():
        print(f"Class {idx}: {name}")
    print("====================================")
    
    # Current denomination map in app.py
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
    
    # Check if all model classes are in our denomination map
    missing_classes = []
    for name in class_names.values():
        if name not in denomination_map:
            missing_classes.append(name)
    
    if missing_classes:
        print("\n❌ MISSING CLASSES IN DENOMINATION MAP:")
        for name in missing_classes:
            print(f"  - {name}")
        print("\nUpdate app.py to include these classes!")
    else:
        print("\n✅ All model classes are included in the denomination map.")
    
except Exception as e:
    print(f"Error loading model: {e}") 