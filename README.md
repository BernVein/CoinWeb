# Philippine Coin Classifier

A web application that uses computer vision to detect and classify Philippine coins.

## Features

- Real-time coin detection using YOLO (You Only Look Once) object detection model
- Classification of Philippine coins (1, 5, 10, and 20 peso coins)
- Total amount calculation based on detected coins
- Visual bounding box display around detected coins

## Project Structure

```
├── app.py                 # Flask web application
├── models/                # Directory containing the model files
│   └── PHCoinClassifier/  
│       └── best.pt        # Trained YOLO model weights
├── static/                # Static files for the web interface
├── templates/             # HTML templates
│   └── index.html         # Main application UI
├── requirements.txt       # Python dependencies
└── render.yaml            # Render deployment configuration
```

## Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python app.py
   ```

3. Visit `http://localhost:8000` in your browser.

## Deploying to Render

### Manual Deployment

1. Create a new Web Service on [Render](https://render.com/).
2. Connect your GitHub repository.
3. Use the following settings:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --workers 1 --timeout 120`
   - **Python Version**: 3.9.12
   - **Environment Variables**:
     - `RENDER`: true

### Automatic Deployment with render.yaml

1. Push your code to GitHub with the included `render.yaml` file.
2. Log in to [Render](https://render.com/) and create a new **Blueprint**.
3. Connect your GitHub repository and Render will automatically set up your web service based on the configuration in `render.yaml`.

### Important Notes for Render Deployment

1. Make sure your `best.pt` model file is uploaded to the repository in the correct location (`models/PHCoinClassifier/best.pt`).
2. The model is large, so be patient during the first deployment as it might take some time to load.
3. Render's free tier may experience cold starts, which can make the initial loading of the model slow.

## Troubleshooting Render Deployment

If you encounter issues deploying to Render:

1. Check that you're using compatible versions in `requirements.txt`.
2. Verify that your model file is in the correct location.
3. Look at the logs in the Render dashboard for specific error messages.
4. The application includes a health check endpoint at `/health` to verify the service is running.
5. Check the model status at `/model-status` to see if the model loaded correctly.

## Model Information

The model used in this project is a YOLOv8 object detection model trained to recognize Philippine coins. The model can detect:
- 1 peso coins (front and back)
- 5 peso coins (front and back)
- 10 peso coins (front and back)
- 20 peso coins (front and back) 