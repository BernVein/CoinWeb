#!/bin/bash
# Make script exit on any error
set -e

# Install specific torch version first to ensure compatibility
echo "Installing PyTorch..."
pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
echo "Installing other requirements..."
pip install -r requirements.txt

# Print Python and pip versions for debugging
echo "Python version:"
python --version
echo "Pip version:"
pip --version
echo "Installed packages:"
pip list

# Check if the model exists
if [ -f "models/PHCoinClassifier/best.pt" ]; then
    echo "Model file found!"
else
    echo "WARNING: Model file not found at models/PHCoinClassifier/best.pt"
    echo "Please ensure the model file is included in your repository"
fi

# Start application with gunicorn and increased timeout
echo "Starting Gunicorn server..."
exec gunicorn app:app --bind 0.0.0.0:$PORT --log-level debug --timeout 180 --workers 1 --preload 