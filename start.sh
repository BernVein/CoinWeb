#!/bin/bash
# Install requirements
pip install -r requirements.txt

# Print Python and pip versions for debugging
echo "Python version:"
python --version
echo "Pip version:"
pip --version
echo "Installed packages:"
pip list

# Start application with gunicorn with increased timeouts
exec gunicorn app:app --bind 0.0.0.0:$PORT --log-level info --timeout 120 --workers 1 