import os
from datetime import datetime
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = os.path.join(BASE_DIR, 'brain_tumor_classifier.h5')

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load model once at startup
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise

# Utility: Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Utility: Preprocess image
def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        return np.expand_dims(image, axis=0)
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")

# Utility: Predict
def predict_tumor(image_path):
    try:
        processed = preprocess_image(image_path)
        prediction = model.predict(processed)
        probability = float(prediction[0][0])
        result = "Tumor Detected" if probability > 0.5 else "No Tumor Detected"
        confidence = probability if probability > 0.5 else 1 - probability
        return {"result": result, "confidence": f"{confidence:.2%}"}
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

# Route: Home
@app.route('/')
def home():
    return render_template('index.html')

# Route: Predict
@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}), 400

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)
        result = predict_tumor(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Always clean up the uploaded file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as cleanup_error:
                print(f"Cleanup failed for {filepath}: {cleanup_error}")

# Optional: Clean old uploads
def cleanup_uploads(hours_old=1):
    now = datetime.now()
    threshold = hours_old * 3600
    for filename in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            created_time = os.path.getctime(path)
            if (now - datetime.fromtimestamp(created_time)).total_seconds() > threshold:
                os.remove(path)
        except Exception as e:
            print(f"Error deleting {path}: {e}")

# Entry point
if __name__ == '__main__':
    cleanup_uploads()
    app.run(debug=True)
