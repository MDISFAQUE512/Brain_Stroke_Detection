import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from datetime import datetime

# Initialize Flask app  
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load Keras model once at startup
MODEL_PATH = 'brain_tumor_classifier.h5'
model = load_model(MODEL_PATH)

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image) / 255.0  # Normalize to [0,1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

def predict_tumor(image_path):
    try:
        processed_image = preprocess_image(image_path)
        prediction = model.predict(processed_image)
        probability = float(prediction[0][0])
        result = "Tumor Detected" if probability > 0.5 else "No Tumor Detected"
        confidence = probability if probability > 0.5 else 1 - probability
        return {
            "result": result,
            "confidence": f"{confidence:.2%}"
        }
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image file (PNG, JPG, JPEG, or GIF)'}), 400

    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(filepath)

        result = predict_tumor(filepath)

        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify(result)

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

def cleanup_uploads():
    current_time = datetime.now()
    for filename in os.listdir(UPLOAD_FOLDER):
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file_time = datetime.fromtimestamp(os.path.getctime(filepath))
        if (current_time - file_time).total_seconds() > 3600:
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Error deleting file {filepath}: {str(e)}")

if __name__ == '__main__':
    cleanup_uploads()
    app.run(debug=True)
