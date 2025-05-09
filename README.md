# 🧠 Brain Stroke Detection Web App

This project is a Flask-based web application that allows users to upload brain MRI images and detect signs of a stroke using a trained deep learning model.

## 🚀 Features

- Upload an MRI image from your device
- Deep learning model predicts whether a stroke is detected
- User-friendly web interface
- Automatically deletes uploaded files after prediction

## 🖼️ Demo

> Upload a brain scan image to check if a stroke is detected.

![Screenshot](uploads/demo_screenshot.png) <!-- You can add a real screenshot later -->

## 🛠️ Tech Stack

- Python
- Flask
- TensorFlow / Keras
- Pillow (PIL)
- HTML/CSS (frontend)

## 🧪 How It Works

1. User uploads an MRI image.
2. Image is preprocessed (resized, normalized).
3. The trained CNN model (`brain_tumor_classifier.h5`) predicts stroke presence.
4. Result and confidence score are displayed to the user.

## 📁 Folder Structure
brain_stroke_detection/
├── templates/ # HTML templates (e.g., index.html)
├── uploads/ # Temporarily stores uploaded images
├── venv/ # Python virtual environment
├── main.py # Flask backend code
├── brain_tumor_classifier.h5 # Trained Keras model
├── requirements.txt # Python dependencies
└── .gitignore # Git ignore rules

## ⚙️ Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/brain_stroke_detection.git
cd brain_stroke_detection

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Flask app
python main.py

