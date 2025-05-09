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

