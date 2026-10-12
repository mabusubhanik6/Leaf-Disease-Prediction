# Leaf Disease Prediction Using Neural Networks

## 1. Project Overview

Leaf Disease Prediction is a deep learning application that uses a Convolutional Neural Network (CNN) to classify potato leaf images into three categories: Potato Early Blight, Potato Healthy, and Potato Late Blight.

The application provides disease information, symptoms, precautions, and plant health tips in English, Telugu, Hindi, and Kannada.

## 2. Objectives

* Detect potato leaf diseases from images.
* Classify leaves into three categories.
* Display prediction confidence.
* Provide understandable disease information and precautions.
* Support regional languages for easier access.

## 3. Technologies Used

* Python
* TensorFlow and Keras
* Convolutional Neural Network (CNN)
* OpenCV and Pillow
* NumPy and Matplotlib
* Streamlit
* JSON for disease information and translations

## 4. Dataset

The dataset contains 900 potato leaf images:

* Potato Early Blight: 470 images
* Potato Healthy: 363 images
* Potato Late Blight: 67 images

An 80/20 class-wise split was used:

* Training: 719 images
* Validation: 181 images

## 5. Model Architecture

The CNN includes:

1. Image rescaling
2. Convolutional layers with ReLU activation
3. Max-pooling layers
4. Flatten layer
5. Dense layer
6. Dropout layer
7. Softmax output layer for three classes

## 6. Evaluation Results

* Validation accuracy: approximately 99.45%
* Validation loss: approximately 0.0147

These results apply to the held-out validation dataset. Real-world performance may differ, and further testing with independent images is recommended.

## 7. Application Features

* Upload a leaf image.
* Capture a leaf photo using a camera.
* Predict the leaf category.
* Display prediction confidence.
* Explain the disease and its symptoms.
* Provide precautions and plant health tips.
* Select English, Telugu, Hindi, or Kannada.

## 8. How to Run

Activate the virtual environment in PowerShell:

```
.\venv\Scripts\Activate.ps1
```

Install dependencies if needed:

```
python -m pip install -r requirements.txt
```

Start the application:

```
python -m streamlit run app.py
```

## 9. Disclaimer

This application provides an AI-based prediction and general plant-health information. For important agricultural decisions, consult a qualified local agriculture expert.
