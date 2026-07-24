# ==========================================================
# predict.py - Prediction functions for ANN
# ==========================================================

import numpy as np
import cv2
from tensorflow import keras
import os


def load_model(model_path='models/mnist_ann_model.h5'):
    """Load the trained ANN model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return keras.models.load_model(model_path)


def preprocess_image(img_path):

    # Read image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # Resize to 28x28
    img = cv2.resize(img, (28, 28))

    # Invert colors if needed (MNIST has white digits on black background)
    # If the image has black digits on white background, invert it
    if np.mean(img) > 127:
        img = 255 - img

    # Normalize to 0-1
    img = img / 255.0

    # Reshape for model input
    img = img.reshape(1, 28, 28)

    return img


def predict_digit(img_path, model):

    # Preprocess the image
    preprocessed_img = preprocess_image(img_path)

    # Make prediction
    predictions = model.predict(preprocessed_img, verbose=0)
    predicted_digit = predictions.argmax(axis=1)[0]

    # ✅ Convert numpy.float32 to Python float
    confidence = float(predictions[0][predicted_digit])  
    probabilities = predictions[0]

    return predicted_digit, confidence, probabilities