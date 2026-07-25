# ==========================================================
# predict.py - Prediction functions for ANN
# ==========================================================

import numpy as np
from PIL import Image
from tensorflow import keras
import os


def load_model(model_path='models/mnist_ann_model.h5'):
    """Load the trained ANN model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return keras.models.load_model(model_path)


def preprocess_image(image):
    """
    image: PIL Image object (from uploaded file)
    """

    # Convert to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')

    # Resize to 28x28 (MNIST size)
    image = image.resize((28, 28))

    # Convert to numpy array
    img = np.array(image, dtype=np.float32)

    # Invert colors if needed (MNIST has white digits on black background)
    # If the image has black digits on white background, invert it
    if np.mean(img) > 127:
        img = 255 - img

    # Normalize to 0-1
    img = img / 255.0

    # Reshape for model input(1, 28, 28)
    img = img.reshape(1, 28, 28)

    return img


def predict_digit(image, model):
    """
        image: PIL Image object
    """

    # Preprocess the image
    preprocessed_img = preprocess_image(image)

    # Make prediction
    predictions = model.predict(preprocessed_img, verbose=0)
    predicted_digit = predictions.argmax(axis=1)[0]

    # ✅ Convert numpy.float32 to Python float
    confidence = float(predictions[0][predicted_digit])  
    probabilities = predictions[0]

    return predicted_digit, confidence, probabilities