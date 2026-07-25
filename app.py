# ==========================================================
# app.py - Handwritten Digit Classifier Web App (ANN)
# ==========================================================

import os
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from predict import load_model, predict_digit

# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="Handwritten Digit Classifier - ANN",
    page_icon="🔢",
    layout="wide"
)

st.title("🔢 Handwritten Digit Classifier using ANN")
st.markdown("Upload an image of a handwritten digit (0-9) and get instant prediction!")


# ==========================================================
# Load Model
# ==========================================================
@st.cache_resource
def get_model():
    """Load the trained ANN model."""
    try:
        return load_model()
    except FileNotFoundError:
        return None


model = get_model()

if model is None:
    st.error("❌ Model not found! Please run: python train_model.py")
    st.stop()

# ==========================================================
# Upload Section
# ==========================================================
uploaded_image = st.file_uploader(
    "Choose an image...",
    type=['png', 'jpg', 'jpeg'],
    help="Upload a clear handwritten digit"
)

if uploaded_image is not None:

    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    # Save uploaded image
    file_path = os.path.join('uploads', uploaded_image.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_image.getbuffer())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Your Image")
        display_image = Image.open(uploaded_image)
        st.image(display_image, width=300)

    with col2:
        st.subheader("🔍 Prediction")
        with st.spinner("Analyzing..."):
            # Get prediction
            digit, confidence, probabilities = predict_digit(file_path, model)

        # Display result
        st.markdown(f"### Predicted Digit: **{digit}**")
        st.markdown(f"**Confidence:** {confidence * 100:.1f}%")
        st.progress(min(confidence, 1.0))

        # Show confidence bar
        if confidence > 0.9:
            st.success("✅ High confidence prediction!")
        elif confidence > 0.7:
            st.warning("⚠️ Moderate confidence")
        else:
            st.error("❌ Low confidence - try a clearer image")

    # Show probability distribution
    st.markdown("---")
    st.subheader("📊 Probability Distribution")

    # Create a bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    digits = list(range(10))
    colors = ['green' if i == digit else 'blue' for i in digits]
    ax.bar(digits, probabilities, color=colors)
    ax.set_xlabel('Digit')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(10))
    st.pyplot(fig)

else:
    st.info("👆 Upload an image to get predictions!")

st.markdown("---")
st.caption("🔢 Powered by TensorFlow + Streamlit")