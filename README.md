🔢 Handwritten Digit Classifier

## 📖 Overview
A Handwritten Digit Classification System built with Streamlit and Artificial Neural Networks (ANN) that recognizes handwritten digits (0-9) from uploaded images. The system uses a neural network trained on the MNIST dataset to provide accurate digit predictions.

---

## ✨ Features
- 🔢 Digit Recognition - Classifies handwritten digits from 0-9 using ANN
- 📸 Image Upload - Supports PNG, JPG, and JPEG formats
- 📊 Probability Distribution - Shows confidence for each digit
- 🎯 Confidence Score - Displays prediction confidence level
- 📈 Visual Feedback - Bar chart showing probability distribution
- 🎨 Clean UI - User-friendly interface with instant results

---


## 🛠️ Technologies Used
- Python 3.10+ - Core programming language
- TensorFlow/Keras - Deep learning framework
- Streamlit - Web application framework
- OpenCV - Image preprocessing
- NumPy - Numerical operations
- Matplotlib - Data visualization
- Scikit-learn - Model evaluation metrics

---

## 📁 Project Structure
```text
handwritten-digit-classifier-ann/
├── app.py                    # Main Streamlit application (UI)
├── predict.py                # Prediction functions
├── train_model.py            # Model training script
├── requirements.txt          # Python dependencies
├── setup.sh                  # Setup script
├── .gitignore               # Git ignore file
├── models/                   # Trained model files (gitignored)
│   └── mnist_ann_model.h5
├── uploads/                  # Temporary uploads (gitignored)
├── saved_visualisations/     # Generated plots (gitignored)
│   └── ann_training_history.png
└── README.md                 # Project documentation
```


---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/anjalitarkar101/handwritten-digit-classifier-ann.git
cd handwritten-digit-classifier
```

### Step 2: Run Setup Script
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create required directories (models/, uploads/, saved_visualisations/)
- Install all dependencies

### Step 3: Train the Model
```bash
python train_model.py
```

This will:
- Load the MNIST dataset
- Train the ANN model
- Save the model to models/ folder

### Step 4: Run the Application
```bash
streamlit run app.py
Open your browser and navigate to http://localhost:8501
```

---

## 📊 How It Works
1. Data Preprocessing
- Loads MNIST dataset (70,000 images of handwritten digits)
- Normalizes pixel values to 0-1 range
- Splits into training and test sets

2. Model Architecture (ANN)
```txt
Input Layer (28x28 pixels)
    ↓
Flatten Layer (784 neurons)
    ↓
Dense Layer (128 neurons, ReLU)
    ↓
Dropout (20%)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Output Layer (10 neurons, Softmax)
    ↓
Prediction (Digit 0-9)
```

3. Prediction Pipeline
- User uploads an image
- Image is preprocessed (resized, inverted, normalized)
- Model predicts the digit
- Results displayed with confidence and probability distribution

---

## 🔧 Dependencies
```txt
tensorflow==2.13.0
numpy==1.24.3
matplotlib==3.7.2
Pillow==10.0.0
opencv-python==4.8.1.78
scikit-learn==1.3.0
streamlit==1.50.0
```

---

## 📝 Usage Guide
1. Upload Image - Click "Choose an image" and select a handwritten digit image
2. View Prediction - See the predicted digit with confidence score
3. Check Distribution - View probability distribution across all digits
4. Confidence Level - High (>90%), Moderate (>70%), or Low (<70%)

---

## 📊 Dataset Information

| Dataset | Size | Description |
|:--------|:----:|:------------|
| **MNIST Training** | 60,000 images | Handwritten digits (0-9) for training |
| **MNIST Test** | 10,000 images | Handwritten digits (0-9) for testing |
---

## 📄 License
This project is licensed under the MIT License.

© 2026 Anjali Tarkar. All rights reserved.

---

## 👩‍💻 Author
**Anjali Tarkar**
- GitHub: https://github.com/anjalitarkar101
- Email: anjalitarkar101@gmail.com

---

## ⭐ Show Your Support
If you find this project useful, please give it a star on GitHub!

---

## 🙏 Acknowledgments
- Yann LeCun, Corinna Cortes, Christopher J.C. Burges - MNIST Dataset 
- TensorFlow/Keras - For the deep learning framework
- Streamlit - For the awesome web framework

