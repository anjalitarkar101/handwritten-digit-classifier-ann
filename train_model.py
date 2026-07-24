# ==========================================================
# train_model.py - Handwritten Digit Classification using ANN
# ==========================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix
import os

print("=" * 60)
print("🔢 Handwritten Digit Classification using ANN")
print("=" * 60)

# ==========================================================
# 1. LOAD THE MNIST DATASET
# ==========================================================
print("\n📥 Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"\n📊 Dataset Information:")
print(f"   Training images: {x_train.shape}")
print(f"   Training labels: {y_train.shape}")
print(f"   Test images: {x_test.shape}")
print(f"   Test labels: {y_test.shape}")
print(f"   Image size: {x_train[0].shape}")
print(f"   Number of classes: {len(np.unique(y_train))}")

# ==========================================================
# 2. NORMALIZATION (Scale pixel values to 0-1)
# ==========================================================
print("\n🔄 Normalizing data...")
print(f"   Before normalization - min: {x_train[0].min()}, max: {x_train[0].max()}")

x_train = x_train / 255.0
x_test = x_test / 255.0

print(f"   After normalization - min: {x_train[0].min()}, max: {x_train[0].max()}")
print("✅ Normalization complete!")

# ==========================================================
# 3. BUILD THE ANN MODEL
# ==========================================================
print("\n🏗️  Building ANN model...")

model = Sequential([
    Flatten(input_shape=(28, 28)),      # Layer 1: Flatten layer - converts 2D to 1D
    Dense(128, activation='relu'),      # Layer 2: First Hidden layer
    Dropout(0.2),                       # Dropout for regularization
    Dense(64, activation='relu'),       # Layer 3: Second Hidden layer
    Dense(10, activation='softmax')     # Layer 4: Output layer
])

# Display model summary
model.summary()

# ==========================================================
# 4. COMPILE THE MODEL
# ==========================================================
print("\n⚙️  Compiling the model...")
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
print("✅ Model compiled!")

# ==========================================================
# 5. TRAIN THE MODEL
# ==========================================================
print("\n🚀 Training the model...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)
print("✅ Training complete!")

# ==========================================================
# 6. EVALUATE ON TEST DATA
# ==========================================================
print("\n📊 Evaluating on test data...")

# Predict probabilities for all test images
y_prob = model.predict(x_test)
y_pred = y_prob.argmax(axis=1)

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\n   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\n🔄 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ==========================================================
# 7. VISUALIZE TRAINING HISTORY
# ==========================================================
print("\n📈 Plotting training history...")

# Plot training and validation loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.set_title('Loss over Epochs')
ax1.grid(True)

# Plot training and validation accuracy
ax2.plot(history.history['accuracy'], label='Training Accuracy')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.set_title('Accuracy over Epochs')
ax2.grid(True)

plt.tight_layout()
plt.savefig('saved_visualisations/ann_training_history.png')
print("✅ Training history saved as 'saved_visualisations/ann_training_history.png'")

# Show plot without blocking
plt.show(block=False)
plt.pause(2)  # Show for 2 seconds
plt.close()

# ==========================================================
# 8. SAVE THE MODEL
# ==========================================================
print("\n💾 Saving the model...")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the model
model.save('models/mnist_ann_model.h5')
print("✅ Model saved as 'models/mnist_ann_model.h5'")

print("\n" + "=" * 60)
print("✅ ANN TRAINING COMPLETE!")
print("=" * 60)
print("\n📌 Next step:")
print("   Run: streamlit run app.py")
print("=" * 60)