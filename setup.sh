#!/bin/bash
echo "=========================================="
echo "🔢 Handwritten Digit Classifier - ANN Setup"
echo "=========================================="

echo "📁 Creating directories..."
mkdir -p models saved_visualisations uploads

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Train model: python train_model.py"
echo "2. Run app: streamlit run app.py"
echo "=========================================="