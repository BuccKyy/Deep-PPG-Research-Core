#!/bin/bash

# Setup script for Deep-PPG-Research-Core
# This script initializes the project for first-time users

set -e  # Exit on error

echo "🧠 Setting up Deep-PPG-Research-Core..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p datasets/{raw,processed,tfrecords}
mkdir -p experiments/{notebooks,results/{figures,logs,metrics}}
mkdir -p models/pretrained
mkdir -p logs

# Create .gitkeep files to preserve empty directories
touch datasets/raw/.gitkeep
touch datasets/processed/.gitkeep
touch datasets/tfrecords/.gitkeep
touch experiments/results/figures/.gitkeep
touch experiments/results/logs/.gitkeep
touch experiments/results/metrics/.gitkeep
touch models/pretrained/.gitkeep

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "⚙️  Creating .env file..."
    cat > .env << EOF
# Dataset paths
DATASET_ROOT=./datasets/
MIMIC_PATH=/path/to/mimic-iii/
PULSEDB_PATH=/path/to/pulsedb/

# Experiment tracking
MLFLOW_TRACKING_URI=http://localhost:5000
TENSORBOARD_LOG_DIR=./experiments/logs/

# Model export
EXPORT_DIR=./deployment/models/
TFLITE_QUANTIZATION=int8

# Hardware
CUDA_VISIBLE_DEVICES=0
TF_GPU_MEMORY_GROWTH=true
EOF
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Download sample data: python scripts/download_datasets.py --dataset sample"
echo "3. Run example training: python training/train.py --config training/config/cnn_lstm_config.yaml"
echo ""
echo "Happy researching! 🚀"
