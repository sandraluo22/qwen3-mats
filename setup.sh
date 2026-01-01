#!/bin/bash
# Quick setup script

echo "Setting up Qwen3 inference environment..."
echo ""

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ nvidia-smi not found. GPU may not be available."
fi

echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete! You can now run:"
echo "  python gsm8k_analysis.py"

