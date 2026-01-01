#!/bin/bash
# Quick setup script for RunPod GPU instances

echo "Setting up Qwen3 inference environment on RunPod..."
echo ""

# Check if running on RunPod
if [ -n "$RUNPOD_POD_ID" ]; then
    echo "✓ Detected RunPod environment (Pod ID: $RUNPOD_POD_ID)"
else
    echo "⚠ Not running on RunPod (or RUNPOD_POD_ID not set)"
fi

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ nvidia-smi not found. Make sure you're using a GPU instance."
fi

echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete! You can now run:"
echo "  python inference.py"

