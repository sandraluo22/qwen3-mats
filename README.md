# Qwen3-4B-Thinking-2507 Inference

This project provides a simple script to run inference with the Qwen3-4B-Thinking-2507 model for basic mathematics prompts. Optimized for RunPod GPU instances.

## RunPod Setup

### 1. Deploy a RunPod GPU Instance

1. Go to [RunPod](https://www.runpod.io/) and create an account
2. Navigate to "Pods" → "+ Deploy Pod"
3. Select a GPU with at least **8GB VRAM** (RTX 3090, A4000, or better recommended)
4. Choose a template:
   - **PyTorch 2.0+** template (recommended)
   - Or **Ubuntu 22.04** with Python 3.10+
5. Set disk space to at least **20GB** (for model download)
6. Deploy the pod

### 2. Connect to Your RunPod Instance

You can connect via:
- **SSH**: Use the provided SSH command from RunPod dashboard
- **Web Terminal**: Click "Connect" → "Web Terminal" in RunPod dashboard
- **JupyterLab**: If available in your template

### 3. Setup on RunPod

Once connected to your RunPod instance:

```bash
# Option 1: Quick setup (recommended)
chmod +x setup_runpod.sh
./setup_runpod.sh

# Option 2: Manual setup
pip install -r requirements.txt
```

### 4. Run Inference

```bash
python inference.py
```

The script will automatically detect and use the GPU. On first run, it will download the model (~8GB), which may take a few minutes.

## Local Setup (Alternative)

If running locally with a GPU:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the inference script:
```bash
python inference.py
```

## Usage

The script includes several example mathematics prompts. You can modify the `prompt` variable in `main()` to use your own prompts.

### Example Prompts

- "What is 15 multiplied by 23?"
- "Solve: 2x + 5 = 17. What is the value of x?"
- "Calculate the area of a circle with radius 7. Use π = 3.14159."
- "If a train travels 120 km in 2 hours, what is its average speed?"

### Custom Usage

You can also import and use the `run_inference` function in your own code:

```python
from inference import run_inference

response = run_inference("What is 25 * 4?")
print(response)
```

## Model Information

- **Model**: Qwen3-4B-Thinking-2507
- **Source**: Hugging Face (Qwen/Qwen3-4B-Thinking-2507)
- **Note**: This is a "Thinking" model that may include reasoning tags in its output

## Requirements

- Python 3.8+
- **RunPod GPU Instance** (recommended):
  - GPU with at least 8GB VRAM (RTX 3090, A4000, or better)
  - 20GB+ disk space for model storage
  - CUDA 11.8+ (usually pre-installed in RunPod templates)
- **Local GPU** (alternative):
  - CUDA-capable GPU with 8GB+ VRAM
  - CUDA toolkit installed
- **CPU** (not recommended, very slow):
  - ~16GB+ RAM
  - Expect 5-10x slower inference

## GPU Verification

The script automatically checks for GPU availability and displays:
- GPU name and model
- Available VRAM
- CUDA version
- Which GPU device is being used

If no GPU is detected, the script will fall back to CPU (with a warning).

