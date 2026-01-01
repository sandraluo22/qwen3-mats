# Qwen3-4B-Thinking-2507 Inference

This project provides scripts to run inference with the Qwen3-4B-Thinking-2507 model for mathematics problems, including GSM8K dataset analysis.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the GSM8K analysis:
```bash
python gsm8k_analysis.py
```

The script will automatically detect and use the GPU. On first run, it will download the model (~8GB), which may take a few minutes.

## Usage

### GSM8K Analysis

The main script (`gsm8k_analysis.py`) performs chain of thought analysis on GSM8K problems:

- Loads 50 problems from GSM8K test set
- Runs Qwen3 on each problem
- Extracts chain of thought sentences
- For each sentence, resamples the question 10 times until stabilized
- Forces final answers from partial chain of thought
- Saves results to `gsm8k_analysis_results.json`

```bash
python gsm8k_analysis.py
```

## Model Information

- **Model**: Qwen3-4B-Thinking-2507
- **Source**: Hugging Face (Qwen/Qwen3-4B-Thinking-2507)
- **Note**: This is a "Thinking" model that may include reasoning tags in its output

## Requirements

- Python 3.8+
- GPU with at least 8GB VRAM (recommended)
- CUDA-capable GPU with CUDA toolkit installed
- 20GB+ disk space for model storage
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
