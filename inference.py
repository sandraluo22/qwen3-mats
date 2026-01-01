#!/usr/bin/env python3
"""
Inference script for Qwen3-4B-Thinking-2507 model with basic mathematics prompts.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os


def check_gpu():
    """Check and display GPU information."""
    print("=" * 60)
    print("GPU INFORMATION")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available!")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name}")
            print(f"    Total Memory: {gpu_memory:.2f} GB")
        
        # Set default device
        device = torch.cuda.current_device()
        print(f"\n  Using GPU: {device} ({torch.cuda.get_device_name(device)})")
        print("=" * 60)
        return True
    else:
        print("✗ CUDA is not available. Will use CPU (slow).")
        print("=" * 60)
        return False


def run_inference(prompt: str, model_name: str = "Qwen/Qwen3-4B-Thinking-2507", max_new_tokens: int = 512):
    """
    Run inference with Qwen3-4B-Thinking-2507 model.
    
    Args:
        prompt: The mathematics prompt/question
        model_name: Hugging Face model identifier
        max_new_tokens: Maximum number of tokens to generate
    
    Returns:
        Generated response from the model
    """
    # Check GPU availability
    use_gpu = check_gpu()
    
    print(f"\nLoading model: {model_name}")
    print("This may take a few minutes on first run (downloading model)...")
    
    # Determine dtype and device map
    if use_gpu:
        dtype = torch.float16  # Use float16 for GPU to save memory
        device_map = "auto"  # Automatically distribute across available GPUs
        print(f"Using GPU with float16 precision")
    else:
        dtype = torch.float32
        device_map = None
        print(f"Using CPU with float32 precision")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,  # May be needed for some Qwen models
    )
    
    # If no device_map, manually move to device
    if not use_gpu and device_map is None:
        model = model.to("cpu")
    
    print("Model loaded successfully!")
    print(f"\nPrompt: {prompt}\n")
    print("Generating response...\n")
    
    # Format the input as a conversation
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize input
    # Get the device from the model
    device = next(model.parameters()).device
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Clear GPU cache if using GPU
    if use_gpu:
        torch.cuda.empty_cache()
    
    # Extract only the newly generated tokens
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
    # Decode the response
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    return response


def main():
    """Main function with example mathematics prompts."""
    
    # Example mathematics prompts
    math_prompts = [
        "What is 15 multiplied by 23?",
        "Solve: 2x + 5 = 17. What is the value of x?",
        "Calculate the area of a circle with radius 7. Use π = 3.14159.",
        "If a train travels 120 km in 2 hours, what is its average speed?",
    ]
    
    # You can modify this to use a custom prompt
    prompt = math_prompts[0]  # Change index to try different prompts
    
    # Or uncomment to use a custom prompt:
    # prompt = "What is 42 + 58?"
    
    try:
        response = run_inference(prompt)
        print("\n" + "=" * 60)
        print("RESPONSE:")
        print("=" * 60)
        print(response)
        print("=" * 60)
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

