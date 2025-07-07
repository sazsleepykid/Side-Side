import os
from langchain_community.llms import LlamaCpp

# Ensure CUDA path is correctly set
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
os.environ["PATH"] = os.path.join(os.environ["CUDA_PATH"], "bin") + ";" + os.environ.get("PATH", "")

def initialize_llama_model(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
    temperature=0.1,
    max_tokens=2000,
    top_p=1.0,
    n_ctx=2048,
    n_gpu_layers=-1,        # ✅ Use all available GPU memory
    n_batch=512,
    f16_kv=True,
    verbose=False
):
    """Initialize the LlamaCpp model with GPU acceleration if available."""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"Initializing Llama model from {model_path} with GPU acceleration...")
    
    try:
        llm = LlamaCpp(
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            f16_kv=f16_kv,
            use_mlock=False,
            use_mmap=True,
            seed=42,
            verbose=verbose
        )
        print("✅ Llama model initialized successfully with GPU acceleration!")
        return llm
    except Exception as e:
        print(f"⚠️ GPU init failed: {e}")
        print("Falling back to CPU mode...")

        llm = LlamaCpp(
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n_ctx=n_ctx,
            verbose=verbose
        )
        print("✅ Llama model initialized in CPU-only mode.")
        return llm


def check_gpu_availability():
    """Check if CUDA GPU is available on the system."""
    try:
        cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9")
        nvcc_path = os.path.join(cuda_path, "bin", "nvcc.exe")

        if not os.path.exists(cuda_path):
            return False, f"CUDA path not found: {cuda_path}"
        if not os.path.exists(nvcc_path):
            return False, f"nvcc not found at: {nvcc_path}"
        
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if "GeForce" in result.stdout:
            return True, "CUDA-compatible NVIDIA GPU detected"
        else:
            return False, "nvidia-smi did not detect an active GPU"
    except Exception as e:
        return False, f"Error detecting GPU: {e}"


def download_llama_model(output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Llama Model Download Instructions ===")
    print("1. Visit: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF")
    print("2. Download: 'llama-2-7b-chat.Q4_K_M.gguf'")
    print(f"3. Place it into: {output_dir}/")
    print(f"4. Final path: {os.path.join(output_dir, 'llama-2-7b-chat.Q4_K_M.gguf')}")
    print("\nNote: Login to Hugging Face and accept model license if prompted.")
    
    return os.path.join(output_dir, "llama-2-7b-chat.Q4_K_M.gguf")


if __name__ == "__main__":
    model_path = download_llama_model()
    print(f"\nModel will be available at: {model_path}")
    
    gpu_available, gpu_info = check_gpu_availability()
    if gpu_available:
        print(f"\n✅ GPU detected: {gpu_info}")
    else:
        print(f"\n⚠️ {gpu_info}")
