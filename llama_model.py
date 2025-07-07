import os
from langchain_community.llms import LlamaCpp

def initialize_llama_model(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
    temperature=0.1,
    max_tokens=2000,
    top_p=1,
    verbose=False,
    n_ctx=2048  # Add this parameter to increase context window
):
    """Initialize and return the Llama model"""
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please download the model first.")
    
    print(f"Initializing Llama model from {model_path}...")
    
    # Initialize Llama model with larger context window
    llm = LlamaCpp(
        model_path=model_path,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        verbose=verbose,
        n_ctx=n_ctx,  # Set context window size
    )
    
    print("Llama model initialized successfully!")
    return llm

def download_llama_model(output_dir="models"):
    """
    Provide instructions for downloading Llama model
    (actual download requires user to follow external steps due to licensing)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Llama Model Download Instructions ===")
    print("1. Visit https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main")
    print("2. Download the file 'llama-2-7b-chat.Q4_K_M.gguf' (or another quantized version)")
    print(f"3. Place the downloaded file in the '{output_dir}' directory")
    print(f"4. Ensure the path is: '{output_dir}/llama-2-7b-chat.Q4_K_M.gguf'")
    print("\nNote: You may need to create a Hugging Face account and accept the model license.")
    
    return os.path.join(output_dir, "llama-2-7b-chat.Q4_K_M.gguf")

if __name__ == "__main__":
    # This allows running this file directly to download the model
    model_path = download_llama_model()
    print(f"\nOnce downloaded, the model will be available at: {model_path}")
    print("You can then run the other scripts to process emails and start the app.")