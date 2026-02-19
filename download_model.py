#!/usr/bin/env python3
"""
Download Qwen3-0.6B model from Hugging Face
"""

import os
from huggingface_hub import snapshot_download

def download_qwen3_model():
    """Download Qwen3-0.6B model"""
    
    # Set HF endpoint to mirror
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    print("🚀 Starting Qwen3-0.6B model download...")
    print("=" * 50)
    
    try:
        # Create models directory
        models_dir = "./models/Qwen3-0.6B"
        os.makedirs(models_dir, exist_ok=True)
        
        print(f"📦 Downloading Qwen/Qwen3-0.6B to {models_dir}")
        
        # Download the model
        model_path = snapshot_download(
            repo_id="Qwen/Qwen3-0.6B",
            local_dir=models_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"✅ Model downloaded successfully!")
        print(f"📁 Model path: {model_path}")
        
        # List downloaded files
        print("\n📋 Downloaded files:")
        for root, dirs, files in os.walk(models_dir):
            level = root.replace(models_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = download_qwen3_model()
    exit(0 if success else 1)