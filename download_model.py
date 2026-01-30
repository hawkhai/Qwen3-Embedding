#!/usr/bin/env python3
"""
Pre-download script for Qwen3-Embedding model and tokenizer.
Run this script to download all required files before using the model.
"""

import os
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download

def download_qwen3_embedding():
    """Download Qwen3-Embedding model and tokenizer."""
    model_name = 'Qwen/Qwen3-Embedding-0.6B'

    print(f"Downloading {model_name}...")
    print("This may take several minutes depending on your internet connection.")

    try:
        # Download tokenizer
        print("\n1. Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer downloaded successfully")

        # Download model
        print("\n2. Downloading model...")
        model = AutoModel.from_pretrained(model_name)
        print("✓ Model downloaded successfully")

        # Alternative: Use snapshot_download to download all files at once
        # This is more efficient for complete downloads
        print("\n3. Ensuring all model files are downloaded...")
        cache_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=None,  # Use default cache directory
            resume_download=True
        )

        print(f"\n✓ All files downloaded successfully!")
        print(f"Model cached in: {cache_dir}")
        print(f"You can now run your main script without internet connection.")

        # Test that everything works
        print("\n4. Testing model loading...")
        test_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        test_model = AutoModel.from_pretrained(model_name)
        print("✓ Model and tokenizer load successfully")

        return True

    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have internet connection")
        print("2. Upgrade transformers: pip install transformers>=4.51.0")
        print("3. Check if you have enough disk space")
        return False

if __name__ == "__main__":
    print("Qwen3-Embedding Model Downloader")
    print("=" * 40)

    # Check transformers version
    try:
        import transformers
        version = transformers.__version__
        print(f"transformers version: {version}")

        # Parse version and check if it's >= 4.51.0
        major, minor, patch = map(int, version.split('.'))
        if major < 4 or (major == 4 and minor < 51):
            print(f"⚠️  WARNING: transformers {version} is too old!")
            print("Please upgrade with: pip install transformers>=4.51.0")
            exit(1)
        else:
            print("✓ transformers version is compatible")

    except ImportError:
        print("✗ transformers not installed!")
        print("Please install with: pip install -r requirements.txt")
        exit(1)

    print()
    success = download_qwen3_embedding()

    if success:
        print("\n🎉 Download completed successfully!")
        print("You can now run: python3 my_corpus.py")
    else:
        print("\n❌ Download failed. Please check the error messages above.")
        exit(1)
