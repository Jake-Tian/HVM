#!/usr/bin/env python3
"""
Download the audio folder from Hugging Face dataset.
Usage: python download_hf_audio.py
"""

from huggingface_hub import snapshot_download
from pathlib import Path

# Configuration
repo_id = "JakeTian/HippoVlog"
folder_path = "audio"  # The folder within the repo
local_dir = Path("data")  # Where to save locally

# Create local directory if it doesn't exist
local_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading folder '{folder_path}' from {repo_id}...")
print(f"Destination: {local_dir / folder_path}")

try:
    # Download only the specific folder
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=f"{folder_path}/**",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    
    print(f"\n✓ Successfully downloaded to {local_dir / folder_path}")
except Exception as e:
    print(f"\n✗ Error downloading: {e}")
    exit(1)
