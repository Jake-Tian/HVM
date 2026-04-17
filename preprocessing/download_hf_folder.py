#!/usr/bin/env python3
"""
Download a folder from Hugging Face dataset/repository.
Usage: python download_hf_folder.py
"""

from huggingface_hub import snapshot_download
from pathlib import Path

# Configuration
repo_id = "ByteDance-Seed/M3-Bench"
folder_path = "subtitles"  # The folder within the repo
local_dir = Path("data")  # Where to save locally (will create subtitles subfolder)

# Create local directory if it doesn't exist
local_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading folder '{folder_path}' from {repo_id}...")
print(f"Destination: {local_dir / folder_path}")

try:
    # Download only the specific folder
    # This will create the folder structure: data/subtitles/
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=f"{folder_path}/**",  # Only download files in the subtitles folder
        local_dir=str(local_dir),  # Base directory
        local_dir_use_symlinks=False,
    )
    
    print(f"\n✓ Successfully downloaded to {local_dir / folder_path}")
except Exception as e:
    print(f"\n✗ Error downloading: {e}")
    print("\nMake sure you have huggingface_hub installed:")
    print("  pip install huggingface_hub")
    print("\nAlternative: Try using the command line:")
    print(f"  huggingface-cli download {repo_id} --repo-type dataset --local-dir {local_dir} --include '{folder_path}/**'")

