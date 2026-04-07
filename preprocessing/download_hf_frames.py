#!/usr/bin/env python3
"""
Download video frames from Hugging Face dataset.
Usage: python download_hf_frames.py video_name
"""

import sys
from pathlib import Path
from huggingface_hub import snapshot_download

def download_frames(video_name):
    repo_id = "JakeTian/HippoVlog"
    local_dir = Path("data/frames")
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading frames for {video_name} from {repo_id}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[f"{video_name}/**"],
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        print(f"✓ Successfully downloaded frames for {video_name}")
        return True
    except Exception as e:
        print(f"✗ Error downloading frames for {video_name}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_hf_frames.py <video_name>")
        sys.exit(1)
    
    video_name = sys.argv[1]
    if not download_frames(video_name):
        sys.exit(1)
