#!/usr/bin/env python3
"""
Download video frames for a specific video ID from Hugging Face dataset.
Usage: python download_hf_frames.py <video_id1> [video_id2] ...
"""

import sys
import os
from pathlib import Path
from huggingface_hub import snapshot_download

# Configuration
REPO_ID = "JakeTian/M3-web"
LOCAL_DIR = Path("data/frames")
HF_TOKEN = os.getenv("HF_TOKEN")

def download_frames(video_id):
    """
    Download frames for a specific video ID from HuggingFace.
    
    Args:
        video_id: Video ID (e.g., "3hsECQqpTw4")
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Create local directory if it doesn't exist
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Local video frames directory
    video_frames_dir = LOCAL_DIR / video_id
    
    # Check if directory already exists and is not empty
    if video_frames_dir.exists() and any(video_frames_dir.iterdir()):
        print(f"✓ Frames for {video_id} already exist, skipping...")
        return True
    
    print(f"Downloading frames for {video_id} from {REPO_ID}...")
    
    try:
        # Download only the specific video folder
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=f"{video_id}/**",
            local_dir=str(LOCAL_DIR),
            local_dir_use_symlinks=False,
            token=HF_TOKEN,
        )
        
        # Verify download
        if video_frames_dir.exists() and any(video_frames_dir.iterdir()):
            print(f"✓ Successfully downloaded frames for {video_id} to {video_frames_dir}")
            return True
        else:
            print(f"✗ Failed to download frames for {video_id}: Folder empty or not created")
            return False
            
    except Exception as e:
        print(f"✗ Error downloading frames for {video_id}: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python download_hf_frames.py <video_id1> [video_id2] ...")
        sys.exit(1)
    
    video_ids = sys.argv[1:]
    
    success_count = 0
    for video_id in video_ids:
        if download_frames(video_id):
            success_count += 1
            
    print(f"\nSummary: Successfully downloaded {success_count}/{len(video_ids)} video frame sets.")

if __name__ == "__main__":
    main()
