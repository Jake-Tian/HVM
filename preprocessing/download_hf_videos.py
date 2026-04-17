#!/usr/bin/env python3
"""
Download videos from Hugging Face dataset.
Usage: python download_hf_videos.py bedroom_01 bedroom_02 ...
"""

import sys
import requests
from pathlib import Path
from tqdm import tqdm

# Configuration
BASE_URL = "https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/resolve/main/videos/robot"
LOCAL_DIR = Path("data/videos")

def download_video(video_name):
    """
    Download a video from HuggingFace.
    
    Args:
        video_name: Video name without extension (e.g., "bedroom_01")
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Ensure video name ends with .mp4
    if not video_name.endswith('.mp4'):
        video_name = f"{video_name}.mp4"
    
    # Construct URL
    url = f"{BASE_URL}/{video_name}?download=true"
    
    # Create local directory if it doesn't exist
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Local file path
    local_file = LOCAL_DIR / video_name
    
    # Check if file already exists
    if local_file.exists():
        print(f"✓ {video_name} already exists, skipping...")
        return True
    
    print(f"Downloading {video_name}...")
    
    try:
        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        # Download and save with progress bar
        with open(local_file, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=video_name, disable=(total_size == 0)) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        if total_size > 0:
                            pbar.update(len(chunk))
        
        print(f"✓ Successfully downloaded {video_name} to {local_file}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error downloading {video_name}: {e}")
        # Clean up partial file if it exists
        if local_file.exists():
            local_file.unlink()
        return False
    except Exception as e:
        print(f"✗ Unexpected error downloading {video_name}: {e}")
        # Clean up partial file if it exists
        if local_file.exists():
            local_file.unlink()
        return False


def main():
    """Main function to download videos from command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python download_hf_videos.py <video_name1> [video_name2] ...")
        print("Example: python download_hf_videos.py bedroom_01 bedroom_02 gym_01")
        sys.exit(1)
    
    video_names = sys.argv[1:]
    print(f"Downloading {len(video_names)} video(s) to {LOCAL_DIR}...")
    print()
    
    success_count = 0
    failed_videos = []
    
    for video_name in video_names:
        # Remove .mp4 extension if provided
        video_name = video_name.replace('.mp4', '')
        
        if download_video(video_name):
            success_count += 1
        else:
            failed_videos.append(video_name)
        print()
    
    # Summary
    print("=" * 60)
    print(f"Download Summary:")
    print(f"  Successful: {success_count}/{len(video_names)}")
    if failed_videos:
        print(f"  Failed: {', '.join(failed_videos)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
