#!/usr/bin/env python3
"""Download pre-extracted frames for one or more videos from Hugging Face."""

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "JakeTian/M3-web"
LOCAL_DIR = Path("data/frames")
HF_TOKEN = os.getenv("HF_TOKEN")


def has_frames(video_dir):
    return video_dir.exists() and any(video_dir.rglob("*.jpg"))


def download_frames(video_id):
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    video_frames_dir = LOCAL_DIR / video_id

    if has_frames(video_frames_dir):
        print(f"✓ Frames for {video_id} already exist, skipping...")
        return True

    print(f"Downloading frames for {video_id} from {REPO_ID}...")

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=f"{video_id}/**",
            local_dir=str(LOCAL_DIR),
            token=HF_TOKEN,
        )
    except Exception as exc:
        print(f"✗ Error downloading frames for {video_id}: {exc}")
        return False

    if has_frames(video_frames_dir):
        print(
            f"✓ Successfully downloaded frames for {video_id} "
            f"to {video_frames_dir}"
        )
        return True

    print(f"✗ Failed to download frames for {video_id}: no JPG frames found")
    return False


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python download_hf_frames.py "
            "<video_id1> [video_id2] ..."
        )
        sys.exit(1)

    video_ids = sys.argv[1:]
    success_count = sum(
        download_frames(video_id)
        for video_id in video_ids
    )

    print(
        f"\nSummary: Successfully downloaded "
        f"{success_count}/{len(video_ids)} video frame sets."
    )

    if success_count != len(video_ids):
        sys.exit(1)


if __name__ == "__main__":
    main()