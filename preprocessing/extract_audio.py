import os
import sys
from pathlib import Path
from moviepy import VideoFileClip

def extract_audio(video_dir, output_dir):
    video_path = Path(video_dir)
    audio_path = Path(output_dir)
    audio_path.mkdir(parents=True, exist_ok=True)

    videos = sorted(list(video_path.glob("*.mp4")))
    print(f"Found {len(videos)} videos to process.")

    for v in videos:
        out_file = audio_path / f"{v.stem}.wav"
        if out_file.exists():
            print(f"Skipping {v.name}, already exists.")
            continue
        
        print(f"Extracting audio from {v.name}...")
        try:
            video = VideoFileClip(str(v))
            # write_audiofile is available on the audio attribute of the VideoFileClip
            video.audio.write_audiofile(str(out_file), fps=16000, nbytes=2, codec='pcm_s16le', logger=None)
            video.close()
            print(f"✓ Saved {out_file.name}")
        except Exception as e:
            print(f"✗ Error processing {v.name}: {e}")

if __name__ == "__main__":
    v_dir = "/research/d7/gds/yztian25/data/HippoVlog/videos"
    a_dir = "HVM-Hippo/data/audio"
    extract_audio(v_dir, a_dir)
