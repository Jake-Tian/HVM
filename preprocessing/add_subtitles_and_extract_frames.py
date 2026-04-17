"""
Add subtitles to video and extract one frame per second with subtitles.
Saves frames to data/frames/{video_name}/{second}/ directory structure.
"""

import cv2
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def parse_srt_file(srt_path: Path) -> List[Dict[str, any]]:
    """
    Parse SRT subtitle file and return list of subtitle entries.
    
    Args:
        srt_path: Path to .srt file
        
    Returns:
        List of dicts with 'start', 'end', and 'text' keys
    """
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match SRT entries
    # Format: number\nHH:MM:SS,mmm --> HH:MM:SS,mmm\ntext\n
    pattern = r'(\d+)\n(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})\n(.*?)(?=\n\d+\n|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    subtitles = []
    for match in matches:
        # Extract timestamp components
        start_h, start_m, start_s, start_ms = map(int, match[1:5])
        end_h, end_m, end_s, end_ms = map(int, match[5:9])
        
        # Convert to seconds
        start_time = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000.0
        end_time = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000.0
        
        # Extract text (remove speaker prefix if present, or keep it)
        text = match[9].strip()
        
        subtitles.append({
            'start': start_time,
            'end': end_time,
            'text': text
        })
    
    return subtitles


def srt_time_to_seconds(time_str: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    parts = time_str.split(',')
    time_part = parts[0].split(':')
    ms = int(parts[1]) if len(parts) > 1 else 0
    
    hours = int(time_part[0])
    minutes = int(time_part[1])
    seconds = int(time_part[2])
    
    return hours * 3600 + minutes * 60 + seconds + ms / 1000.0


def get_subtitle_at_time(subtitles: List[Dict], time_seconds: float) -> str:
    """Get the active subtitle text at a given time."""
    for sub in subtitles:
        if sub['start'] <= time_seconds <= sub['end']:
            return sub['text']
    return None


def wrap_text(text: str, font, font_scale: float, thickness: int, max_width: int) -> List[str]:
    """
    Wrap text into multiple lines that fit within max_width.
    """
    words = text.split(' ')
    lines = []
    current_line = ''
    
    for word in words:
        test_line = current_line + (' ' if current_line else '') + word
        (line_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        
        if line_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines if lines else [text]


def draw_subtitle_on_frame(frame: np.ndarray, subtitle_text: str, 
                          font_size: int = 24, font_color: Tuple[int, int, int] = (255, 255, 255),
                          bg_color: Tuple[int, int, int] = (0, 0, 0),
                          position: str = 'bottom') -> np.ndarray:
    """
    Draw subtitle text on a frame.
    
    Args:
        frame: BGR image array
        subtitle_text: Text to display
        font_size: Font size
        font_color: BGR color tuple for text
        bg_color: BGR color tuple for background
        position: 'bottom' or 'center'
        
    Returns:
        Frame with subtitle drawn
    """
    if not subtitle_text:
        return frame
    
    height, width = frame.shape[:2]
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = font_size / 30.0
    thickness = max(1, int(font_size / 20))
    line_type = cv2.LINE_AA
    
    # Wrap text
    max_text_width = int(width * 0.9)
    text_lines = wrap_text(subtitle_text, font, font_scale, thickness, max_text_width)
    
    # Calculate dimensions
    line_heights = []
    line_widths = []
    for line in text_lines:
        (line_width, line_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        line_heights.append(line_height + baseline)
        line_widths.append(line_width)
    
    # Box dimensions
    padding = 10
    line_spacing = 5
    box_width = max(line_widths) + 2 * padding
    box_height = sum(line_heights) + (len(text_lines) - 1) * line_spacing + 2 * padding
    
    # Position
    if position == 'bottom':
        y_pos = height - box_height - 20
    else:  # center
        y_pos = (height - box_height) // 2
    
    x_pos = (width - box_width) // 2
    
    # Draw background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x_pos, y_pos),
        (x_pos + box_width, y_pos + box_height),
        bg_color,
        -1
    )
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Draw text
    text_x = x_pos + padding
    current_y = y_pos + padding + line_heights[0] if line_heights else y_pos + padding
    
    for i, line in enumerate(text_lines):
        line_width = line_widths[i]
        line_x = x_pos + (box_width - line_width) // 2
        
        cv2.putText(
            frame,
            line,
            (line_x, current_y),
            font,
            font_scale,
            font_color,
            thickness,
            line_type
        )
        
        if i < len(text_lines) - 1:
            current_y += line_heights[i] + line_spacing
    
    return frame


def process_video_with_subtitles(video_path: Path, srt_path: Path, 
                                  output_frames_dir: Path,
                                  frames_per_second: int = 1):
    """
    Process video: extract one frame per second and add subtitles.
    
    Args:
        video_path: Path to input video
        srt_path: Path to SRT subtitle file
        output_frames_dir: Directory to save frames (e.g., data/frames/bedroom_01)
        frames_per_second: Number of frames to extract per second (default: 1)
    """
    # Parse subtitles
    print(f"Parsing subtitle file: {srt_path}")
    subtitles = parse_srt_file(srt_path)
    print(f"Found {len(subtitles)} subtitle entries")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0
    
    print(f"Video: {width}x{height}, {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s duration")
    
    # Create output directory
    output_frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frames - one per second, grouped into folders of 30 frames each
    # Track which seconds we've already extracted
    extracted_seconds = set()
    frames_saved = 0
    frames_per_folder = 30
    
    print("\nExtracting frames with subtitles...")
    print(f"Grouping frames into folders of {frames_per_folder} frames each")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_count / fps if fps > 0 else 0.0
        current_second = int(current_time)
        
        # Extract one frame per second (only once per second)
        if current_second not in extracted_seconds and current_second >= 0:
            # Get subtitle for this time
            subtitle_text = get_subtitle_at_time(subtitles, current_time)
            
            # Draw subtitle on frame
            frame_with_subtitle = draw_subtitle_on_frame(
                frame.copy(),
                subtitle_text,
                font_size=28,
                font_color=(255, 255, 255),
                bg_color=(0, 0, 0),
                position='bottom'
            )
            
            # Calculate folder number (1-indexed): every 30 frames go into a new folder
            folder_num = (frames_saved // frames_per_folder) + 1
            # Calculate frame number within folder (1-indexed)
            frame_num_in_folder = (frames_saved % frames_per_folder) + 1
            
            # Create directory for this folder (e.g., data/frames/bedroom_01/1/)
            folder_dir = output_frames_dir / str(folder_num)
            folder_dir.mkdir(parents=True, exist_ok=True)
            
            # Save frame (numbered within folder: 1.jpg, 2.jpg, ..., 30.jpg)
            frame_filename = folder_dir / f"{frame_num_in_folder}.jpg"
            cv2.imwrite(str(frame_filename), frame_with_subtitle)
            frames_saved += 1
            extracted_seconds.add(current_second)
            
            if frames_saved % 10 == 0:
                print(f"  Saved {frames_saved} frames (up to {current_second}s, folder {folder_num})")
        
        frame_count += 1
    
    cap.release()
    print(f"\n✓ Extracted {frames_saved} frames to {output_frames_dir}")
    return frames_saved


def main():
    """Process videos with matching subtitle files.
    
    Usage:
        python add_subtitles_and_extract_frames.py [video_name1] [video_name2] ...
        
    If video names are provided, processes only those videos.
    If no video names are provided, processes all videos in data/videos that have
    matching subtitle files in data/subtitles/robot.
        
    Examples:
        python add_subtitles_and_extract_frames.py bedroom_01  # Process single video
        python add_subtitles_and_extract_frames.py bedroom_01 bedroom_02  # Process two videos
        python add_subtitles_and_extract_frames.py              # Process all videos
    """
    import sys
    
    # Paths
    videos_dir = Path("data/videos")
    subtitles_dir = Path("data/subtitles/robot")
    frames_base_dir = Path("data/frames")
    
    # Get video names from command line arguments (if provided)
    if len(sys.argv) > 1:
        # Process specified videos
        video_names = [arg.replace('.mp4', '') for arg in sys.argv[1:]]
        videos_to_process = []
        
        for video_name in video_names:
            video_path = videos_dir / f"{video_name}.mp4"
            srt_path = subtitles_dir / f"{video_name}.srt"
        
            if not video_path.exists():
                print(f"✗ Video file not found: {video_path}")
                continue
        
            if not srt_path.exists():
                print(f"✗ Subtitle file not found: {srt_path}")
                continue
                
            videos_to_process.append((video_name, video_path, srt_path))
        
            if not videos_to_process:
                print("No valid videos to process.")
                return
        
        # Process each video
        for video_name, video_path, srt_path in videos_to_process:
            print(f"\n{'='*60}")
            print(f"Processing: {video_name}")
            print(f"{'='*60}")
            print(f"Video: {video_path}")
            print(f"Subtitles: {srt_path}")
            print(f"Output: {frames_base_dir / video_name}")
            print(f"{'='*60}\n")
            
            try:
                frames_saved = process_video_with_subtitles(
                    video_path=video_path,
                    srt_path=srt_path,
                    output_frames_dir=frames_base_dir / video_name,
                    frames_per_second=1
                )
                print(f"\n✓ Successfully processed {video_name}: {frames_saved} frames saved")
            except Exception as e:
                print(f"\n✗ Error processing {video_name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Process all videos with matching subtitle files
        video_files = list(videos_dir.glob("*.mp4"))
        if not video_files:
            print("No video files found in data/videos/")
            return
        
        # Find videos with matching subtitle files
        videos_to_process = []
        for video_file in video_files:
            video_name = video_file.stem
            srt_path = subtitles_dir / f"{video_name}.srt"
            if srt_path.exists():
                videos_to_process.append((video_name, video_file, srt_path))
            else:
                print(f"⚠ Skipping {video_name}: No matching subtitle file found")
        
        if not videos_to_process:
            print("No videos with matching subtitle files found.")
            return
        
        print(f"\n{'='*60}")
        print(f"Found {len(videos_to_process)} video(s) with matching subtitles")
        print(f"{'='*60}\n")
        
        # Process each video
        successful = 0
        failed = 0
        
        for i, (video_name, video_path, srt_path) in enumerate(videos_to_process, 1):
            output_frames_dir = frames_base_dir / video_name
            
            print(f"\n{'='*60}")
            print(f"[{i}/{len(videos_to_process)}] Processing: {video_name}")
            print(f"{'='*60}")
            print(f"Video: {video_path}")
            print(f"Subtitles: {srt_path}")
            print(f"Output: {output_frames_dir}")
            print(f"{'='*60}\n")
            
            try:
                frames_saved = process_video_with_subtitles(
                    video_path=video_path,
                    srt_path=srt_path,
                    output_frames_dir=output_frames_dir,
                    frames_per_second=1
                )
                print(f"\n✓ Successfully processed {video_name}: {frames_saved} frames saved")
                successful += 1
            except Exception as e:
                print(f"\n✗ Error processing {video_name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Processing Summary")
        print(f"{'='*60}")
        print(f"Total videos: {len(videos_to_process)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

