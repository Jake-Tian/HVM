"""
General utility helpers.
"""
import sys
from pathlib import Path


class Tee:
    """Write to both file and stdout."""
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout
        
    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.stdout.write(text)
        self.stdout.flush()
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()

def strip_code_fences(text) -> str:
    """
    Remove surrounding Markdown code fences (``` or ```json) from a string.
    Preserves inner content exactly.
    """
    if text is None:
        return ""
    if isinstance(text, tuple):
        text = text[0] if text else ""
    if not isinstance(text, str):
        return str(text)

    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the first fence line
        lines = stripped.splitlines()
        if lines:
            # Remove the opening fence (could be ``` or ```json)
            lines = lines[1:]
        # If the last line is a closing fence, drop it
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def load_video_list(video_list_path="video_list.txt"):
    """Load video names from video_list.txt."""
    video_names = []
    with open(video_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                video_names.append(line)
    return video_names

