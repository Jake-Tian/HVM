import numpy as np
import sys
from pathlib import Path
from utils.llm import get_embedding


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


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def find_pkl_files(graph_dir="data/graphs"):
    """
    List all video names (without .pkl extension) in the graph directory.

    Args:
        graph_dir: Directory containing graph pickle files.

    Returns:
        list[str]: Sorted video names derived from *.pkl filenames.
    """
    graph_path = Path(graph_dir)
    if not graph_path.exists():
        return []

    pkl_files = sorted(graph_path.glob("*.pkl"))
    return [f.stem for f in pkl_files]


def extract_choice_from_content(content) -> str:
    """
    Post-process a final LLM content string for MCQ extraction.

    Rules:
    1) Keep only text after the last newline.
    2) Remove all non-letter characters from that line.
    3) If the last remaining character is one of A/B/C/D, return that character.
       Otherwise, return the whole processed line.
    """
    if content is None:
        return ""
    if not isinstance(content, str):
        content = str(content)

    last_line = content.rsplit("\n", 1)[-1]
    letters_only = "".join(ch for ch in last_line if ch.isalpha())
    if not letters_only:
        return ""

    last_char = letters_only[-1].upper()
    if last_char in {"A", "B", "C", "D"}:
        return last_char
    return letters_only

