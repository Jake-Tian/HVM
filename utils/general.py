import numpy as np
import os
import sys
from pathlib import Path
from utils.embedding import get_embedding


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


class QuietStdout:
    """Redirect stdout to a file only (terminal stays quiet).

    tqdm writes to stderr by default, so progress bars remain visible in the
    terminal while all `print(...)` output is captured to the log file.
    """
    def __init__(self, file):
        self.file = file

    def write(self, text):
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.file.flush()


def verbose_terminal() -> bool:
    """HVM_VERBOSE=1 restores the old verbose terminal output (Tee to both)."""
    return os.environ.get("HVM_VERBOSE", "0") == "1"

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


def merge_character_appearances(characters_appearance, appearance_dict, similarity_threshold=0.85):
    """
    Merge/update character appearances into appearance_dict.

    Args:
        characters_appearance: Iterable of objects with .name and .appearance fields
        appearance_dict: Dict mapping character name -> [appearance_text, embedding]
        similarity_threshold: Threshold for matching unknown placeholders
    """
    equivalence_list = []
    for character in characters_appearance:
        # old character
        if character.name in appearance_dict:
            if appearance_dict[character.name][0] != character.appearance:
                appearance_dict[character.name][0] = character.appearance
                appearance_dict[character.name][1] = get_embedding(character.appearance)
            continue

        embedding = get_embedding(character.appearance)
        best_similarity = 0.0
        best_match = None
        # new character
        for char_name, char_appearance in appearance_dict.items():
            if char_name.startswith("<character_"):
                similarity = cosine_similarity(embedding, char_appearance[1])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = char_name

        if best_similarity > similarity_threshold:
            # <character_X> → <character_Y>
            if character.name.startswith("<character_"):
                # Keep the smaller character_X key in appearance_dict.
                current_idx = int(character.name[len("<character_"):-1])
                best_idx = int(best_match[len("<character_"):-1])
                if current_idx < best_idx:
                    appearance_dict.pop(best_match, None)
                    appearance_dict[character.name] = [character.appearance, embedding]
                    equivalence_list.append([best_match, character.name])
                else:
                    appearance_dict[best_match] = [character.appearance, embedding]
                    equivalence_list.append([character.name, best_match])
            # named character → <character_X>
            else:
                appearance_dict.pop(best_match, None)
                appearance_dict[character.name] = [character.appearance, embedding]
                equivalence_list.append([best_match, character.name])
        else:
            appearance_dict[character.name] = [character.appearance, embedding]

    return equivalence_list


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

    pkl_files = sorted(
        path for path in graph_path.glob("*.pkl")
        if not path.stem.endswith("_preabstraction")
    )
    return [f.stem for f in pkl_files]
