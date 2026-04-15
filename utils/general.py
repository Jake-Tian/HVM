import numpy as np
import sys
import re
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
            # Check if it's an unknown placeholder format: <name_number> (e.g., <male_1>, <police_1>)
            is_placeholder = bool(re.match(r"^<\w+_\d+>$", char_name))
            if is_placeholder:
                similarity = cosine_similarity(embedding, char_appearance[1])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = char_name

        if best_similarity > similarity_threshold:
            # placeholder → placeholder
            is_char_placeholder = bool(re.match(r"^<\w+_\d+>$", character.name))
            if is_char_placeholder:
                # Merge logic: if both are placeholders, we might need a way to decide which one to keep.
                # For now, let's keep the best match from the dictionary.
                appearance_dict[best_match] = [character.appearance, embedding]
                equivalence_list.append([character.name, best_match])
            # named character → placeholder
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

    pkl_files = sorted(graph_path.glob("*.pkl"))
    return [f.stem for f in pkl_files]

