from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from utils.llm import get_embedding


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    a = np.array(vec1, dtype=float)
    b = np.array(vec2, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return -1.0
    return float(np.dot(a, b) / denom)


def _collect_json_files(file_directory: str | Path) -> list[Path]:
    target = Path(file_directory)
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")
    if target.is_file():
        return [target]
    return sorted(target.glob("*.json"))


def _parse_behavior_row(row) -> tuple[str, str] | None:
    if not isinstance(row, list) or len(row) < 2:
        return None
    timestamp, content = row[0], row[1]
    if not isinstance(timestamp, str) or not isinstance(content, str):
        return None
    return timestamp, content


def _parse_conversation_row(row) -> tuple[str, str, str] | None:
    if not isinstance(row, dict):
        return None
    timestamp = row.get("start_time")
    speaker = row.get("speaker")
    content = row.get("content")
    if not isinstance(timestamp, str) or not isinstance(speaker, str) or not isinstance(content, str):
        return None
    return timestamp, speaker, content


def search_behavior(
    question: str,
    k: int,
    file_directory: str | Path,
    context_window: int = 1,
) -> list[list[str]]:
    """
    Search behavior files and return top-k hits expanded with context.
    For each selected hit, include `context_window` rows before and after.

    Behavior file format:
      [
        [start_time_hhmmss, content, embedding],
        ...
      ]
    """
    if k <= 0:
        return []

    query_embedding = get_embedding(question)
    scored: list[tuple[float, int, int]] = []  # score, file_idx, row_idx
    file_data: list[list] = []

    for file_idx, json_path in enumerate(_collect_json_files(file_directory)):
        with json_path.open("r", encoding="utf-8") as infile:
            data = json.load(infile)
        file_data.append(data if isinstance(data, list) else [])

        for row_idx, item in enumerate(file_data[file_idx]):
            if not isinstance(item, list) or len(item) < 3:
                continue
            parsed = _parse_behavior_row(item)
            if parsed is None:
                continue
            _, _ = parsed
            embedding = item[2]
            if not isinstance(embedding, list):
                continue
            score = _cosine_similarity(query_embedding, embedding)
            scored.append((score, file_idx, row_idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    anchors = scored[:k]

    selected: list[tuple[str, str]] = []
    seen = set()
    for _, file_idx, row_idx in anchors:
        rows = file_data[file_idx]
        start = max(0, row_idx - context_window)
        end = min(len(rows) - 1, row_idx + context_window)
        for idx in range(start, end + 1):
            parsed = _parse_behavior_row(rows[idx])
            if parsed is None:
                continue
            key = (file_idx, idx)
            if key in seen:
                continue
            seen.add(key)
            selected.append(parsed)

    selected.sort(key=lambda x: x[0])
    return [[timestamp, content] for timestamp, content in selected]


def search_conversation(
    question: str,
    speaker_strict: list[str] | None,
    file_directory: str | Path,
    k: int = 50,
    context_window: int = 1,
) -> list[list[str]]:
    """
    Search conversation files and return top-k hits expanded with context.
    For each selected hit, include `context_window` rows before and after.

    Conversation file format:
      [
        {
          "start_time": "hhmmss",
          "speaker": "...",
          "content": "...",
          "embedding": [...]
        },
        ...
      ]
    """
    if k <= 0:
        return []

    query_embedding = get_embedding(question)
    speaker_filter = {s for s in speaker_strict} if speaker_strict else None
    scored: list[tuple[float, int, int]] = []  # score, file_idx, row_idx
    file_data: list[list] = []

    for file_idx, json_path in enumerate(_collect_json_files(file_directory)):
        with json_path.open("r", encoding="utf-8") as infile:
            data = json.load(infile)
        file_data.append(data if isinstance(data, list) else [])

        for row_idx, item in enumerate(file_data[file_idx]):
            parsed = _parse_conversation_row(item)
            if parsed is None:
                continue
            _, speaker, _ = parsed
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                continue
            if speaker_filter is not None and speaker not in speaker_filter:
                continue
            score = _cosine_similarity(query_embedding, embedding)
            scored.append((score, file_idx, row_idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    anchors = scored[:k]

    selected: list[tuple[str, str, str]] = []
    seen = set()
    for _, file_idx, row_idx in anchors:
        rows = file_data[file_idx]
        start = max(0, row_idx - context_window)
        end = min(len(rows) - 1, row_idx + context_window)
        for idx in range(start, end + 1):
            parsed = _parse_conversation_row(rows[idx])
            if parsed is None:
                continue
            key = (file_idx, idx)
            if key in seen:
                continue
            seen.add(key)
            selected.append(parsed)

    selected.sort(key=lambda x: x[0])
    return [[timestamp, speaker, content] for timestamp, speaker, content in selected]