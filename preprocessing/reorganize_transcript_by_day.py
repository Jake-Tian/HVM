#!/usr/bin/env python3
"""
Reorganize A1_JAKE transcript SRT files by day: English lines only, with embeddings.

Reads:  data/EgoLife/EgoLifeCap/Transcript/A1_JAKE/DAY*/A1_JAKE_DAY*_*.srt
Writes: data/conversations/DAY1.json ... DAY7.json

Each entry is a dict:
  start_time: str (hhmmss, wall clock: filename hour + SRT mm:ss)
  speaker: str
  content: str (English only)
  embedding: list[float]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from utils.llm import get_multiple_embeddings

# e.g. .../Transcript/A1_JAKE/DAY3/A1_JAKE_DAY3_12000000.srt
# Hour block from filename: A1_JAKE_DAY1_11000000.srt -> 11
FILENAME_HOUR_PATTERN = re.compile(r"A1_JAKE_DAY[1-7]_(\d{2})\d{6}\.srt$", re.IGNORECASE)
# Speaker: text
SPEAKER_LINE_PATTERN = re.compile(r"^\s*([^:]+):\s*(.*)$")
# CJK (common) — skip these lines for English content selection
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

OUTPUT_DIR = Path("data/conversations")
EMBED_BATCH_SIZE = 100
DEFAULT_INPUT_ROOT = Path("data/EgoLife/EgoLifeCap/Transcript/A1_JAKE")


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def extract_english_content(lines: list[str]) -> list[tuple[str, str]]:
    """
    From subtitle text lines like 'Jake: ...' / 'Jake: English',
    return list of (speaker, english_content) for lines whose content has no CJK.
    """
    pairs: list[tuple[str, str]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = SPEAKER_LINE_PATTERN.match(line)
        if not m:
            continue
        speaker = m.group(1).strip()
        content = m.group(2).strip()
        if not content:
            continue
        if has_cjk(content):
            continue
        pairs.append((speaker, content))
    return pairs


def parse_srt_time_start(time_line: str) -> tuple[int, int] | None:
    """Return (minute, second) from 'HH:MM:SS,mmm --> ...' start part uses HH:MM:SS."""
    if "-->" not in time_line:
        return None
    start = time_line.split("-->", 1)[0].strip()
    parts = start.replace(",", ".").split(":")
    if len(parts) != 3:
        return None
    try:
        m = int(parts[1])
        sec_part = parts[2]
        s = int(float(sec_part))
    except ValueError:
        return None
    # Segment files use 00:MM:SS for the hour block; wall minute = M, second = S
    return m, s


def wall_hhmmss(file_hour: int, minute: int, second: int) -> str:
    return f"{file_hour % 24:02d}{minute % 60:02d}{second % 60:02d}"


def filename_hour(path: Path) -> int | None:
    m = FILENAME_HOUR_PATTERN.match(path.name)
    if not m:
        return None
    return int(m.group(1))


def day_from_path(path: Path) -> str | None:
    for part in path.parts:
        if len(part) == 4 and part.startswith("DAY") and part[3:].isdigit():
            return part.upper()
    return None


def parse_srt_blocks(content: str) -> list[tuple[str, list[str]]]:
    """List of (time_line, text_lines)."""
    blocks: list[tuple[str, list[str]]] = []
    raw_blocks = re.split(r"\n\s*\n", content.strip())
    for block in raw_blocks:
        lines = [ln.rstrip("\r") for ln in block.strip().split("\n")]
        if len(lines) < 2:
            continue
        i = 0
        if lines[0].strip().isdigit():
            i = 1
        if i >= len(lines):
            continue
        time_line = lines[i]
        if "-->" not in time_line:
            continue
        text_lines = lines[i + 1 :]
        if not text_lines:
            continue
        blocks.append((time_line, text_lines))
    return blocks


def load_srt_file(path: Path, file_hour: int) -> list[tuple[str, str, str]]:
    """
    Returns list of (start_hhmmss, speaker, english_content).
    """
    out: list[tuple[str, str, str]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for time_line, text_lines in parse_srt_blocks(text):
        t = parse_srt_time_start(time_line)
        if t is None:
            continue
        minute, second = t
        start_hhmmss = wall_hhmmss(file_hour, minute, second)
        pairs = extract_english_content(text_lines)
        if not pairs:
            continue
        # One subtitle cue: usually CN + EN same speaker — take last English line
        speaker, content = pairs[-1]
        out.append((start_hhmmss, speaker, content))
    return out


def embed_entries(
    entries: list[tuple[str, str, str]], batch_size: int
) -> list[dict]:
    """Build list of dicts with embeddings."""
    result: list[dict] = []
    for i in range(0, len(entries), batch_size):
        batch = entries[i : i + batch_size]
        texts = [c for _, _, c in batch]
        embeddings = get_multiple_embeddings(texts)
        for (start_time, speaker, content), emb in zip(batch, embeddings):
            result.append(
                {
                    "start_time": start_time,
                    "speaker": speaker,
                    "content": content,
                    "embedding": emb,
                }
            )
    return result


def collect_by_day(input_root: Path) -> dict[str, list[tuple[str, str, str]]]:
    by_day: dict[str, list[tuple[str, str, str]]] = {f"DAY{d}": [] for d in range(1, 8)}
    for srt_path in sorted(input_root.rglob("*.srt")):
        day = day_from_path(srt_path)
        if day is None:
            continue
        fh = filename_hour(srt_path)
        if fh is None:
            continue
        by_day[day].extend(load_srt_file(srt_path, fh))
    for day in by_day:
        by_day[day].sort(key=lambda x: (x[0], x[1], x[2]))
    return by_day


def write_outputs(by_day: dict[str, list[tuple[str, str, str]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for day_idx in range(1, 8):
        day = f"DAY{day_idx}"
        entries = by_day.get(day, [])
        records = embed_entries(entries, EMBED_BATCH_SIZE) if entries else []
        out_path = OUTPUT_DIR / f"{day}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"{day}: {len(records)} entries -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-day conversation JSON with English text and embeddings."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root folder containing DAY1/.../DAY7/ with SRT files.",
    )
    args = parser.parse_args()
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")

    by_day = collect_by_day(args.input_root)
    write_outputs(by_day)


if __name__ == "__main__":
    main()
