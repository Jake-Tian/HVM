#!/usr/bin/env python3
"""
Reorganize translated dense-caption JSONL files by day and add embeddings.

Output format per item:
  [start_time_hhmmss, translated_text, embedding]

One output file is written per day (DAY1 ... DAY7).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from utils.llm import get_multiple_embeddings


FILENAME_DAY_PATTERN = re.compile(r".*_(DAY[1-7])_\d{8}\.jsonl$")
CUSTOM_ID_TIME_PATTERN = re.compile(r".*-(\d{6})-(\d{6})$")
OUTPUT_DIR = Path("data/EgoLife/behaviors")
EMBED_BATCH_SIZE = 100


def discover_translated_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*.jsonl"))


def extract_day_from_filename(file_path: Path) -> str | None:
    match = FILENAME_DAY_PATTERN.match(file_path.name)
    if not match:
        return None
    return match.group(1)


def extract_start_time(custom_id: str) -> str | None:
    match = CUSTOM_ID_TIME_PATTERN.match(custom_id)
    if not match:
        return None
    return match.group(1)


def load_day_records(input_files: list[Path]) -> dict[str, list[tuple[str, str]]]:
    """
    Returns:
        day -> list of (start_time_hhmmss, translated_text)
    """
    day_records: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for file_path in input_files:
        day = extract_day_from_filename(file_path)
        if day is None:
            continue

        with file_path.open("r", encoding="utf-8") as infile:
            for raw in infile:
                line = raw.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                custom_id = record.get("custom_id")
                text = record.get("translated_text", "")
                if not isinstance(custom_id, str) or not isinstance(text, str):
                    continue
                text = text.strip()
                if not text:
                    continue

                start_time = extract_start_time(custom_id)
                if start_time is None:
                    continue
                day_records[day].append((start_time, text))

    for day in day_records:
        day_records[day].sort(key=lambda x: x[0])
    return day_records


def embed_records(records: list[tuple[str, str]], batch_size: int) -> list[list]:
    """
    Convert records to [[time, text, embedding], ...].
    """
    output: list[list] = []
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        texts = [text for _, text in batch]
        embeddings = get_multiple_embeddings(texts)
        for (start_time, text), embedding in zip(batch, embeddings):
            output.append([start_time, text, embedding])
    return output


def write_day_outputs(
    day_records: dict[str, list[tuple[str, str]]],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for day_idx in range(1, 8):
        day = f"DAY{day_idx}"
        records = day_records.get(day, [])
        with_embeddings = embed_records(records, EMBED_BATCH_SIZE) if records else []
        output_path = OUTPUT_DIR / f"{day}.json"
        with output_path.open("w", encoding="utf-8") as outfile:
            json.dump(with_embeddings, outfile, ensure_ascii=False, indent=2)
        print(f"{day}: {len(with_embeddings)} entries -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reorganize translated captions by day and add embeddings."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/EgoLife/EgoLifeCap/DenseCaption/translated"),
        help="Directory containing translated JSONL files.",
    )
    args = parser.parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    input_files = discover_translated_files(args.input_dir)
    if not input_files:
        raise FileNotFoundError(f"No JSONL files found in: {args.input_dir}")

    day_records = load_day_records(input_files)
    write_day_outputs(day_records)


if __name__ == "__main__":
    main()
