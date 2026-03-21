#!/usr/bin/env python3
"""
Extract TimeTriple structures from translated dense-caption JSONL files.

Example:
  python preprocessing/extract_timetriples.py \
    --input data/EgoLife/EgoLifeCap/DenseCaption/translated
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from classes.output_structure import TimeTripleList
from utils.llm import generate_text_response
from utils.prompts import prompt_extract_timetriples


CUSTOM_ID_PATTERN = re.compile(r".*-(\d{6})-(\d{6})$")
INTERVAL_MINUTES = 5


def parse_start_time(custom_id: str) -> tuple[str, int, int]:
    match = CUSTOM_ID_PATTERN.match(custom_id)
    if not match:
        raise ValueError(f"Unable to parse time range from custom_id: {custom_id}")
    start_hms_raw = match.group(1)
    hour = int(start_hms_raw[0:2])
    minute = int(start_hms_raw[2:4])
    return start_hms_raw, hour, minute


def load_translated_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_no, raw in enumerate(infile, start=1):
            line = raw.strip()
            if not line:
                continue
            record = json.loads(line)
            custom_id = record.get("custom_id")
            translated_text = record.get("translated_text", "")
            if not custom_id or not isinstance(translated_text, str):
                continue
            try:
                start_hhmmss, hour, minute = parse_start_time(custom_id)
            except ValueError:
                # Skip malformed entries without halting the whole file.
                continue
            rows.append(
                {
                    "custom_id": custom_id,
                    "text": translated_text.strip(),
                    "start_hhmmss": start_hhmmss,
                    "hour": hour,
                    "minute": minute,
                    "line_no": line_no,
                }
            )
    rows.sort(key=lambda x: (x["start_hhmmss"], x["line_no"]))
    return rows


def split_into_interval_dicts(rows: list[dict]) -> list[dict]:
    """
    Return a list of interval payloads.
    Each interval payload contains a dictionary:
      { "hhmmss": "translated_text", ... }
    """
    grouped: defaultdict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        interval_minute = (row["minute"] // INTERVAL_MINUTES) * INTERVAL_MINUTES
        grouped[(row["hour"], interval_minute)].append(row)

    interval_payloads: list[dict] = []
    for hour, interval_minute in sorted(grouped.keys()):
        entries = sorted(grouped[(hour, interval_minute)], key=lambda x: (x["start_hhmmss"], x["line_no"]))
        interval_dict: dict[str, str] = {}
        for entry in entries:
            key = entry["start_hhmmss"]
            value = entry["text"]
            if not value:
                continue
            if key in interval_dict and interval_dict[key]:
                # Keep one dictionary key per timestamp while preserving all text.
                interval_dict[key] = f"{interval_dict[key]} {value}".strip()
            else:
                interval_dict[key] = value
        interval_start = f"{hour:02d}{interval_minute:02d}00"
        interval_payloads.append(
            {
                "interval_start": interval_start,
                "events": interval_dict,
            }
        )
    return interval_payloads


def build_interval_prompt(events: dict[str, str]) -> str:
    return prompt_extract_timetriples + "\n" + json.dumps(events, ensure_ascii=False)


def extract_interval_triples(events: dict[str, str], retries: int = 1) -> tuple[list[dict], int]:
    prompt = build_interval_prompt(events)
    attempts = retries + 1
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            parsed, tokens = generate_text_response(prompt, TimeTripleList)
            triples = [{"time": item.time, "triple": item.triple} for item in parsed.triples]
            return triples, int(tokens or 0)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    raise RuntimeError(f"Failed to parse TimeTripleList after {attempts} attempts: {last_error}")


def process_file(input_path: Path, output_dir: Path) -> dict:
    rows = load_translated_jsonl(input_path)
    interval_payloads = split_into_interval_dicts(rows)

    total_tokens = 0
    output_intervals = []
    for interval in interval_payloads:
        events = interval["events"]
        if not events:
            continue
        triples, tokens = extract_interval_triples(events)
        total_tokens += tokens
        output_intervals.append(
            {
                "interval_start": interval["interval_start"],
                "events": events,
                "event_count": len(events),
                "token_usage": tokens,
                "triples": triples,
            }
        )

    result = {
        "source_file": str(input_path),
        "interval_minutes": INTERVAL_MINUTES,
        "total_lines": len(rows),
        "total_intervals": len(output_intervals),
        "total_token_usage": total_tokens,
        "intervals": output_intervals,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.timetriples.json"
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(result, outfile, ensure_ascii=False, indent=2)

    print(
        f"Processed {input_path.name}: {len(rows)} lines, "
        f"{len(output_intervals)} intervals, tokens={total_tokens} -> {output_path}"
    )
    return result


def discover_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract time-stamped triples from translated dense-caption JSONL files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/EgoLife/EgoLifeCap/DenseCaption/translated"),
        help="Input JSONL file or directory containing translated caption JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/EgoLife/EgoLifeCap/DenseCaption/translated_triples"),
        help="Directory to write extracted TimeTriple JSON outputs.",
    )
    args = parser.parse_args()

    files = discover_input_files(args.input)
    if not files:
        raise FileNotFoundError(f"No JSONL files found under: {args.input}")

    print(
        f"Found {len(files)} file(s). Extracting TimeTriples every "
        f"{INTERVAL_MINUTES} minute(s)."
    )
    for file_path in files:
        process_file(file_path, args.output_dir)


if __name__ == "__main__":
    main()
