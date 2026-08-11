"""Compare clean 100/60 reasoning accuracy with 2%, 5%, and 10% noise."""

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTION_PATH = PROJECT_ROOT / "data/web_100.json"
CONFIGS = {
    "clean_100_60": PROJECT_ROOT / "data/ablation/reasoning_abs/100_60",
    "noise_2%": PROJECT_ROOT / "data/ablation/reasoning_noise/p2",
    "noise_5%": PROJECT_ROOT / "data/ablation/reasoning_noise/p5",
    "noise_10%": PROJECT_ROOT / "data/ablation/reasoning_noise/p10",
}


def load_result(path, expected):
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or len(data) != expected or expected == 0:
        return None
    if any(
        not isinstance(record, dict)
        or not isinstance(record.get("reasoning"), dict)
        or record["reasoning"].get("error")
        for record in data.values()
    ):
        return None
    return data


def main():
    questions = json.loads(QUESTION_PATH.read_text(encoding="utf-8"))
    videos = sys.argv[1:]
    if not videos:
        videos = sorted(
            path.stem
            for directory in list(CONFIGS.values())[1:]
            for path in directory.glob("*.json")
        )
        videos = sorted(set(videos))

    loaded = {name: {} for name in CONFIGS}
    for name, directory in CONFIGS.items():
        for video in videos:
            expected = len(questions.get(video, {}).get("qa_list", []))
            result = load_result(directory / f"{video}.json", expected)
            if result is not None:
                loaded[name][video] = result

    common_videos = set(videos)
    for results in loaded.values():
        common_videos &= set(results)
    common_videos = sorted(common_videos)

    print(f"Requested videos: {len(videos)}")
    for name in CONFIGS:
        missing = sorted(set(videos) - set(loaded[name]))
        print(f"{name}: valid={len(loaded[name])}/{len(videos)} missing={' '.join(missing)}")
    print(f"Common complete videos: {len(common_videos)}")
    if not common_videos:
        return 1

    rows = []
    for name in CONFIGS:
        records = [
            record
            for video in common_videos
            for record in loaded[name][video].values()
        ]
        correct = sum(bool(record["reasoning"].get("evaluate_correct")) for record in records)
        tokens = sum(
            (record["reasoning"].get("token_summaries") or {}).get("total", 0) or 0
            for record in records
        )
        rows.append((name, correct, len(records), tokens))

    clean_accuracy = rows[0][1] / rows[0][2] * 100
    print("\nconfig          correct     accuracy      delta       tokens")
    print("---------------------------------------------------------------")
    for name, correct, total, tokens in rows:
        accuracy = correct / total * 100
        print(
            f"{name:14s} {correct:3d}/{total:<3d}   {accuracy:8.2f}%   "
            f"{accuracy-clean_accuracy:+7.2f}pp   {tokens:10d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
