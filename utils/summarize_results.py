#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "correct"}
    return False


def _extract_video_name(qid: str, item: Dict[str, Any]) -> str:
    for key in ("video_name", "video", "video_id"):
        if isinstance(item.get(key), str) and item[key].strip():
            return item[key].strip()
    match = re.match(r"^(.*)_Q\d+$", qid)
    if match:
        return match.group(1)
    return "unknown"


def _get_types(item: Dict[str, Any]) -> List[str]:
    types_val = item.get("type")
    if isinstance(types_val, list):
        return [t for t in types_val if isinstance(t, str) and t.strip()]
    if isinstance(types_val, str) and types_val.strip():
        return [types_val.strip()]
    return []


def _count_video_watches(item: Dict[str, Any]) -> int:
    outputs = item.get("video_answer_outputs")
    if not isinstance(outputs, list):
        return 0
    return len(outputs)


def _is_search(item: Dict[str, Any]) -> bool:
    semantic = item.get("semantic_video_output")
    if isinstance(semantic, str):
        if re.search(r"Action:\s*\[Search\]", semantic, flags=re.IGNORECASE):
            return True
    return False


def summarize_results(results: Dict[str, Any]) -> Dict[str, Any]:
    total = 0
    correct = 0
    total_search = 0
    total_video_watch_events = 0
    type_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    video_stats = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "search": 0,
        "video_watch_events": 0,
    })

    for qid, item in results.items():
        if not isinstance(item, dict):
            continue
        total += 1
        is_correct = _safe_bool(item.get("evaluator_correct"))
        if is_correct:
            correct += 1

        video_name = _extract_video_name(qid, item)
        video_stats[video_name]["total"] += 1
        if is_correct:
            video_stats[video_name]["correct"] += 1

        if _is_search(item):
            total_search += 1
            video_stats[video_name]["search"] += 1

        watch_events = _count_video_watches(item)
        total_video_watch_events += watch_events
        video_stats[video_name]["video_watch_events"] += watch_events

        for t in _get_types(item):
            type_stats[t]["total"] += 1
            if is_correct:
                type_stats[t]["correct"] += 1

    def accuracy(c: int, n: int) -> float:
        return round((c / n) * 100, 2) if n else 0.0

    summary = {
        "overall": {
            "total": total,
            "correct": correct,
            "accuracy_percent": accuracy(correct, total),
            "search_count": total_search,
        },
        "video_watch": {
            "total_watch_events": total_video_watch_events,
            "avg_watch_events_per_question": round(total_video_watch_events / total, 2) if total else 0.0,
        },
        "by_type": {},
        "by_video": {},
    }

    for t, stats in sorted(type_stats.items()):
        summary["by_type"][t] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy_percent": accuracy(stats["correct"], stats["total"]),
        }

    for video_name, stats in sorted(video_stats.items()):
        summary["by_video"][video_name] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy_percent": accuracy(stats["correct"], stats["total"]),
            "search_count": stats["search"],
            "video_watch_events": stats["video_watch_events"],
        }

    return summary


def _print_summary(summary: Dict[str, Any]) -> None:
    overall = summary["overall"]
    watch = summary["video_watch"]
    print("Overall correctness:")
    print(f"  Total questions: {overall['total']}")
    print(f"  Correct: {overall['correct']}")
    print(f"  Accuracy: {overall['accuracy_percent']}%")
    print(f"  Search actions: {overall['search_count']}")
    print("")
    print("Video watch stats (from video_answer_outputs):")
    print(f"  Watch events: {watch['total_watch_events']}")
    print(f"  Avg watch events/question: {watch['avg_watch_events_per_question']}")
    print("")
    print("Breakdown by question type:")
    for t, stats in summary["by_type"].items():
        print(f"  {t}: {stats['correct']}/{stats['total']} ({stats['accuracy_percent']}%)")
    print("")
    print("Correctness by video:")
    header = ["video", "correct/total", "accuracy", "search", "watches"]
    rows = []
    for v, stats in summary["by_video"].items():
        rows.append([
            v,
            f"{stats['correct']}/{stats['total']}",
            f"{stats['accuracy_percent']}%",
            str(stats["search_count"]),
            str(stats["video_watch_events"]),
        ])
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    header_line = "  " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header))
    sep_line = "  " + "-+-".join("-" * widths[i] for i in range(len(widths)))
    print(header_line)
    print(sep_line)
    for row in rows:
        print("  " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))))


def _load_question_types(questions_path: str) -> Dict[str, List[str]]:
    """Load robot.json and build question_id -> list of type strings."""
    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    qid_to_types: Dict[str, List[str]] = {}
    for video_id, video_data in data.items():
        if not isinstance(video_data, dict):
            continue
        qa_list = video_data.get("qa_list")
        if not isinstance(qa_list, list):
            continue
        for qa in qa_list:
            if not isinstance(qa, dict):
                continue
            qid = qa.get("question_id")
            if not qid:
                continue
            types_val = qa.get("type")
            if isinstance(types_val, list):
                types = [t for t in types_val if isinstance(t, str) and t.strip()]
            elif isinstance(types_val, str) and types_val.strip():
                types = [types_val.strip()]
            else:
                types = []
            qid_to_types[qid] = types
    return qid_to_types


def summarize_by_category(
    results: Dict[str, Any],
    qid_to_types: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Compute accuracy breakdown by question category from robot.json types."""
    type_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    matched = 0
    unmatched = 0

    for qid, item in results.items():
        if not isinstance(item, dict):
            continue
        is_correct = _safe_bool(item.get("evaluator_correct"))
        types = qid_to_types.get(qid, [])
        if not types:
            unmatched += 1
            continue
        matched += 1
        for t in types:
            type_stats[t]["total"] += 1
            if is_correct:
                type_stats[t]["correct"] += 1

    def accuracy(c: int, n: int) -> float:
        return round((c / n) * 100, 2) if n else 0.0

    by_category = {}
    for t in sorted(type_stats.keys()):
        stats = type_stats[t]
        by_category[t] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy_percent": accuracy(stats["correct"], stats["total"]),
        }

    return {
        "by_category": by_category,
        "matched_questions": matched,
        "unmatched_questions": unmatched,
    }


def _print_category_breakdown(category_summary: Dict[str, Any]) -> None:
    print("Accuracy breakdown by question category (from robot.json):")
    print("")
    by_cat = category_summary["by_category"]
    if not by_cat:
        print("  (no categories found)")
        return
    # Sort by category name for consistent output
    for cat, stats in sorted(by_cat.items()):
        print(f"  {cat}: {stats['correct']}/{stats['total']} ({stats['accuracy_percent']}%)")
    print("")
    print(f"  Matched questions: {category_summary['matched_questions']}")
    print(f"  Unmatched (no type in robot.json): {category_summary['unmatched_questions']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize results.json correctness and watch stats.")
    parser.add_argument(
        "--input",
        default="data/results/result_egolife.json",
        help="Path to results.json (default: data/results/result_egolife.json)",
    )
    parser.add_argument(
        "--questions",
        default="data/questions/robot.json",
        help="Path to robot.json for category breakdown (default: data/questions/robot.json)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write summary as JSON",
    )
    parser.add_argument(
        "--categories-only",
        action="store_true",
        help="Only print accuracy by question category (from robot.json)",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    if args.categories_only:
        qid_to_types = _load_question_types(args.questions)
        category_summary = summarize_by_category(results, qid_to_types)
        _print_category_breakdown(category_summary)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(category_summary, f, ensure_ascii=False, indent=2)
    else:
        summary = summarize_results(results)
        _print_summary(summary)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        # Also print category breakdown if questions file exists
        try:
            qid_to_types = _load_question_types(args.questions)
            category_summary = summarize_by_category(results, qid_to_types)
            print("")
            _print_category_breakdown(category_summary)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
