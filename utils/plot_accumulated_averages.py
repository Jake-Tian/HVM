import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def compute_clip_averages(stats_data: dict, max_clip: int = 73) -> dict[str, list[float]]:
    """
    Compute average accumulated metrics by clip index across videos.

    For each clip c in [1, max_clip], only videos that contain clip c are
    included in the average for that clip.
    """
    clip_to_triplets: dict[int, list[float]] = {i: [] for i in range(1, max_clip + 1)}
    clip_to_msgs: dict[int, list[float]] = {i: [] for i in range(1, max_clip + 1)}
    clip_to_nodes: dict[int, list[float]] = {i: [] for i in range(1, max_clip + 1)}

    for _, video_data in stats_data.items():
        clip_stats = video_data.get("clip_stats", [])
        if not isinstance(clip_stats, list):
            continue

        for row in clip_stats:
            if not isinstance(row, dict):
                continue
            clip_id = row.get("clip_id")
            if not isinstance(clip_id, int) or clip_id < 1 or clip_id > max_clip:
                continue

            t = row.get("accumulated_triples")
            m = row.get("accumulated_conversation_messages")
            n = row.get("accumulated_unique_nodes")

            if isinstance(t, (int, float)):
                clip_to_triplets[clip_id].append(float(t))
            if isinstance(m, (int, float)):
                clip_to_msgs[clip_id].append(float(m))
            if isinstance(n, (int, float)):
                clip_to_nodes[clip_id].append(float(n))

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    x = list(range(1, max_clip + 1))
    avg_triples = [avg(clip_to_triplets[c]) for c in x]
    avg_messages = [avg(clip_to_msgs[c]) for c in x]
    avg_nodes = [avg(clip_to_nodes[c]) for c in x]
    sample_size = [len(clip_to_triplets[c]) for c in x]

    return {
        "clip_ids": x,
        "avg_accumulated_triples": avg_triples,
        "avg_accumulated_conversation_messages": avg_messages,
        "avg_accumulated_unique_nodes": avg_nodes,
        "num_videos_used_per_clip": sample_size,
    }


def save_plot(averages: dict[str, list[float]], output_png: Path) -> None:
    x = averages["clip_ids"]
    y_triples = averages["avg_accumulated_triples"]
    y_msgs = averages["avg_accumulated_conversation_messages"]
    y_nodes = averages["avg_accumulated_unique_nodes"]

    plt.figure(figsize=(11, 6))
    plt.plot(x, y_triples, label="Avg accumulated triples", linewidth=2.2)
    plt.plot(x, y_msgs, label="Avg accumulated conversation messages", linewidth=2.2)
    plt.plot(x, y_nodes, label="Avg accumulated unique nodes", linewidth=2.2)

    plt.title("Average Accumulated Metrics Over Clips")
    plt.xlabel("Clip ID")
    plt.ylabel("Average Accumulated Count")
    plt.xlim(min(x), max(x))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read episodic_memory_accumulated_stats.json and plot average "
            "accumulated triples/conversation/nodes over clips."
        )
    )
    parser.add_argument(
        "--input",
        default="data/episodic_memory_accumulated_stats.json",
        help="Path to accumulated stats JSON.",
    )
    parser.add_argument(
        "--max-clip",
        type=int,
        default=73,
        help="Maximum clip id for x-axis (default: 73).",
    )
    parser.add_argument(
        "--output-plot",
        default="data/avg_accumulated_metrics_by_clip.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--output-json",
        default="data/avg_accumulated_metrics_by_clip.json",
        help="Output JSON path for averaged values.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/avg_accumulated_metrics_by_clip.csv",
        help="Output CSV path for averaged values.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_plot = Path(args.output_plot)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    with input_path.open("r", encoding="utf-8") as f:
        stats_data = json.load(f)

    averages = compute_clip_averages(stats_data, max_clip=args.max_clip)
    save_plot(averages, output_plot)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(averages, f, indent=2, ensure_ascii=False)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "clip_id",
                "avg_accumulated_triples",
                "avg_accumulated_conversation_messages",
                "avg_accumulated_unique_nodes",
                "num_videos_used",
            ]
        )
        for i, clip_id in enumerate(averages["clip_ids"]):
            writer.writerow(
                [
                    clip_id,
                    averages["avg_accumulated_triples"][i],
                    averages["avg_accumulated_conversation_messages"][i],
                    averages["avg_accumulated_unique_nodes"][i],
                    averages["num_videos_used_per_clip"][i],
                ]
            )

    print(f"Saved plot: {output_plot}")
    print(f"Saved averages JSON: {output_json}")
    print(f"Saved averages CSV: {output_csv}")


if __name__ == "__main__":
    main()
