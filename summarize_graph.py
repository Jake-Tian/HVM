import argparse
import pickle
from pathlib import Path


def format_edge(edge):
    target = edge.target if edge.target is not None else "null"
    confidence = getattr(edge, "confidence", None)
    if confidence is None:
        return f"[clip={edge.clip_id}] {edge.source} --{edge.content}--> {target}"
    return f"[clip={edge.clip_id}] {edge.source} --{edge.content}--> {target} (confidence={confidence})"


def main():
    parser = argparse.ArgumentParser(
        description="Print all high-level edges connected to all characters."
    )
    parser.add_argument(
        "video_id",
        nargs="?",
        default="_A9R3dlxh_o",
        help="Video ID used to locate data/graphs/<video_id>.pkl",
    )
    args = parser.parse_args()

    graph_path = Path(f"data/graphs/{args.video_id}.pkl")
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    print(f"Graph: {graph_path}")
    print(f"Characters: {len(graph.characters)}")
    print("=" * 80)

    for character_name in sorted(graph.characters.keys()):
        connected_edge_ids = graph.edges_of(character_name)
        high_level_edges = [
            graph.edges[edge_id]
            for edge_id in connected_edge_ids
            if edge_id in graph.edges
            and graph.edges[edge_id].clip_id == 0
            and graph.edges[edge_id].scene == "high-level"
        ]

        print(f"\n{character_name} ({len(high_level_edges)} high-level edges)")
        if not high_level_edges:
            print("  - None")
            continue

        for edge in sorted(high_level_edges, key=lambda e: (e.source, str(e.target), e.content, e.id)):
            print(f"  - {format_edge(edge)}")


if __name__ == "__main__":
    main()