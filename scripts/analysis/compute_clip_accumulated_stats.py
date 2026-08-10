import argparse
import json
from pathlib import Path
from typing import Any


def is_character_node(node_name: str) -> bool:
    """Return True when node is a plain character token like <Lily>."""
    value = node_name.strip()
    return value.startswith("<") and value.endswith(">")


def collect_nodes_from_triple(triple: Any) -> list[str]:
    """
    Extract subject/object nodes from one triple.
    Triple format is expected to be [subject, predicate, object].
    """
    if not isinstance(triple, list) or len(triple) < 3:
        return []

    nodes: list[str] = []
    subject = triple[0]
    obj = triple[2]

    if isinstance(subject, str):
        s = subject.strip()
        if s:
            nodes.append(s)

    if isinstance(obj, str):
        o = obj.strip()
        if o:
            nodes.append(o)

    return nodes


def compute_accumulated_stats(memory_data: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}

    for video_name, clips in memory_data.items():
        if not isinstance(clips, dict):
            continue

        # Clip keys are "1", "2", ...; keep deterministic numeric order.
        sorted_clip_ids = sorted(clips.keys(), key=lambda k: int(k))

        cumulative_triples = 0
        cumulative_messages = 0
        unique_nodes: set[str] = set()
        unique_characters: set[str] = set()
        unique_objects: set[str] = set()

        per_clip: list[dict[str, Any]] = []

        for clip_id in sorted_clip_ids:
            clip_data = clips.get(clip_id, {})
            triples = clip_data.get("triples") or []
            conversation = clip_data.get("conversation") or []

            # 1) Cumulative triple count
            triples_count = len(triples) if isinstance(triples, list) else 0
            cumulative_triples += triples_count

            # 2) Cumulative conversation message count
            messages_count = len(conversation) if isinstance(conversation, list) else 0
            cumulative_messages += messages_count

            # 3) Cumulative unique nodes (character + object) from triples
            if isinstance(triples, list):
                for triple in triples:
                    for node in collect_nodes_from_triple(triple):
                        unique_nodes.add(node)
                        if is_character_node(node):
                            unique_characters.add(node)
                        else:
                            unique_objects.add(node)

            per_clip.append(
                {
                    "clip_id": int(clip_id),
                    "accumulated_triples": cumulative_triples,
                    "accumulated_conversation_messages": cumulative_messages,
                    "accumulated_unique_nodes": len(unique_nodes),
                    "accumulated_unique_characters": len(unique_characters),
                    "accumulated_unique_objects": len(unique_objects),
                }
            )

        output[video_name] = {
            "num_clips": len(per_clip),
            "clip_stats": per_clip,
        }

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute accumulated per-clip statistics from episodic memory JSON."
    )
    parser.add_argument(
        "--input",
        default="data/episodic_memory/episodic_memory.json",
        help="Path to episodic memory JSON input.",
    )
    parser.add_argument(
        "--output",
        default="data/episodic_memory/episodic_memory_accumulated_stats.json",
        help="Path to output JSON file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        memory_data = json.load(f)

    results = compute_accumulated_stats(memory_data)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved accumulated stats to: {output_path}")


if __name__ == "__main__":
    main()
