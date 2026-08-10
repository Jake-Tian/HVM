"""
Re-run threshold-based abstraction at a custom frequency without re-invoking the MLLM.

Workflow:
    1. Run memorization once:  python process_full_video.py <video>
       This produces data/graphs/<video>_preabstraction.pkl  (low-level edges +
       conversation-derived high-level edges + appearance edges + node
       embeddings, NO threshold-based abstraction edges) and the default
       data/graphs/<video>.pkl.
    2. For each ablation config, re-run abstraction from the checkpoint:
       python abstraction_ablation.py <video> --config configs/abs_frequent.json \
           --out data/graphs/<video>_frequent.pkl
    3. Run reasoning on each variant pkl.

The script loads the checkpoint, runs graph.run_abstraction(config), re-inserts
embeddings for the newly added high-level/appearance edges, and saves the result.
No MLLM calls, no triple re-extraction.
"""
import argparse
import os
import pickle
import sys
import traceback
from pathlib import Path

from utils.abstraction_config import AbstractionConfig


def main():
    parser = argparse.ArgumentParser(
        description="Re-run abstraction at a custom frequency from a pre-abstraction checkpoint."
    )
    parser.add_argument("video_name", help="Video name (matches data/graphs/<name>_preabstraction.pkl)")
    parser.add_argument("--config", required=True, help="Path to an AbstractionConfig JSON file")
    parser.add_argument(
        "--out", required=True,
        help="Output pkl path, e.g. data/graphs/<video>_frequent.pkl",
    )
    parser.add_argument(
        "--checkpoint-dir", default="data/graphs",
        help="Directory containing the _preabstraction.pkl checkpoint (default: data/graphs)",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_dir) / f"{args.video_name}_preabstraction.pkl"
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("  Run `python process_full_video.py <video>` first to generate it.")
        sys.exit(1)

    config = AbstractionConfig.from_json(args.config)
    print(f"Loaded AbstractionConfig: {config.to_json()}")

    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        graph = pickle.load(f)

    print(f"Checkpoint loaded: {len(graph.characters)} characters, {len(graph.edges)} edges")
    n_highlevel_before = sum(
        1 for e in graph.edges.values()
        if e.clip_id == 0 and e.scene == "high-level"
    )
    print(f"  high-level edges before: {n_highlevel_before}")

    try:
        abs_tokens = graph.run_abstraction(config)
    except Exception as e:
        print(f"✗ Error during run_abstraction: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Re-insert embeddings for the newly added high-level/appearance edges.
    # (Option A: re-embed all high-level+appearance edges; cheap since counts are small.)
    try:
        graph.insert_high_level_and_appearance_embeddings()
    except Exception as e:
        print(f"✗ Error inserting high-level/appearance embeddings: {e}")
        traceback.print_exc()

    n_highlevel_after = sum(
        1 for e in graph.edges.values()
        if e.clip_id == 0 and e.scene == "high-level"
    )
    print(f"  high-level edges after:  {n_highlevel_after} (+{n_highlevel_after - n_highlevel_before})")
    print(f"  abstraction tokens: attributes={abs_tokens.get('attributes_tokens', 0)}, "
          f"relationships={abs_tokens.get('relationships_tokens', 0)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"✓ Saved ablation graph to {out_path}")


if __name__ == "__main__":
    main()
