"""Build one threshold-ablation graph from a pre-abstraction checkpoint."""

import argparse
import pickle
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.abstraction_config import AbstractionConfig


def main():
    parser = argparse.ArgumentParser(
        description="Re-run abstraction without repeating video memorization."
    )
    parser.add_argument("video_name")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint-dir", default="data/graphs")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint_dir) / f"{args.video_name}_preabstraction.pkl"
    if not checkpoint.exists():
        parser.error(f"checkpoint not found: {checkpoint}")

    config = AbstractionConfig.from_json(args.config)
    print(f"Loading checkpoint: {checkpoint}")
    print(f"Abstraction config:\n{config.to_json()}")
    with checkpoint.open("rb") as f:
        graph = pickle.load(f)

    try:
        usage = graph.run_abstraction(config)
        graph.insert_high_level_and_appearance_embeddings()
        # New checkpoints already contain OCR embeddings. This is a no-op for
        # them and keeps older checkpoints compatible.
        graph.ocr_embedding_insertion()
    except Exception as exc:
        print(f"✗ Failed to build abstraction graph: {exc}")
        traceback.print_exc()
        return 1

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(graph, f)

    print(
        f"✓ Saved {output} | edges={len(graph.edges)} | "
        f"attribute_tokens={usage.get('attributes_tokens', 0)} | "
        f"relationship_tokens={usage.get('relationships_tokens', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
