"""
Noise-injection experiment for R1W3 (quantitative robustness to hallucinated triples).

Workflow:
    1. Memorization already done: data/graphs/<video>_preabstraction.pkl exists
       (low-level edges + conversation-derived high-level edges + appearance edges +
       node embeddings, NO threshold-based abstraction edges).
    2. Inject noise: sample p% of the target graph's low-level edge count from OTHER
       videos' low-level edges, add them to the checkpoint graph (foreign entities:
       keep source character/object names, add missing nodes), assign a random valid
       clip_id and a random target scene, compute content + scene embeddings so the
       noise edges are retrievable by search.
    3. Re-run abstraction at 50/30 frequency (configs/abs_50_30.json).
    4. Save the injected graph + a manifest of injected edge ids (graph.noise_edge_ids)
       and a noise-log path (graph.noise_log_path) so the reasoning-time search
       instrumentation can count how often noise edges get retrieved.

No MLLM calls. Embeddings use the OpenAI text-embedding-3-small API (needs OPENAI_API_KEY).

Usage:
    python noise_injection.py study_03 --noise-rate 0.02 \
        --config configs/abs_50_30.json \
        --out data/ablation/graphs_noise/p2/study_03.pkl \
        --noise-log data/ablation/reasoning_noise/p2/study_03.noise.jsonl
"""
import argparse
import os
import pickle
import random
import sys
import traceback
from pathlib import Path

from classes.edge_class import Edge
from utils.abstraction_config import AbstractionConfig
from utils.embedding import get_embedding

# The 13 benchmark videos used in the ablation studies.
ALL_VIDEOS = [
    "bedroom_01", "bedroom_06", "kitchen_09", "kitchen_17",
    "living_room_02", "living_room_15", "living_room_22", "office_01",
    "study_03", "study_05", "study_06", "study_18", "study_23",
]

# Scenes that are NOT low-level (abstraction / appearance / conversation namespaces).
_NON_LOWLEVEL_SCENES = {"high-level", "appearance", "conversation", None}


def _is_character(name):
    return isinstance(name, str) and name.startswith("<") and name.endswith(">")


def _ensure_node(graph, name):
    """Ensure a node exists in the graph. Characters use add_character; objects use
    _get_or_create_object_node. Returns the normalized node name."""
    if name is None:
        return None
    if _is_character(name):
        return graph.add_character(name)
    # object (or bare string) -> create object node
    graph._get_or_create_object_node(name)
    return name


def _low_level_edges(graph):
    """Return the list of low-level edges (clip_id>0, scene not in abstraction ns)."""
    return [
        e for e in graph.edges.values()
        if e.clip_id > 0 and e.scene not in _NON_LOWLEVEL_SCENES
    ]


def build_noise_pool(pool_videos, pool_dir):
    """Load low-level edges from other videos' final graphs into a flat list."""
    pool = []
    for v in pool_videos:
        path = Path(pool_dir) / f"{v}.pkl"
        if not path.exists():
            print(f"  [pool] skip {v} (not found: {path})")
            continue
        with open(path, "rb") as f:
            g = pickle.load(f)
        ll = _low_level_edges(g)
        pool.extend(ll)
        print(f"  [pool] {v}: {len(ll)} low-level edges")
    return pool


def inject_noise(target_graph, pool_edges, noise_rate, rng):
    """Inject noise_rate * len(target_low_level) edges sampled from pool_edges.

    Foreign-entity policy: keep the source edge's source/target names as-is; create
    missing character/object nodes in the target graph. Assign a random valid clip_id
    (1..max_target_clip) and a random target scene so the noise edge integrates
    temporally and spatially. Compute content + scene embeddings for retrievability.

    Returns the list of injected edge ids.
    """
    target_ll = _low_level_edges(target_graph)
    n_inject = int(round(noise_rate * len(target_ll)))
    print(f"  target low-level edges: {len(target_ll)} -> injecting {n_inject} noise edges "
          f"({noise_rate*100:.1f}%)")

    if n_inject == 0:
        return []

    # Candidate target scenes (low-level scenes only) and max clip_id.
    target_scenes = list({
        e.scene for e in target_ll
        if e.scene and e.scene not in _NON_LOWLEVEL_SCENES
    })
    max_clip = max((e.clip_id for e in target_ll), default=1)

    # CRITICAL: avoid Edge id collision. The class-level _id_counter resets to 0 on
    # fresh import (class attrs are not pickled), so new Edge() would reuse ids 1,2,...
    # and overwrite existing edges in graph.edges. Reset it above the current max id.
    existing_ids = list(target_graph.edges.keys())
    Edge._id_counter = max(existing_ids) if existing_ids else 0

    sampled = rng.sample(pool_edges, min(n_inject, len(pool_edges)))
    injected_ids = []
    n_added = 0
    for src_edge in sampled:
        # ensure source node
        src_name = _ensure_node(target_graph, src_edge.source)
        # ensure target node (may be None for attribute-style edges; low-level usually has target)
        tgt_name = _ensure_node(target_graph, src_edge.target) if src_edge.target is not None else None

        clip_id = rng.randint(1, max_clip)
        scene = rng.choice(target_scenes) if target_scenes else src_edge.scene

        # compute embeddings so the noise edge is retrievable
        try:
            content_emb = get_embedding(str(src_edge.content))
        except Exception as e:
            print(f"    [warn] content embedding failed for '{src_edge.content}': {e}")
            content_emb = None
        try:
            scene_emb = get_embedding(str(scene))
        except Exception as e:
            print(f"    [warn] scene embedding failed for '{scene}': {e}")
            scene_emb = None

        edge = Edge(
            clip_id=clip_id,
            source=src_name,
            target=tgt_name,
            content=src_edge.content,
            scene=scene,
            confidence=src_edge.confidence,
            embedding=content_emb,
            scene_embedding=scene_emb,
        )
        try:
            target_graph.add_edge(edge)
            injected_ids.append(edge.id)
            n_added += 1
        except Exception as e:
            print(f"    [warn] add_edge failed for noise edge {src_edge}: {e}")
    print(f"  injected {n_added}/{n_inject} noise edges (ids: {injected_ids[:5]}{'...' if len(injected_ids)>5 else ''})")
    return injected_ids


def main():
    parser = argparse.ArgumentParser(
        description="Inject cross-video noise edges into a pre-abstraction checkpoint, "
                    "then re-run abstraction at 50/30 frequency."
    )
    parser.add_argument("video_name", help="Target video name (matches data/graphs/<name>_preabstraction.pkl)")
    parser.add_argument("--noise-rate", type=float, default=0.02,
                        help="Fraction of target low-level edge count to inject (default 0.02 = 2%%)")
    parser.add_argument("--config", default="configs/abs_50_30.json",
                        help="AbstractionConfig JSON (default 50/30 frequency)")
    parser.add_argument("--out", required=True, help="Output pkl path")
    parser.add_argument("--noise-log", required=True,
                        help="Path for the reasoning-time noise-retrieval log (jsonl)")
    parser.add_argument("--checkpoint-dir", default="data/graphs",
                        help="Dir with <name>_preabstraction.pkl (default data/graphs)")
    parser.add_argument("--pool-dir", default="data/graphs",
                        help="Dir with other videos' final <name>.pkl to sample noise from")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    checkpoint_path = Path(args.checkpoint_dir) / f"{args.video_name}_preabstraction.pkl"
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("  Run `python process_full_video.py <video>` first to generate it.")
        sys.exit(1)

    config = AbstractionConfig.from_json(args.config)
    print(f"Loaded AbstractionConfig: {config.to_json()}")
    print(f"Noise rate: {args.noise_rate*100:.1f}%  seed: {args.seed}")

    # Load target checkpoint.
    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        graph = pickle.load(f)
    n_ll_before = len(_low_level_edges(graph))
    n_hl_before = sum(1 for e in graph.edges.values() if e.clip_id == 0 and e.scene == "high-level")
    print(f"Checkpoint: {len(graph.characters)} chars, {len(graph.edges)} edges "
          f"(low-level={n_ll_before}, high-level={n_hl_before})")

    # Build noise pool from the OTHER videos.
    pool_videos = [v for v in ALL_VIDEOS if v != args.video_name]
    print(f"Building noise pool from {len(pool_videos)} other videos...")
    pool = build_noise_pool(pool_videos, args.pool_dir)
    print(f"Pool total: {len(pool)} low-level edges")

    # Inject.
    print("Injecting noise...")
    injected_ids = inject_noise(graph, pool, args.noise_rate, rng)

    # Embed any newly created nodes (foreign characters/objects) in batch.
    print("Embedding new nodes...")
    try:
        graph.node_embedding_insertion()
    except Exception as e:
        print(f"  [warn] node_embedding_insertion failed: {e}")

    # Re-run abstraction at 50/30 frequency.
    print("Running abstraction (50/30)...")
    try:
        abs_tokens = graph.run_abstraction(config)
    except Exception as e:
        print(f"✗ run_abstraction failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Re-insert embeddings for newly added high-level/appearance edges.
    try:
        graph.insert_high_level_and_appearance_embeddings()
    except Exception as e:
        print(f"  [warn] insert_high_level_and_appearance_embeddings failed: {e}")
        traceback.print_exc()

    n_hl_after = sum(1 for e in graph.edges.values() if e.clip_id == 0 and e.scene == "high-level")
    n_ll_after = len(_low_level_edges(graph))
    print(f"After abstraction: low-level={n_ll_after}, high-level={n_hl_after} "
          f"(high-level delta={n_hl_after - n_hl_before})")
    print(f"  abstraction tokens: attributes={abs_tokens.get('attributes_tokens', 0)}, "
          f"relationships={abs_tokens.get('relationships_tokens', 0)}")

    # Attach manifest + noise-log path so reasoning-time instrumentation can count
    # how often the injected edges get retrieved.
    graph.noise_edge_ids = set(injected_ids)
    graph.noise_log_path = str(args.noise_log)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"✓ Saved injected graph to {out_path} ({len(graph.edges)} edges, "
          f"{len(injected_ids)} noise edges)")

    # Also write the manifest alongside for offline analysis.
    manifest_path = out_path.with_suffix(".noise_manifest.json")
    import json
    with open(manifest_path, "w") as f:
        json.dump({
            "video": args.video_name,
            "noise_rate": args.noise_rate,
            "seed": args.seed,
            "n_injected": len(injected_ids),
            "noise_edge_ids": injected_ids,
            "n_low_level_before": n_ll_before,
            "n_low_level_after": n_ll_after,
            "n_high_level_before": n_hl_before,
            "n_high_level_after": n_hl_after,
        }, f, indent=2)
    print(f"✓ Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
