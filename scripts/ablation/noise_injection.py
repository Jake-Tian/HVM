"""Inject cross-video low-level triples, then rebuild 100/60 abstraction."""

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes.edge_class import Edge
from classes.ocr import OCR
from utils.abstraction_config import AbstractionConfig
from utils.embedding import get_multiple_embeddings


NON_LOW_LEVEL_SCENES = {"high-level", "appearance", "conversation", None}


def low_level_edges(graph):
    return [
        edge
        for edge in graph.edges.values()
        if edge.clip_id > 0 and edge.scene not in NON_LOW_LEVEL_SCENES
    ]


def ensure_node(graph, name):
    if name is None:
        return None
    if isinstance(name, str) and name.startswith("<") and name.endswith(">"):
        return graph.add_character(name)
    graph._get_or_create_object_node(name)
    return name


def discover_pool_videos(pool_dir, target_video):
    return sorted(
        path.stem
        for path in Path(pool_dir).glob("*.pkl")
        if not path.name.endswith("_preabstraction.pkl")
        and path.stem != target_video
        and path.stat().st_size > 0
    )


def build_noise_pool(pool_dir, pool_videos):
    edge_pool = []
    ocr_pool = []
    for video in pool_videos:
        path = Path(pool_dir) / f"{video}.pkl"
        with path.open("rb") as handle:
            graph = pickle.load(handle)
        edges = low_level_edges(graph)
        edge_pool.extend(
            (edge.source, edge.target, edge.content, edge.scene, edge.confidence)
            for edge in edges
        )
        ocr_pool.extend((ocr.context, ocr.content) for ocr in graph.ocr_info)
        print(
            f"[pool] {video}: {len(edges)} low-level edges, "
            f"{len(graph.ocr_info)} OCR records"
        )
    return edge_pool, ocr_pool


def inject_noise(graph, pool, noise_rate, rng):
    target_edges = low_level_edges(graph)
    requested = int(round(noise_rate * len(target_edges)))
    if requested == 0:
        return [], len(target_edges)
    if not pool:
        raise ValueError("noise pool is empty")

    sampled = rng.sample(pool, min(requested, len(pool)))
    target_scenes = sorted(
        {
            edge.scene
            for edge in target_edges
            if edge.scene not in NON_LOW_LEVEL_SCENES
        }
    )
    max_clip = max((edge.clip_id for edge in target_edges), default=1)
    planned = [
        (*record, rng.randint(1, max_clip), rng.choice(target_scenes) if target_scenes else record[3])
        for record in sampled
    ]

    contents = [str(record[2]) for record in planned]
    scenes = [str(record[6]) for record in planned]
    embeddings = get_multiple_embeddings(contents + scenes)
    content_embeddings = embeddings[: len(planned)]
    scene_embeddings = embeddings[len(planned) :]

    Edge._id_counter = max(graph.edges, default=0)
    injected_ids = []
    for record, content_embedding, scene_embedding in zip(
        planned, content_embeddings, scene_embeddings
    ):
        source, target, content, _source_scene, confidence, clip_id, scene = record
        edge = Edge(
            clip_id=clip_id,
            source=ensure_node(graph, source),
            target=ensure_node(graph, target),
            content=content,
            scene=scene,
            confidence=confidence,
            embedding=content_embedding,
            scene_embedding=scene_embedding,
        )
        graph.add_edge(edge)
        injected_ids.append(edge.id)

    print(
        f"target low-level edges={len(target_edges)}; "
        f"requested={requested}; injected={len(injected_ids)}"
    )
    return injected_ids, len(target_edges)


def inject_ocr_noise(graph, pool, noise_rate, rng):
    original_count = len(graph.ocr_info)
    requested = int(noise_rate * original_count)
    if requested == 0:
        return [], original_count
    if not pool:
        raise ValueError("OCR noise pool is empty")

    sampled = rng.sample(pool, min(requested, len(pool)))
    max_clip = max(
        [edge.clip_id for edge in low_level_edges(graph)]
        + [ocr.clip_id for ocr in graph.ocr_info]
        + [1]
    )
    injected = []
    for context, content in sampled:
        ocr = OCR(
            clip_id=rng.randint(1, max_clip),
            context=context,
            content=content,
        )
        graph.ocr_info.append(ocr)
        injected.append(
            {"clip_id": ocr.clip_id, "context": ocr.context, "content": ocr.content}
        )

    print(
        f"target OCR records={original_count}; "
        f"requested={requested}; injected={len(injected)}"
    )
    return injected, original_count


def main():
    parser = argparse.ArgumentParser(
        description="Inject cross-video noise into a pre-abstraction graph."
    )
    parser.add_argument("video_name")
    parser.add_argument("--noise-rate", required=True, type=float)
    parser.add_argument("--config", default="configs/abs_100_60.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint-dir", default="data/graphs")
    parser.add_argument("--pool-dir", default="data/graphs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.noise_rate < 1:
        parser.error("--noise-rate must be between 0 and 1")

    checkpoint = Path(args.checkpoint_dir) / f"{args.video_name}_preabstraction.pkl"
    if not checkpoint.is_file():
        parser.error(f"checkpoint not found: {checkpoint}")

    pool_videos = discover_pool_videos(args.pool_dir, args.video_name)
    if not pool_videos:
        parser.error(f"no other final graphs found in {args.pool_dir}")

    print(f"Loading checkpoint: {checkpoint}")
    with checkpoint.open("rb") as handle:
        graph = pickle.load(handle)

    print(f"Building noise pool from {len(pool_videos)} other videos")
    edge_pool, ocr_pool = build_noise_pool(args.pool_dir, pool_videos)
    rng = random.Random(args.seed)
    injected_ids, low_level_before = inject_noise(
        graph, edge_pool, args.noise_rate, rng
    )
    injected_ocr, ocr_before = inject_ocr_noise(
        graph, ocr_pool, args.noise_rate, rng
    )

    graph.node_embedding_insertion()
    config = AbstractionConfig.from_json(args.config)
    config_json = config.to_json()
    usage = graph.run_abstraction(config)
    graph.insert_high_level_and_appearance_embeddings()
    graph.ocr_embedding_insertion()
    graph.noise_edge_ids = set(injected_ids)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(graph, handle)

    manifest = {
        "video": args.video_name,
        "noise_rate": args.noise_rate,
        "seed": args.seed,
        "config": args.config,
        "config_json": config_json,
        "pool_videos": pool_videos,
        "n_low_level_before": low_level_before,
        "n_injected": len(injected_ids),
        "noise_edge_ids": injected_ids,
        "n_ocr_before": ocr_before,
        "n_ocr_injected": len(injected_ocr),
        "injected_ocr": injected_ocr,
        "abstraction_tokens": usage,
    }
    manifest_path = output.with_suffix(".noise_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved graph: {output}")
    print(f"Saved manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
