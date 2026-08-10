"""Rebuild triples and graph memory from saved episodic-memory JSON.

This path reuses prior MLLM outputs and never reads video frames.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import tempfile
import time
from pathlib import Path


def parse_appearance_records(raw) -> list[dict[str, str]]:
    """Parse the saved ``str(list[Appearance])`` without using eval."""
    if raw in (None, "", []):
        return []
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, str):
        expression = ast.parse(raw, mode="eval").body
        if not isinstance(expression, ast.List):
            raise ValueError("characters_appearance must be a list")
        records = []
        for item in expression.elts:
            if not (
                isinstance(item, ast.Call)
                and isinstance(item.func, ast.Name)
                and item.func.id == "Appearance"
                and not item.args
            ):
                raise ValueError("unsupported characters_appearance entry")
            values = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in item.keywords
                if keyword.arg is not None
            }
            records.append(values)
    else:
        raise ValueError("characters_appearance has an unsupported type")

    parsed = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("characters_appearance entry must be an object")
        name = record.get("name")
        appearance = record.get("appearance")
        if not isinstance(name, str) or not isinstance(appearance, str):
            raise ValueError("appearance entries require string name and appearance")
        parsed.append({"name": name, "appearance": appearance})
    return parsed


def load_source(video_name: str) -> tuple[Path, dict, dict]:
    source_path = Path(f"data/memorization/{video_name}.json")
    if not source_path.is_file():
        raise FileNotFoundError(f"missing memorization file: {source_path}")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    episodic_memory = payload.get("episodic_memory")
    if not isinstance(episodic_memory, dict) or not episodic_memory:
        raise ValueError(f"invalid episodic_memory in {source_path}")
    for clip_id, clip in episodic_memory.items():
        int(clip_id)
        if not isinstance(clip, dict):
            raise ValueError(f"clip {clip_id} is not an object")
        if not isinstance(clip.get("characters_behavior", []), list):
            raise ValueError(f"clip {clip_id} has invalid characters_behavior")
        if not isinstance(clip.get("conversation", []), list):
            raise ValueError(f"clip {clip_id} has invalid conversation")
        parse_appearance_records(clip.get("characters_appearance"))
        if not isinstance(clip.get("scene"), str):
            raise ValueError(f"clip {clip_id} has invalid scene")
    return source_path, payload, episodic_memory


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_name = temp_file.name
            temp_file.write(content)
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def rebuild(video_name: str) -> dict:
    from classes.hetero_graph import HeteroGraph
    from classes.output_structure import Appearance, TripleExtraction
    from utils.abstraction_config import AbstractionConfig
    from utils.general import merge_character_appearances
    from utils.llm_gpt import (
        MODEL,
        generate_text_response,
    )
    from utils.prompts import prompt_extract_triples
    from utils.token_usage import add_stage_usage, build_token_summary

    source_path, payload, episodic_memory = load_source(video_name)
    graph = HeteroGraph()
    appearance_dict = {}
    previous_conversation = False
    stage_usage = {}
    prior_usage = payload.get("memory_token_summaries", {})
    if isinstance(prior_usage, dict) and prior_usage.get("mllm") is not None:
        add_stage_usage(stage_usage, "mllm", prior_usage["mllm"])

    print(
        f"[{video_name}] rebuilding {len(episodic_memory)} clips "
        f"with {MODEL}"
    )
    for clip_id_text in sorted(episodic_memory, key=lambda value: int(value)):
        clip_id = int(clip_id_text)
        clip = episodic_memory[clip_id_text]
        behaviors = clip.get("characters_behavior") or []
        conversation = clip.get("conversation") or []
        scene = clip["scene"]

        if (
            previous_conversation
            and not conversation
            and graph.current_conversation_id is not None
        ):
            _, tokens = graph.extract_conversation_summary(
                graph.current_conversation_id
            )
            add_stage_usage(stage_usage, "conversation", tokens)

        if conversation:
            graph.update_conversation(
                clip_id,
                conversation,
                previous_conversation=previous_conversation,
            )
            previous_conversation = True
        else:
            previous_conversation = False

        if behaviors:
            behavior_prompt = (
                prompt_extract_triples
                + "\n"
                + json.dumps(behaviors, ensure_ascii=False)
            )
            try:
                response, tokens = generate_text_response(
                    behavior_prompt,
                    text_format=TripleExtraction,
                )
            except Exception as error:
                print(
                    f"[{video_name}] clip {clip_id}: "
                    f"triple extraction failed, retrying once: {error}"
                )
                response, tokens = generate_text_response(
                    behavior_prompt,
                    text_format=TripleExtraction,
                )
            triples = [
                [triple.source, triple.content, triple.target]
                for triple in response.triples
            ]
            add_stage_usage(stage_usage, "triples", tokens)
        else:
            triples = []

        graph.insert_triples(triples, clip_id, scene)
        clip["triples"] = triples

        appearances = [
            Appearance(**record)
            for record in parse_appearance_records(
                clip.get("characters_appearance")
            )
        ]
        for old_name, new_name in merge_character_appearances(
            appearances,
            appearance_dict,
        ):
            graph.rename_character(old_name, new_name)

        print(
            f"[{video_name}] clip {clip_id}: "
            f"behaviors={len(behaviors)} triples={len(triples)}"
        )

    if previous_conversation and graph.current_conversation_id is not None:
        _, tokens = graph.extract_conversation_summary(
            graph.current_conversation_id
        )
        add_stage_usage(stage_usage, "conversation", tokens)

    graph.insert_character_appearances({
        name: value[0]
        for name, value in appearance_dict.items()
    })
    graph.node_embedding_insertion()
    preabstraction_bytes = pickle.dumps(
        graph,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    abstraction_usage = graph.run_abstraction(AbstractionConfig())
    add_stage_usage(
        stage_usage,
        "attributes",
        abstraction_usage.get(
            "attributes_usage",
            abstraction_usage.get("attributes_tokens", 0),
        ),
    )
    add_stage_usage(
        stage_usage,
        "relationships",
        abstraction_usage.get(
            "relationships_usage",
            abstraction_usage.get("relationships_tokens", 0),
        ),
    )
    graph.insert_high_level_and_appearance_embeddings()

    token_summaries = build_token_summary(stage_usage)
    output_payload = {
        "memory_token_summaries": token_summaries,
        "episodic_memory": episodic_memory,
    }
    final_graph_bytes = pickle.dumps(
        graph,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    json_bytes = json.dumps(
        output_payload,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    atomic_write(
        Path(f"data/graphs/{video_name}_preabstraction.pkl"),
        preabstraction_bytes,
    )
    atomic_write(
        Path(f"data/graphs/{video_name}.pkl"),
        final_graph_bytes,
    )
    atomic_write(source_path, json_bytes)

    return {
        "clips": len(episodic_memory),
        "edges": len(graph.edges),
        "characters": len(graph.characters),
        "tokens": token_summaries.get("total", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild triples and graphs without invoking the MLLM."
    )
    parser.add_argument("video_names", nargs="+")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate saved MLLM outputs without API calls or writes",
    )
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required")

    for video_name in args.video_names:
        started = time.time()
        _, _, episodic_memory = load_source(video_name)
        if args.dry_run:
            print(f"OK {video_name}: {len(episodic_memory)} reusable clips")
            continue
        result = rebuild(video_name)
        print(
            f"DONE {video_name}: clips={result['clips']} "
            f"edges={result['edges']} chars={result['characters']} "
            f"tokens={result['tokens']} elapsed={time.time() - started:.0f}s"
        )


if __name__ == "__main__":
    main()
