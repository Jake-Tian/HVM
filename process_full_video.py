import glob
import json
import pickle
import sys
import time
import traceback
from pathlib import Path
from tqdm import tqdm
from classes.hetero_graph import HeteroGraph
from classes.output_structure import EpisodicFormat, TripleExtraction
from utils.llm_gpt import generate_text_response
from utils.mllm_gpt import generate_messages, get_response
from utils.prompts import prompt_generate_episodic_memory, prompt_extract_triples
from utils.general import (
    Tee, QuietStdout, verbose_terminal,
    load_video_list, merge_character_appearances,
)
from utils.token_usage import (
    add_stage_usage,
    build_token_summary,
    usage_total,
)


def extract_behavior_triples(behaviors):
    behavior_prompt = (
        prompt_extract_triples
        + "\n"
        + json.dumps(behaviors, ensure_ascii=False)
    )
    response, tokens = generate_text_response(
        behavior_prompt,
        text_format=TripleExtraction,
    )
    triples = [
        [triple.source, triple.content, triple.target]
        for triple in response.triples
    ]
    return triples, tokens


def process_full_video(video_name):

    frames_dir = Path(f"data/frames/{video_name}")
    output_graph_path = f"data/graphs/{video_name}.pkl"
    output_json_path = f"data/memorization/{video_name}.json"
    if not frames_dir.exists() or not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    
    image_folders = sorted(
        [str(folder) for folder in frames_dir.iterdir() if folder.is_dir()],
        key=lambda x: int(Path(x).name)
    )
    if not image_folders:
        raise ValueError(f"No frame clip folders found in {frames_dir}")

    # tqdm stays visible because logs redirect stdout, not stderr.
    pbar = tqdm(total=len(image_folders) + 3, desc=f"memo {video_name}", file=sys.stderr)
    
    previous_conversation = False
    appearance_dict = dict()    # character name → [appearance description, embedding]
    episodic_memory = dict()
    graph = HeteroGraph()
    stage_usage = {}
    
    try: 
        for folder in image_folders:
            print("--------------------------------")
            print("Processing folder: ", folder)
            clip_id = int(Path(folder).name)
            current_images = sorted(
                glob.glob(f"{folder}/*.jpg"),
                key=lambda p: int(Path(p).stem) if Path(p).stem.isdigit() else p,
            ) 

            #--------------------------------
            # Episodic Memory
            #--------------------------------
            # Exclude embeddings from the appearance prompt.
            appearance_prompt_dict = {name: value[0] for name, value in appearance_dict.items()}
            prompt = "Character appearance from previous videos: \n" + json.dumps(appearance_prompt_dict) + "\n" + prompt_generate_episodic_memory
            messages = generate_messages(current_images, prompt)
            try:
                response, tokens = get_response(messages, EpisodicFormat)
                print("MLLM tokens: ", tokens)
                add_stage_usage(stage_usage, "mllm", tokens)
            except Exception as e:
                print(f"MLLM call failed, retrying... Error: {e}")
                try:
                    response, tokens = get_response(messages, EpisodicFormat)
                    add_stage_usage(stage_usage, "mllm", tokens)
                except Exception as e:
                    print(f"MLLM call failed, proceeding with empty response... Error: {e}")
                    traceback.print_exc()
                    continue

            if usage_total(stage_usage.get("mllm")) > 7000000:
                print(f"MLLM token limit reached. Stop processing this video.")
                print("Prompt: \n", prompt)
                break

            try:
                behaviors = response.behaviors
                conversation = response.conversation
                characters_appearance = response.characters_appearance
                scene = response.scene
            except Exception as e:
                print(f"Error parsing response: {e}. Continuing to next clip...")
                traceback.print_exc()
                continue

            if behaviors and len(behaviors) > 0 and behaviors[0].startswith("Equivalence:"):
                equivalence_parts = behaviors[0].split(":")[1].split(",")
                print(equivalence_parts)
                if len(equivalence_parts) >= 2:
                    behaviors = behaviors[1:]
                    old_name = equivalence_parts[0].strip()
                    new_name = equivalence_parts[1].strip()
                    graph.rename_character(old_name, new_name)
                    if old_name in appearance_dict:
                        appearance_dict[new_name] = appearance_dict[old_name]
                        del appearance_dict[old_name]
                else:
                    print(f"Warning: Malformed equivalence line '{behaviors[0]}', skipping rename")

            if previous_conversation and len(conversation) == 0 and graph.current_conversation_id is not None:
                try:
                    print(f"Extracting summary for completed conversation {graph.current_conversation_id}...")
                    result, tokens = graph.extract_conversation_summary(graph.current_conversation_id)
                    add_stage_usage(stage_usage, "conversation", tokens)
                    print(f"✓ Conversation summary extracted. Attributes: {len(result['character_attributes'])}, Relationships: {len(result['characters_relationships'])}")
                except Exception as e:
                    print(f"✗ Error extracting conversation summary: {e}")
                    traceback.print_exc()

            if len(conversation) > 0:
                graph.update_conversation(clip_id, conversation, previous_conversation=previous_conversation)
                previous_conversation = True
            else:
                previous_conversation = False

            #--------------------------------
            # Graph Construction
            #--------------------------------
            if behaviors:
                try:
                    triples, tokens = extract_behavior_triples(behaviors)
                    add_stage_usage(stage_usage, "triples", tokens)
                except Exception as e:
                    print(f"LLM call failed, retrying... Error: {e}")
                    triples, tokens = extract_behavior_triples(behaviors)
                    add_stage_usage(stage_usage, "triples", tokens)
            else:
                triples = []
            
            graph.insert_triples(triples, clip_id, scene)
            print(f"Inserted {len(triples)} triples into graph for clip {clip_id}")

            equivalence_list = merge_character_appearances(characters_appearance, appearance_dict)
            if equivalence_list:
                for equivalence in equivalence_list:
                    graph.rename_character(equivalence[0], equivalence[1])

            episodic_memory[clip_id] = {
                "folder": folder,
                "characters_behavior": behaviors,
                "conversation": conversation,
                "characters_appearance": str(characters_appearance),
                "scene": scene,
                "triples": triples
            }
            pbar.update(1)

        if previous_conversation and graph.current_conversation_id is not None:
            try:
                print(f"Extracting summary for final conversation {graph.current_conversation_id}...")
                result, tokens = graph.extract_conversation_summary(graph.current_conversation_id)
                add_stage_usage(stage_usage, "conversation", tokens)
                print(f"✓ Final conversation summary extracted. Attributes: {len(result['character_attributes'])}, Relationships: {len(result['characters_relationships'])}")
            except Exception as e:
                print(f"✗ Error extracting final conversation summary: {e}")
                traceback.print_exc()

        print("Inserting character appearances...")
        print("Number of edges: ", len(graph.edges))
        try:
            appearance_text_dict = {name: value[0] for name, value in appearance_dict.items()}
            graph.insert_character_appearances(appearance_text_dict)
        except Exception as e:
            print(f"✗ Error inserting character appearances: {e}")
            traceback.print_exc()

        # Complete node embeddings before saving the reusable checkpoint.
        try:
            graph.node_embedding_insertion()
        except Exception as e:
            print(f"✗ Error inserting node embeddings: {e}")
            traceback.print_exc()

        pbar.set_postfix_str("appearances")
        pbar.update(1)

        # --------------------------------
        # Pre-abstraction checkpoint
        # --------------------------------
        # This checkpoint lets ablations rerun abstraction without the MLLM.
        try:
            checkpoint_path = Path(f"data/graphs/{video_name}_preabstraction.pkl")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "wb") as f:
                pickle.dump(graph, f)
            print(f"✓ Saved pre-abstraction checkpoint to {checkpoint_path}")
        except Exception as e:
            print(f"✗ Error saving pre-abstraction checkpoint: {e}")
            traceback.print_exc()

        # --------------------------------
        # Abstract Memory (incremental + final)
        # --------------------------------
        from utils.abstraction_config import AbstractionConfig
        print("Running threshold-based abstraction...")
        print("Number of edges before abstraction: ", len(graph.edges))
        try:
            abs_tokens = graph.run_abstraction(AbstractionConfig())
            add_stage_usage(
                stage_usage,
                "attributes",
                abs_tokens.get(
                    "attributes_usage",
                    abs_tokens.get("attributes_tokens", 0),
                ),
            )
            add_stage_usage(
                stage_usage,
                "relationships",
                abs_tokens.get(
                    "relationships_usage",
                    abs_tokens.get("relationships_tokens", 0),
                ),
            )
        except Exception as e:
            print(f"✗ Error during run_abstraction: {e}")
            traceback.print_exc()
        print("Abstraction complete.")
        print("Number of edges: ", len(graph.edges))

        pbar.set_postfix_str("abstraction")
        pbar.update(1)

        try:
            graph.insert_high_level_and_appearance_embeddings()
        except Exception as e:
            print(f"✗ Error inserting embeddings: {e}")
            traceback.print_exc()

        pbar.set_postfix_str("save")
        pbar.update(1)
    except Exception as e:
        print(f"✗ Error during memorization for {video_name}: {e}")
        traceback.print_exc()
    finally:
        pbar.close()

    output_graph_path = Path(output_graph_path)
    output_graph_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_graph_path, "wb") as f:
        pickle.dump(graph, f)

    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    token_summaries = build_token_summary(stage_usage)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({"memory_token_summaries": token_summaries, "episodic_memory": episodic_memory}, f, indent=2)
    print(f"\n✓ Saved episodic memory and token summaries for {video_name} to {output_json_path}")
    return graph, episodic_memory, token_summaries


def main():
    real_stdout = sys.stdout
    Path("data/logs").mkdir(parents=True, exist_ok=True)

    if len(sys.argv) < 2: # If no video names are provided, process all videos
        video_names = load_video_list()
    else:
        video_names = sys.argv[1:]

    for video_name in video_names:
        log_path = Path(f"data/logs/{video_name}_memo.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")
        sys.stdout = Tee(log_file) if verbose_terminal() else QuietStdout(log_file)
        start_time = time.time()
        try:
            graph, episodic_memory, token_summaries = process_full_video(video_name)
        except Exception as e:
            sys.stdout = real_stdout
            log_file.close()
            print(f"✗ [{video_name}] memo failed: {e}")
            traceback.print_exc()
            continue
        elapsed = time.time() - start_time
        sys.stdout = real_stdout
        log_file.close()
        total_tokens = token_summaries.get("total", 0)
        print(
            f"✓ [{video_name}] memo {elapsed:.0f}s | "
            f"clips={len(episodic_memory)} edges={len(graph.edges)} chars={len(graph.characters)} | "
            f"tokens={total_tokens}"
        )

if __name__ == "__main__":
    main()
