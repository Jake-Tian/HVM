import glob
import json
import pickle
import sys
import time
import traceback
from pathlib import Path
from classes.hetero_graph import HeteroGraph
from classes.output_structure import EpisodicFormat
from utils.llm import generate_text_response
from utils.mllm_gpt import generate_messages, get_response
from utils.prompts import prompt_generate_episodic_memory, prompt_extract_triples
from utils.general import Tee, strip_code_fences, load_video_list, merge_character_appearances


def process_full_video(video_name):

    frames_dir = Path(f"data/frames/{video_name}")
    output_graph_path = f"data/graphs/{video_name}.pkl"
    output_json_path = f"data/memorization/{video_name}.json"
    if not frames_dir.exists() or not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    
    # Get sorted image folders
    image_folders = sorted(
        [str(folder) for folder in frames_dir.iterdir() if folder.is_dir()],
        key=lambda x: int(Path(x).name)
    )
    if not image_folders:
        raise ValueError(f"No frame clip folders found in {frames_dir}")

    # image_folders = image_folders[:2] # Uncomment for quick debugging
    
    previous_conversation = False
    appearance_dict = dict()    # character name → [appearance description, embedding]
    episodic_memory = dict()
    graph = HeteroGraph()
    token_summaries = {"mllm": 0, "triples": 0, "attributes": 0, "relationships": 0, "conversation": 0}
    
    try: 
        for folder in image_folders:
            print("--------------------------------")
            print("Processing folder: ", folder)
            clip_id = int(Path(folder).name)
            # Collect images in the current folder
            current_images = sorted(
                glob.glob(f"{folder}/*.jpg"),
                key=lambda p: int(Path(p).stem) if Path(p).stem.isdigit() else p,
            ) 

            #--------------------------------
            # Episodic Memory
            #--------------------------------
            # Only pass appearance text to prompt (exclude embeddings).
            appearance_prompt_dict = {name: value[0] for name, value in appearance_dict.items()}
            prompt = "Character appearance from previous videos: \n" + json.dumps(appearance_prompt_dict) + "\n" + prompt_generate_episodic_memory
            messages = generate_messages(current_images, prompt)
            try:
                response, tokens = get_response(messages, EpisodicFormat)
                print(response)
                token_summaries["mllm"] += int(tokens or 0)
            except Exception as e:
                print(f"MLLM call failed, retrying... Error: {e}")
                try: 
                    response, tokens = get_response(messages)
                    print(response)
                    token_summaries["mllm"] += int(tokens or 0)
                except Exception as e:
                    print(f"MLLM call failed, retrying... Error: {e}")
                    traceback.print_exc()
                    continue

            if token_summaries["mllm"] > 7000000:
                print(f"MLLM token limit reached. Stop processing this video.")
                print("Prompt: \n", prompt)
                break

            try:
                behaviors = response.behaviors
                conversation = response.conversation
                characters_appearance = response.characters_appearance
                scene = getattr(response, 'scene', None)
                ocr = response.ocr
            except Exception as e:
                print(f"Error parsing response: {e}. Continuing to next clip...")
                traceback.print_exc()
                continue

            if ocr:
                graph.add_ocr_info(clip_id, ocr)

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

            # Extract summary before creating/updating conversation
            if previous_conversation and len(conversation) == 0 and graph.current_conversation_id is not None:
                try:
                    print(f"Extracting summary for completed conversation {graph.current_conversation_id}...")
                    result, tokens = graph.extract_conversation_summary(graph.current_conversation_id)
                    token_summaries["conversation"] += int(tokens or 0)
                    print(f"✓ Conversation summary extracted. Attributes: {len(result['character_attributes'])}, Relationships: {len(result['characters_relationships'])}")
                except Exception as e:
                    print(f"✗ Error extracting conversation summary: {e}")
                    traceback.print_exc()

            if len(conversation) > 0:
                graph.update_conversation(clip_id, conversation, previous_conversation=previous_conversation)
                previous_conversation = True  # Set to True for next iteration
            else:
                previous_conversation = False  # No conversation in this clip, reset for next iteration

            #--------------------------------
            # Graph Construction
            #--------------------------------
            if behaviors:
                behavior_prompt = prompt_extract_triples + "\n" + "\n".join(str(b) for b in behaviors)
                try:
                    triples_response, tokens = generate_text_response(behavior_prompt)
                    token_summaries["triples"] += int(tokens or 0)
                except Exception as e:
                    print(f"LLM call failed, retrying... Error: {e}")
                    triples_response, tokens = generate_text_response(behavior_prompt)
                    token_summaries["triples"] += int(tokens or 0)
                triples_response = strip_code_fences(triples_response)
                try:
                    triples = json.loads(triples_response)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse triples JSON for clip {clip_id}: {e}")
                    print(f"Raw triples response: {triples_response}")
                    triples = []
                except Exception as e:
                    print(f"Warning: Unexpected error while parsing triples for clip {clip_id}: {e}")
                    print(f"Raw triples response: {triples_response}")
                    triples = []

                if not isinstance(triples, list):
                    print(
                        f"Warning: Triples payload is not a list for clip {clip_id} "
                        f"(got {type(triples).__name__}), skipping triples."
                    )
                    triples = []
            else:
                triples = []
            
            # Pass character_appearance to insert_triples for matching and merging
            graph.insert_triples(triples, clip_id, scene)
            print(f"Inserted {len(triples)} triples into graph for clip {clip_id}")

            equivalence_list = merge_character_appearances(characters_appearance, appearance_dict)
            if equivalence_list:
                for equivalence in equivalence_list:
                    graph.rename_character(equivalence[0], equivalence[1])

            # Store episodic memory for this clip
            episodic_memory[clip_id] = {
                "folder": folder,
                "characters_behavior": behaviors,
                "conversation": conversation,
                "characters_appearance": str(characters_appearance),
                "scene": scene,
                "ocr": [ocr_item.model_dump() for ocr_item in ocr] if ocr else [],
                "triples": triples
            }

        # Extract summary for any remaining active conversation at the end
        if previous_conversation and graph.current_conversation_id is not None:
            try:
                print(f"Extracting summary for final conversation {graph.current_conversation_id}...")
                result, tokens = graph.extract_conversation_summary(graph.current_conversation_id)
                token_summaries["conversation"] += int(tokens or 0)
                print(f"✓ Final conversation summary extracted. Attributes: {len(result['character_attributes'])}, Relationships: {len(result['characters_relationships'])}")
            except Exception as e:
                print(f"✗ Error extracting final conversation summary: {e}")
                traceback.print_exc()

        # Insert character appearances
        print("Inserting character appearances...")
        print("Number of edges: ", len(graph.edges))
        try:
            appearance_text_dict = {name: value[0] for name, value in appearance_dict.items()}
            graph.insert_character_appearances(appearance_text_dict)
        except Exception as e:
            print(f"✗ Error inserting character appearances: {e}")
            traceback.print_exc()

        # --------------------------------
        # Abstract Memory
        # --------------------------------
        # Generate character attributes
        print("Generating character attributes...")
        print("Number of edges: ", len(graph.edges))
        degrees = graph.get_node_degrees()
        # Select all characters whose degree is greater than 10
        characters = [character for character in graph.characters if degrees.get(character, 0) > 10]

        for character in characters:
            try: 
                tokens = graph.character_attributes(character)
                token_summaries["attributes"] += int(tokens or 0)
            except Exception as e:
                print(f"✗ Error generating character attributes for {character}: {e}")
                traceback.print_exc()
                print("Continuing to next character...")
                continue
        print("Character attributes generated.")
        print("Number of edges: ", len(graph.edges))

        # Generate character relationships
        for i in range(len(characters)-1):
            for j in range(i+1, len(characters)):
                try:
                    tokens = graph.character_relationships(characters[i], characters[j])
                    token_summaries["relationships"] += int(tokens or 0)
                except Exception as e:
                    print(f"✗ Error generating character relationships for {characters[i]} and {characters[j]}: {e}")
                    traceback.print_exc()
                    print("Continuing to next character pair...")
                    continue
        print("Character relationships generated.")
        print("Number of edges: ", len(graph.edges))

        try: 
            graph.node_embedding_insertion()
            graph.insert_high_level_and_appearance_embeddings()
            graph.ocr_embedding_insertion()
        except Exception as e:
            print(f"✗ Error inserting embeddings: {e}")
            traceback.print_exc()
    except Exception as e:
        print(f"✗ Error during memorization for {video_name}: {e}")
        traceback.print_exc()

    # Save the graph to a file
    output_graph_path = Path(output_graph_path)
    output_graph_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_graph_path, "wb") as f:
        pickle.dump(graph, f)

    # Save the episodic memory and token summaries to a JSON file
    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    token_summaries["total"] = int(sum(v for v in token_summaries.values()))
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({"memory_token_summaries": token_summaries, "episodic_memory": episodic_memory}, f, indent=2)
    print(f"\n✓ Saved episodic memory and token summaries for {video_name} to {output_json_path}")
    return graph, episodic_memory, token_summaries


def main():
    # Example usage: python process_full_video.py meeting_room_03

    original_stdout = sys.stdout
    log_file = open("log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(log_file)
    
    if len(sys.argv) < 2: # If no video names are provided, process all videos
        video_names = load_video_list()
    else:
        video_names = sys.argv[1:]

    for video_name in video_names:
        try:
            start_time = time.time()
            print(f"\nProcessing {video_name}...")
            graph, episodic_memory, token_summaries = process_full_video(video_name)
            print(f"✓ {video_name} complete. Graph has {len(graph.characters)} characters and {len(graph.edges)} edges.")
            print(f"Token summaries: {token_summaries}")
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds")
        except Exception as e:
            print(f"✗ Error processing video {video_name}: {e}")
            traceback.print_exc()
            print("Continuing to next video...")
            continue
    
    sys.stdout = original_stdout
    log_file.close()

if __name__ == "__main__":
    main()

